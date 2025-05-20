from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from utilities.preprocessing import pick_vessel, drop_class_b, filter_relevant_columns, drop_duplicates, resample_to_fixed_interval, normalize_columns
from utilities.data_validation import missing_data_check
from utilities.sliding_window import sliding_windows
from utilities.inference import AISInferenceModel
from utilities.zone_check import load_cable_lines, any_forecast_in_zone, build_buffered_zone, forecast_path_crosses_zone, vessel_near_any_cable
from shapely.geometry import Point
import numpy as np
import yaml
import json
import pandas as pd
import io
import uvicorn
import os

with open("utilities/inference_config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_path = config["data_path"]
window_size = config["window_size"]
step_size = config["step_size"]
cable_path = config["cable_coordinates_path"]
classifier_path = config["model_paths"]["classifier"]
forecaster_path = config["model_paths"]["forecaster"]

script_dir = os.path.dirname(os.path.abspath(__file__))
stats_path = os.path.join(script_dir, "./data/train_norm_stats.json")

with open(stats_path, "r") as f:
    norm_stats = json.load(f)

model = AISInferenceModel(classifier_path=classifier_path, forecaster_path=forecaster_path, verbose=False)
cable_lines = load_cable_lines(cable_path)

buffered_zone_1602 = build_buffered_zone(cable_lines, buffer_meters=1602)
buffered_zone_2136 = build_buffered_zone(cable_lines, buffer_meters=2136)


def denormalize_column(values, col_name):
    mean = norm_stats[col_name]["mean"]
    std = norm_stats[col_name]["std"]
    return [(v * std + mean) for v in values]

def convert(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

app = FastAPI()

class AISDataPoint(BaseModel):
    Latitude: float
    Longitude: float
    ROT: float
    SOG: float
    COG: float
    Heading: float
    Width: float
    Length: float
    Draught: float

def preprocess_input(data: List[AISDataPoint]) -> np.ndarray:
    features = [
        [
            point.Latitude,
            point.Longitude,
            point.ROT,
            point.SOG,
            point.COG,
            point.Heading,
            point.Width,
            point.Length,
            point.Draught
        ]
        for point in data
    ]
    array = np.array(features, dtype=np.float32)
    return array.reshape(1, len(data), 9)

@app.get("/")
async def root():
    return {"message": "AIS Forecasting Service is running."}

@app.post("/predict")
async def predict(data: List[AISDataPoint]):
    input_tensor = preprocess_input(data)
    label, probability, logit, forecast = model.predict(input_tensor)

    response = {
        "logit": float(logit),
        "probability": float(probability),
        "label": label
    }

    if forecast is not None:
        response["forecast"] = forecast.tolist()
        response["zone_alert"] = any_forecast_in_zone(forecast, buffered_zone_1602)

    return response

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...), mmsi: int = Form(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    df = pick_vessel(df, mmsi)
    df_raw = df.copy()
    df_raw.to_csv("data/last_input.csv", index=False)
    df = drop_class_b(df)
    df = filter_relevant_columns(df)
    df = drop_duplicates(df)
    df = resample_to_fixed_interval(df)

    df = normalize_columns(df, stats_path=stats_path, exclude=["trawling"])
    df = df.drop(["# Timestamp", "MMSI", "trawling"], axis=1)

    results = []

    for window in sliding_windows(df, window_size, step_size):

        if missing_data_check(window, window_size):
            input_tensor = window[[
                "Latitude", "Longitude", "ROT", "SOG", "COG",
                "Heading", "Width", "Length", "Draught"
            ]].values.astype(np.float32).reshape(1, window_size, 9)

            label, probability, logit, forecast = model.predict(input_tensor)
            if forecast is not None:
                start_lat, start_lon = forecast[0]
                start_point = Point(start_lon, start_lat)

                crosses_critical = forecast_path_crosses_zone(forecast, buffered_zone_1602)
                inside_entry_zone = buffered_zone_2136.contains(start_point)

                # Risk level logic
                if label == 1:  # Trawling
                    if crosses_critical:
                        risk_level = 3
                    elif inside_entry_zone:
                        risk_level = 2
                    else:
                        risk_level = 1


                lat_norm = window["Latitude"].values
                lon_norm = window["Longitude"].values
                lat = denormalize_column(lat_norm, "Latitude")
                lon = denormalize_column(lon_norm, "Longitude")
                input_coords = list(zip(lat, lon))

                results.append({
                    "forecast": forecast,
                    "fishing_confidence": round(probability, 4),
                    "risk_level": risk_level,
                    "input": input_coords
                })


    # Save results to JSON file for visualization
    results_path = os.path.join(script_dir, "data/results.json")

    with open(results_path, "w") as f:
        json.dump({"results": results}, f, indent=2, default = convert)
    for result in results:
        if isinstance(result.get("fishing_confidence"), np.floating):
            result["fishing_confidence"] = float(result["fishing_confidence"])
    print(f"Saved prediction results to {results_path}")

    # Return results in API response
    return json.loads(json.dumps({"results": results}, default=convert))

@app.get("/map")
def get_map():
    map_path = os.path.join(script_dir, "forecast_map.html")

    try:
        # Dynamically generate map
        import subprocess
        subprocess.run(["python3", "visualize_forecast.py"], check=True, cwd=script_dir)
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to generate map: {str(e)}"}

    if not os.path.exists(map_path):
        return {"error": f"Map file not found at {map_path}"}

    return FileResponse(map_path, media_type="text/html")

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
