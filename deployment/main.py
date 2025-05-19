from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from utilities.preprocessing import pick_vessel, drop_class_b, filter_relevant_columns, drop_duplicates, resample_to_fixed_interval, normalize_columns
from utilities.data_validation import missing_data_check
from utilities.sliding_window import sliding_windows
from utilities.inference import AISInferenceModel
from utilities.zone_check import load_cable_lines, any_forecast_in_zone, build_buffered_zone, all_forecast_steps_in_zone, vessel_near_any_cable
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

model = AISInferenceModel(classifier_path=classifier_path, forecaster_path=forecaster_path, verbose=False)
cable_lines = load_cable_lines(cable_path)
buffered_zone = build_buffered_zone(cable_lines)

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
        response["zone_alert"] = any_forecast_in_zone(forecast, buffered_zone)

    return response

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...), mmsi: int = Form(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    df = pick_vessel(df, mmsi)
    df_raw = df.copy()
    df_raw.to_csv("data/last_input.csv", index=False)
    latest_row = df.sort_values("# Timestamp").dropna(subset=["Latitude", "Longitude"]).iloc[-1]
    curr_lat = latest_row["Latitude"]
    curr_lon = latest_row["Longitude"]
    if not vessel_near_any_cable(curr_lat, curr_lon, cable_lines, radius_m=100000):
        return {"message": "Vessel is outside of risk zone, no prediction needed."}
    df = drop_class_b(df)
    df = filter_relevant_columns(df)
    df = drop_duplicates(df)
    df = resample_to_fixed_interval(df)

    print(f"\n--- DEBUG FOR MMSI {mmsi} ---")
    print(f"Data points after preprocessing: {len(df)}")
    print(f"Window size: {window_size}")

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
                zone_steps = all_forecast_steps_in_zone(forecast, buffered_zone)

                results.append({
                    "forecast": forecast.tolist(),
                    "zone_alert": len(zone_steps) > 0,
                    "fishing_confidence": round(probability, 4),
                    "zone_entry_step": zone_steps[0] if zone_steps else None,
                    "zone_steps": zone_steps
                })

    # Save results to JSON file for visualization
    results_path = os.path.join(script_dir, "data/results.json")

    with open(results_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"Saved prediction results to {results_path}")

    # Return results in API response
    return {"results": results}

@app.get("/map")

def get_map():
    map_path = os.path.join(script_dir, "forecast_map.html")
    if not os.path.exists(map_path):
        return {"error": f"Map file not found at {map_path}"}

    return FileResponse(map_path, media_type="text/html")
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
