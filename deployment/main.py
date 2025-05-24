from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from utilities.preprocessing import preprocess_vessel_df, denormalize_column
from utilities.data_validation import missing_data_check, validate_csv_rows
from utilities.sliding_window import sliding_windows
from utilities.inference import AISInferenceModel
from utilities.zone_check import load_cable_lines, any_forecast_in_zone, build_buffered_zone, forecast_path_crosses_zone, vessel_near_any_cable
from utils import save_results
from shapely.geometry import Point
import numpy as np
import yaml
import json
import pandas as pd
import io
import uvicorn
import os
import subprocess



with open("utilities/inference_config.yaml", "r") as f:
    config = yaml.safe_load(f)

window_size = config["window_size"]
step_size = config["step_size"]
cable_path = config["cable_coordinates_path"]

stats_path = config["normalization_stats_path"]
script_dir = os.path.dirname(os.path.realpath(__file__))

with open(stats_path, "r") as f:
    norm_stats = json.load(f)

model = AISInferenceModel(
    classifier_path=config["model_paths"]["classifier"],
    forecaster_path=config["model_paths"]["forecaster"],
    fallback_classifier_path=config["model_paths"]["fallback_classifier"],
    fallback_forecaster_path=config["model_paths"]["fallback_forecaster"],
    verbose=False
)

cable_lines = load_cable_lines(cable_path)
buffered_zone_1602 = build_buffered_zone(cable_lines, buffer_meters=1602)
buffered_zone_2136 = build_buffered_zone(cable_lines, buffer_meters=3738)

app = FastAPI()



@app.get("/")
async def root():
    return {"message": "AIS Forecasting Service is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...), mmsi: int = Form(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    validate_csv_rows(df)
    window_size = 30  # explicitly enforced

    df = preprocess_vessel_df(df, mmsi, stats_path)

    df = df.drop(["# Timestamp", "MMSI", "trawling"], axis=1)

    results = []

    # Check if a vessel is near any undersea cable
    current_lat_norm = df["Latitude"].iloc[-1]
    current_lon_norm = df["Longitude"].iloc[-1]

    current_lat = denormalize_column(np.array([current_lat_norm]), "Latitude", norm_stats)[0]
    current_lon = denormalize_column(np.array([current_lon_norm]), "Longitude", norm_stats)[0]

    if not vessel_near_any_cable(current_lat, current_lon, cable_lines, radius_m=3738):
        return {"warning": "Vessel not near any cable. No prediction performed."}

    # Check for missing data
    use_fallback = not missing_data_check(df, window_size, verbose=False)
    if use_fallback:
        print("Detected missing values or fully null columns in input window. Using fallback models.")
        input_tensor = df[["Latitude", "Longitude"]].values.astype(np.float32).reshape(1, window_size, 2)
    else:
        input_tensor = df[[
            "Latitude", "Longitude", "ROT", "SOG", "COG",
            "Heading", "Width", "Length", "Draught"
        ]].values.astype(np.float32).reshape(1, window_size, 9)

    # Perform prediction
    label, probability, logit, forecast = model.predict(input_tensor, use_fallback=use_fallback)

    response = {
        "logit": float(logit),
        "probability": float(probability),
        "label": label
    }

    if forecast is not None:
        # Determine spatial alert zones
        start_lat, start_lon = forecast[0]
        start_point = Point(start_lon, start_lat)
        crosses_critical = forecast_path_crosses_zone(forecast, buffered_zone_1602)
        inside_entry_zone = buffered_zone_2136.contains(start_point)

        # Risk level logic
        if label == 1:
            if crosses_critical:
                risk_level = 3
            elif inside_entry_zone:
                risk_level = 2
            else:
                risk_level = 1
        else:
            risk_level = 0

        # Denormalize input coordinates for response
        lat_norm = df["Latitude"].values
        lon_norm = df["Longitude"].values
        lat = denormalize_column(lat_norm, "Latitude", norm_stats)
        lon = denormalize_column(lon_norm, "Longitude", norm_stats)
        input_coords = list(zip(lat, lon))

        response.update({
            "forecast": forecast.tolist(),
            "zone_alert": any_forecast_in_zone(forecast, buffered_zone_1602),
            "risk_level": risk_level,
            "input": input_coords,
            "used_fallback": use_fallback
        })
    results.append(response)
    save_results(script_dir, config, results)

@app.post("/sliding_window_predict")
async def rolling_window_predict(file: UploadFile = File(...), mmsi: int = Form(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    validate_csv_rows(df)
    df = preprocess_vessel_df(df, mmsi, stats_path)
    df = df.drop(["# Timestamp", "MMSI", "trawling"], axis=1)

    results = []

    for window in sliding_windows(df, window_size, step_size):
        # Extract the last point in the window as the current location
        current_lat_norm = window["Latitude"].iloc[-1]
        current_lon_norm = window["Longitude"].iloc[-1]

        current_lat = denormalize_column(np.array([current_lat_norm]), "Latitude", norm_stats)[0]
        current_lon = denormalize_column(np.array([current_lon_norm]), "Longitude", norm_stats)[0]

        if not vessel_near_any_cable(current_lat, current_lon, cable_lines, radius_m=3738):
            print(f"Skipping window: vessel not near cable at ({current_lat}, {current_lon})")
            continue


        use_fallback = not missing_data_check(window, window_size, verbose = False)

        if use_fallback:
            print("Detected missing values or fully null columns in input window. Using fallback models.")


            input_tensor = (window[["Latitude", "Longitude"]]
                            .values.astype(np.float32).reshape(1, window_size, 2))
        else:
            input_tensor = window[[
                "Latitude", "Longitude", "ROT", "SOG", "COG",
                "Heading", "Width", "Length", "Draught"
            ]].values.astype(np.float32).reshape(1, window_size, 9)


        label, probability, logit, forecast = model.predict(input_tensor, use_fallback=use_fallback)

        if forecast is not None:
            start_lat, start_lon = forecast[0]
            start_point = Point(start_lon, start_lat)

            crosses_critical = forecast_path_crosses_zone(forecast, buffered_zone_1602)
            inside_entry_zone = buffered_zone_2136.contains(start_point)

            if label == 1:
                if crosses_critical:
                    risk_level = 3
                elif inside_entry_zone:
                    risk_level = 2
                else:
                    risk_level = 1
            else:
                risk_level = 0

            lat_norm = window["Latitude"].values
            lon_norm = window["Longitude"].values
            lat = denormalize_column(lat_norm, "Latitude", norm_stats)
            lon = denormalize_column(lon_norm, "Longitude", norm_stats)
            input_coords = list(zip(lat, lon))

            results.append({
                "forecast": forecast.tolist(),
                "fishing_confidence": round(probability, 4),
                "risk_level": risk_level,
                "input": input_coords,
                "used_fallback": use_fallback
            })


    save_results(script_dir, config, results)

@app.get("/map")
def get_map():
    map_path = config["forecast_map_path"]

    try:
        subprocess.run(["python3", "visualize_forecast.py"], check=True, cwd=script_dir)
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to generate map: {str(e)}"}

    if not os.path.exists(map_path):
        return {"error": f"Map file not found at {map_path}"}

    return FileResponse(map_path, media_type="text/html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
