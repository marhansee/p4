from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List

from preprocessing import pick_vessel, drop_class_b, filter_relevant_columns, drop_duplicates, resample_to_fixed_interval, normalize_columns, denormalize_column, reduce_skewness
from data_validation import missing_data_check
from inference import AISInferenceModel
from zone_check import load_cable_lines, build_buffered_zone, forecast_path_crosses_zone
from shapely.geometry import Point
import numpy as np
import yaml
import json
import pandas as pd
import io
import os
import time

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


@app.post("/test_inference_timing")
async def test_inference_timing(file: UploadFile = File(...), mmsi: int = Form(...)):

    t0 = time.perf_counter()

    contents = await file.read()
    t1 = time.perf_counter()

    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df = pick_vessel(df, mmsi)
    df = drop_class_b(df)
    df = filter_relevant_columns(df)
    df = drop_duplicates(df)
    df = df.sort_values("# Timestamp")

    try:
        df = resample_to_fixed_interval(df)
    except ValueError as e:
        return {"error": f"Resampling failed: {str(e)}"}
    df = reduce_skewness(df)
    if len(df) < window_size:
        return {"error": f"Not enough resampled rows to form a window of size {window_size}"}


    window = df.tail(window_size).copy()
    t_start = pd.to_datetime(window["# Timestamp"].iloc[0])
    t_end = pd.to_datetime(window["# Timestamp"].iloc[-1])
    real_acquisition_seconds = (t_end - t_start).total_seconds()


    with open(stats_path, "r") as f:
        norm_stats = json.load(f)


    for col in ["Latitude", "Longitude", "ROT", "SOG", "COG", "Heading", "Width", "Length", "Draught"]:
        if col in window.columns:
            window[col] = normalize_columns(window[col], col)

    window = window.drop(["# Timestamp", "MMSI", "trawling"], axis=1, errors="ignore")

    t2 = time.perf_counter()

    use_fallback = not missing_data_check(window, window_size, verbose=False)

    input_tensor = (
        window[["Latitude", "Longitude"]].values.astype(np.float32).reshape(1, window_size, 2)
        if use_fallback else
        window[[
            "Latitude", "Longitude", "ROT", "SOG", "COG",
            "Heading", "Width", "Length", "Draught"
        ]].values.astype(np.float32).reshape(1, window_size, 9)
    )
    t3 = time.perf_counter()


    label, probability, logit, forecast = model.predict(input_tensor, use_fallback=use_fallback)

    t4 = time.perf_counter()

    if forecast is None:
        return {"error": "Model did not produce a forecast"}


    start_lat, start_lon = forecast[0]
    start_point = Point(start_lon, start_lat)
    crosses_critical = forecast_path_crosses_zone(forecast, buffered_zone_1602)
    inside_entry_zone = buffered_zone_2136.contains(start_point)

    risk_level = 0
    if label == 1:
        if crosses_critical:
            risk_level = 3
        elif inside_entry_zone:
            risk_level = 2
        else:
            risk_level = 1

    lat_norm = window["Latitude"].values
    lon_norm = window["Longitude"].values
    lat = denormalize_column(lat_norm, "Latitude", norm_stats)
    lon = denormalize_column(lon_norm, "Longitude", norm_stats)
    input_coords = list(zip(lat, lon))

    result = {
        "forecast": forecast.tolist(),
        "fishing_confidence": float(probability),
        "risk_level": int(risk_level),
        "input": input_coords,
        "used_fallback": use_fallback
    }

    return {
        "result": result,
        "timing_seconds": {
            "estimated_real_acquisition_after_resampling": real_acquisition_seconds,
            "file_read": round(t1 - t0, 4),
            "preprocessing_and_resampling": round(t2 - t1, 4),
            "tensor_building": round(t3 - t2, 4),
            "inference": round(t4 - t3, 4),
            "total_runtime_excluding_acquisition": round(t4 - t0, 4),
            "total_runtime_including_acquisition": round(t4 - t0 + real_acquisition_seconds, 4)
        }
    }
