from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
import numpy as np
import yaml
import json
import pandas as pd
import io
from utilities.preprocessing import pick_vessel, drop_class_b, filter_relevant_columns, drop_duplicates, resample_to_fixed_interval, normalize_columns
from utilities.data_validation import missing_data_check
from utilities.sliding_window import sliding_windows
from utilities.inference import AISInferenceModel
from utilities.zone_check import load_cable_lines, any_forecast_in_zone

with open("utilities/inference_config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_path = config["data_path"]
window_size = config["window_size"]
step_size = config["step_size"]
cable_path = config["cable_coordinates_path"]
classifier_path = config["model_paths"]["classifier"]
forecaster_path = config["model_paths"]["forecaster"]

model = AISInferenceModel(classifier_path=classifier_path, forecaster_path=forecaster_path, verbose=False)
cable_lines = load_cable_lines(cable_path)

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
        response["zone_alert"] = any_forecast_in_zone(forecast, cable_lines)

    return response

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...), mmsi: int = Form(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    df = pick_vessel(df, mmsi)
    df = drop_class_b(df)
    df = filter_relevant_columns(df)
    df = drop_duplicates(df)
    df = resample_to_fixed_interval(df)
    df = normalize_columns(df, stats_path="/home/martin-birch/p4/data/train_norm_stats.json", exclude=["trawling"])
    df = df.drop(["# Timestamp", "MMSI", "trawling"], axis=1)

    results = []
    for window in sliding_windows(df, window_size, step_size):
        if missing_data_check(window, window_size):
            input_tensor = window[[
                "Latitude", "Longitude", "ROT", "SOG", "COG",
                "Heading", "Width", "Length", "Draught"
            ]].values.astype(np.float32).reshape(1, window_size, 9)
            label, probability, logit, forecast = model.predict(input_tensor)

            result = {
                "logit": float(logit),
                "probability": float(probability),
                "label": label
            }
            if forecast is not None:
                result["forecast"] = forecast.tolist()
                result["zone_alert"] = any_forecast_in_zone(forecast, cable_lines)

            results.append(result)

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
