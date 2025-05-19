from utilities.preprocessing import preprocess_vessel_df
from utilities.data_validation import missing_data_check
from utilities.sliding_window import sliding_windows
from utilities.inference import AISInferenceModel
from utilities.zone_check import load_cable_lines, any_forecast_in_zone
import numpy as np
import yaml
import json

with open("utilities/inference_config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_path = config["data_path"]
MMSI = config["MMSI"]
window_size = config["window_size"]
step_size = config["step_size"]
cable_path = config["cable_coordinates_path"]
classifier_path = config["model_paths"]["classifier"]
forecaster_path = config["model_paths"]["forecaster"]

def export_window_as_json(input_tensor, window):
    points = window[[
        "Latitude", "Longitude", "ROT", "SOG", "COG",
        "Heading", "Width", "Length", "Draught"
    ]].to_dict(orient="records")
    print(json.dumps(points, indent=2))

def process_vessel_stream(input_path: str, MMSI: int, window_size: int, step_size: int, model_fn=None, log_fn=print):
    df = preprocess_vessel_df(input_path, MMSI)
    df = df.drop(["# Timestamp","MMSI","trawling"],axis=1)
    valid_windows = 0
    skipped_windows = 0

    for window in sliding_windows(df, window_size, step_size=1):
        if missing_data_check(window, window_size=window_size, mmsi=MMSI):
            input_tensor = window[[
                "Latitude", "Longitude", "ROT", "SOG", "COG",
                "Heading", "Width", "Length", "Draught"
            ]].values.astype(np.float32).reshape(1, window_size, 9)

            if model_fn is not None:
                model_fn(input_tensor, window)
            valid_windows += 1
            break  # only export the first valid window
        else:
            skipped_windows += 1
    log_fn(f" MMSI {MMSI}: Valid windows {valid_windows} \n Skipped windows:{skipped_windows}")

    return {
        "mmsi": MMSI,
        "valid_windows": valid_windows,
        "skipped_windows": skipped_windows
    }

process_vessel_stream(input_path, MMSI, window_size, step_size, model_fn=export_window_as_json)
