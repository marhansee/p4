from utilities.preprocessing import preprocess_vessel_df
from utilities.data_validation import missing_data_check
from utilities.sliding_window import sliding_windows
from utilities.inference import AISInferenceModel
from utilities.zone_check import load_cable_lines, any_forecast_in_zone
import numpy as np

input_path = "aisdk-2024-06-04_fishing_labeled.csv"
MMSI = 220141000

cable_coordinates_path = "/home/martin/p4/data/cable_coordinates.csv"

model = AISInferenceModel(classifier_path="/home/martin/p4/deployment/models/lstm_finalv3.onnx",
                          forecaster_path="/home/martin/p4/deployment/models/s2s_bigru.onnx",verbose = False)

def run_model(input_tensor, window):

    cable_lines = load_cable_lines(cable_coordinates_path)

    label, prob, logit, forecast = model.predict(input_tensor)
    print(f"Prediction → label: {label}, prob: {prob:.4f}")
    if forecast is not None:
        if any_forecast_in_zone(forecast, cable_lines):
            print("Trajectory forecast in critical zone!")

def process_vessel_stream(input_path: str, MMSI: int, model_fn=None, window_size = 60, step_size = 1, log_fn=print):
    df = preprocess_vessel_df(input_path, MMSI)
    df = df.drop(["# Timestamp","MMSI","trawling"],axis=1)
    valid_windows = 0
    skipped_windows = 0

    for window in sliding_windows(df, window_size=60, step_size=1):
        if missing_data_check(window, MMSI):
            input_tensor = window[
                ["Latitude", "Longitude", "ROT", "SOG", "COG",
                 "Heading", "Width", "Length", "Draught"]
            ].values.astype(np.float32).reshape(1, 60, 9)

            if model_fn is not None:
                model_fn(input_tensor, window)
            valid_windows += 1
        else:
            skipped_windows += 1
    log_fn(f" MMSI {MMSI}: Valid windows {valid_windows} \n Skipped windows:{skipped_windows}")

    return {
        "mmsi": MMSI,
        "valid_windows": valid_windows,
        "skipped_windows": skipped_windows
    }

process_vessel_stream(input_path, MMSI, model_fn=run_model)
