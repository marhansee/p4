from utilities.preprocessing import preprocess_vessel_df
from utilities.data_validation import missing_data_check
from utilities.sliding_window import sliding_windows

import numpy as np

input_path = "aisdk-2024-06-04_fishing_labeled.csv"
MMSI = 220141000

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
            ].values.astype(np.float32)

            if model_fn is not None:
                model_fn(input_tensor, window)

            # Insert model inference

            valid_windows += 1

        else:
            skipped_windows += 1
    log_fn(f" MMSI {MMSI}: Valid windows {valid_windows} \n Skipped windows:{skipped_windows}")

    return {
        "mmsi": MMSI,
        "valid_windows": valid_windows,
        "skipped_windows": skipped_windows
    }

process_vessel_stream(input_path, MMSI)
