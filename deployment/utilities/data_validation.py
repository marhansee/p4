import numpy as np
import pandas as pd
from pydantic import ValidationError, BaseModel
from typing import List


def missing_data_check(df, window_size, mmsi=None, log_fn=print, verbose=False):
    """
    Validates whether a 60-step input window is safe for inference.

    Requirements:
      - All 9 expected features must exist and not be fully null
      - Must contain exactly 60 rows

    Args:
        df (pd.DataFrame): The input window (should be 60 time steps).
        mmsi (str or int, optional): For logging context.
        log_fn (function): Logging or print function.

    Returns:
        bool: True if safe to infer, False if not.
    """

    expected_cols = [
        "Latitude", "Longitude", "ROT", "SOG", "COG",
        "Heading", "Width", "Length", "Draught"
    ]

    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        if verbose:
            log_fn(f"❌ MMSI {mmsi or '?'} — missing columns: {missing}")
        return False

    fully_null = [col for col in expected_cols if df[col].isnull().all()]
    if fully_null:
        if verbose:
            log_fn(f"❌ MMSI {mmsi or '?'} — fully null columns: {fully_null}")
        return False

    partly_null = [col for col in expected_cols if df[col].isnull().any()]
    if partly_null:
        if verbose:
            log_fn(f"⚠️ MMSI {mmsi or '?'} — partly null columns: {partly_null}")
        return False

    if len(df) < window_size:
        if verbose:
            log_fn(f"❌ MMSI {mmsi or '?'} — window too short: {len(df)} < {window_size}")
        return False

    if verbose:
        log_fn(f"✅ MMSI {mmsi or '?'} — window is valid.")
    return True

def convert(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

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

def validate_csv_rows(df: pd.DataFrame) -> List[AISDataPoint]:
    validated_rows = []
    for i, row in df.iterrows():
        try:
            validated_rows.append(AISDataPoint(**row.to_dict()))
        except ValidationError as e:
            raise ValueError(f"Row {i} failed validation: {e}")
    return validated_rows
