import pandas as pd
import numpy as np

continuous_cols: list[str] = ["Latitude", "Longitude", "ROT", "SOG", "COG", "Heading", "Draught"]
static_cols: list[str] = ["Width", "Length", "trawling"]

# Feature bounds (used for clamping)
clamp_limits = {
    "SOG": (0, 40),
    "ROT": (-90, 90),
}


def load_csv(input_path):
    df = pd.read_csv(input_path, header=0)
    return df


def pick_vessel(df, MMSI):
    return df[df['MMSI'] == MMSI].copy()


def drop_class_b(df):
    return df[df['Type of mobile'] == 'Class A']


def filter_relevant_columns(df):
    cols = ['# Timestamp', 'MMSI'] + continuous_cols + static_cols
    return df[cols]


def drop_duplicates(df):
    df = df.drop_duplicates(subset=['# Timestamp', 'MMSI'])
    return df


def resample_to_fixed_interval(df):
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])

    df = df.sort_values("# Timestamp")
    df = df.set_index('# Timestamp')

    # Resample to 10s intervals
    df_resampled = df.resample('10s').asfreq()

    # Forward-fill static features
    for col in ["MMSI"] + static_cols:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].ffill()

    for col in ["MMSI"] + static_cols:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].bfill()

    # Interpolate continuous features
    for col in continuous_cols:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].interpolate(method='linear', limit_direction='both')
    # Clamp known limits
    for col, (low, high) in clamp_limits.items():
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].clip(lower=low, upper=high)

    return df_resampled.reset_index()



def preprocess_vessel_df(input_path: str, MMSI: int):
    df = load_csv(input_path)
    df = pick_vessel(df, MMSI)
    df = drop_class_b(df)
    df = filter_relevant_columns(df)
    df = drop_duplicates(df)
    df = resample_to_fixed_interval(df)

    return df

