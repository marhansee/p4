import pandas as pd
import json
import numpy as np

continuous_cols: list[str] = ["Latitude", "Longitude", "ROT", "SOG", "COG", "Heading", "Draught"]
static_cols: list[str] = ["Width", "Length", "trawling"]

skewed_positive: list[str] = ["SOG", "Draught"]
skewed_signed: list[str] = ["ROT"]

# Feature bounds (used for clamping)
clamp_limits = {
    "SOG": (0, 40),
    "ROT": (-90, 90),
}

def normalize_columns(df, stats_path: str, exclude: list = []):

    with open(stats_path, "r") as f:
        stats = json.load(f)

    df = df.copy()
    for col in df.columns:
        if col in stats and col not in exclude:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            if std != 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0
    return df

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
    df = df.drop_duplicates(subset=['# Timestamp'])
    return df

def resample_to_fixed_interval(df):
    # Aggregate duplicates first
    df = df.groupby('# Timestamp').agg('first').reset_index()

    # Convert and set datetime index
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])
    df = df.sort_values("# Timestamp").set_index('# Timestamp')

    # Resample to 10s intervals
    df_resampled = df.resample('10s').asfreq()

    # Forward-fill static features
    for col in ["MMSI"] + static_cols:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].ffill().bfill()

    # Interpolate continuous features
    for col in continuous_cols:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].interpolate(method='linear', limit_direction='both')

    # Clamp known limits
    for col, (low, high) in clamp_limits.items():
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].clip(lower=low, upper=high)

    return df_resampled.reset_index()

def reduce_skewness(df, skewed_positive=[], skewed_signed=[]):
    """Automatically log-transform columns with high skewness."""
    df = df.copy()  # Avoid modifying in-place
    for col in skewed_positive:
        skew_val = df[col].skew()
        if abs(skew_val) > 1:
            df[col] = np.log1p(df[col])

    for col in skewed_signed:
        skew_val = df[col].skew()
        if abs(skew_val) > 1:
            df[col] = np.where(
                df[col] >= 0,
                np.log1p(df[col]),
                -np.log1p(-df[col])
            )

    return df

def denormalize_column(values, col_name, norm_stats):
    mean = norm_stats[col_name]["mean"]
    std = norm_stats[col_name]["std"]
    return [(v * std + mean) for v in values]



def preprocess_vessel_df(df: pd.DataFrame, MMSI: int, stats_path: str) -> pd.DataFrame:
    df = pick_vessel(df, MMSI)
    df = drop_class_b(df)
    df = filter_relevant_columns(df)
    df = drop_duplicates(df)
    df = resample_to_fixed_interval(df)
    df = reduce_skewness(df)
    df = normalize_columns(df, stats_path=stats_path, exclude=["trawling"])
    return df

