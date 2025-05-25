import pyarrow.parquet as pq
import numpy as np
import os
import sys
import json
import yaml
import glob
import pandas as pd
import psutil
import torch


def load_data(parquet_files, input_features, target_columns=None):
    """
    Load data from multiple Parquet files and extract features and target labels.

    Parameters:
    - parquet_files: List of Parquet file paths to read data from.
    - input_features: List of input feature columns.
    - target_columns: List of target column names to be used for y (default is None).

    Returns:
    - X: NumPy array of input features.
    - y: NumPy array of target labels.
    """

    all_features = []
    all_target_data = []

    # Read and process each Parquet file
    for parquet_file in parquet_files:
        # Read the Parquet file
        table = pq.read_table(parquet_file)

        # Convert to pandas to work with smaller chunks (since PyArrow handles larger data well)
        df = table.to_pandas()

        # Sort by MMSI and timestamp
        df = df.sort_values(by=['MMSI', 'timestamp_epoch'])

        # Extract features and targets
        features_data = df[input_features].values
        target_data = df[target_columns].values if target_columns else None

        all_features.append(features_data)
        if target_data is not None:
            all_target_data.append(target_data)

    # Stack all features and target data from different files into arrays
    X = np.vstack(all_features)
    y = np.vstack(all_target_data) if target_columns else None


    X = pd.DataFrame(X, columns=input_features)
    y = pd.DataFrame(y, columns=target_columns)
    print("Processed and sorted the data!")

    return X, y

def make_sequences(X_df, y_df, seq_len, group_col):
    sequences = []
    labels = []

    for _, group in X_df.groupby(group_col):
        group = group.sort_values('timestamp_epoch')
        group_y = y_df.loc[group.index]

        # Drop non-feature columns BEFORE converting to NumPy
        group_features = group.drop(columns=[group_col, 'timestamp_epoch'])

        X_array = group_features.values
        y_array = group_y.values

        if len(group) < seq_len:
            continue

        for i in range(len(group) - seq_len + 1):
            x_seq = X_array[i:i+seq_len]
            y_target = y_array[i+seq_len-1]
            sequences.append(x_seq)
            labels.append(y_target)


    return np.array(sequences), np.array(labels)


def load_config_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Unexpected error: {e}")

def load_scaler_json(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    
    try:
        with open(file_path, 'r') as file:
            scaler = json.load(file)
        return scaler
    except Exception as e:
        print(f"Unexpected error: {e}")


def scale_data(scaler, X_train):
    """
    Scale the input features based on the provided scaler, keeping non-scaled features unchanged.

    Parameters:
    - scaler: A dictionary containing mean and std values for scaling each feature.
    - X_train: Pandas DataFrame of input features.

    Returns:
    - X_train_scaled: Scaled version of X_train with original features kept unchanged.
    """
    # Create a copy of X_train to avoid modifying the original
    X_train_scaled = X_train.copy()
    
    # Loop through the scaler dictionary and scale the corresponding columns
    for feature, stats in scaler.items():
        if feature in X_train.columns:  # Check if feature is in X_train columns
            mean_feature = stats['mean']
            std_feature = stats['std']
            
            # Scale the feature (subtract mean, divide by std)
            X_train_scaled[feature] = (X_train[feature] - mean_feature) / std_feature
            
    print("Selected features have been scaled!")
    
    return X_train_scaled

def inverse_scale_lat_lon(lat_scaled, lon_scaled, scaler):
    """
    Inversely scale normalized lat/lon using the provided scaler dict.
    """
    lat_mean = scaler['Latitude']['mean']
    lat_std = scaler['Latitude']['std']
    lon_mean = scaler['Longitude']['mean']
    lon_std = scaler['Longitude']['std']

    lat_original = lat_scaled * lat_std + lat_mean
    lon_original = lon_scaled * lon_std + lon_mean

    return lat_original, lon_original

def print_memory_stats(device='cpu'):
    print(f"\n[MEMORY USAGE]")
    print(f"CPU RAM: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU VRAM: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
