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


    # X = pd.DataFrame(X, columns=input_features)
    # y = pd.DataFrame(y, columns=target_columns)
    print("Processed and sorted the data!")

    return X, y


def load_config_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
        # return None
    
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
        # return None
    
    try:
        with open(file_path, 'r') as file:
            scaler = json.load(file)
        return scaler
    except Exception as e:
        print(f"Unexpected error: {e}")


def scale_data(scaler, X_train, features_to_scale):
    """
    Scale the input features based on the provided scaler, keeping non-scaled features unchanged.

    Parameters:
    - scaler: A json-file containing mean and std values for scaling each feature.
    - X_train: NumPy array of input features.
    - features_to_scale: List of feature columns to be scaled.

    Returns:
    - X_train_scaled: Scaled version of X_train with original features kept unchanged.
    """
    # Identify indices of features to scale
    scale_indices = [i for i, feature in enumerate(features_to_scale)]
    
    # Create a copy of X_train to avoid modifying the original
    X_train_scaled = X_train.copy()

    # Loop through the features to scale and scale the corresponding columns
    for i, feature in enumerate(features_to_scale):
        mean_feature = scaler[f'{feature}_mean']
        std_feature = scaler[f'{feature}_std']
        
        # Scale the feature (subtract mean, divide by std)
        X_train_scaled[:, scale_indices[i]] = (X_train[:, scale_indices[i]] - mean_feature) / std_feature
    
    print("Selected features have been scaled!")

    return X_train_scaled

def print_memory_stats(device='cpu'):
    print(f"\n[MEMORY USAGE]")
    print(f"CPU RAM: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU VRAM: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

def main():
    from data_loader import Forecasting_Dataloader
    from torch.utils.data import DataLoader
    # Load data
    train_data_folder_path = os.path.abspath('data/parquet_test/')
    train_parquet_files = glob.glob(os.path.join(train_data_folder_path, '*.parquet'))

    # print(train_parquet_files)  # Debug print

    input_features = ['Latitude', 'Longitude', 'ROT', 'SOG', 'COG', 'Heading', 
                      'Width', 'Length', 'Draught']
    # target_features = [f'future_lat_{i}' for i in range(1, 21)]
    target_features = ['trawling']
    
    X_train, y_train = load_data(
        parquet_files=train_parquet_files,
        input_features=input_features,
        target_columns=target_features
    )

    print_memory_stats()  # After loading

    unique, counts = np.unique(y_train, return_counts=True)
    print("Class distribution in 'trawling':")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c}")



    sys.exit()
    train_dataset = Forecasting_Dataloader(
        X=X_train,
        y=y_train,
        seq_length=60
    )

    # Load dataloaders
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=32,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    
    
    print_memory_stats()  # After loading
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to('cpu'), target.to('cpu')
        # print(data)
        # print(target)
        print(f"Batch {batch_idx}")
        print("Data shape:", data.shape)
        print("Target shape:", target.shape)
        print_memory_stats()  # After loading
        if batch_idx == 1:
            break  # Only inspect the first batch

if __name__ == '__main__':
    main()