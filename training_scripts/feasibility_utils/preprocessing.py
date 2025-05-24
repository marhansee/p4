import pandas as pd 
import numpy as np
from tqdm import tqdm

def split_data(input_csv , train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=None):
    """
    Splits the data into train, validation, and test sets.
    
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - train_ratio (float): The proportion of the data to be used for training (default 0.7).
    - val_ratio (float): The proportion of the data to be used for validation (default 0.1).
    - test_ratio (float): The proportion of the data to be used for testing (default 0.2).
    - output_prefix (str): Prefix for the output files.
    
    Returns:
    - train_df: DataFrame for Train split
    - val_df: DataFrame for Validation split
    - test_df: DataFrame for Test split
    """
    #load data
    df = pd.read_csv(input_csv)

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Count unique trajectory IDs per label
    trajectory_counts = df.groupby("label")["trajectory_id"].nunique()
    print(f"Unique Trajectories Count: \nSailing (0): {trajectory_counts.get(0, 0)} \nFishing (1): {trajectory_counts.get(1, 0)}")

    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Empty lists to store final splits
    train_data = []
    val_data = []
    test_data = []

    for label, group in df.groupby("label"):
        unique_trajectories = group["trajectory_id"].unique()
        np.random.shuffle(unique_trajectories)

        # Compute split sizes
        num_train = int(len(unique_trajectories) * train_ratio)
        num_val = int(len(unique_trajectories) * val_ratio)

        # Assing trajectories to train, val, and test sets
        train_trajectories = unique_trajectories[:num_train]
        val_trajectories = unique_trajectories[num_train:num_train + num_val]
        test_trajectories = unique_trajectories[num_train + num_val:]

        # Split the actual data based on trajectory assignment
        train_data.append(group[group["trajectory_id"].isin(train_trajectories)])
        val_data.append(group[group["trajectory_id"].isin(val_trajectories)])
        test_data.append(group[group["trajectory_id"].isin(test_trajectories)])

    # Concatenate back to form final datasets
    train_df = pd.concat(train_data).reset_index(drop=True)
    val_df = pd.concat(val_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)

    return train_df, val_df, test_df


def add_lagged_value(df, num_lag:list, target_features=['latitude','longitude'], ):
    """
    Function that adds lagged values to features except the following feautures:
        - trajectory_id, t, latitude, longitude

    The original features are dropped (except longitude and latitude)

    NaNs are dropped.

    Args:
        - df: The DataFrame that will receive lagged features
        - num_lag (list): List containing the number of lags to be added
            - For instance, [1, 3] adds lag=1 and lag=3
        - target_features: Target features

    returns:
        - df: DataFrame with lagged features
    """
    if not "trajectory_id" in df.columns:
        raise AssertionError("trajectory_id must be in the DataFrame!")

    df = df.sort_values(['trajectory_id', 't']).copy()

    # if not isinstance(num_lag, list):
        # raise AssertionError("num_lag-feature must be a list!")

    if not num_lag:
        return df
    elif num_lag == 0:
        return df

    # Specify features to shift
    feature_cols = [col for col in df.columns if col not in ['trajectory_id', 't']]

    # Add lag
    for feature in feature_cols:
        for lag in num_lag:
            df[f'{feature}_lag{lag}'] = df.groupby('trajectory_id')[feature].shift(lag)


    # Drop original features
    # feature_drop = [feature for feature in feature_cols if feature not in target_features]
    # df = df.drop(columns=feature_drop)

    # Drop NaN rows 
    df = df.dropna().reset_index(drop=True)

    return df

def add_future_value(df, horizon=5, target_features=['latitude','longitude']):
    if not "trajectory_id" in df.columns:
        raise AssertionError("trajectory_id must be in the DataFrame!")

    df = df.sort_values(['trajectory_id', 't']).copy()

    # Create future target values (shifted backwards)
    for target in target_features:
        df[f'{target}_future{horizon}'] = df.groupby('trajectory_id')[target].shift(-horizon)

    # Drop NaN rows 
    df = df.dropna().reset_index(drop=True)

    return df

def apply_downsampling(df, time_interval):

    """
    Downsamples each trajectory by selecting a single data sample per a specified time interval.

    Parameters:
    - df: Pandas DataFrame with data
    - time_interval (int): Time interval in seconds to keep a single data point.

    Returns:
    - df: Modified DataFrame
    """

    # Sort by trajectory and timestamp to maintain order
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values(by=["trajectory_id", "t"])  

    downsampled_trajectories = []

    # Downsample per trajectory
    for traj_id, group in df.groupby("trajectory_id"):
        group = group.set_index("t")
        sampled = group.resample(f"{time_interval}s").first().dropna().reset_index()
        sampled["trajectory_id"] = traj_id
        downsampled_trajectories.append(sampled)

    downsampled_df = pd.concat(downsampled_trajectories, ignore_index=True)

    print(f"Downsampled dataset with {time_interval}s interval. Original size: {len(df)}, New size: {len(downsampled_df)}")

    # Drop missing values
    downsampled_df = downsampled_df.dropna()

    return downsampled_df

def apply_resampling(df, time_interval):
    """
    Resamples each trajectory at fixed time intervals using linear interpolation.

    Parameters:
    - df: Pandas DataFrame with data
    - time_interval (int): Time interval in seconds to keep a single data point.

    Returns:
    - df: Modified DataFrame
    """

    # Sort by trajectory and timestamp to maintain order
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values(by=["trajectory_id", "t"])

    # Function to resample and interpolate per trajectory
    def resample_and_interpolate(group):
        # Ensure timestamps are unique within each group (trajectory)
        group = group.sort_values("t")  # Sort by timestamp just in case
        group = group[~group["t"].duplicated(keep="first")]  # Drop duplicate timestamps

        group = group.set_index("t")
        group = group.resample(f"{time_interval}s").asfreq()


        first_valid = group.first_valid_index()
        if first_valid is not None:
            group = group.loc[first_valid:]

        # Fill missing 'id' with the same one from the trajectory
        if "id" in group.columns:
            group["id"] = group["id"].ffill().bfill()

        # Drop non-numeric cols we don’t wanna interpolate
        #group = group.drop(columns=["id"], errors="ignore")

        # Dtype fix
        group = group.infer_objects(copy=False)

        # Interpolate numerics
        group = group.interpolate(method="linear")

        # Handle label separately (categorical)
        if "label" in group.columns:
            group["label"] = group["label"].ffill().bfill()
        
        return group.reset_index()    

    # Apply resampling and interpolation per trajectory
    resampled_df = df.groupby("trajectory_id", group_keys=False).apply(resample_and_interpolate, include_groups=True)

    print(f"Resampled dataset at {time_interval}s intervals. Original size: {len(df)}, New size: {len(resampled_df)}")
    
    # Drop missing values
    resampled_df = resampled_df.dropna()

    # Save dataset
    return resampled_df

def sliding_window_features(df, features, target_columns, window_size=7, step_size=1, forecast_horizon=10):
    """
    Extracts statistical features using a sliding window per trajectory, 
    and retrieves the target value after `window_size + forecast_horizon` steps.

    Parameters:
    - df: Pandas DataFrame
    - features: List of column names to extract statistics from (excluding target)
    - target_columns: List of column names for the target variable (e.g., latitude, longitude)
    - window_size (int): Number of samples per sliding window.
    - step_size (int): Step size for moving the window.
    - forecast_horizon (int): Number of time steps ahead to forecast the target (e.g., 10).

    Returns:
    - df: Modified DataFrame with statistical features and target value.
    """
    
    # Sort by trajectory and timestamp to maintain order
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values(by=["trajectory_id", "t"])

    # List to store processed windows
    feature_rows = []

    # Function to extract features from a window
    def compute_window_stats(group):
        for start in range(0, len(group) - window_size - forecast_horizon + 1, step_size):
            window = group.iloc[start:start + window_size]  # Slice the window
            stats = {
                "trajectory_id": group["trajectory_id"].iloc[0],
                "window_start": window["t"].iloc[0],
                "window_end": window["t"].iloc[-1],
            }

            # Compute statistical features excluding the target columns
            for feature in features:
                stats[f"{feature}_mean"] = window[feature].mean()
                stats[f"{feature}_median"] = window[feature].median()
                stats[f"{feature}_min"] = window[feature].min()
                stats[f"{feature}_max"] = window[feature].max()
                stats[f"{feature}_std"] = window[feature].std()

            # Add label (optional) if the label exists in the window
            if "label" in window.columns:
                stats["label"] = window["label"].mode()[0]  # Most common label in the window

            # Calculate target index (window end + forecast horizon)
            target_idx = start + window_size + forecast_horizon - 1

            # Ensure target index is within bounds
            if target_idx < len(group):
                # Add target values for prediction (shifted by forecast_horizon)
                for target in target_columns:
                    stats[f"{target}_target"] = group[target].iloc[target_idx]
            else:
                # Handle cases where the target index is out of bounds (optional)
                for target in target_columns:
                    stats[f"{target}_target"] = None  # Or set to np.nan if preferred

            feature_rows.append(stats)

    # Apply function per trajectory
    for _, group in tqdm(df.groupby("trajectory_id"), desc="Sliding thru dem trajectories"):
        compute_window_stats(group)

    feature_df = pd.DataFrame(feature_rows)

    print(f"Sliding window stats extracted. Original size: {len(df)}, New size: {len(feature_df)}")

    return feature_df


def preprocess_data(train_df, val_df, test_df, num_lags, horizon):
    if num_lags: 
        # Add lags to each datasplit
        train_df = add_lagged_value(train_df, num_lag=num_lags)
        val_df = add_lagged_value(val_df, num_lag=num_lags)
        test_df = add_lagged_value(test_df, num_lag=num_lags)


    # Add future target values
    train_df = add_future_value(train_df, horizon=horizon)
    val_df = add_future_value(val_df, horizon=horizon)
    test_df = add_future_value(test_df, horizon=horizon)

    # Drop non-numerical features
    features_to_drop = ['t','trajectory_id']
    train_df.drop(columns=features_to_drop, inplace=True)
    val_df.drop(columns=features_to_drop, inplace=True)
    test_df.drop(columns=features_to_drop, inplace=True)

    # Define input and output
    X_train = train_df.drop(columns=[f'latitude_future{horizon}',
                                     f'longitude_future{horizon}'])
    X_val = val_df.drop(columns=[f'latitude_future{horizon}',
                                     f'longitude_future{horizon}'])
    X_test = test_df.drop(columns=[f'latitude_future{horizon}',
                                     f'longitude_future{horizon}'])
    
    y_train = train_df[[f'latitude_future{horizon}',
                        f'longitude_future{horizon}']]
    
    y_val = val_df[[f'latitude_future{horizon}',
                    f'longitude_future{horizon}']]
    
    y_test = test_df[[f'latitude_future{horizon}',
                    f'longitude_future{horizon}']]
    
    print("Preprocessing complete!")
    return X_train, X_val, X_test, y_train, y_val, y_test