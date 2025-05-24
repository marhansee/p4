import pandas as pd 
import numpy as np
from tqdm import tqdm
import time
import random
import os
from itertools import combinations

def split_data(input_csv , train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, output_prefix="Split_"):
    """
    Splits the data into train, validation, and test sets.
    
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - train_ratio (float): The proportion of the data to be used for training (default 0.7).
    - val_ratio (float): The proportion of the data to be used for validation (default 0.1).
    - test_ratio (float): The proportion of the data to be used for testing (default 0.2).
    - output_prefix (str): Prefix for the output files.
    
    Returns:
    - None: The function saves the splits as CSV files.
    """
    #load data
    df = pd.read_csv(input_csv)

    # Count unique trajectory IDs per label
    trajectory_counts = df.groupby("label")["trajectory_id"].nunique()
    print("Unique Trajectories Count: \nSailing (0): {trajectory_counts.get(0, 0)} \nFishing (1): {trajectory_counts.get(1, 0)}")

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

    # Save splits to CSV
    train_df.to_csv(f"{output_prefix}Train.csv", index=False)
    val_df.to_csv(f"{output_prefix}Validation.csv", index=False)    
    test_df.to_csv(f"{output_prefix}Test.csv", index=False)

    # Print sizes to verify balance
    print(f"Train size: {len(train_df)}, Sailing: {len(train_df[train_df['label'] == 0])}, Fishing: {len(train_df[train_df['label'] == 1])}")
    print(f"Validation size: {len(val_df)}, Sailing: {len(val_df[val_df['label'] == 0])}, Fishing: {len(val_df[val_df['label'] == 1])}")
    print(f"Test size: {len(test_df)}, Sailing: {len(test_df[test_df['label'] == 0])}, Fishing: {len(test_df[test_df['label'] == 1])}")

def missing_values(input_csv, output_csv, missing_dict):
    """"
    Introduces missing values into specified features as defined percentages"
        
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the modified CSV file.
    - missing_dict (dict): Dictionary with {feature_name: missing_percentage} pairs.
    
    Returns:
    - None:  The function saves the modified DataFrame to a CSV file.
    """

    #Load data
    df = pd.read_csv(input_csv)

    for feature, percentage in missing_dict.items():
        if feature in df.columns:
            # Determine how many values to remove
            num_missing = int(len(df) * (percentage / 100))

            # Randomly select indices to set as NaN
            missing_indices = np.random.choice(df.index, num_missing, replace=False)
            df.loc[missing_indices, feature] = np.nan

            print(f"Introduced {num_missing} missing values in '{feature}' {percentage}%")
        else:
            print(f"Feature '{feature}' not found in DataFrame. Skipping.")

    # Save modified data
    df.to_csv(output_csv, index=False)

def duplicates(input_csv, output_csv, duplicate_percentage=30):
    """
    Introduces duplicate rows into the dataset at a specified percentage.
    
    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the modified CSV file.
    - duplicate_percentage (float): Percentage of the dataset to duplicate (0-100).
    
    Returns:
    - None: Saves the modified DataFrame as a CSV file.
    """

    # Load the data
    df = pd.read_csv(input_csv)

    # How many rows to duplicate
    num_dupes = int(len(df) * (duplicate_percentage / 100))

    if num_dupes == 0:
        print(f"Bruh, 0% duplication? That's just the same dataset. Try again.")
        return
    
    # Randomly select rows to duplicate
    dupe_rows = df.sample(n=num_dupes, replace=True)

    # Append and shuffle to avoid stacked dupes
    df = pd.concat([df, dupe_rows]).sample(frac=1).reset_index(drop=True)

    print(f"Introduced {num_dupes} duplicate rows ({duplicate_percentage}%)")

    # Save modified dataset
    df.to_csv(output_csv, index=False)

def label_noise(input_csv, output_csv, noise_percentage):
    """
    Introduces label noise by randomly flipping some 'fishing' (1) labels to 'sailing' (0).

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the modified CSV file.
    - noise_percentage (float): Percentage of fishing (1) labels to flip to sailing (0).
    
    Returns:
    - None: Saves the modified DataFrame as a CSV file.
    """

    # Load data
    df = pd.read_csv(input_csv)

    # Get all trajectory IDs
    fishing_trajectories = df[df["label"] == 1]["trajectory_id"].unique()

    # Determine how many trajectories to corrupt
    num_noisy_trajectories = int(len(fishing_trajectories) * (noise_percentage / 100))

    if num_noisy_trajectories == 0:
        print("Bruh, 0% noise? You just running the same dataset fr. Try again.")
        return
    
    # Randomly select trajectories
    noisy_trajectories = np.random.choice(fishing_trajectories, num_noisy_trajectories, replace=False)
 
    # Flip labels of selected trajectories
    df.loc[df["trajectory_id"].isin(noisy_trajectories), "label"] = 0

    print(f"Introduced noise to {num_noisy_trajectories} fishing trajectories ({noise_percentage}%).")

    # Save dataset
    df.to_csv(output_csv, index=False)

def downsampling(input_csv, output_csv, time_interval):

    """
    Downsamples each trajectory by selecting a single data sample per a specified time interval.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the downsampled CSV file.
    - time_interval (int): Time interval in seconds to keep a single data point.

    Returns:
    - None: Saves the downsampled DataFrame as a CSV file.
    """

    # Load data
    df = pd.read_csv(input_csv)

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

    downsampled_df.to_csv(output_csv, index=False)

def resampling(input_csv, output_csv, time_interval):
    """
    Resamples each trajectory at fixed time intervals using linear interpolation.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the resampled CSV file.
    - time_interval (int): Time interval in seconds for resampling.

    Returns:
    - None: Saves the resampled DataFrame as a CSV file.
    """

    # Load data
    df = pd.read_csv(input_csv)

    # Sort by trajectory and timestamp to maintain order
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values(by=["trajectory_id", "t"])

    # Function to resample and interpolate per trajectory
    def resample_and_interpolate(group):
        # Determine label for entire trajectory
        label_val = group["label"].dropna().unique()
        if len(label_val) == 1:
            label_val = label_val[0]
        else:
            label_val = group["label"].mode().iloc[0] if not group["label"].mode().empty else np.nan

        group = group.sort_values("t")
        group = group[~group["t"].duplicated(keep="first")]

        group = group.set_index("t")
        group = group.resample(f"{time_interval}s").asfreq()

        # Drop everything before the first real data point
        first_valid = group.first_valid_index()
        group = group.loc[first_valid:]

        # Restore 'id' if needed
        if "id" in group.columns:
            group["id"] = group["id"].ffill().bfill()

        # Interpolate numerical columns
        group = group.infer_objects(copy=False)
        group = group.interpolate(method="linear", limit_direction="both")

        # Restore label
        group["label"] = label_val

        # Drop rows where all columns except label are NaN
        exempt_cols = ["label"]
        non_exempt = [col for col in group.columns if col not in exempt_cols]
        group = group.dropna(subset=non_exempt, how="all")

        return group.reset_index()  

    # Apply resampling and interpolation per trajectory
    resampled_df = df.groupby("trajectory_id", group_keys=False).apply(resample_and_interpolate, include_groups=True)

    print(f"Resampled dataset at {time_interval}s intervals. Original size: {len(df)}, New size: {len(resampled_df)}")

    # Save dataset
    resampled_df.to_csv(output_csv, index=False)

def upsampling(input_csv, output_dir):
    intervals = [1, 11]
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    df["t"] = pd.to_datetime(df["t"])

    def upsample_and_interpolate(group, interval):
        label_val = group["label"].dropna().unique()
        label_val = label_val[0] if len(label_val) == 1 else (
            group["label"].mode().iloc[0] if not group["label"].mode().empty else np.nan
        )

        group = group.sort_values("t").drop_duplicates(subset="t", keep="first")
        group = group.set_index("t").resample(f"{interval}s").asfreq()

        group = group.loc[group.first_valid_index():]
        if "id" in group.columns:
            group["id"] = group["id"].ffill().bfill()

        group = group.infer_objects(copy=False).interpolate(method="linear", limit_direction="both")
        group["label"] = label_val

        exempt = ["label"]
        non_exempt = [col for col in group.columns if col not in exempt]
        group = group.dropna(subset=non_exempt, how="all")
        return group.reset_index()

    for interval in intervals:
        output_path = os.path.join(
            output_dir, f"{os.path.splitext(os.path.basename(input_csv))[0]}_upsampled_{interval}s.csv"
        )
        wrote_header = False

        for _, traj in df.groupby("trajectory_id", group_keys=False):
            try:
                upsampled = upsample_and_interpolate(traj.copy(), interval)
                upsampled.to_csv(output_path, index=False, mode='a', header=not wrote_header)
                wrote_header = True
            except Exception as e:
                print(f"Failed trajectory: {traj['trajectory_id'].iloc[0]} — {e}")


def sliding_window_features(input_csv, output_csv, window_size, step_size, features):

    """
    Extracts statistical features using a sliding window per trajectory.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the feature dataset.
    - window_size (int): Number of samples per sliding window.
    - step_size (int): Step size for moving the window.
    - features (list): List of column names to extract statistics from.

    Returns:
    - None: Saves the new DataFrame as a CSV file.
    """

    # Load data
    df = pd.read_csv(input_csv)

    # Sort by trajectory and timestamp to maintain order
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values(by=["trajectory_id", "t"])

    # List to store processed windows
    feature_rows = []

    # Function to extract features from a window
    def compute_window_stats(group):
        for start in range(0, len(group) - window_size +1, step_size):
            window = group.iloc[start:start + window_size] # Slice the window
            stats = {
                "trajectory_id": group["trajectory_id"].iloc[0],
                "window_start" : window["t"].iloc[0],
                "window_end": window["t"].iloc[-1],
            }
            for feature in features:
                stats[f"{feature}_mean"] = window[feature].mean()
                stats[f"{feature}_median"] = window[feature].median()
                stats[f"{feature}_min"] = window[feature].min()
                stats[f"{feature}_max"] = window[feature].max()
                stats[f"{feature}_std"] = window[feature].std()

            # Add label 
            if "label" in window.columns:
                stats["label"] = window["label"].mode()[0]  # Most common label in the window
            feature_rows.append(stats)

    # Apply function per trajectory
    for _, group in tqdm(df.groupby("trajectory_id"), desc="Sliding thru dem trajectories"):
        compute_window_stats(group)

    feature_df = pd.DataFrame(feature_rows)

    print(f"Sliding window stats extracted. Original size: {len(df)}, New size: {len(feature_df)}")

    feature_df.to_csv(output_csv, index=False)

def measure_inference_time(model, X_val, n_runs=100):
    """
    Measure average inference time of a model.

    Args:
        model: Trained classifier
        X_val: Validation features (pandas DataFrame or ndarray)
        n_runs: Number of samples to time

    Returns:
        avg_time: Average prediction time per sample (in milliseconds)
    """
    sample_indices = np.random.choice(len(X_val), size=min(n_runs, len(X_val)), replace=False)
    samples = X_val.iloc[sample_indices] if hasattr(X_val, 'iloc') else X_val[sample_indices]

    start = time.time()
    for i in range(len(samples)):
        _ = model.predict(samples[i:i+1])
    end = time.time()

    total_time = end - start
    avg_time_ms = (total_time / len(samples)) * 1000
    return avg_time_ms

def remove_random_percentage(group):
    percentage_to_remove = random.uniform(1, 10)  # random float between 1 and 10
    num_rows_to_remove = int(len(group) * (percentage_to_remove / 100))
    rows_to_drop = group.sample(n=num_rows_to_remove, random_state=42).index
    return group.drop(rows_to_drop)

def synthesize_irregular_sampling(df):
    df = df.groupby('trajectory_id').apply(remove_random_percentage).reset_index(drop=True)
    return df

def synth_irregular_file(input_path, output_path):
    df = pd.read_csv(input_path)
    corrupted_df = synthesize_irregular_sampling(df)
    corrupted_df.to_csv(output_path, index=False)

def make_all_corrupted_versions(base_train_csv, output_dir, corruption_settings):
    """
    Generates all individual + combo corrupted datasets.

    Parameters:
    - base_train_csv (str): Path to clean train.csv
    - output_dir (str): Directory to save all corrupted versions
    - corruption_settings (dict): Dict with corruption types & params
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pull static corruptions
    missing_config = corruption_settings["missing"]
    dupes_config = corruption_settings["dupes"]
    label_noise_levels = corruption_settings["labelnoise"]

    corruption_funcs = {
        "missing": lambda inp, out: missing_values(inp, out, missing_config),
        "dupes": lambda inp, out: duplicates(inp, out, dupes_config),
        "labelnoise": lambda inp, out, lvl: label_noise(inp, out, lvl),
        "irregular" : lambda inp, out: synth_irregular_file(inp, out)
    }

    base_combo_types = ["missing", "dupes", "labelnoise", "irregular"]

    # Create individual + all combo datasets
    for r in range(1, len(base_combo_types)+1):
        for combo in combinations(base_combo_types, r):
            noise_levels = label_noise_levels if "labelnoise" in combo else [None]

            for noise_level in noise_levels:
                combo_name_parts = []
                for c in combo:
                    if c == "labelnoise":
                        combo_name_parts.append(f"{c}_{noise_level}")
                    else:
                        combo_name_parts.append(c)
                combo_name = "_".join(combo_name_parts)

                current_file = os.path.join(output_dir, f"train_{combo_name}.csv")


                temp_input = base_train_csv
                for i, c_type in enumerate(combo):
                    is_last = i == len(combo) - 1
                    temp_output = current_file if is_last else f"temp_{combo_name}_{i}.csv"
                    if c_type == "labelnoise":
                        corruption_funcs[c_type](temp_input, temp_output, noise_level)
                    else:
                        corruption_funcs[c_type](temp_input, temp_output)

                    temp_input = temp_output

                # Clean temp files
                for i in range(len(combo) - 1):
                    try:
                        os.remove(f"temp_{combo_name}_{i}.csv")
                    except FileNotFoundError:
                        pass

    print("All corrupted datasets generated.")


#sliding_window_features("Trajectory_IDs.csv", "window_dataset.csv", window_size=5, step_size=1, features=["euc_speed", "distanceToShore", "signed_turn", "bearing"])

#df = pd.read_csv("window_dataset.csv")
#print(df.head())
#print(f" Extracted {len(df)} window features, check 'windowed_dataset.csv' for full dump.")
#Is this per traject? ^

#corrupted_settings = {"missing" : {"signed_turn" : 29,"bearing" :20, "euc_speed" :1, "distanceToShore" :70}, "dupes" : 30, "labelnoise" : [10,15,20,30], "irregular" : True}

#make_all_corrupted_versions("Split_Train.csv", "Corrupted_Datasets", corruption_settings=corrupted_settings)

