import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("Binary_synthetic.csv")
"""
if df is None or df.empty:
    print("No data to process")

else:
    # Convert timestamp to datetime
    df["t"] = pd.to_datetime(df["t"])

    # Sort id and timestampt to maintain chronological order
    df = df.sort_values(by=["id", "t"])

    # Create a column to track the previous label for comparison
    df["previous_label"] = df.groupby("id")["label"].shift(1)

    # Identify cases where label changes
    label_changes = df["label"] != df["previous_label"]

    # Identify cases where time_gap is greater than 1 hour
    large_time_gap = df["time_gap"] >= 3600

    # Generate unique trajectory ID whenever label changes, or when it's the first record for a ship
    df["trajectory_id"] = (label_changes | large_time_gap).cumsum()

    # Drop previous_label column
    df = df.drop(columns=["previous_label"])

    # Save modified data
    df.to_csv("Trajectory_IDs.csv", index=False)
"""
# Load data
df = pd.read_csv("Trajectory_IDs.csv")
"""

# Count unique trajectory IDs per label
trajectory_counts = df.groupby("label")["trajectory_id"].nunique()

print("Unique Trajectories Count:")
print(f"Sailing (0): {trajectory_counts.get(0, 0)}")
print(f"Fishing (1): {trajectory_counts.get(1, 0)}")

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

# Empty lists to store final splits
train_data = []
val_data = []
test_data = []

# Split seperately for sailing and fishing
for label, group in df.groupby("label"):
    unique_trajectories = group["trajectory_id"].unique()
    np.random.shuffle(unique_trajectories)

    # Compute split sizes
    num_train = int(len(unique_trajectories) * train_ratio)
    num_val = int(len(unique_trajectories) * val_ratio)

    # Assign trajectories to train, val, and test sets
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

train_df.to_csv("Train.csv", index=False)
val_df.to_csv("Validation.csv", index=False)    
test_df.to_csv("Test.csv", index=False)

# Print sizes to verify balance
print(f"Train size: {len(train_df)}, Sailing: {len(train_df[train_df['label'] == 0])}, Fishing: {len(train_df[train_df['label'] == 1])}")
print(f"Validation size: {len(val_df)}, Sailing: {len(val_df[val_df['label'] == 0])}, Fishing: {len(val_df[val_df['label'] == 1])}")
print(f"Test size: {len(test_df)}, Sailing: {len(test_df[test_df['label'] == 0])}, Fishing: {len(test_df[test_df['label'] == 1])}")

"""

# Count unique trajectory IDs per label
trajectory_counts = df.groupby("label")["trajectory_id"].nunique()

print("Unique Trajectories Count:")
print(f"Sailing (0): {trajectory_counts.get(0, 0)}")
print(f"Fishing (1): {trajectory_counts.get(1, 0)}")