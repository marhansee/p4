import pandas as pd
import numpy as np
import random
from utils.preprocessing import apply_downsampling, apply_resampling

def add_duplicates(df, duplicate_percentage=30):
    """
    Introduces duplicate rows into the dataset at a specified percentage.
    
    Parameters:
    - df: Pandas DataFrame with data
    - duplicate_percentage (float): Percentage of the dataset to duplicate (0-100).
    
    Returns:
    - df: Modified DataFrame
    """

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
    return df

def add_label_noise(df, noise_percentage):
    """
    Introduces label noise by randomly flipping some 'fishing' (1) labels to 'sailing' (0).

    Parameters:
    - df: Pandas DataFrame with data
    - noise_percentage (float): Percentage of fishing (1) labels to flip to sailing (0).
    
    Returns:
    - df: Modified DataFrame
    """

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
    return df

def add_missing_values(df, missing_dict):
    """
    Introduces missing values into specified features as defined percentages"
        
    Parameters:
    - df: Pandas DataFrame with data
    - missing_dict (dict): Dictionary with {feature_name: missing_percentage} pairs.
    
    Returns:
    - df: Modified DataFrame
    """

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
    return df

def remove_from_group(group, min_percentage, max_percentage, random_state=None):
    """
    Removes a random percentage of rows from a trajectory within a given range.

    Args:
        - group: The feature to group by
        - min_percentage: Minimum percentage of rows to be removed from the group
        - max_percentage: Maximum percentage of rows to be removed from the group
    """

    # Set random state
    if random_state is not None:
        random.seed(random_state + hash(group.name)) # Random state for each group
    
    # Randomly select a percentage between min_percentage and max_percentage
    percentage_to_remove = random.uniform(min_percentage, max_percentage)
    
    # Calculate the number of rows to remove from the group
    num_rows_to_remove = int(len(group) * (percentage_to_remove / 100))
    
    # Randomly select the rows to drop
    rows_to_drop = group.sample(n=num_rows_to_remove, random_state=random_state).index
    
    # Drop the selected rows
    return group.drop(rows_to_drop)

def synthesize_irregular_sampling(df, min_percentage=1, max_percentage=10, random_state=42):
    """
    Removes a random percentage of rows from each group of "trajectory_id" in the dataframe within a given range.
    
    Args:
        - df (pd.DataFrame): The input dataframe with a column "trajectory_id".
        - min_percentage (float): The minimum percentage of rows to remove from each group (default is 1%).
        - max_percentage (float): The maximum percentage of rows to remove from each group (default is 10%).
    
    Returns:
        - df: The dataframe with the random rows removed.
    """
    
    # Apply the function to each group and return the result
    df = df.groupby('trajectory_id').apply(
        remove_from_group, 
        min_percentage=min_percentage, 
        max_percentage=max_percentage,
        random_state=random_state).reset_index(drop=True)
    
    return df

def add_degradations(config, train_df, val_df, test_df):
    # Define missing value percentages
    missing_values_features = {
        'signed_turn': 29,
        'bearing': 21,
        'euc_speed': 1,
        'distanceToShore': 70
    }


    if config['degradations']['apply_irregular_sampling']:
        train_df = synthesize_irregular_sampling(train_df,
                    min_percentage=1, max_percentage=10, random_state=42)
        val_df = synthesize_irregular_sampling(val_df,
                    min_percentage=1, max_percentage=10, random_state=42)
        test_df = synthesize_irregular_sampling(test_df,
                    min_percentage=1, max_percentage=10, random_state=42)


    if config['degradations']['apply_downsampling']:
        train_df = apply_downsampling(train_df, 
                   time_interval=config['degradations']['sampling_interval'])
        val_df = apply_downsampling(val_df,
                   time_interval=config['degradations']['sampling_interval'])
        test_df = apply_downsampling(test_df,
                   time_interval=config['degradations']['sampling_interval'])
        
    if config['degradations']['apply_resampling']:
        train_df = apply_resampling(train_df,
                   time_interval=config['degradations']['sampling_interval'])
        val_df = apply_resampling(val_df,
                   time_interval=config['degradations']['sampling_interval'])
        test_df = apply_resampling(test_df,
                   time_interval=config['degradations']['sampling_interval'])


    # Add degradations
    if config['degradations']['add_missing_values']:
        train_df = add_missing_values(train_df, 
                   missing_dict=missing_values_features)
        val_df = add_missing_values(val_df,
                   missing_dict=missing_values_features)
        test_df = add_missing_values(test_df,
                   missing_dict=missing_values_features)
        if config['degradations']['impute_strategy'] == 'zero':
            train_df.fillna(0, inplace=True)
            val_df.fillna(0, inplace=True)
            test_df.fillna(0, inplace=True)
        elif config['degradations']['impute_strategy'] == 'linear':
            train_df.interpolate(method='linear', inplace=True, limit_direction='both')
            val_df.interpolate(method='linear', inplace=True, limit_direction='both')
            test_df.interpolate(method='linear', inplace=True, limit_direction='both')
        elif config['degradations']['impute_strategy'] == 'mean':
            for feature in missing_values_features.keys():
                train_df[feature].fillna(train_df[feature].mean(), inplace=True)
                val_df[feature].fillna(val_df[feature].mean(), inplace=True)
                test_df[feature].fillna(test_df[feature].mean(), inplace=True)


    if config['degradations']['add_duplicates']:
        train_df = add_duplicates(train_df, duplicate_percentage=30)
        val_df = add_duplicates(val_df, duplicate_percentage=30)
        test_df = add_duplicates(test_df, duplicate_percentage=30)

    print("Added degradations!")
    return train_df, val_df, test_df