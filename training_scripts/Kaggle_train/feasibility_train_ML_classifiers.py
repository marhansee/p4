import pandas as pd
import os
from tqdm import tqdm   
from testing_scripts.feasibility_tests.feasibility_train_test import evaluate_model, train_all_classifiers

feature_columns = ["signed_turn", "bearing", "time_gap", "distance_gap", "euc_speed", "distanceToShore", "longitude", "latitude"]

results = []

folders = [("preprocessed_datasets", "clean"), ("Corrupted_Datasets", "corrupted")]
val_csv = "Split_Test.csv"

val_df = pd.read_csv(val_csv)
if val_df["label"].isna().any():
    print("⚠️ NaNs found in validation labels – dropping those rows.")
    val_df = val_df.dropna(subset=["label"])
val_df.to_csv("temp_val.csv", index=False) 

for folder, corruption_type in folders:
    all_csvs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]

    for train_csv in tqdm(all_csvs, desc=f"Training from {folder}"):
        train_df = pd.read_csv(train_csv)

        if train_df["label"].isna().any():
            print(f"⚠️ NaNs found in train labels for {os.path.basename(train_csv)} – dropping those rows.")
            train_df = train_df.dropna(subset=["label"])
            train_df.to_csv("temp_train.csv", index=False)  # Save cleaned version
            train_csv = "temp_train.csv" 
        else:
            pass

        model_metrics = train_all_classifiers(train_csv, "temp_val.csv", feature_columns)
        for res in model_metrics:
            res["corruption"] = corruption_type
        results.extend(model_metrics)

results_df = pd.DataFrame(results)
results_df.to_csv("all_classifiers_results.csv", index=False)

summary = results_df.groupby(["corruption", "model"])[["accuracy", "f1_score"]].mean().round(3)
print("Summary of Results (Grouped by Corruption + Model)")
print(summary)