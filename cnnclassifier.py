from CNN1D import CNN1DClassifier, train_cnn_classifier
import os
from tqdm import tqdm
import pandas as pd

feature_columns = ["signed_turn", "bearing", "time_gap", "distance_gap", "euc_speed", "distanceToShore", "longitude", "latitude"]

cnn_results = []

folders = [("preprocessed_datasets", "clean"), ("Corrupted_Datasets", "corrupted")]
val_csv = "Split_Test.csv"

for folder, corruption_type in folders:
    all_csvs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

    for train_csv in tqdm(all_csvs, desc=f"Training CNN from {folder}"):
        metrics = train_cnn_classifier(train_csv, val_csv, feature_columns)
        metrics["corruption"] = corruption_type
        cnn_results.append(metrics)

cnn_df = pd.DataFrame(cnn_results)
cnn_df.to_csv("cnn_results.csv", index=False)   