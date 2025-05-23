import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import numpy as np
from function_house import missing_values, duplicates, downsampling

train_clean = "Split_Train.csv" 
test_clean = "Split_Validation.csv"

train_missing = "train_with_missing.csv"
test_missing = "test_with_missing.csv"

train_duped = "train_with_duplicates.csv"
test_duped = "test_with_duplicates.csv"

train_final = "train_final.csv"
test_final = "test_final.csv"

missing_cfg = {"signed_turn": 29, "bearing": 20, "euc_speed": 1, "distanceToShore": 70}
dupe_percent = 30
downsample_seconds = 60

missing_values(train_clean, train_missing, missing_cfg)
missing_values(test_clean, test_missing, missing_cfg)

duplicates(train_missing, train_duped, dupe_percent)
duplicates(test_missing, test_duped, dupe_percent)

downsampling(train_duped, train_final, downsample_seconds)
downsampling(test_duped, test_final, downsample_seconds)

train_csv = "train_corrupted_final.csv"
test_csv = "test_corrupted_final.csv"

pd.read_csv(train_final).to_csv(train_csv, index=False)
pd.read_csv(test_final).to_csv(test_csv, index=False)

train_csv = "train_corrupted_final.csv"
test_csv = "test_corrupted_final.csv"

full_features = ["signed_turn", "bearing", "time_gap", "distance_gap", "euc_speed", "distanceToShore", "longitude", "latitude"]
reduced_features = ["distance_gap", "euc_speed", "distanceToShore", "longitude", "latitude"]

def evaluate_rf_model(train_csv, test_csv, feature_columns, title_suffix):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()

    train_df[feature_columns] = train_df[feature_columns].interpolate(method='linear', limit_direction='both')
    test_df[feature_columns] = test_df[feature_columns].interpolate(method='linear', limit_direction='both')

    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Inference time
    t0 = time.time()
    _ = model.predict(X_test)
    t1 = time.time()
    inf_time = (t1 - t0) / len(X_test) * 1000  # ms per sample

    print(f"\n=== {title_suffix} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Inference: {inf_time:.4f} ms/sample")

    # Plot feature importances
    importances = model.feature_importances_
    plot_feature_importances(feature_columns, importances, f"Feature Importances ({title_suffix})")

def plot_feature_importances(features, importances, title):
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(features)[sorted_idx]
    sorted_importances = importances[sorted_idx]

    plt.figure(figsize=(10, 5))
    plt.barh(sorted_features, sorted_importances, color='dodgerblue')
    plt.xlabel("Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

evaluate_rf_model(train_csv, test_csv, full_features, "Full Feature Set")
evaluate_rf_model(train_csv, test_csv, reduced_features, "Reduced Feature Set")
