import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

class CNN1DClassifier(nn.Module):
    def __init__(self, n_features):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4, 100)  # input length 10 -> conv(3) = 8, pool(2) = 4
        self.out = nn.Linear(100, 1)       # output raw logits

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)  # no sigmoid here
        return x

def create_sequences(df, features, seq_len):
    X, y = [], []
    for i in range(len(df) - seq_len + 1):
        seq = df.iloc[i:i+seq_len]
        label_seq = seq["label"].values
        if np.isnan(label_seq).any():
            continue
        features_seq = seq[features].values
        majority_label = int(np.clip(np.round(label_seq.mean()), 0, 1))
        X.append(features_seq)
        y.append(majority_label)
    return np.array(X), np.array(y)

def train_cnn_classifier(train_csv, val_csv, feature_columns, sequence_length=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    train_df = pd.read_csv(train_csv).dropna(subset=["label"])
    val_df = pd.read_csv(val_csv).dropna(subset=["label"])

    X_train, y_train = create_sequences(train_df, feature_columns, sequence_length)
    X_val, y_val = create_sequences(val_df, feature_columns, sequence_length)

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(np.clip(y_train, 0, 1), dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model = CNN1DClassifier(n_features=len(feature_columns)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        logits = model(X_val.to(device)).cpu().numpy().flatten()
        end_time = time.time()
        inference_time = end_time - start_time

        probs = 1 / (1 + np.exp(-logits))  # manual sigmoid
        preds = (probs > 0.5).astype(int)
        y_val_np = y_val.numpy().flatten()

    return {
        "model": "1D_CNN",
        "accuracy": accuracy_score(y_val_np, preds),
        "precision": precision_score(y_val_np, preds, zero_division=0),
        "recall": recall_score(y_val_np, preds, zero_division=0),
        "f1_score": f1_score(y_val_np, preds, zero_division=0),
        "inference_time_sec": inference_time,
        "train_csv": train_csv
    }

    
