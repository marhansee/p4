import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length=10, horizon=10):
        """
        Args:
        - X: Input features (Pandas DataFrame)
        - y: Target values (Pandas DataFrame)
        - seq_length: Number of past timesteps to use
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.seq_length = seq_length
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - (self.seq_length + self.horizon)

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]  # Sequence of input features
        y_target = self.y[idx+self.seq_length+self.horizon]  # Target [latitude, longitude]
        return X_seq, y_target
