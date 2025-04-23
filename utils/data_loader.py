import torch
from torch.utils.data import Dataset, DataLoader

class Forecasting_Dataloader(Dataset):
    def __init__(self, X, y, seq_length=10):
        """
        Args:
        - X: Input features (Pandas DataFrame or PySpark DataFrame)
        - y: Target values (Pandas DataFrame)
        - seq_length: Number of past timesteps to use
        """
        self.X = torch.tensor(X.values, dtype=torch.float32)  # Assuming X is a Pandas DataFrame
        self.y = torch.tensor(y.values, dtype=torch.float32)  # y contains pairs of lat and lon for each timestep
        self.seq_length = seq_length
        self.horizon = self.y.shape[1] // 2  # Number of pairs (lat/lon)

    def __len__(self):
        return len(self.X) - (self.seq_length + self.horizon)

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]  # Sequence of input features
        y_target = self.y[idx+self.seq_length:idx+self.seq_length+self.horizon]  # Target: Latitude and Longitude pairs
        return X_seq, y_target
    

class Classifier_Dataloader(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]  # Sequence of features
        y_target = self.y[idx+self.seq_length-1]   # Single classification label
        return X_seq, y_target
