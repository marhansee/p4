import torch
from torch.utils.data import Dataset, DataLoader
# from petastorm.pytorch import DataLoader
# from petastorm.reader import make_batch_reader

class Forecasting_Dataloader(Dataset):
    def __init__(self, X, y, seq_length=10):
        """
        Args:
        - X: Input features (NumPy array)
        - y: Target values (NumPy array)
        - seq_length: Number of past timesteps to use
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # Assuming X is a NumPy array
        self.y = torch.tensor(y, dtype=torch.float32)  # y contains pairs of lat and lon for each timestep
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - (self.seq_length) + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]  # Sequence of input features
        y_target = self.y[idx+self.seq_length-1]  # Target: Latitude and Longitude pairs
        return X_seq, y_target
    

class Classifier_Dataloader(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]  # Sequence of features
        y_target = self.y[idx+self.seq_length-1]   # Single classification label
        return X_seq, y_target

class Classifier_Dataloader2(Dataset):
    def __init__(self, X_sequences, y_labels):
        self.X = torch.tensor(X_sequences, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# def get_sequence_data_loader(data_path, batch_size=32, num_epochs=1):
#     return DataLoader(
#         make_batch_reader(dataset_url=data_path, num_epochs=num_epochs),
#         batch_size=batch_size
#     )