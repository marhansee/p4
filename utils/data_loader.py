import torch
from torch.utils.data import Dataset, DataLoader

class System_Dataloader(Dataset):
    def __init__(self, X_sequences, y_labels):
        self.X = torch.tensor(X_sequences, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
