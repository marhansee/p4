import torch
from thop import profile

import torch.nn as nn
import torch

"""
This script has been copied from Rahman, Mijanur (2022).
SOURCE: https://medium.com/@mijanr/different-ways-to-combine-cnn-and-lstm-networks-for-time-series-classification-tasks-b03fc37e91b6
"""

class CNN_LSTM(nn.Module):
    def __init__(self, n_features, out_channels=64, hidden_size=64, num_layers=2, num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=out_channels, 
                      kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        # self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        #cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc1(out[:, -1, :])
        # out = self.relu(out)
        # out = self.fc2(out)

        return out

def main():
    n_features = 10
    sequence_length = 60
    out_channel = 64
    dummy_input = torch.randn(1,sequence_length, n_features) # 1-D CNN expects [batch, seq_length, 10]
    model = CNN_LSTM(n_features=n_features, out_channels=out_channel, num_classes=1)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 1)
    
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()