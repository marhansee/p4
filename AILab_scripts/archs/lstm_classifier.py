import torch.nn as nn
import torch
from thop import profile
from torchsummary import summary

import torch.nn as nn
import torch

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout_prob=0.2, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob
        )

        self.fc = nn.Linear(hidden_size, hidden_size)  # Output one value (logit) for binary classification
        self.fc2 = nn.Linear(hidden_size, self.num_classes)  # Final output layer
        self.relu = nn.ReLU()
        #self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden)
        last_hidden = out[:, -1, :]  # Use the last timestep's hidden state
        out = self.fc(last_hidden)  # (batch, 1)
       # out = self.norm(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out 


def main():
    n_features = 10
    sequence_length = 60
    dummy_input = torch.randn(1, sequence_length, n_features)
    model = LSTMClassifier(n_features, hidden_size=64, num_layers=2)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 20, 2)

    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()
