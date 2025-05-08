import torch.nn as nn
import torch
from thop import profile
from torchsummary import summary

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_seq_len=20, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden)
        out = out[:, -self.output_seq_len:, :]  # Take the last 20 steps
        out = self.fc(out)  # (batch, 20, 2)
        return out

def main():
    n_features = 10
    sequence_length = 60
    output_sequence_length = 20
    dummy_input = torch.randn(1, sequence_length, n_features)
    model = LSTMModel(input_size=n_features, hidden_size=64, num_layers=2, output_size=2, output_seq_len=output_sequence_length)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 20, 2)

    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()