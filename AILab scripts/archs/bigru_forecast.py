import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

class BiGRUModel(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, output_seq_len=20, output_size=2, dropout_prob=0.2):
        super(BiGRUModel, self).__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        self.output_size = output_size

        self.gru = nn.GRU(
            self.n_features, self.hidden_size, self.num_layers,
            batch_first=True, 
            dropout=dropout_prob,
            bidirectional=True
        )

        # Fully connected layer for each timestep
        self.fc = nn.Linear(hidden_size * 2, output_size) # output two features

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.gru(x, h0)

        # Select the last 20 timesteps
        out = out[:, -self.output_seq_len:, :]  # Shape: (batch, 20, hidden*2)

        out = self.fc(out)  # Apply fc to each of the 20 timesteps -> (batch, 20, 2)

        return out
    
def main():
    n_features = 10
    sequence_length = 60
    output_seq_length = 20
    dummy_input = torch.randn(1, sequence_length, n_features)
    model = BiGRUModel(n_features=n_features, hidden_size=64,dropout_prob=0.2, 
                       num_layers=2, output_seq_len=output_seq_length)

    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 20, 2)


    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()
