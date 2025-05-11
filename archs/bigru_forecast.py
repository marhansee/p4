import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

class BiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_seq_len=20, dropout_prob=0.2):
        super(BiGRUModel, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.output_seq_len = output_seq_len

        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim,
            batch_first=True, dropout=dropout_prob,
            bidirectional=True
        )

        # Fully connected layer for each timestep
        self.fc = nn.Linear(hidden_dim * 2, 2) # output two features

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device)

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
    model = BiGRUModel(input_dim=n_features, hidden_dim=64,dropout_prob=0.2, 
                       layer_dim=2, output_seq_len=output_seq_length)

    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 20, 2)


    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()