
import torch
import torch.nn as nn
from thop import profile

class CNN1DForecaster(nn.Module):
    def __init__(self, n_features, seq_len, out_channels=64, output_size=2, output_seq_len=20, dropout_prob=0.2):
        super(CNN1DForecaster, self).__init__()
        self.output_size = output_size
        self.output_seq_len = output_seq_len
        
        # Convolutional block using Sequential
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=out_channels, 
                      kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, 
                      kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, seq_len)
            x = self.feature_extractor(dummy_input)
            self.flattened_size = x.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # Add dropout here
            nn.Linear(out_channels, self.output_size * self.output_seq_len)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = x.view(x.size(0), self.output_seq_len, self.output_size)  # Output shape: (batch, samples, features)
        return x

    
def main():
    n_features = 10
    sequence_length = 60
    dummy_input = torch.randn(1, sequence_length, n_features)
    model = CNN1DForecaster(n_features=n_features, seq_len=sequence_length, output_seq_len=15)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 20, 2)
    
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()