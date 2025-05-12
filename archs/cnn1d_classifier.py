
import torch.nn as nn
import torch
from thop import profile

class CNN1DClassifier(nn.Module):
    def __init__(self, n_features, seq_len, out_channels=64, num_classes=1):
        super(CNN1DClassifier, self).__init__()

        # Convolutional block using Sequential
        self.feature_extractor = nn.Sequential(
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

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, seq_len)
            x = self.feature_extractor(dummy_input)
            self.flattened_size = x.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

def main():
    n_features = 10
    sequence_length = 60
    out_channel = 64
    dummy_input = torch.randn(1,sequence_length, n_features) # 1-D CNN expects [batch, seq_length, 10]
    model = CNN1DClassifier(n_features=n_features, seq_len=sequence_length, out_channels=out_channel, num_classes=1)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 1)
    
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()