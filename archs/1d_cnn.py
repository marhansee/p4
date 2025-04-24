
import torch.nn as nn
import torch

class One_D_CNN(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(One_D_CNN, self).__init__()
        
        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 5, 512)  # 128 filters, reduced by pooling, assuming seq_length=10
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, input_size, seq_length)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Flatten the output
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x