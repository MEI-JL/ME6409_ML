import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse time dim -> 1
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, channels, window_size)
        x = self.conv_block(x)   # (batch, 64, window_size)
        x = self.pool(x)         # (batch, 64, 1)
        x = x.squeeze(-1)        # (batch, 64)
        x = self.fc(x)           # (batch, 1)
        return x