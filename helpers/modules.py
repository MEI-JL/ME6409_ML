import torch
import torch.nn as nn

def init_model_params(model:nn.Module) -> None:

    def _init_params(m:nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Linear, 
                        nn.GRU,nn.RNN, nn.LSTM)):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    model.apply(_init_params)


class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,
                 kernel_size:int=5, padding:int=2, dilation:int=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 
                        kernel_size = kernel_size,
                        padding = padding,
                        dilation = dilation),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class KneeCNN(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse time dim -> 1

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, window_size)
        x = self.encoder(x)      # (batch, 64, window_size)

        x = self.pool(x)         # (batch, 64, 1)
        x = x.squeeze(-1)        # (batch, 64)
        
        # import pdb; pdb.set_trace()
        x = self.regressor(x)    # (batch, 1)
        # pdb.set_trace()
        return x


class KneeTCN(nn.Module):
    # similar, but dilate and do asymmetric padding for causal conv
    # https://arxiv.org/pdf/1803.01271
    def __init__(self, in_channels:int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse time dim -> 1

        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: (batch, channels, window_size)
        x = self.encoder(x)      # (batch, 64, window_size)
        x = self.pool(x)         # (batch, 64, 1)
        x = x.squeeze(-1)        # (batch, 64)
        x = self.regressor(x)    # (batch, 1)
        return x
    
    
class KneeLSTM(nn.Module):
    # remember to permute input data!
    def __init__(self, in_channels:int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse time dim -> 1

        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        # x: (batch, channels, window_size)
        x = x.permute(2, 0, 1) # change back to (window_size, batch, channels)

        x = self.encoder(x)      # (batch, 64, window_size)
        x = self.pool(x)         # (batch, 64, 1)
        x = x.squeeze(-1)        # (batch, 64)
        x = self.regressor(x)    # (batch, 1)
        return x