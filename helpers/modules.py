import torch
import torch.nn as nn
from typing import Literal
from helpers.constants import *

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

# for model input channels, messy workaround
def _get_ablated_channels_n(ablated_sensor: Literal[
                     "angle", "velocity", "imu_sim", "imu_thigh", "imu_shank"
                     ] | None = None):
    n_channels = 0
    if ablated_sensor != "angle":
        n_channels = n_channels + 1
    if ablated_sensor != "velocity":
        n_channels = n_channels + 1
    if ablated_sensor != "imu_sim":
        if ablated_sensor != "imu_thigh":
            n_channels = n_channels + len(IMU_THIGH_COLS)
        if ablated_sensor != "imu_shank":
            n_channels = n_channels + len(IMU_SHANK_COLS)

    return n_channels

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
    def __init__(self, 
                 ablated_sensor: Literal[
                "angle", "velocity", "imu_sim", "imu_thigh", "imu_shank"
                ] | None = None
                ):
        in_channels = _get_ablated_channels_n(ablated_sensor)

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
        x = self.regressor(x)    # (batch, 1)
        return x


class DilatedCausalConvBlock(nn.Module):
    # in each block, pad x zeros at the beginning? Weight Norm?
    # Lout = floor( (Lin + left_pad - dilation*(kernel_size - 1) - 1 )/stride + 1 )
    # let stride = 1 for simplicity:
    # Lout = Lin + left_pad - dilation*(kernel_size - 1) 
    # padding = dilation*(kernel_size - 1) 

    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    def __init__(self, in_channels:int, out_channels:int,
                 kernel_size:int=5, dilation:int=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 
                        kernel_size = kernel_size,
                        padding = (dilation*(kernel_size - 1), 0), # zeros
                        dilation = dilation),
            # add WeightNorm? if we do so, do the same for regular cnn.
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.weight_norm.html
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)



class KneeTCN(nn.Module):
    # simple implementation without residual layer.
    def __init__(self, 
                 ablated_sensor: Literal[
                "angle", "velocity", "imu_sim", "imu_thigh", "imu_shank"
                ] | None = None
                ):
        in_channels = _get_ablated_channels_n(ablated_sensor)

        super().__init__()
        self.encoder = nn.Sequential(
            DilatedCausalConvBlock(in_channels, 32, dilation=1),
            DilatedCausalConvBlock(32, 64, dilation = 2),
            DilatedCausalConvBlock(64, 64, dilation = 4),
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
    # also output need to be the same length as input, change that in dataset.
    # input: (L, Hin); output: (L, Hout).
    def __init__(self, 
                 ablated_sensor: Literal[
                "angle", "velocity", "imu_sim", "imu_thigh", "imu_shank"
                ] | None = None
                ):
        in_channels = _get_ablated_channels_n(ablated_sensor)

        super().__init__()
        self.lstm = nn.LSTM(in_channels, 64, 4, dropout=0.1)

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        # x: (batch, channels, window_size)
        x = x.permute(2, 0, 1) # change back to (window_size, batch, channels)
        x = self.lstm(x) # (window_size, batch, 1)
        x = x.permute(0, 2, 1) # change back to (batch, channels, window_size) for loss calculation
        return x