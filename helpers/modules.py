import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List, Any
from helpers.constants import *

def init_model_params(model:nn.Module) -> None:

    def _init_params(m:nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()
            m.bias.data.fill_(0.01) # try this for relu activation
        elif isinstance(m,(nn.RNN, nn.LSTM)):
            for layer_params in m.all_weights: # n layers
                # ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
                nn.init.xavier_normal_(layer_params[0])
                nn.init.xavier_normal_(layer_params[1])
                layer_params[2].data.zero_()
                layer_params[3].data.zero_()
        elif isinstance(m, (nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    model.apply(_init_params)

# for model input channels, messy workaround
def _get_ablated_channels_n(
        ablated_sensors: List[SENSOR_NAMES|Any] = []
    ) -> int:

    n_channels = 0
    if "angle" not in ablated_sensors:
        n_channels = n_channels + 1
    if "velocity" not in ablated_sensors:
        n_channels = n_channels + 1
    if "imu_thigh" not in ablated_sensors:
        n_channels = n_channels + len(IMU_THIGH_COLS)
    if "imu_shank" not in ablated_sensors:
        n_channels = n_channels + len(IMU_SHANK_COLS)

    return n_channels

class ConvBlock(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9751372
    def __init__(
            self, 
            in_channels:int, 
            out_channels:int,
            kernel_size:int=5, 
        ):
        super().__init__()
        padding = int(kernel_size/2) # floor
        self.conv = nn.Sequential(
                nn.Conv1d(
                in_channels, out_channels, 
                kernel_size = kernel_size,
                padding = padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# weightnorm might not be fully reset...
class KneeCNN(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9751372
    def __init__(
            self, 
            ablated_sensors: List[SENSOR_NAMES|Any] = [],
            hidden_layer_size: int = HIDDEN_LAYER_SIZE
        ):
        # saw some time series data model that starts with larger kernel 
        # larger receptive field 
        super().__init__()
        in_channels = _get_ablated_channels_n(ablated_sensors)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_layer_size),
            ConvBlock(hidden_layer_size, hidden_layer_size),
            ConvBlock(hidden_layer_size, hidden_layer_size),
            # ConvBlock(hidden_layer_size, hidden_layer_size),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse time dim -> 1
        self.regressor = nn.Sequential(
            nn.Linear(hidden_layer_size, int(hidden_layer_size/2)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(hidden_layer_size/2), 1),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, window_size)
        x = self.encoder(x)      # (batch, HIDDEN_LAYER_SIZE, window_size)
        x = self.pool(x)         # (batch, HIDDEN_LAYER_SIZE, 1)
        x = x.squeeze(-1)        # (batch, HIDDEN_LAYER_SIZE)
        x = self.regressor(x)    # (batch, 1)
        return x


class DilatedCausalConvBlock(nn.Module):
    # in each block, pad x zeros at the beginning
    # let stride = 1 for simplicity:
    # Lout = Lin + left_pad - dilation*(kernel_size - 1) 
    # padding = dilation*(kernel_size - 1) 
    def __init__(
            self, 
            in_channels:int, 
            out_channels:int,
            kernel_size:int=2, 
            dilation:int=1
        ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConstantPad1d((dilation*(kernel_size - 1), 0), 0.0,),
            weight_norm(nn.Conv1d(
                in_channels, out_channels, 
                kernel_size = kernel_size,
                dilation = dilation
                )
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class KneeTCN(nn.Module):
    # simple implementation without residual layer.
    # from Molinaro 2024 paper and https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    def __init__(
            self, 
            ablated_sensors: List[SENSOR_NAMES|Any] = [],
            hidden_layer_size: int = HIDDEN_LAYER_SIZE
        ):
        super().__init__()
        in_channels = _get_ablated_channels_n(ablated_sensors)
        self.encoder = nn.Sequential(
            DilatedCausalConvBlock(in_channels, hidden_layer_size, dilation= 1),
            DilatedCausalConvBlock(hidden_layer_size, hidden_layer_size, dilation= 2),
            DilatedCausalConvBlock(hidden_layer_size, hidden_layer_size, dilation= 4),
            DilatedCausalConvBlock(hidden_layer_size, hidden_layer_size, dilation= 8),
            DilatedCausalConvBlock(hidden_layer_size, hidden_layer_size, dilation= 16),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse time dim -> 1
        self.regressor = nn.Sequential(
            nn.Linear(hidden_layer_size, int(hidden_layer_size/2)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(hidden_layer_size/2), 1),
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: (batch, channels, window_size)
        x = self.encoder(x)      # (batch, 64, window_size)
        x = self.pool(x)         # (batch, 64, 1)
        x = x.squeeze(-1)        # (batch, 64)
        x = self.regressor(x)    # (batch, 1)
        return x
    
    
class KneeLSTM(nn.Module):
    # this modules outputs perdiction results for all timesteps.
    # input: (L, Hin); output: (L, Hout).
    def __init__(
            self, 
            ablated_sensors: List[SENSOR_NAMES|Any] = [],
            hidden_layer_size: int = HIDDEN_LAYER_SIZE
        ):
        super().__init__()
        in_channels = _get_ablated_channels_n(ablated_sensors)
        self.encoder = nn.LSTM(in_channels, int(hidden_layer_size/2), 4, dropout=0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse time dim -> 1
        self.regressor = nn.Sequential(
            nn.Linear(int(hidden_layer_size/2), int(hidden_layer_size/2)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(hidden_layer_size/2), 1),
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        ''' input x: (batch, channels, window_size) 32,14,50'''
        # change to (window_size, batch, channels)
        x = x.permute(2, 0, 1) # 50,32,14
        x, _ = self.encoder(x) # (window_size, batch, channels), (hn, cn)
        # instead of pooling, pass each state into regressor.
        # (l,b,c) -> (l*b, c) -> (l,b,c) -> (b, c, l)
        l,b,_ = x.shape # 50,32,64
        x = x.flatten(0,1)
        x = self.regressor(x)    # (batch, 1), (l,b,c) -> (l*b,c)
        x = x.unflatten(0,(l,b)) # inverse of flatten(), (l*b,c) -> (l,b,c)

        x = x.permute(1, 2, 0) # (l,b,c) -> (b,c,l)

        return x