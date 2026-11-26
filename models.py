import torch
import torch.nn as nn
from config import DROPOUT_RATE


class DNN_NSB(nn.Module):
    """
    fully-connected DNN for NSB weight inference,
    applies 50% dropout after FC 1 and FC 2 only
    """
    def __init__(self, N: int, num_layers: int, hidden_nodes: int, dropout_rate: float = DROPOUT_RATE):
        super().__init__()
        in_dim = N * (N + 1)
        out_dim = 2 * N

        layers = []
        curr_dim = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_nodes))
            layers.append(nn.ReLU())
            if i < 2:  # Dropout on FC 1 and FC 2 only
                layers.append(nn.Dropout(p=dropout_rate))
            curr_dim = hidden_nodes

        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN_MVDR(nn.Module):
    """
    CNN baseline for MVDR weight inference
    """
    def __init__(self, N: int, num_conv_layers: int = 3, base_channels: int = 32):
        super().__init__()
        conv_blocks = []
        in_ch = 2
        for i in range(num_conv_layers):
            out_ch = base_channels if i < 2 else base_channels * 2
            conv_blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_blocks.append(nn.ReLU())
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_blocks)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * N * N, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2 * N),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))