"""Small causal TCN with a direct H x sensor forecast head."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Module):
    def __init__(self, inputs, channels, kernel, dilation):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(inputs, channels, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class Block(nn.Module):
    def __init__(self, inputs, channels, kernel, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(CausalConv(inputs, channels, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
                                    CausalConv(channels, channels, kernel, dilation), nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(inputs, channels, 1) if inputs != channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.layers(x) + self.skip(x))


class TCN(nn.Module):
    def __init__(self, input_features, targets, config):
        super().__init__()
        blocks = []
        channels = input_features
        for dilation in config['dilations']:
            blocks.append(Block(channels, config['channels'], config['kernel_size'], dilation, config['dropout']))
            channels = config['channels']
        self.encoder = nn.Sequential(*blocks)
        self.horizon, self.targets = config['horizon_steps'], targets
        self.head = nn.Linear(channels, self.horizon * targets)
        # Start at the persistence solution; learn direct multi-horizon corrections.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, current):
        hidden = self.encoder(x.transpose(1, 2))[:, :, -1]
        delta = self.head(hidden).reshape(-1, self.horizon, self.targets)
        return current[:, None, :] + delta


class IndependentTCN(nn.Module):
    """Five separately parameterized local models; input order is CO2/mask/age/calendar."""
    def __init__(self, input_features, targets, config):
        super().__init__()
        if input_features != targets * 3 + 4:
            raise ValueError('IndependentTCN requires CO2-only feature order')
        self.targets = targets
        self.models = nn.ModuleList([TCN(7, 1, config) for _ in range(targets)])

    def forward(self, x, current):
        return torch.cat([model(x[:, :, [j, self.targets+j, 2*self.targets+j] +
                                  list(range(3*self.targets, 3*self.targets+4))], current[:, j:j+1])
                          for j, model in enumerate(self.models)], dim=2)
