"""Small channel-shared patch Transformer inspired by PatchTST, not a reproduction."""
import torch
from torch import nn


class PatchTransformer(nn.Module):
    """Each sensor uses its own CO2/mask/age and shared calendar context."""
    def __init__(self, input_features, targets, config):
        super().__init__()
        if input_features != 3*targets+4:
            raise ValueError('Expected CO2-only features')
        self.targets = targets
        self.length = config['lookback_steps']
        self.patch = config.get('patch_length', 8)
        self.stride = config.get('patch_stride', 4)
        width = config.get('patch_width', 16)
        if self.patch < 1 or self.stride < 1 or self.patch > self.length or (self.length-self.patch) % self.stride:
            raise ValueError('Patch geometry must exactly cover lookback')
        count = 1 + (self.length-self.patch)//self.stride
        self.embed = nn.Linear(self.patch*7, width)
        self.position = nn.Parameter(torch.zeros(1, count, width))
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=2, dim_feedforward=width*2,
                                           dropout=config['dropout'], batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(count*width, config['horizon_steps'])
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, current):
        batch, length, features = x.shape
        if length != self.length or features != 3*self.targets+4:
            raise ValueError('Invalid input shape')
        sensor = torch.stack([x[:, :, i*self.targets:(i+1)*self.targets] for i in range(3)], dim=-1)
        sensor = sensor.permute(0, 2, 1, 3)
        cal = x[:, None, :, -4:].expand(-1, self.targets, -1, -1)
        tokens = torch.cat([sensor, cal], dim=-1).unfold(2, self.patch, self.stride)
        tokens = tokens.reshape(batch*self.targets, tokens.shape[2], -1)
        encoded = self.encoder(self.embed(tokens) + self.position)
        delta = self.head(encoded.flatten(1)).reshape(batch, self.targets, -1).transpose(1, 2)
        return current[:, None, :] + delta
