"""Small direct linear forecasting baselines with explicitly different input contracts.

NLinear/DLinear follow the architectural definitions at cure-lab/LTSF-Linear;
these are local implementations, not a reproduction of that paper's benchmark.
"""
import torch
from torch import nn
from torch.nn import functional as F


class NLinear(nn.Module):
    """Shared lag-to-horizon map on each CO2 history, centered on the current value."""
    def __init__(self, input_features, targets, config):
        super().__init__()
        if input_features != targets*3+4:
            raise ValueError('Expected CO2-only input order')
        self.targets=targets
        self.map=nn.Linear(config['lookback_steps'],config['horizon_steps'])

    def forward(self,x,current):
        history=x[:,:,:self.targets]-current[:,None,:]
        return self.map(history.transpose(1,2)).transpose(1,2)+current[:,None,:]


class DLinear(nn.Module):
    """Shared trend/residual temporal maps; decomposition sees only the input window."""
    def __init__(self,input_features,targets,config):
        super().__init__()
        if input_features != targets*3+4:
            raise ValueError('Expected CO2-only input order')
        self.targets=targets
        self.kernel=config.get('decomposition_kernel',25)
        if self.kernel<1 or self.kernel%2!=1:
            raise ValueError('Decomposition kernel must be a positive odd number')
        self.trend=nn.Linear(config['lookback_steps'],config['horizon_steps'])
        self.residual=nn.Linear(config['lookback_steps'],config['horizon_steps'])

    def forward(self,x,current):
        history=x[:,:,:self.targets].transpose(1,2)
        trend=F.avg_pool1d(F.pad(history,(self.kernel//2,self.kernel//2),mode='replicate'),self.kernel,stride=1)
        return (self.trend(trend)+self.residual(history-trend)).transpose(1,2)


class MaskAwareLinear(nn.Module):
    """Direct joint residual map using the full CO2/mask/age/calendar window."""
    def __init__(self,input_features,targets,config):
        super().__init__()
        self.horizon,self.targets=config['horizon_steps'],targets
        self.map=nn.Linear(input_features*config['lookback_steps'],self.horizon*targets)
        nn.init.zeros_(self.map.weight);nn.init.zeros_(self.map.bias)

    def forward(self,x,current):
        return current[:,None,:]+self.map(x.flatten(1)).reshape(-1,self.horizon,self.targets)
