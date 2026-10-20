"""Window-only structured candidates, without identified BA or ventilation parameters.

DecompositionStateRNN is inspired by PIAD-SRNN, not its original reproduction.
RelaxationForecast is a positive convex relaxation in ppm, not a calibrated room.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DecompositionStateRNN(nn.Module):
    """Shared per-point temporal recurrence; independent state for every sample."""
    def __init__(self, input_features, targets, config):
        super().__init__()
        if input_features != targets * 3 + 4:
            raise ValueError('Expected CO2-only input order')
        self.targets = targets
        self.kernel = config.get('decomposition_kernel', 25)
        if self.kernel < 1 or self.kernel % 2 != 1:
            raise ValueError('Decomposition kernel must be positive and odd')
        hidden = config.get('state_hidden', 16)
        self.state = nn.RNN(7, hidden, batch_first=True, nonlinearity='relu')
        nn.init.orthogonal_(self.state.weight_hh_l0, gain=.25)
        self.seasonal_head = nn.Linear(hidden, config['horizon_steps'])
        self.trend_head = nn.Linear(config['lookback_steps'], config['horizon_steps'])
        for layer in (self.seasonal_head, self.trend_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, current):
        batch, length, _ = x.shape
        history = x[:, :, :self.targets].transpose(1, 2)
        trend = F.avg_pool1d(F.pad(history, (self.kernel//2, self.kernel//2),
                                 mode='replicate'), self.kernel, stride=1)
        local = torch.stack((history-trend,
                             x[:, :, self.targets:2*self.targets].transpose(1, 2),
                             x[:, :, 2*self.targets:3*self.targets].transpose(1, 2)), dim=-1)
        calendar = x[:, None, :, -4:].expand(-1, self.targets, -1, -1)
        sequence = torch.cat((local, calendar), dim=-1).reshape(batch*self.targets, length, 7)
        # nn.RNN creates a zero initial state for each independent sequence.
        _, state = self.state(sequence)
        seasonal = self.seasonal_head(state[-1]).reshape(batch, self.targets, -1)
        delta = seasonal + self.trend_head(trend-trend[:, :, -1:])
        return current[:, None, :] + delta.transpose(1, 2)


class RelaxationForecast(nn.Module):
    """C(h)=C0*exp(-k*h)+Ceq*(1-exp(-k*h)), with positive Ceq and k.

    Context predicts one equilibrium/rate per point, held fixed over the horizon.
    Rates are per five-minute step, bounded by 0.25; they are latent predictive
    parameters, with no physical identification claim. Nonnegative C0 is required.
    """
    def __init__(self, input_features, targets, config):
        super().__init__()
        self.horizon, self.targets = config['horizon_steps'], targets
        self.register_buffer('mean', torch.tensor(config['target_mean'], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(config['target_scale'], dtype=torch.float32))
        if self.mean.numel() != targets or self.scale.numel() != targets or (self.scale <= 0).any():
            raise ValueError('Invalid training-only target normalization')
        self.context = nn.Sequential(nn.Linear(input_features*config['lookback_steps'], 16), nn.Tanh())
        self.head = nn.Linear(16, 2*targets)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[targets:] = -2.

    def forward(self, x, current):
        raw_current = current*self.scale+self.mean
        delta, rate_logits = self.head(self.context(x.flatten(1))).split(self.targets, dim=-1)
        equilibrium = 100.*F.softplus(raw_current/100.+delta)
        rate = .25*torch.sigmoid(rate_logits)
        steps = torch.arange(1, self.horizon+1, device=x.device, dtype=x.dtype)[None, :, None]
        weight = -torch.expm1(-steps*rate[:, None, :])
        ppm = raw_current[:, None, :]*(1.-weight)+equilibrium[:, None, :]*weight
        return (ppm-self.mean)/self.scale
