"""Direct multi-horizon alternatives sharing the TCN window interface."""
import numpy as np
import torch
from torch import nn


class DirectLSTM(nn.Module):
    def __init__(self, input_features, targets, config):
        super().__init__()
        self.encoder = nn.LSTM(input_features, config['channels'], batch_first=True)
        self.head = nn.Linear(config['channels'], config['horizon_steps'] * targets)
        self.horizon, self.targets = config['horizon_steps'], targets
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, current):
        _, (hidden, _) = self.encoder(x)
        return current[:, None, :] + self.head(hidden[-1]).reshape(-1, self.horizon, self.targets)


def ridge_design(x, origins, lookback):
    """Same historical information, summarized by latest, mean and net change."""
    return np.stack([np.concatenate((x[t], x[t-lookback+1:t+1].mean(axis=0),
                                    x[t] - x[t-lookback+1])) for t in origins]).astype(np.float64)


class DirectRidge:
    def __init__(self, alpha):
        if alpha <= 0:
            raise ValueError('Ridge penalty must be positive')
        self.alpha = alpha

    def fit(self, design, residual, observed):
        self.shape = residual.shape[1:]
        target, masks = residual.reshape(len(design), -1), observed.reshape(len(design), -1)
        self.coef = np.zeros((design.shape[1], target.shape[1]))
        self.intercept = np.zeros(target.shape[1])
        # Only observed training labels enter each direct horizon/sensor fit.
        for j in range(target.shape[1]):
            select = masks[:, j]
            if not select.any():
                raise ValueError('No observed training labels for Ridge output')
            a, b = design[select], target[select, j]
            center, offset = a.mean(axis=0), b.mean()
            a = torch.as_tensor(a - center, dtype=torch.float64)
            b = torch.as_tensor(b - offset, dtype=torch.float64)
            gram = a.T @ a + torch.eye(a.shape[1], dtype=torch.float64) * self.alpha
            self.coef[:, j] = torch.linalg.solve(gram, a.T @ b).numpy()
            self.intercept[j] = offset - center @ self.coef[:, j]
        return self

    def predict(self, design, current):
        return current[:, None, :] + (design @ self.coef + self.intercept).reshape(-1, *self.shape)
