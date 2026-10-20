"""Historical windows and observed future labels for forecasting."""
import numpy as np
from torch.utils.data import Dataset


class Windows(Dataset):
    def __init__(self, x, current, y, origins, config):
        self.x, self.current, self.y, self.origins, self.config = x, current, y, origins, config

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        t = self.origins[i]
        target = self.y[t + 1:t + self.config['horizon_steps'] + 1]
        return (self.x[t - self.config['lookback_steps'] + 1:t + 1], self.current[t],
                np.nan_to_num(target), np.isfinite(target))


