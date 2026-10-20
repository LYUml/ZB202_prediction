"""Train-only normalization and causal model inputs (legacy numerical behavior)."""
import numpy as np
from .data import SENSORS


def build_inputs(data, names, train_end):
    values = data['values']
    valid = np.isfinite(values[:train_end])
    count = valid.sum(axis=0)
    mean = np.nansum(values[:train_end], axis=0) / np.maximum(count, 1)
    variance = np.where(valid, (values[:train_end] - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    scale = np.maximum(np.sqrt(variance), 1e-3)
    norm = np.where(np.isfinite(values), (values - mean) / scale, 0)
    mask = np.isfinite(values).astype(np.float32)
    age = np.minimum(data['ages'], 86400) / 86400
    timestamps = data['timestamps']
    # HK calendar clock, derived from UTC without any future values.
    local = timestamps + 8 * 3600
    day_phase = 2 * np.pi * (local % 86400) / 86400
    week_phase = 2 * np.pi * (local % (86400 * 7)) / (86400 * 7)
    calendar = np.stack([np.sin(day_phase), np.cos(day_phase), np.sin(week_phase), np.cos(week_phase)], axis=1)
    x = np.concatenate([norm, mask, age, calendar], axis=1).astype(np.float32)
    cols = [names.index(s + '/co2_ppm') for s in SENSORS]
    target_mean, target_scale = mean[cols], scale[cols]
    y = ((data['targets'] - target_mean) / target_scale).astype(np.float32)
    current = ((values[:, cols] - target_mean) / target_scale).astype(np.float32)
    return x, current, y, {'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
                           'target_mean': target_mean.tolist(), 'target_scale': target_scale.tolist()}


