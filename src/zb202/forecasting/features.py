"""Versioned train-only encoding; legacy mode is reserved for frozen-run replay."""
import numpy as np
from .data import SENSORS


def build_inputs(data, names, train_end, encoding='safe-v2'):
    if encoding not in ('legacy-v1', 'safe-v2'):
        raise ValueError('Unknown input encoding: ' + encoding)
    if not 0 < train_end <= len(data['values']):
        raise ValueError('Invalid training boundary')
    values = data['values']
    valid = np.isfinite(values[:train_end])
    count = valid.sum(axis=0)
    mean = np.nansum(values[:train_end], axis=0) / np.maximum(count, 1)
    variance = np.where(valid, (values[:train_end] - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    scale = np.maximum(np.sqrt(variance), 1e-3)
    unsupported = count == 0
    constant = variance < 1e-6
    binary = np.array([name.endswith(('/last_occupancy_event', '/last_magnet_event')) for name in names])
    if encoding == 'safe-v2':
        # Event magnitudes stay 0/1; a rare event must not become a large z-score.
        mean[binary] = 0
        scale[binary | constant | unsupported] = 1
    norm = np.where(np.isfinite(values), (values - mean) / scale, 0)
    mask = np.isfinite(values).astype(np.float32)
    age = np.minimum(data['ages'], 86400) / 86400
    if encoding == 'safe-v2':
        # Freeze all channels of train-unseen features, including mask and age.
        # Constant continuous values also have no learned response to future changes.
        norm[:, unsupported | (constant & ~binary)] = 0
        mask[:, unsupported] = 0
        age[:, unsupported] = 0
    timestamps = data['timestamps']
    # HK calendar clock, derived from UTC without any future values.
    local = timestamps + 8 * 3600
    day_phase = 2 * np.pi * (local % 86400) / 86400
    week_phase = 2 * np.pi * (local % (86400 * 7)) / (86400 * 7)
    calendar = np.stack([np.sin(day_phase), np.cos(day_phase), np.sin(week_phase), np.cos(week_phase)], axis=1)
    x = np.concatenate([norm, mask, age, calendar], axis=1).astype(np.float32)
    cols = [names.index(s + '/co2_ppm') for s in SENSORS]
    target_mean, target_scale = mean[cols], scale[cols]
    if encoding == 'safe-v2' and unsupported[cols].any():
        raise ValueError('A target CO2 sensor has no training observations')
    y = ((data['targets'] - target_mean) / target_scale).astype(np.float32)
    current = ((values[:, cols] - target_mean) / target_scale).astype(np.float32)
    metadata = {'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
                'target_mean': target_mean.tolist(), 'target_scale': target_scale.tolist()}
    if encoding == 'safe-v2':
        metadata.update(encoding=encoding, train_observation_count=count.tolist(),
                        disabled_features=[names[j] for j in np.flatnonzero(unsupported)],
                        frozen_constant_values=[names[j] for j in np.flatnonzero(constant & ~binary)],
                        binary_features=[names[j] for j in np.flatnonzero(binary)],
                        future_observed_bins_for_disabled={names[j]: int(np.isfinite(values[train_end:, j]).sum())
                                                           for j in np.flatnonzero(unsupported)})
    return x, current, y, metadata


