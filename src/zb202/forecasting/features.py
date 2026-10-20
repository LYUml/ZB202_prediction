"""Versioned train-only encoding; legacy mode is reserved for frozen-run replay."""
import numpy as np
from .data import SENSORS


def encode_inputs(data, names, metadata):
    """Apply frozen training statistics; future labels never enter feature encoding."""
    mean = np.asarray(metadata['feature_mean'])
    scale = np.asarray(metadata['feature_scale'])
    if mean.shape != (len(names),) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError('Normalization does not match feature schema')
    values = data['values']
    norm = np.where(np.isfinite(values), (values-mean)/scale, 0)
    mask = np.isfinite(values).astype(np.float32)
    age = np.minimum(data['ages'], 86400)/86400
    if metadata.get('encoding', 'legacy-v1') == 'safe-v2':
        disabled = [names.index(name) for name in metadata['disabled_features']]
        constant = [names.index(name) for name in metadata['frozen_constant_values']]
        norm[:, disabled+constant] = 0
        mask[:, disabled] = 0
        age[:, disabled] = 0
    local = data['timestamps']+8*3600
    day = 2*np.pi*(local % 86400)/86400
    week = 2*np.pi*(local % (86400*7))/(86400*7)
    calendar = np.stack([np.sin(day), np.cos(day), np.sin(week), np.cos(week)], axis=1)
    x = np.concatenate([norm, mask, age, calendar], axis=1).astype(np.float32)
    cols = [names.index(s+'/co2_ppm') for s in SENSORS]
    current = ((values[:, cols]-np.asarray(metadata['target_mean'])) /
               np.asarray(metadata['target_scale'])).astype(np.float32)
    return x, current


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
    cols = [names.index(s + '/co2_ppm') for s in SENSORS]
    target_mean, target_scale = mean[cols], scale[cols]
    if encoding == 'safe-v2' and unsupported[cols].any():
        raise ValueError('A target CO2 sensor has no training observations')
    y = ((data['targets'] - target_mean) / target_scale).astype(np.float32)
    metadata = {'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
                'target_mean': target_mean.tolist(), 'target_scale': target_scale.tolist()}
    if encoding == 'safe-v2':
        metadata.update(encoding=encoding, train_observation_count=count.tolist(),
                        disabled_features=[names[j] for j in np.flatnonzero(unsupported)],
                        frozen_constant_values=[names[j] for j in np.flatnonzero(constant & ~binary)],
                        binary_features=[names[j] for j in np.flatnonzero(binary)],
                        future_observed_bins_for_disabled={names[j]: int(np.isfinite(values[train_end:, j]).sum())
                                                           for j in np.flatnonzero(unsupported)})
    x, current = encode_inputs(data, names, metadata)
    return x, current, y, metadata


