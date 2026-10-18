"""Causal time alignment of real Influx observations; no future interpolation."""
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

SENSORS = ['AM103_05', 'AM103_06', 'AM103_07', 'AM103_08', 'AM308_01']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iso(timestamp):
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()


def observations(path):
    header = None
    with gzip.open(path, 'rt', newline='') as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith('#'):
                continue
            if '_time' in row and '_value' in row:
                header = row
                continue
            if header is None or len(header) != len(row):
                raise ValueError('Unexpected annotated CSV table/row')
            record = dict(zip(header, row))
            if not record.get('_time') or not record.get('_field'):
                raise ValueError('Missing time or field in CSV')
            yield record


def align_series(times, values, grid, max_age):
    """Features use latest <= origin; labels use actual readings in (t-step,t]."""
    order = np.argsort(times, kind='stable')
    times, values = np.asarray(times)[order], np.asarray(values)[order]
    latest = np.searchsorted(times, grid, side='right') - 1
    exists = latest >= 0
    idx = np.maximum(latest, 0)
    age = np.where(exists, grid - times[idx], np.inf)
    fresh = exists & (age <= max_age)
    feature = np.where(fresh, values[idx], np.nan)
    # Exact-bin observation masks are separate from last-value feature masks.
    step = grid[1] - grid[0]
    observed = exists & (age < step)
    label = np.where(observed, values[idx], np.nan)
    return feature, age, label


def prepare(project_root, config):
    root = Path(project_root)
    snapshot = root / config['snapshot']
    source = snapshot / 'zb202_iot.csv.gz'
    source_hash = digest(source)
    raw_manifest = json.loads((snapshot / 'manifest.json').read_text())
    expected = next(x['sha256'] for x in raw_manifest['sources'] if x['bucket'] == 'zb202_iot')
    if source_hash != expected:
        raise ValueError('Raw snapshot checksum mismatch')
    series = {}
    raw_rows = invalid = 0
    for r in observations(source):
        raw_rows += 1
        m, f = r['_measurement'], r['_field']
        if m in SENSORS and f in ('co2_ppm', 'temperature_c', 'relative_humidity_pct'):
            name, max_age = m + '/' + f, config['environment_max_age_seconds']
        elif m.startswith('VS341_') and f == 'occupancy':
            name, max_age = m + '/last_occupancy_event', config['event_max_age_seconds']
        elif m.startswith('WS301_') and f == 'magnet_status':
            name, max_age = m + '/last_magnet_event', config['event_max_age_seconds']
        elif m.startswith('WS523_') and f == 'active_power':
            name, max_age = m + '/' + f, config['power_max_age_seconds']
        else:
            continue
        try:
            value = float(r['_value'])
            time = datetime.fromisoformat(r['_time'].replace('Z', '+00:00')).timestamp()
        except ValueError:
            invalid += 1
            continue
        if not np.isfinite(value) or (f == 'co2_ppm' and value < 0):
            invalid += 1
            continue
        entry = series.setdefault(name, {'times': [], 'values': [], 'max_age': max_age})
        entry['times'].append(time)
        entry['values'].append(value)
    if any(s + '/co2_ppm' not in series for s in SENSORS):
        raise ValueError('Missing target CO2 sensor')
    step = config['step_seconds']
    minimum = min(min(v['times']) for v in series.values())
    maximum = max(max(v['times']) for v in series.values())
    start, stop = int(np.ceil(minimum / step) * step), int(np.ceil(maximum / step) * step)
    grid = np.arange(start, stop + step, step, dtype=np.int64)
    names = sorted(series)
    features = np.full((len(grid), len(names)), np.nan, dtype=np.float32)
    ages = np.full_like(features, np.inf)
    targets = np.full((len(grid), len(SENSORS)), np.nan, dtype=np.float32)
    quality = {}
    for j, name in enumerate(names):
        entry = series[name]
        feature, age, label = align_series(np.array(entry['times']), np.array(entry['values']), grid, entry['max_age'])
        features[:, j], ages[:, j] = feature, age
        times = np.sort(entry['times'])
        quality[name] = {'observations': len(times), 'feature_available_bins': int(np.isfinite(feature).sum()),
                         'actually_observed_bins': int(np.isfinite(label).sum()),
                         'largest_gap_hours': float(np.diff(times).max() / 3600) if len(times) > 1 else None}
        if name.endswith('/co2_ppm'):
            sensor = name.split('/')[0]
            targets[:, SENSORS.index(sensor)] = label
    identity = hashlib.sha256((source_hash + json.dumps(config, sort_keys=True) + digest(__file__)).encode()).hexdigest()[:12]
    output = root / 'data/processed/zb202_iot' / snapshot.name / (config['version'] + '-' + identity)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'aligned.npz', timestamps=grid, values=features, ages=ages, targets=targets)
    manifest = {'source': 'processed-real-observations', 'raw_sha256': source_hash,
                'config': config, 'preprocessor_sha256': digest(__file__), 'feature_names': names,
                'target_names': SENSORS, 'grid_start_utc': iso(grid[0]), 'grid_end_utc': iso(grid[-1]),
                'grid_rows': len(grid), 'raw_field_rows_read': raw_rows, 'invalid_selected_rows': invalid,
                'series': quality, 'aligned_sha256': digest(output / 'aligned.npz'),
                'label_definition': 'Latest actual observation in (target_bin_end - step, target_bin_end]; never forward-filled.',
                'causality_limit': 'Uses recorded event time; ingestion arrival delays are unavailable.',
                'event_limit': config['event_semantics']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output


def sample_origins(data, feature_names, config):
    n = len(data['timestamps'])
    train_end = int(n * config['train_fraction'])
    val_end = int(n * (config['train_fraction'] + config['validation_fraction']))
    co2_cols = [feature_names.index(s + '/co2_ppm') for s in SENSORS]
    availability = np.isfinite(data['values'][:, co2_cols])
    current_available = availability.all(axis=1)
    cumulative = np.concatenate([[0], np.cumsum(availability.mean(axis=1))])
    lookback, horizon = config['lookback_steps'], config['horizon_steps']
    result = {k: [] for k in ['train', 'validation', 'test']}
    for t in range(lookback - 1, n - horizon):
        coverage = (cumulative[t + 1] - cumulative[t - lookback + 1]) / lookback
        if not current_available[t] or coverage < config['minimum_co2_input_coverage']:
            continue
        if not np.isfinite(data['targets'][t + 1:t + horizon + 1]).any():
            continue
        if t + horizon < train_end:
            result['train'].append(t)
        elif t >= train_end and t + horizon < val_end:
            result['validation'].append(t)
        elif t >= val_end:
            result['test'].append(t)
    return {k: np.asarray(v, dtype=np.int64) for k, v in result.items()}, train_end, val_end
