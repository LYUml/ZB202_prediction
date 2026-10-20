"""Audit a frozen run; interventions are sensitivity checks, never new test scores."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SENSORS, digest, iso, observations, sample_origins
from .features import build_inputs
from .windows import Windows
from .evaluation import predict, scores
from .tcn import TCN


def diagnose(root, run):
    root, run = Path(root), Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    processed = root / manifest['processed']
    pm = json.loads((processed / 'manifest.json').read_text())
    if digest(processed / 'aligned.npz') != pm['aligned_sha256']:
        raise ValueError('Aligned data checksum mismatch')
    if digest(processed / 'manifest.json') != manifest['processed_manifest_sha256']:
        raise ValueError('Processed manifest checksum mismatch')
    for name, expected in manifest['outputs'].items():
        if digest(run / name) != expected:
            raise ValueError(f'Frozen output checksum mismatch: {name}')
    data = dict(np.load(processed / 'aligned.npz'))
    # Only load the local, trusted project checkpoint; never an arbitrary remote pickle.
    checkpoint = torch.load(run / 'tcn.pt', map_location='cpu', weights_only=False)
    cfg, names = checkpoint['config'], checkpoint['feature_names']
    origins, train_end, val_end = sample_origins(data, names, cfg)
    x, current, y, norm = build_inputs(data, names, train_end)
    for key in norm:
        np.testing.assert_allclose(norm[key], checkpoint['normalization'][key], rtol=1e-7)
    torch.set_num_threads(4)
    model = TCN(x.shape[1], len(SENSORS), cfg)
    model.load_state_dict(checkpoint['state_dict'])
    mean, scale = np.array(norm['target_mean']), np.array(norm['target_scale'])
    def infer(inputs, split):
        return predict(model, DataLoader(Windows(inputs, current, y, origins[split], cfg),
                       batch_size=cfg['batch_size']), mean, scale)
    def paired(p, a, m, b):
        return {'tcn': scores(p, a, m), 'persistence': scores(b, a, m),
                'min_prediction_ppm': float(p.min()), 'negative_predictions': int((p < 0).sum())}
    result = {'run_id': manifest['run_id'], 'purpose': 'Exposed-test forensic audit; no training or model selection',
              'intervention_limit': 'Frozen-weight input sensitivity only; not deployable improvements or unbiased test estimates.',
              'input_artifact_hashes': {p.name: digest(p) for p in [run/'manifest.json', run/'tcn.pt', run/'test_predictions.npz']},
              'features': [], 'splits': {}}
    for j, name in enumerate(names):
        v = data['values'][:, j]
        entry = {'name': name, 'train_mean': norm['feature_mean'][j], 'train_scale': norm['feature_scale'][j]}
        for split, left, right in [('train', 0, train_end), ('validation', train_end, val_end), ('test', val_end, len(v))]:
            valid = v[left:right][np.isfinite(v[left:right])]
            entry[split] = {'finite_bins': len(valid), 'min': float(valid.min()) if len(valid) else None,
                            'max': float(valid.max()) if len(valid) else None,
                            'max_abs_normalized': float(abs(x[left:right, j]).max())}
        result['features'].append(entry)
    unseen = [j for j in range(len(names)) if not np.isfinite(data['values'][:train_end, j]).any()]
    interventions = {'original': x}
    masked = x.copy()
    masked[:, unseen] = 0
    interventions['unseen_value_zero'] = masked
    co2 = names.index('AM103_05/co2_ppm')
    # Retain target, current anchor, masks and ages. Replace only unusually high historical input values.
    bounded = x.copy()
    cap = np.nanmax(data['values'][:train_end, co2])
    high = data['values'][:, co2] > cap
    bounded[high, co2] = (cap - norm['feature_mean'][co2]) / norm['feature_scale'][co2]
    interventions['co2_input_train_max_cap'] = bounded
    both = bounded.copy(); both[:, unseen] = 0
    interventions['both'] = both
    saved = np.load(run / 'test_predictions.npz')
    for split in ['validation', 'test']:
        rows = {}
        for label, inputs in interventions.items():
            p, a, m, b = infer(inputs, split)
            rows[label] = paired(p, a, m, b)
            if label != 'original':
                continue
            if split == 'test':
                result['replay_max_abs_diff_ppm'] = float(abs(p - saved['predictions_ppm']).max())
                np.testing.assert_allclose(p, saved['predictions_ppm'], atol=.01, rtol=1e-6)
            ix = origins[split]
            history_high = np.array([high[t-cfg['lookback_steps']+1:t+1].any() for t in ix])
            event_high = np.array([(x[t-cfg['lookback_steps']+1:t+1, unseen] > 100).any() for t in ix])
            rows['groups'] = {}
            for group, select in [('co2_high_history', history_high), ('unseen_event_high_history', event_high),
                                  ('neither', ~(history_high | event_high))]:
                rows['groups'][group] = {'origins': int(select.sum()), **paired(p[select], a[select], m[select], b[select])} if select.any() else {'origins': 0}
            err = np.where(m, abs(p-a), -np.inf)
            worst = np.unravel_index(np.argmax(err), err.shape)
            i, h, sensor = map(int, worst)
            rows['worst_observed_error'] = {'origin_utc': iso(data['timestamps'][ix[i]]),
                'target_bin_end_utc': iso(data['timestamps'][ix[i]+h+1]), 'sensor': SENSORS[sensor],
                'prediction': float(p[worst]), 'actual': float(a[worst]), 'persistence': float(b[worst])}
            neg = np.unravel_index(np.argmin(p), p.shape)
            i, h, sensor = map(int, neg)
            rows['minimum_prediction'] = {'origin_utc': iso(data['timestamps'][ix[i]]),
                'target_bin_end_utc': iso(data['timestamps'][ix[i]+h+1]), 'sensor': SENSORS[sensor],
                'prediction': float(p[neg]), 'actual': float(a[neg]) if m[neg] else None,
                'current_co2_all_sensors': data['values'][ix[i], [names.index(s+'/co2_ppm') for s in SENSORS]].tolist()}
        result['splits'][split] = rows
    source = root / cfg['snapshot'] / 'zb202_iot.csv.gz'
    if digest(source) != manifest['raw_sha256']:
        raise ValueError('Raw snapshot checksum mismatch')
    extremes, context, positive_events = [], [], []
    for r in observations(source):
        if r['_field'] == 'co2_ppm' and r['_measurement'] in SENSORS and '2026-10-15T07:00' <= r['_time'] <= '2026-10-15T07:30':
            context.append({k: r[k] for k in ['_time', '_measurement', '_field', '_value']})
        if r['_measurement'] == 'VS341_04' and r['_field'] == 'occupancy' and float(r['_value']) == 1:
            positive_events.append(r['_time'])
        if r['_field'] == 'co2_ppm' and r['_measurement'] in SENSORS and float(r['_value']) > cap:
            extremes.append({'time_utc': r['_time'], 'sensor': r['_measurement'], 'co2_ppm': float(r['_value'])})
    result['raw_co2_above_AM103_05_train_max'] = extremes
    result['raw_peak_context'] = context
    result['VS341_04_positive_event_times_utc'] = positive_events
    result['case_study_limit'] = 'Raw context timestamps and VS341_04 channel are specific to the first-run audit.'
    output = run / 'diagnostics'
    output.mkdir(exist_ok=True)
    path = output / 'audit.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return path
