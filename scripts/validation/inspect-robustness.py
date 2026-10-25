"""Independently recompute stored errors and summarize every robustness outcome."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from zb202.forecasting.data import digest
from zb202.forecasting.robustness import single_inputs, external_origins
from zb202.forecasting.baselines import DirectRidge, ridge_design


def inspect(run):
    run = Path(run)
    record = json.loads((run/'manifest.json').read_text())
    if record['status'] != 'complete':
        raise ValueError('Completed robustness run required')
    checks = []
    for filename, expected in record['outputs'].items():
        if digest(run/filename) != expected:
            raise ValueError('Artifact hash mismatch: '+filename)
        checks.append('sha256:'+filename)

    def check_predictions(path, metrics):
        with np.load(path) as data:
            truth, pred, baseline = (data[key] for key in ('actual_ppm', 'predictions_ppm', 'persistence_ppm'))
            mask = np.isfinite(truth)
            for key, values in [('model', pred), ('persistence', baseline)]:
                error = (values.astype(np.float64)-truth.astype(np.float64))[mask]
                # Stored baselines can be float32; tolerate only numerical accumulation differences.
                np.testing.assert_allclose(np.abs(error).mean(), metrics[key]['mae_ppm'], rtol=2e-6, atol=1e-5)
                np.testing.assert_allclose(np.sqrt((error**2).mean()), metrics[key]['rmse_ppm'], rtol=2e-6, atol=1e-5)
                if int(mask.sum()) != metrics[key]['observed_labels']:
                    raise ValueError('Observed label count mismatch')
                checks.append('recomputed:'+path.name+':'+key)
            return dict(origins=len(truth), unique_target_sensor_pairs=len(set(
                (float(stamp+300*(h+1)), int(sensor))
                for origin, h, sensor in np.argwhere(mask & (truth >= 1000))
                for stamp in [data['origins'][origin]])))

    for filename, expected in record['code_sha256'].items():
        if digest(run/'source_snapshot/zb202/forecasting'/filename) != expected:
            raise ValueError('Source snapshot hash mismatch: '+filename)
        checks.append('source-sha256:'+filename)
    grouped = defaultdict(list)
    for entry in record['local_stress']:
        key = json.dumps(entry['scenario'], sort_keys=True)
        grouped[key].append(entry)
        if entry['accepted_origins']:
            scenario_index = record['protocol']['scenarios'].index(entry['scenario'])
            check_predictions(run/f"local-{scenario_index}-{entry['seed']}.npz", entry['metrics'])
    summary = dict(status='validated', run_manifest_sha256=digest(run/'manifest.json'),
                   inspector_sha256=digest(Path(__file__)), local_stress=[], external_results=[],
                   transfer_results=record['transfer_results'])
    for entries in grouped.values():
        item = dict(scenario=entries[0]['scenario'], corruption_seeds=[e['seed'] for e in entries],
                    availability_mean=float(np.mean([e['availability_fraction'] for e in entries])),
                    availability_range=[min(e['availability_fraction'] for e in entries), max(e['availability_fraction'] for e in entries)])
        valid = [e for e in entries if e['accepted_origins']]
        if valid:
            item.update(mae_mean_ppm=float(np.mean([e['metrics']['model']['mae_ppm'] for e in valid])),
                        mae_range_ppm=[min(e['metrics']['model']['mae_ppm'] for e in valid), max(e['metrics']['model']['mae_ppm'] for e in valid)],
                        persistence_mae_mean_ppm=float(np.mean([e['metrics']['persistence']['mae_ppm'] for e in valid])),
                        clean_same_origins_mae_mean_ppm=float(np.mean([e['clean_same_origins']['model']['mae_ppm'] for e in valid])))
        summary['local_stress'].append(item)
    historical = check_predictions(run/'historical-exposed.npz', record['historical_exposed_stress'])
    summary['historical_exposed_stress'] = dict(record['historical_exposed_stress'], **historical)
    for name, source in record['external_sources'].items():
        if digest(ROOT/'data/raw/public'/name/'source.zip') != source['archive_sha256']:
            raise ValueError('Public archive hash mismatch: '+name)
        checks.append('public-archive-sha256:'+name)
        with np.load(run/f'{name}-aligned.npz') as loaded:
            data = dict(loaded)
        cfg = record['protocol']['external_model']
        origins, boundaries = external_origins(data, cfg, record['protocol']['external_split_fractions'])
        train_end, val_end, _ = boundaries
        ranges = {}
        for label, start, stop in [('train', 0, train_end), ('validation', train_end, val_end), ('test', val_end, len(data['timestamps']))]:
            vals = data['targets'][start:stop]; vals = vals[np.isfinite(vals)]
            ranges[label] = dict(observed_bins=len(vals), min_ppm=float(vals.min()), max_ppm=float(vals.max()), mean_ppm=float(vals.mean()))
        # Deterministic Ridge re-fit verifies the recorded comparator and saves its predictions.
        x, current, y, norm = single_inputs(data, train_end)
        ix = origins['train']; future = ix[:, None]+np.arange(1, cfg['horizon_steps']+1)
        ridge = DirectRidge(record['protocol']['external_ridge_alpha']).fit(ridge_design(x, ix, cfg['lookback_steps']),
                    y[future]-current[ix, None], np.isfinite(y[future]))
        ix = origins['test']; future = ix[:, None]+np.arange(1, cfg['horizon_steps']+1)
        pred = ridge.predict(ridge_design(x, ix, cfg['lookback_steps']), current[ix])*np.asarray(norm['target_scale'])+np.asarray(norm['target_mean'])
        path = run/f'{name}-ridge-audited.npz'
        np.savez_compressed(path, origins=data['timestamps'][ix], actual_ppm=data['targets'][future], predictions_ppm=pred,
                            persistence_ppm=np.repeat(data['values'][ix, None], cfg['horizon_steps'], axis=1),
                            coef=ridge.coef, intercept=ridge.intercept)
        for entry in record['external_results']:
            if entry['dataset'] != name:
                continue
            stem = 'ridge-audited' if entry['model'] == 'ridge' else ('tcn-seed-mean' if entry['model'] == 'tcn_seed_mean' else f"tcn-{entry['seed']}")
            check_predictions(run/f'{name}-{stem}.npz', entry['metrics'])
            summary['external_results'].append(dict(entry, source_ranges=ranges))
    for entry in record['transfer_results']:
        check_predictions(run/f"transfer-{entry['source']}-to-{entry['target']}.npz", entry['metrics'])
    summary['checks'] = checks; summary['check_count'] = len(checks)
    (run/'validated-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': summary['status'], 'checks': len(checks), 'historical_unique_ge1000ppm': historical['unique_target_sensor_pairs']}))
    return summary


if __name__ == '__main__':
    inspect(Path(sys.argv[1]))
