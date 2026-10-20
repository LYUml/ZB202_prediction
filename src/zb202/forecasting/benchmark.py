"""Predeclared chronological development comparison and exposed-test stress replay."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader
from .baselines import DirectLSTM, DirectRidge, ridge_design
from .data import SENSORS, digest, iso, sample_origins
from .evaluation import predict, scores
from .features import build_inputs
from .tcn import TCN, IndependentTCN
from .linear import NLinear, DLinear, MaskAwareLinear
from .structured import DecompositionStateRNN, RelaxationForecast
from .windows import Windows


def fold_origins(data, names, config, boundaries):
    """All labels stay in train/inner-validation/outer-assessment, including horizon."""
    train_end, inner_end, outer_end = boundaries
    cut = {k: v[:outer_end] for k, v in data.items()}
    cfg = dict(config, train_fraction=1., validation_fraction=0.)
    eligible, _, _ = sample_origins(cut, names, cfg)
    eligible = eligible['train']
    h = cfg['horizon_steps']
    origins = {'train': eligible[eligible+h < train_end],
               'validation': eligible[(eligible >= train_end) & (eligible+h < inner_end)],
               'test': eligible[eligible >= inner_end]}
    if any(len(idx) == 0 for idx in origins.values()):
        raise ValueError('An empty temporal fold')
    return origins


def summarize(pred, truth, mask, baseline):
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite model predictions')
    error = abs(pred-truth)[mask]
    result = scores(pred, truth, mask)
    result.update(p99_abs_error_ppm=float(np.quantile(error, .99)),
                  median_abs_error_ppm=float(np.median(error)),
                  negative_predictions=int((pred < 0).sum()),
                  prediction_count=pred.size, min_prediction_ppm=float(pred.min()),
                  by_horizon={}, by_sensor={})
    for step in [2, 6, 12]:
        if step <= pred.shape[1]:
            result['by_horizon'][str(step*5)] = scores(pred[:, step-1], truth[:, step-1], mask[:, step-1])
    for j, sensor in enumerate(SENSORS):
        result['by_sensor'][sensor] = scores(pred[:, :, j], truth[:, :, j], mask[:, :, j])
    change = mask[:, -1] & (abs(truth[:, -1] - baseline[:, -1]) > 20)
    result['change_gt20_at60min'] = scores(pred[:, -1], truth[:, -1], change)
    return result


def build_model(kind, input_features, targets, cfg, norm=None):
    """Construct the same model for fitting and checkpoint inference."""
    factories = {'tcn': TCN, 'lstm': DirectLSTM, 'independent_tcn': IndependentTCN,
                 'nlinear': NLinear, 'dlinear': DLinear, 'mask_linear': MaskAwareLinear,
                 'decomp_srnn': DecompositionStateRNN, 'relaxation': RelaxationForecast}
    if kind not in factories:
        raise ValueError('Unknown neural model: ' + kind)
    if kind == 'relaxation':
        if norm is None:
            raise ValueError('Relaxation requires training-only normalization')
        cfg = dict(cfg, target_mean=norm['target_mean'], target_scale=norm['target_scale'])
    return factories[kind](input_features, targets, cfg)


def fit_neural(kind, x, current, y, origins, cfg, norm, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_model(kind, x.shape[1], current.shape[1], cfg, norm)
    loaders = {k: DataLoader(Windows(x, current, y, idx, cfg), batch_size=cfg['batch_size'],
                             shuffle=k == 'train', num_workers=0) for k, idx in origins.items()}
    mean, scale = np.array(norm['target_mean']), np.array(norm['target_scale'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    best, stale, state, history = float('inf'), 0, None, []
    for epoch in range(cfg['epochs']):
        model.train()
        for inputs, latest, target, mask in loaders['train']:
            prediction = model(inputs, latest)
            loss = abs(prediction-target)[mask].mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        vp, vy, vm, _ = predict(model, loaders['validation'], mean, scale)
        value = scores(vp, vy, vm)['mae_ppm']
        history.append({'epoch': epoch+1, 'inner_validation_mae_ppm': value})
        if value < best:
            best, stale, state = value, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    model.load_state_dict(state)
    return model, predict(model, loaders['test'], mean, scale), history


def benchmark(root, config_path):
    root = Path(root)
    protocol = json.loads(Path(config_path).read_text())
    cfg = json.loads((root / protocol['base_config']).read_text())
    cfg.update(epochs=protocol['epochs'], patience=protocol['patience'], input_encoding=protocol['input_encoding'])
    processed = root / protocol['processed']
    pm = json.loads((processed / 'manifest.json').read_text())
    if digest(processed / 'aligned.npz') != pm['aligned_sha256']:
        raise ValueError('Processed data hash mismatch')
    if digest(root / cfg['snapshot'] / 'zb202_iot.csv.gz') != pm['raw_sha256']:
        raise ValueError('Raw data hash mismatch')
    data = dict(np.load(processed / 'aligned.npz'))
    names = pm['feature_names']
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-benchmark-' + uuid4().hex[:8]
    output = root / 'artifacts/forecasting' / run_id
    output.mkdir(parents=True)
    record = {'run_id': run_id, 'status': 'running', 'protocol': protocol, 'model_config': cfg,
              'raw_sha256': pm['raw_sha256'], 'aligned_sha256': pm['aligned_sha256'],
              'processed_manifest_sha256': digest(processed/'manifest.json'),
              'code_sha256': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'versions': {'numpy': np.__version__, 'torch': torch.__version__},
              'folds': [], 'results': [],
              'limits': ['Development windows and old test are not new untouched data.',
                         'Overlapping windows are correlated; seed SD is not statistical confidence.',
                         'Ridge uses compressed history, not the full neural sequence representation.',
                         'No raw labels removed or clipped; no output clipping.',
                         'No BA inputs, no causal damper response claims.']}
    def save():
        (output/'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    save()  # Persist the protocol and code before any model sees assessment labels.
    folds = [(f'rolling-{i+1}', f, protocol['feature_sets']) for i, f in enumerate(protocol['fold_fractions'])]
    folds.append(('exposed-stress', protocol['exposed_stress_fractions'], protocol['exposed_stress_feature_sets']))
    for fold, fractions, sets in folds:
        boundaries = [int(len(data['timestamps'])*f) for f in fractions]
        train_end, inner_end, outer_end = boundaries
        origins = fold_origins(data, names, cfg, boundaries)
        x_all, current, y, norm = build_inputs(data, names, train_end, cfg['input_encoding'])
        record['folds'].append({'name': fold, 'boundaries': boundaries, 'normalization': norm,
                               'ranges_utc': {k: {'origins': len(idx), 'first': iso(data['timestamps'][idx[0]]),
                                                  'last': iso(data['timestamps'][idx[-1]])} for k, idx in origins.items()}})
        ix = origins['test']
        future = ix[:, None] + np.arange(1, cfg['horizon_steps']+1)
        truth, mask = data['targets'][future], np.isfinite(data['targets'][future])
        mean, scale = np.array(norm['target_mean']), np.array(norm['target_scale'])
        baseline = np.repeat(current[ix, None, :], cfg['horizon_steps'], axis=1) * scale + mean
        record['results'].append({'fold': fold, 'features': 'shared', 'model': 'persistence', 'seed': None,
                                  'seconds': 0., 'parameters': 0, 'metrics': summarize(baseline, truth, mask, baseline)})
        np.savez_compressed(output/f'{fold}-labels.npz', origins=data['timestamps'][ix],
                            target_bin_end_utc=data['timestamps'][future], actual_ppm=truth,
                            observed_mask=mask, persistence_ppm=baseline)
        for feature_set in sets:
            if feature_set == 'all':
                x = x_all
            elif feature_set == 'co2_only':
                cols = [names.index(s+'/co2_ppm') for s in SENSORS]
                x = x_all[:, cols + [len(names)+j for j in cols] + [2*len(names)+j for j in cols] + list(range(3*len(names), x_all.shape[1]))]
            else:
                raise ValueError('Unknown feature set')
            started = time.perf_counter()
            train_ix = origins['train']
            train_future = train_ix[:, None] + np.arange(1, cfg['horizon_steps']+1)
            ridge = DirectRidge(protocol['ridge_alpha']).fit(
                ridge_design(x, train_ix, cfg['lookback_steps']),
                y[train_future] - current[train_ix, None, :], np.isfinite(y[train_future]))
            pred = ridge.predict(ridge_design(x, ix, cfg['lookback_steps']), current[ix]) * scale + mean
            record['results'].append({'fold': fold, 'features': feature_set, 'model': 'ridge', 'seed': None,
                                      'seconds': time.perf_counter()-started, 'parameters': ridge.coef.size+ridge.intercept.size,
                                      'metrics': summarize(pred, truth, mask, baseline)})
            np.savez_compressed(output/f'{fold}-{feature_set}-ridge.npz', predictions_ppm=pred,
                                coef=ridge.coef, intercept=ridge.intercept)
            save()
            for kind in ['tcn', 'lstm']:
                for seed in protocol['seeds']:
                    print(f'Start {fold} {feature_set} {kind} seed={seed}', flush=True)
                    started = time.perf_counter()
                    model, (pred, actual, observed, persisted), history = fit_neural(kind, x, current, y, origins, cfg, norm, seed)
                    np.testing.assert_allclose(actual[mask], truth[mask], atol=.002)
                    np.testing.assert_array_equal(observed, mask)
                    np.testing.assert_allclose(persisted, baseline, atol=.002)
                    entry = {'fold': fold, 'features': feature_set, 'model': kind, 'seed': seed,
                             'seconds': time.perf_counter()-started, 'parameters': sum(p.numel() for p in model.parameters()),
                             'history': history, 'best_epoch': min(history, key=lambda v:v['inner_validation_mae_ppm'])['epoch'],
                             'metrics': summarize(pred, truth, mask, baseline)}
                    record['results'].append(entry)
                    stem = f'{fold}-{feature_set}-{kind}-{seed}'
                    np.savez_compressed(output/f'{stem}.npz', predictions_ppm=pred)
                    torch.save({'state_dict': model.state_dict(), 'model': kind, 'config': cfg,
                                'normalization': norm, 'feature_set': feature_set,
                                'input_features': x.shape[1], 'seed': seed}, output/f'{stem}.pt')
                    print(f"Done {stem}: MAE={entry['metrics']['mae_ppm']:.3f}, seconds={entry['seconds']:.1f}", flush=True)
                    save()
    record['status'] = 'complete'
    record['outputs'] = {p.name: digest(p) for p in output.iterdir() if p.name != 'manifest.json'}
    save()
    # Reports are curated manually; machine-readable results stay with the run.
    manifest_path = output / 'manifest.json'
    print('Results:', manifest_path, flush=True)
    return output, manifest_path
