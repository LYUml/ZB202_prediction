"""Single frozen chronological split, validation early stopping, then one test pass."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import random
import sys
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SENSORS, digest, iso, prepare, sample_origins
from .tcn import TCN
from .features import build_inputs
from .windows import Windows
from .evaluation import predict, scores


def train(project_root, config_path):
    root = Path(project_root)
    config = json.loads(Path(config_path).read_text())
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    processed = prepare(root, config)
    data = dict(np.load(processed / 'aligned.npz'))
    processed_manifest = json.loads((processed / 'manifest.json').read_text())
    names = processed_manifest['feature_names']
    origins, train_end, val_end = sample_origins(data, names, config)
    if any(len(v) == 0 for v in origins.values()):
        raise ValueError('Insufficient eligible samples in one or more chronological splits')
    x, current, y, normalization = build_inputs(data, names, train_end, config.get('input_encoding', 'safe-v2'))
    loaders = {key: DataLoader(Windows(x, current, y, idx, config), batch_size=config['batch_size'],
                              shuffle=key == 'train', num_workers=0) for key, idx in origins.items()}
    model = TCN(x.shape[1], len(SENSORS), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    mean, scale = np.array(normalization['target_mean']), np.array(normalization['target_scale'])
    best, stale, best_state, history = float('inf'), 0, None, []
    for epoch in range(config['epochs']):
        model.train()
        total, count = 0, 0
        for inputs, latest, target, mask in loaders['train']:
            prediction = model(inputs, latest)
            loss = torch.abs(prediction - target)[mask].mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * mask.sum().item(); count += mask.sum().item()
        vp, vy, vm, _ = predict(model, loaders['validation'], mean, scale)
        value = scores(vp, vy, vm)['mae_ppm']
        history.append({'epoch': epoch + 1, 'train_normalized_mae': total / count, 'validation_mae_ppm': value})
        print(f"Epoch {epoch+1}: validation MAE {value:.2f} ppm", flush=True)
        if value < best:
            best, stale, best_state = value, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config['patience']:
            break
    if best_state is None:
        raise ValueError('No finite validation checkpoint')
    model.load_state_dict(best_state)
    # Test is accessed for metrics only after model selection is finished.
    pred, truth, mask, baseline = predict(model, loaders['test'], mean, scale)
    metrics = {'overall': {'tcn': scores(pred, truth, mask), 'persistence': scores(baseline, truth, mask)},
               'by_horizon': {}, 'by_sensor': {}}
    for step in [2, 6, 12]:
        if step > config['horizon_steps']:
            continue
        j = step - 1
        metrics['by_horizon'][str(step * config['step_seconds'] // 60)] = {
            'tcn': scores(pred[:, j], truth[:, j], mask[:, j]),
            'persistence': scores(baseline[:, j], truth[:, j], mask[:, j])}
    for j, sensor in enumerate(SENSORS):
        metrics['by_sensor'][sensor] = {'tcn': scores(pred[:, :, j], truth[:, :, j], mask[:, :, j]),
                                         'persistence': scores(baseline[:, :, j], truth[:, :, j], mask[:, :, j])}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    output = root / 'artifacts/forecasting' / run_id
    output.mkdir(parents=True)
    torch.save({'state_dict': best_state, 'config': config, 'input_features': x.shape[1],
                'target_names': SENSORS, 'feature_names': names, 'normalization': normalization}, output / 'tcn.pt')
    np.savez_compressed(output / 'test_predictions.npz', origins=data['timestamps'][origins['test']],
                        target_bin_end_utc=data['timestamps'][origins['test'][:, None] + np.arange(1, config['horizon_steps'] + 1)],
                        predictions_ppm=pred, actual_ppm=np.where(mask, truth, np.nan),
                        observed_mask=mask, persistence_ppm=baseline)
    manifest = {'status': 'first-single-split-experiment', 'run_id': run_id, 'source': 'real-observations',
                'config': config, 'config_sha256': digest(config_path), 'processed': str(processed.relative_to(root)),
                'processed_manifest_sha256': digest(processed / 'manifest.json'), 'raw_sha256': processed_manifest['raw_sha256'],
                'normalization': normalization, 'model_parameters': sum(p.numel() for p in model.parameters()),
                'split_boundaries': {'validation_start_utc': iso(data['timestamps'][train_end]), 'test_start_utc': iso(data['timestamps'][val_end])},
                'splits': {k: {'forecast_origins': len(idx), 'first_origin_utc': iso(data['timestamps'][idx[0]]),
                                'last_origin_utc': iso(data['timestamps'][idx[-1]])} for k, idx in origins.items()},
                'history': history, 'metrics': metrics, 'device': 'cpu', 'versions': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__},
                'code_sha256': {str(p.relative_to(root)): digest(p) for p in sorted((root / 'src/zb202/forecasting').glob('*.py'))},
                'outputs': {p.name: digest(p) for p in output.iterdir()},
                'limitations': ['One seed, one chronological split; not rolling-origin validation or external validation.',
                                'Overlapping forecast windows are correlated, not independent test examples.',
                                'No BA inputs; no inference about effects of new damper actions.',
                                'Target timestamps denote right-closed bins, not exactly synchronized sensor readings.',
                                'Event freshness limits are modelling assumptions; no verified people counts.',
                                'Arrival timestamps unavailable; causality checked against stored measurement timestamps only.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Reports are curated manually; machine-readable results stay with the run.
    manifest_path = output / 'manifest.json'
    print('Results:', manifest_path, flush=True)
    return output, manifest_path
