"""Reproducible frozen-input stress tests and honest single-channel external checks.

External checks deliberately use a one-target TCN, not fabricated copies of five
sensors. Forecast targets are always untouched; corruptions affect inputs only.
"""
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import time
from uuid import uuid4
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import DirectRidge, ridge_design
from .benchmark import fit_neural
from .data import SENSORS, digest, iso
from .evaluation import scores, predict
from .inference import FrozenForecaster
from .windows import Windows

PUBLIC_SOURCES = {
    'uci-occupancy-detection': {
        'landing_url': 'https://archive.ics.uci.edu/dataset/357/occupancy+detection',
        'download_url': 'https://archive.ics.uci.edu/static/public/357/occupancy%2Bdetection.zip',
        'doi': '10.24432/C5X01N', 'citation': 'Candanedo, L. (2016). Occupancy Detection. UCI Machine Learning Repository.',
        'license': 'CC BY 4.0', 'co2_field': 'CO2'},
    'uci-room-occupancy-estimation': {
        'landing_url': 'https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation',
        'download_url': 'https://archive.ics.uci.edu/static/public/864/room%2Boccupancy%2Bestimation.zip',
        'doi': '10.24432/C5P605', 'citation': 'Singh, A. & Chaudhari, S. (2018). Room Occupancy Estimation. UCI Machine Learning Repository.',
        'license': 'CC BY 4.0', 'co2_field': 'S5_CO2'},
}


def align_observations(timestamps, values, step=300, max_age=900):
    """Right-closed bins; only past readings, exact-bin labels, never fill gaps."""
    timestamps, values = np.asarray(timestamps), np.asarray(values)
    order = np.argsort(timestamps, kind='stable')
    timestamps, values = timestamps[order], values[order]
    if len(timestamps) == 0 or not np.isfinite(timestamps).all() or not np.isfinite(values).all():
        raise ValueError('Finite nonempty raw observations required')
    if np.any(np.diff(timestamps) == 0):
        raise ValueError('Resolve duplicate raw timestamps explicitly before alignment')
    grid = np.arange(np.ceil(timestamps[0]/step)*step, np.floor(timestamps[-1]/step)*step+1, step)
    pos = np.searchsorted(timestamps, grid, side='right')-1
    age = grid-timestamps[pos]
    fresh = age <= max_age
    observed = age < step  # (end-step, end], including reading exactly at end.
    series = np.where(fresh, values[pos], np.nan).astype(np.float32)[:, None]
    targets = np.where(observed, values[pos], np.nan).astype(np.float32)[:, None]
    return dict(timestamps=grid, values=series, ages=age[:, None].astype(np.float32), targets=targets)


def read_public(root, dataset):
    """Read authoritative archives without extracting or using occupancy labels."""
    path = root/'data/raw/public'/dataset/'source.zip'
    times, values, by_file = [], [], {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(('.txt', '.csv')):
                continue
            rows = csv.reader(io.StringIO(archive.read(name).decode('utf-8-sig')))
            header = next(rows)
            count = 0
            for row in rows:
                if not row:
                    continue
                # UCI357 has an unnamed leading record id on each data row.
                if len(row) == len(header)+1:
                    row = row[1:]
                if len(row) != len(header):
                    raise ValueError('Unexpected public CSV width')
                record = dict(zip(header, row))
                raw_time = record.get('date') or (record['Date'].replace('/', '-')+' '+record['Time'])
                parsed = datetime.fromisoformat(raw_time)
                if parsed.tzinfo is not None:
                    raise ValueError('Recheck timezone policy for changed source')
                # Nominal epoch for ordering, not a claim that local source time is UTC.
                times.append(parsed.replace(tzinfo=timezone.utc).timestamp())
                values.append(float(record[PUBLIC_SOURCES[dataset]['co2_field']]))
                count += 1
            by_file[name] = count
    data = align_observations(times, values)
    unique = sorted(set(times))
    meta = dict(PUBLIC_SOURCES[dataset], archive_sha256=digest(path), rows=len(times),
                raw_files=by_file, nominal_first_time=iso(min(times)), nominal_last_time=iso(max(times)),
                co2_min_ppm=min(values), co2_max_ppm=max(values),
                raw_max_gap_seconds=float(np.diff(unique).max()), grid_bins=len(data['timestamps']),
                observed_bins=int(np.isfinite(data['targets']).sum()),
                timezone='Unverified naive local time; nominal epoch only; calendar features disabled')
    return data, meta


def external_origins(data, cfg, fractions):
    """Split by observed-bin ranks, retaining the original grid across long gaps."""
    observed = np.flatnonzero(np.isfinite(data['targets'][:, 0]))
    boundaries = [int(observed[min(int(len(observed)*f), len(observed)-1)])
                  if f < 1 else len(data['timestamps']) for f in fractions]
    train_end, val_end, stop = boundaries
    origins = {key: [] for key in ('train', 'validation', 'test')}
    length, horizon = cfg['lookback_steps'], cfg['horizon_steps']
    for origin in range(length-1, stop-horizon):
        if not np.isfinite(data['values'][origin]).all():
            continue
        if np.isfinite(data['values'][origin-length+1:origin+1]).mean() < .8:
            continue
        if not np.isfinite(data['targets'][origin+1:origin+horizon+1]).all():
            continue
        if origin+horizon < train_end:
            origins['train'].append(origin)
        elif origin >= train_end and origin+horizon < val_end:
            origins['validation'].append(origin)
        elif origin >= val_end:
            origins['test'].append(origin)
    origins = {key: np.asarray(value, dtype=np.int64) for key, value in origins.items()}
    if any(len(value) == 0 for value in origins.values()):
        raise ValueError('Empty external temporal partition')
    return origins, boundaries


def single_inputs(data, train_end=None, norm=None):
    """One real CO2 channel plus mask/age; four calendar channels fixed to zero."""
    values = data['values']
    if norm is None:
        if train_end is None or not 0 < train_end <= len(values):
            raise ValueError('Training boundary required')
        train = values[:train_end]
        norm = dict(target_mean=[float(np.nanmean(train))],
                    target_scale=[max(float(np.nanstd(train)), .001)])
    mean, scale = np.asarray(norm['target_mean']), np.asarray(norm['target_scale'])
    current = ((values-mean)/scale).astype(np.float32)
    x = np.concatenate([np.nan_to_num(current), np.isfinite(values).astype(np.float32),
                        np.minimum(data['ages'], 86400)/86400,
                        np.zeros((len(values), 4), dtype=np.float32)], axis=1).astype(np.float32)
    y = ((data['targets']-mean)/scale).astype(np.float32)
    return x, current, y, norm


def metric_bundle(prediction, truth, baseline, timestamps, change_reference=None):
    mask = np.isfinite(truth)
    result = dict(model=scores(prediction, truth, mask), persistence=scores(baseline, truth, mask))
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite prediction')
    result['skill_vs_persistence'] = 1-result['model']['mae_ppm']/max(result['persistence']['mae_ppm'], 1e-12)
    result['p99_abs_error_ppm'] = float(np.quantile(abs(prediction-truth)[mask], .99))
    result['negative_predictions'] = int((prediction < 0).sum())
    result['by_horizon'] = {str(h*5): dict(model=scores(prediction[:, h-1], truth[:, h-1], mask[:, h-1]),
                                            persistence=scores(baseline[:, h-1], truth[:, h-1], mask[:, h-1]))
                            for h in (2, 6, 12)}
    # Nonoverlap sensitivity is anchored to the first accepted origin, not best-case selection.
    select = []
    last = -np.inf
    for i, stamp in enumerate(timestamps):
        if stamp-last >= 3600:
            select.append(i); last = stamp
    result['hour_spaced'] = dict(origins=len(select), model=scores(prediction[select], truth[select], mask[select]),
                                 persistence=scores(baseline[select], truth[select], mask[select]))
    result['by_sensor'] = [dict(model=scores(prediction[:, :, j], truth[:, :, j], mask[:, :, j]),
                                persistence=scores(baseline[:, :, j], truth[:, :, j], mask[:, :, j]))
                           for j in range(prediction.shape[2])]
    change_reference = baseline if change_reference is None else change_reference
    result['change_gt20_at60min'] = dict(
        model=scores(prediction[:, -1], truth[:, -1], mask[:, -1] & (abs(truth[:, -1]-change_reference[:, -1]) > 20)),
        persistence=scores(baseline[:, -1], truth[:, -1], mask[:, -1] & (abs(truth[:, -1]-change_reference[:, -1]) > 20)))
    dates = (np.asarray(timestamps)//86400).astype(int)
    result['daily'] = [dict(nominal_day=int(day), origins=int((dates == day).sum()),
                            model=scores(prediction[dates == day], truth[dates == day], mask[dates == day]),
                            persistence=scores(baseline[dates == day], truth[dates == day], mask[dates == day]))
                       for day in np.unique(dates)]
    return result


def corrupt_history(data, columns, scenario, seed):
    """Pure input-only perturbation; preserve timestamps and label arrays."""
    out = {k: v.copy() if k in ('values', 'ages') else v for k, v in data.items()}
    values = out['values'][:, columns].copy()
    rng = np.random.default_rng(seed)
    kind = scenario['kind']
    if kind == 'noise':
        values += rng.normal(0, scenario['sigma_ppm'], values.shape)
    elif kind == 'bias':
        values += scenario['offset_ppm']
    elif kind == 'single_sensor_bias':
        values[:, scenario['sensor']] += scenario['offset_ppm']
    elif kind == 'spikes':
        values += (rng.random(values.shape) < scenario['probability'])*scenario['offset_ppm']
    elif kind == 'missing':
        values[rng.random(values.shape) < scenario['probability']] = np.nan
    elif kind == 'periodic_outage':
        # Absolute nominal five-minute index gives a reproducible network outage phase.
        select = (out['timestamps']//300).astype(int) % scenario['period_steps'] < scenario['duration_steps']
        values[select, scenario['sensor']] = np.nan
    elif kind == 'sensor_outage':
        values[:, scenario['sensor']] = np.nan
    elif kind == 'stale_current':
        out['ages'][:, columns] = scenario['age_seconds']
    elif kind != 'clean':
        raise ValueError('Unknown stress kind')
    out['values'][:, columns] = values
    return out


def eligible_history(data, columns, origins, cfg):
    """Vectorized inference eligibility, including freshness, with no target access."""
    indices = origins[:, None]+np.arange(-cfg['lookback_steps']+1, 1)
    values = data['values'][indices][:, :, columns]
    ages = data['ages'][indices][:, :, columns]
    inconsistent = np.isfinite(values) & ((ages < 0) | ~np.isfinite(ages) |
                                         (ages > cfg['environment_max_age_seconds']))
    return (~inconsistent.any(axis=(1, 2)) &
            (np.isfinite(values).mean(axis=(1, 2)) >= cfg['minimum_co2_input_coverage']) &
            np.isfinite(data['values'][origins][:, columns]).all(axis=1))


def local_stress(root, protocol, output, record, save):
    frozen = json.loads((root/protocol['frozen_protocol']).read_text())
    confirmation = root/protocol['confirmation_run']
    prior = json.loads((confirmation/'manifest.json').read_text())
    processed = root/prior['processed']
    pm = json.loads((processed/'manifest.json').read_text())
    if digest(processed/'aligned.npz') != prior['aligned_sha256']:
        raise ValueError('Local aligned data hash mismatch')
    data = dict(np.load(processed/'aligned.npz'))
    names = pm['feature_names']
    model = FrozenForecaster([root/p for p in frozen['models']], names,
                             [frozen['sha256'][p] for p in frozen['models']])
    cols = [names.index(s+'/co2_ppm') for s in SENSORS]
    if digest(confirmation/'predictions.npz') != prior['outputs']['predictions.npz']:
        raise ValueError('Reference prediction hash mismatch')
    labels = dict(np.load(confirmation/'predictions.npz'))
    origins = np.searchsorted(data['timestamps'], labels['origins'])
    np.testing.assert_array_equal(data['timestamps'][origins], labels['origins'])
    truth = labels['actual_ppm']
    clean = model.predict(data, names, origins)
    np.testing.assert_allclose(clean, labels['predictions_ppm'], atol=1e-5)
    record['local_source'] = dict(confirmation_manifest_sha256=digest(confirmation/'manifest.json'),
                                 aligned_sha256=prior['aligned_sha256'], target_names=SENSORS,
                                 checkpoint_sha256=frozen['sha256'], origins=len(origins),
                                 first_origin_utc=iso(labels['origins'][0]), last_origin_utc=iso(labels['origins'][-1]))
    for scenario_index, scenario in enumerate(protocol['scenarios']):
        seeds = protocol['corruption_seeds'] if scenario['kind'] in ('noise', 'missing', 'spikes') else [None]
        for seed in seeds:
            changed = corrupt_history(data, cols, scenario, seed)
            accepted = eligible_history(changed, cols, origins, model.cfg)
            selected = origins[accepted]
            entry = dict(scenario=scenario, seed=seed, candidate_origins=len(origins),
                         accepted_origins=int(accepted.sum()), availability_fraction=float(accepted.mean()))
            if len(selected):
                pred = model.predict(changed, names, selected)
                baseline = np.repeat(changed['values'][selected][:, None, cols], model.cfg['horizon_steps'], axis=1)
                entry['metrics'] = metric_bundle(pred, truth[accepted], baseline, labels['origins'][accepted],
                                                 change_reference=labels['persistence_ppm'][accepted])
                clean_base = labels['persistence_ppm'][accepted]
                entry['clean_same_origins'] = metric_bundle(clean[accepted], truth[accepted], clean_base, labels['origins'][accepted])
                entry['mae_ratio_to_clean_same_origins'] = (entry['metrics']['model']['mae_ppm']/
                                                          entry['clean_same_origins']['model']['mae_ppm'])
                np.savez_compressed(output/f'local-{scenario_index}-{seed}.npz', origins=labels['origins'][accepted],
                                    predictions_ppm=pred, actual_ppm=truth[accepted], persistence_ppm=baseline)
            record['local_stress'].append(entry); save()
            print('Local', scenario, seed, 'availability', round(entry['availability_fraction'], 3),
                  'MAE', entry.get('metrics', {}).get('model', {}).get('mae_ppm'), flush=True)
    # Retain an earlier high-concentration period without claiming it is unseen.
    reference = json.loads((root/frozen['reference_run']/'manifest.json').read_text())
    old = dict(np.load(root/reference['protocol']['processed']/'aligned.npz'))
    start = old['timestamps'][int(len(old['timestamps'])*.85)]
    stop = old['timestamps'][-1]
    candidates = np.flatnonzero((data['timestamps'] >= start) & (data['timestamps'] <= stop-model.cfg['horizon_steps']*300))
    candidates = candidates[candidates >= model.cfg['lookback_steps']-1]
    candidates = candidates[eligible_history(data, cols, candidates, model.cfg)]
    future = candidates[:, None]+np.arange(1, model.cfg['horizon_steps']+1)
    actual = data['targets'][future]
    keep = np.isfinite(actual).any(axis=(1, 2)); candidates, actual = candidates[keep], actual[keep]
    pred = model.predict(data, names, candidates)
    baseline = np.repeat(data['values'][candidates][:, None, cols], model.cfg['horizon_steps'], axis=1)
    record['historical_exposed_stress'] = metric_bundle(pred, actual, baseline, data['timestamps'][candidates])
    mask = np.isfinite(actual) & (actual >= 1000)
    record['historical_exposed_stress']['labels_ge1000ppm'] = dict(model=scores(pred, actual, mask), persistence=scores(baseline, actual, mask))
    np.savez_compressed(output/'historical-exposed.npz', origins=data['timestamps'][candidates], predictions_ppm=pred,
                        actual_ppm=actual, persistence_ppm=baseline)
    save()


def external_checks(root, protocol, output, record, save):
    cfg = protocol['external_model']
    cache, models = {}, {}
    for name in PUBLIC_SOURCES:
        data, source = read_public(root, name)
        origins, boundaries = external_origins(data, cfg, protocol['external_split_fractions'])
        x, current, y, norm = single_inputs(data, boundaries[0])
        ix = origins['test']; future = ix[:, None]+np.arange(1, cfg['horizon_steps']+1)
        truth = data['targets'][future]
        baseline = np.repeat(data['values'][ix, None], cfg['horizon_steps'], axis=1)
        record['external_sources'][name] = dict(source, boundaries=boundaries, normalization=norm,
            partitions={k: dict(origins=len(v), nominal_first_origin=iso(data['timestamps'][v[0]]),
                                nominal_last_origin=iso(data['timestamps'][v[-1]])) for k, v in origins.items()},
            boundary_nominal_times=[iso(data['timestamps'][b]) if b < len(data['timestamps']) else None for b in boundaries])
        cache[name] = (data, origins, truth, baseline)
        models[name] = []
        np.savez_compressed(output/f'{name}-aligned.npz', **data)
        save()
        train = origins['train']; train_future = train[:, None]+np.arange(1, cfg['horizon_steps']+1)
        ridge = DirectRidge(protocol['external_ridge_alpha']).fit(ridge_design(x, train, cfg['lookback_steps']),
            y[train_future]-current[train, None], np.isfinite(y[train_future]))
        prediction = ridge.predict(ridge_design(x, ix, cfg['lookback_steps']), current[ix])*np.asarray(norm['target_scale'])+np.asarray(norm['target_mean'])
        record['external_results'].append(dict(dataset=name, model='ridge',
            metrics=metric_bundle(prediction, truth, baseline, data['timestamps'][ix])))
        predictions = []
        for seed in protocol['training_seeds']:
            print('External train', name, 'seed', seed, flush=True); start = time.perf_counter()
            model, (pred, actual, observed, persisted), history = fit_neural('tcn', x, current, y, origins, cfg, norm, seed)
            np.testing.assert_allclose(actual, truth, atol=.002)
            np.testing.assert_allclose(persisted, baseline, atol=.002)
            models[name].append((model, norm)); predictions.append(pred)
            record['external_results'].append(dict(dataset=name, model='tcn', seed=seed, seconds=time.perf_counter()-start,
                history=history, metrics=metric_bundle(pred, truth, baseline, data['timestamps'][ix])))
            torch.save(dict(model='tcn', state_dict=model.state_dict(), config=cfg, normalization=norm,
                            input_features=7, targets=1, calendar='disabled', source=name, seed=seed),
                       output/f'{name}-tcn-{seed}.pt')
            np.savez_compressed(output/f'{name}-tcn-{seed}.npz', origins=data['timestamps'][ix], predictions_ppm=pred,
                                actual_ppm=truth, persistence_ppm=baseline)
            save()
        avg = np.mean(predictions, axis=0)
        record['external_results'].append(dict(dataset=name, model='tcn_seed_mean',
            metrics=metric_bundle(avg, truth, baseline, data['timestamps'][ix])))
        np.savez_compressed(output/f'{name}-tcn-seed-mean.npz', origins=data['timestamps'][ix], predictions_ppm=avg,
                            actual_ppm=truth, persistence_ppm=baseline); save()
    # Same one-channel contract in both directions; all source weights/statistics frozen.
    for source, source_models in models.items():
        for target, (data, origins, truth, baseline) in cache.items():
            if source == target:
                continue
            ix = origins['test']; predictions = []
            for model, norm in source_models:
                x, current, y, _ = single_inputs(data, norm=norm)
                loader = DataLoader(Windows(x, current, y, ix, cfg), batch_size=cfg['batch_size'], shuffle=False)
                pred, actual, _, _ = predict(model, loader, np.asarray(norm['target_mean']), np.asarray(norm['target_scale']))
                np.testing.assert_allclose(actual, truth, atol=.002)
                predictions.append(pred)
            avg = np.mean(predictions, axis=0)
            record['transfer_results'].append(dict(source=source, target=target, normalization='source only; no target fitting',
                metrics=metric_bundle(avg, truth, baseline, data['timestamps'][ix])))
            np.savez_compressed(output/f'transfer-{source}-to-{target}.npz', origins=data['timestamps'][ix],
                                predictions_ppm=avg, actual_ppm=truth, persistence_ppm=baseline); save()


def run_robustness(root, config_path):
    root = Path(root); config_path = Path(config_path)
    protocol = json.loads(config_path.read_text())
    output = root/'artifacts/forecasting'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-robustness-'+uuid4().hex[:8])
    output.mkdir(parents=True)
    record = dict(status='running', protocol=protocol, protocol_sha256=digest(config_path),
                  versions={'numpy': np.__version__, 'torch': torch.__version__},
                  local_stress=[], external_sources={}, external_results=[], transfer_results=[],
                  code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    def save():
        (output/'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    for path in Path(__file__).parent.glob('*.py'):
        dest = output/'source_snapshot/zb202/forecasting'/path.name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(path.read_bytes())
    (output/'protocol.json').write_bytes(config_path.read_bytes())
    save(); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    try:
        local_stress(root, protocol, output, record, save)
        external_checks(root, protocol, output, record, save)
    except Exception as exc:
        record['status'] = 'failed'; record['error'] = str(exc); save(); raise
    record['status'] = 'complete'
    record['outputs'] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'}
    save(); print('Results:', output/'manifest.json', flush=True)
    return output, output/'manifest.json'
