"""One frozen, later-time confirmation with no fitting or new-data normalization."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch
from .benchmark import summarize
from .data import digest, iso, prepare, sample_origins, SENSORS
from .inference import FrozenForecaster


def confirm(root, protocol_path, snapshot):
    root = Path(root); protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    frozen = root/'artifacts/research/confirmation-freeze-20261020/protocol.json'
    if digest(protocol_path) != digest(frozen):
        raise ValueError('Confirmation protocol changed after freeze')
    reference = root/protocol['reference_run']
    base = json.loads((reference/'manifest.json').read_text())
    original_processed = root/base['protocol']['processed']
    if digest(original_processed/'manifest.json') != base['processed_manifest_sha256']:
        raise ValueError('Original schema source changed')
    original_names = json.loads((original_processed/'manifest.json').read_text())['feature_names']
    model = FrozenForecaster([root/p for p in protocol['models']], original_names,
                             [protocol['sha256'][p] for p in protocol['models']])
    cfg = dict(model.cfg, snapshot=str(snapshot))
    processed = prepare(root, cfg)
    pm = json.loads((processed/'manifest.json').read_text())
    if pm['feature_names'] != original_names:
        raise ValueError('New data schema differs; no automatic remapping during confirmation')
    data = dict(np.load(processed/'aligned.npz'))
    # Eligibility uses labels solely to exclude origins having no observed future label.
    origins = sample_origins(data, original_names, dict(cfg, train_fraction=1., validation_fraction=0.))[0]['train']
    cutoff = datetime.fromisoformat(protocol['after_utc'].replace('Z', '+00:00')).timestamp()
    raw_manifest = json.loads((root/snapshot/'manifest.json').read_text())
    stop = datetime.fromisoformat(raw_manifest['stop'].replace('Z', '+00:00')).timestamp()
    complete_end = np.floor(stop/cfg['step_seconds'])*cfg['step_seconds']
    origins = origins[(data['timestamps'][origins] >= cutoff) &
                      (data['timestamps'][origins+cfg['horizon_steps']] <= complete_end)]
    if not len(origins):
        raise ValueError('No eligible later-time origins')
    output = root/'artifacts/forecasting'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-confirmation-'+uuid4().hex[:8])
    output.mkdir(parents=True)
    record = dict(status='running', protocol=protocol, protocol_sha256=digest(protocol_path),
                  raw_sha256=pm['raw_sha256'], processed=str(processed.relative_to(root)),
                  processed_manifest_sha256=digest(processed/'manifest.json'), aligned_sha256=pm['aligned_sha256'],
                  source_kind='real-later-time-unseen-at-freeze', origins=len(origins),
                  first_origin_utc=iso(data['timestamps'][origins[0]]), last_origin_utc=iso(data['timestamps'][origins[-1]]),
                  code_sha256={p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                  limitations=['Retrospective new-time confirmation, not prospective live validation.',
                               'About four days, overlapping windows; no independent-sample confidence claim.',
                               'Ingestion arrival times unavailable; causality relies on recorded timestamps.',
                               'No BA features or changed-action prediction, calibration or real control claim.',
                               'After reporting this period is exposed; later changes need another confirmation period.'])
    for p in Path(__file__).parent.glob('*.py'):
        dest = output/'source_snapshot/zb202/forecasting'/p.name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    def save():
        (output/'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    save()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    pred = model.predict(data, original_names, origins, batch_size=cfg['batch_size'])
    future = origins[:, None]+np.arange(1, cfg['horizon_steps']+1)
    truth = data['targets'][future]; mask = np.isfinite(truth)
    cols = [original_names.index(s+'/co2_ppm') for s in SENSORS]
    persistence = np.repeat(data['values'][origins[:, None], cols][:, None, :], cfg['horizon_steps'], axis=1)
    # Data values are float32: match inference's float32 normalization/ppm conversion.
    mean, scale = np.asarray(model.norm['target_mean']), np.asarray(model.norm['target_scale'])
    persistence = ((persistence-mean)/scale).astype(np.float32)*scale+mean
    record['metrics'] = summarize(pred, truth, mask, persistence)
    record['persistence_metrics'] = summarize(persistence, truth, mask, persistence)
    record['mae_improvement_fraction'] = 1-record['metrics']['mae_ppm']/record['persistence_metrics']['mae_ppm']
    dates = np.array([datetime.fromtimestamp(float(t), timezone(timedelta(hours=8))).date().isoformat()
                      for t in data['timestamps'][origins]])
    record['daily'] = []
    for day in sorted(set(dates)):
        ix = dates == day
        record['daily'].append(dict(origin_date_hk=day, origins=int(ix.sum()),
                                   model=summarize(pred[ix], truth[ix], mask[ix], persistence[ix]),
                                   persistence=summarize(persistence[ix], truth[ix], mask[ix], persistence[ix])))
    np.savez_compressed(output/'predictions.npz', origins=data['timestamps'][origins],
                        target_bin_end_utc=data['timestamps'][future], predictions_ppm=pred,
                        actual_ppm=truth, observed_mask=mask, persistence_ppm=persistence)
    record['status'] = 'complete'
    record['outputs'] = {'predictions.npz': digest(output/'predictions.npz')}; save()
    # Reports are curated manually; machine-readable results stay with the run.
    manifest_path = output / 'manifest.json'
    print('Results:', manifest_path, flush=True)
    print(json.dumps({k: record[k] for k in ['origins', 'metrics', 'persistence_metrics', 'mae_improvement_fraction']}), flush=True)
    return output, manifest_path
