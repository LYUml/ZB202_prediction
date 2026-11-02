"""Train a preselected single model and evaluate on the reserved latest time block."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
from uuid import uuid4
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from zb202.forecasting.ablation import feature_columns
from zb202.forecasting.benchmark import fit_neural, fold_origins, summarize
from zb202.forecasting.data import digest, iso
from zb202.forecasting.features import build_inputs
from zb202.forecasting.inference import FrozenForecaster


def main():
    p = argparse.ArgumentParser()
    p.add_argument('selection', type=Path, help='Precommitted model selection JSON')
    args = p.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection['kind'] not in ('tcn', 'lstm', 'relaxation') or selection['seed'] != 42:
        raise ValueError('Unexpected model/seed; selection must follow fixed protocol')
    benchmark = ROOT/selection['benchmark']
    bm = json.loads((benchmark/'manifest.json').read_text())
    if bm['status'] != 'complete' or digest(benchmark/'manifest.json') != selection['benchmark_sha256']:
        raise ValueError('Benchmark is incomplete or changed')
    physics = selection.get('physics')
    if physics and digest(ROOT/physics/'manifest.json') != selection['physics_sha256']:
        raise ValueError('Physics comparison changed')
    processed = ROOT/bm['protocol']['processed']
    pm = json.loads((processed/'manifest.json').read_text())
    if digest(processed/'manifest.json') != bm['processed_manifest_sha256'] or digest(processed/'aligned.npz') != bm['aligned_sha256']:
        raise ValueError('Data mismatch')
    cfg = bm['model_config']
    if cfg['input_encoding'] != 'safe-v2':
        raise ValueError('Expected safe encoding')
    data = dict(np.load(processed/'aligned.npz'))
    names = pm['feature_names']
    n = len(data['timestamps'])
    bounds = [int(n*f) for f in (.80, .90, 1.)]
    origins = fold_origins(data, names, cfg, bounds)
    raw = json.loads((ROOT/cfg['snapshot']/'manifest.json').read_text())
    stop = datetime.fromisoformat(raw['stop'].replace('Z','+00:00')).timestamp()
    complete_end = np.floor(stop/cfg['step_seconds'])*cfg['step_seconds']
    origins['test'] = origins['test'][data['timestamps'][origins['test']+cfg['horizon_steps']] <= complete_end]
    if not len(origins['test']):
        raise ValueError('No complete holdout origins')
    x, current, y, norm = build_inputs(data, names, bounds[0], cfg['input_encoding'])
    cols = feature_columns(names, 'co2_only')
    x = x[:, cols]
    output = ROOT/'artifacts/forecasting'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-latest-holdout-'+uuid4().hex[:8])
    output.mkdir(parents=True)
    record = {'status':'running', 'selection':selection, 'selection_sha256':digest(args.selection),
              'script_sha256':digest(__file__), 'aligned_sha256':bm['aligned_sha256'], 'bounds':bounds,
              'ranges_utc':{k:{'count':len(idx),'first':iso(data['timestamps'][idx[0]]),'last':iso(data['timestamps'][idx[-1]])} for k,idx in origins.items()},
              'limitations':['Retrospective holdout, previously exposed to frozen model and feedback experiment.',
                             'Single predetermined seed; no selection on holdout.',
                             'No inference about causal damper response; labels and recorded times retain original uncertainty.',
                             'Overlapping windows are correlated; performance can shift over time.']}
    def save():
        (output/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    save()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    model, (pred, actual, mask, persistence), history = fit_neural(selection['kind'], x, current, y, origins, cfg, norm, selection['seed'])
    idx = origins['test']
    truth = data['targets'][idx[:,None]+np.arange(1,cfg['horizon_steps']+1)]
    np.testing.assert_array_equal(mask, np.isfinite(truth))
    np.testing.assert_allclose(actual[mask], truth[mask], atol=.002)
    protocol = json.loads((ROOT/'configs/forecasting/confirmation.json').read_text())
    frozen = FrozenForecaster([ROOT/p for p in protocol['models']], names, [protocol['sha256'][p] for p in protocol['models']])
    old = frozen.predict(data, names, idx)
    record.update(history=history, best_epoch=min(history,key=lambda h:h['inner_validation_mae_ppm'])['epoch'],
                  metrics={k:summarize(v, truth, mask, persistence) for k,v in {'candidate':pred,'frozen':old,'persistence':persistence}.items()})
    dates = np.array([datetime.fromtimestamp(float(t),timezone(timedelta(hours=8))).date().isoformat() for t in data['timestamps'][idx]])
    record['daily'] = [{'date_hk':day,'origins':int((dates==day).sum()),'metrics':{
        k:summarize(v[dates==day], truth[dates==day], mask[dates==day], persistence[dates==day])
        for k,v in {'candidate':pred,'frozen':old,'persistence':persistence}.items()}} for day in sorted(set(dates))]
    torch.save({'state_dict':model.state_dict(),'model':selection['kind'],'config':cfg,'normalization':norm,
                'feature_columns':cols,'feature_names':names,'input_features':x.shape[1],'seed':selection['seed']},output/'model.pt')
    restored = FrozenForecaster([output/'model.pt'], names, [digest(output/'model.pt')])
    np.testing.assert_allclose(restored.predict(data, names, idx), pred, atol=.002, rtol=1e-6)
    np.savez_compressed(output/'predictions.npz',origins=data['timestamps'][idx],target_bin_end_utc=data['timestamps'][idx[:,None]+np.arange(1,cfg['horizon_steps']+1)],
                        candidate_ppm=pred,frozen_ppm=old,persistence_ppm=persistence,actual_ppm=truth,observed_mask=mask)
    record['outputs']={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    record['status']='complete';save()
    print(output)
    print({k:round(v['mae_ppm'],3) for k,v in record['metrics'].items()})


if __name__ == '__main__':
    main()
