"""Recompute paired development errors against the original frozen forecaster."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from zb202.forecasting.benchmark import summarize
from zb202.forecasting.data import digest
from zb202.forecasting.inference import FrozenForecaster


def main():
    p = argparse.ArgumentParser()
    p.add_argument('benchmark', type=Path)
    p.add_argument('--physics', type=Path)
    args = p.parse_args()
    base = json.loads((args.benchmark/'manifest.json').read_text())
    if base['status'] != 'complete':
        raise ValueError('Benchmark is not complete')
    processed = ROOT/base['protocol']['processed']
    pm = json.loads((processed/'manifest.json').read_text())
    if digest(processed/'aligned.npz') != base['aligned_sha256'] or digest(processed/'manifest.json') != base['processed_manifest_sha256']:
        raise ValueError('Processed data does not match experiment')
    frozen_protocol = ROOT/'configs/forecasting/confirmation.json'
    frozen = json.loads(frozen_protocol.read_text())
    model = FrozenForecaster([ROOT/p for p in frozen['models']], pm['feature_names'],
                             [frozen['sha256'][p] for p in frozen['models']])
    data = dict(np.load(processed/'aligned.npz'))
    torch.set_num_threads(4)
    physics_manifest = None
    if args.physics:
        physics_manifest = json.loads((args.physics/'manifest.json').read_text())
        if physics_manifest['status'] != 'complete' or physics_manifest['base_manifest_sha256'] != digest(args.benchmark/'manifest.json'):
            raise ValueError('Physics run does not match completed benchmark')
    output = ROOT/'artifacts/forecasting'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-paired-development-'+uuid4().hex[:8])
    output.mkdir(parents=True)
    result = {'status':'running', 'benchmark':str(args.benchmark), 'benchmark_sha256':digest(args.benchmark/'manifest.json'),
              'physics':str(args.physics) if args.physics else None, 'frozen_protocol_sha256':digest(frozen_protocol),
              'script_sha256':digest(__file__), 'folds':[],
              'limitations':['Development data have been used for model comparison; not independent test.',
                             'Overlapping windows are correlated; seeds are not independent test datasets.',
                             'Original frozen model may have been trained on some of these historical periods.',
                             'No weighted combination of different methods.']}
    for fold in [f['name'] for f in base['folds']]:
        label_path = args.benchmark/(fold+'-labels.npz')
        if digest(label_path) != base['outputs'][label_path.name]:
            raise ValueError('Label checksum mismatch')
        z = np.load(label_path)
        origins = np.searchsorted(data['timestamps'], z['origins'])
        if (origins >= len(data['timestamps'])).any() or not np.array_equal(data['timestamps'][origins], z['origins']):
            raise ValueError('Origins not on aligned grid')
        actual, mask, persistence = z['actual_ppm'], z['observed_mask'], z['persistence_ppm']
        frozen_pred = model.predict(data, pm['feature_names'], origins)
        predictions = {'persistence':persistence, 'frozen':frozen_pred}
        for x in base['results']:
            if x['fold'] != fold or x['model'] == 'persistence':
                continue
            stem = f"{fold}-{x['features']}-{x['model']}" + (f"-{x['seed']}" if x['seed'] is not None else '') + '.npz'
            if digest(args.benchmark/stem) != base['outputs'][stem]:
                raise ValueError('Benchmark prediction checksum mismatch')
            predictions[f"{x['features']}-{x['model']}-{x['seed']}"] = np.load(args.benchmark/stem)['predictions_ppm']
        if physics_manifest and fold in [f['name'] for f in physics_manifest['folds']]:
            check = np.load(args.physics/(fold+'-labels.npz'))
            for key in ('origins','actual_ppm','observed_mask'):
                np.testing.assert_equal(check[key], z[key])
            for x in physics_manifest['results']:
                if x['fold'] == fold:
                    stem = f"{fold}-{x['model']}-{x['seed']}.npz"
                    if digest(args.physics/stem) != physics_manifest['outputs'][stem]:
                        raise ValueError('Physics prediction checksum mismatch')
                    predictions[f"{x['model']}-{x['seed']}"] = np.load(args.physics/stem)['predictions_ppm']
        metrics = {name:summarize(pred, actual, mask, persistence) for name,pred in predictions.items()}
        result['folds'].append({'fold':fold, 'origins':len(origins), 'first_origin_utc':datetime.fromtimestamp(float(z['origins'][0]),timezone.utc).isoformat(),
                                'last_origin_utc':datetime.fromtimestamp(float(z['origins'][-1]),timezone.utc).isoformat(), 'metrics':metrics})
        np.savez_compressed(output/(fold+'-paired.npz'), origins=z['origins'], actual_ppm=actual, observed_mask=mask, **predictions)
    result['status'] = 'complete'
    result['outputs'] = {p.name:digest(p) for p in output.glob('*.npz')}
    (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(output)
    for fold in result['folds']:
        print(fold['fold'], {k:round(v['mae_ppm'],3) for k,v in fold['metrics'].items()})


if __name__ == '__main__':
    main()
