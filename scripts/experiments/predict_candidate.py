"""Forecast with a verified, single-model candidate from an experiment run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from zb202.forecasting.data import digest, SENSORS
from zb202.forecasting.inference import FrozenForecaster


def predict(run, processed, output):
    run, processed, output = map(Path, (run, processed, output))
    record = json.loads((run/'manifest.json').read_text())
    pm = json.loads((processed/'manifest.json').read_text())
    if record['status'] != 'complete' or digest(run/'model.pt') != record['outputs']['model.pt']:
        raise ValueError('Candidate incomplete or checkpoint altered')
    if digest(processed/'aligned.npz') != pm['aligned_sha256']:
        raise ValueError('Processed input hash mismatch')
    model = FrozenForecaster([run/'model.pt'], pm['feature_names'], [record['outputs']['model.pt']])
    for key in ('step_seconds','environment_max_age_seconds','event_max_age_seconds','power_max_age_seconds'):
        if pm['config'][key] != model.cfg[key]:
            raise ValueError('Incompatible history rule: '+key)
    raw = json.loads((ROOT/pm['config']['snapshot']/'manifest.json').read_text())
    stop = datetime.fromisoformat(raw['stop'].replace('Z','+00:00')).timestamp()
    timestamp = int(np.floor(stop/model.cfg['step_seconds'])*model.cfg['step_seconds'])
    with np.load(processed/'aligned.npz') as arrays:
        data = {key:arrays[key] for key in ('values','ages','timestamps')}
    found = np.flatnonzero(data['timestamps'] == timestamp)
    if len(found) != 1:
        raise ValueError('Latest complete bin unavailable')
    origin = int(found[0]); data = {key:val[:origin+1] for key,val in data.items()}
    torch.set_num_threads(4)
    pred = model.predict(data, pm['feature_names'], np.array([origin]))[0]
    result = {'source':'experimental-single-model-forecast-from-real-history', 'origin_utc':datetime.fromtimestamp(timestamp,timezone.utc).isoformat(),
              'target_names':SENSORS,'target_bin_end_unix_utc':(timestamp+np.arange(1,model.cfg['horizon_steps']+1)*model.cfg['step_seconds']).tolist(),
              'predictions_ppm':pred.tolist(),'model_sha256':record['outputs']['model.pt'], 'run_manifest_sha256':digest(run/'manifest.json'),
              'processed_sha256':pm['aligned_sha256'], 'limitation':'Retrospectively selected candidate; no changed-damper causal response or prospective performance claim.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(output)
    return output


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('processed');p.add_argument('output');a=p.parse_args()
    predict(a.run,a.processed,a.output)
