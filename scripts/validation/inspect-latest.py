"""Independently check a saved single-model holdout and day-block uncertainty."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from zb202.forecasting.data import digest


def inspect(run):
    run=Path(run);r=json.loads((run/'manifest.json').read_text())
    if r['status']!='complete': raise ValueError('Incomplete run')
    for name,expected in r['outputs'].items():
        if digest(run/name)!=expected:raise ValueError('Changed output: '+name)
    z=np.load(run/'predictions.npz');mask=z['observed_mask'];truth=z['actual_ppm'];t=z['origins'];target=z['target_bin_end_utc']
    if mask.shape!=truth.shape or target.shape!=truth.shape[:2] or np.any(np.diff(t)<=0) or np.any(target<=t[:,None]):
        raise ValueError('Prediction shape or chronology invalid')
    if not np.array_equal(mask,np.isfinite(truth)):
        raise ValueError('Observed mask differs from truth')
    metrics={}
    for name in ('candidate','frozen','persistence'):
        e=(z[name+'_ppm']-truth)[mask]
        mae=float(np.abs(e).mean());rmse=float(np.sqrt((e*e).mean()))
        np.testing.assert_allclose([mae,rmse],[r['metrics'][name]['mae_ppm'],r['metrics'][name]['rmse_ppm']],atol=1e-9)
        metrics[name]={'mae_ppm':mae,'rmse_ppm':rmse}
    dates=np.array([datetime.fromtimestamp(float(x),timezone(timedelta(hours=8))).date().isoformat() for x in t]);days=np.unique(dates)
    sums=[]
    for day in days:
        i=dates==day;m=mask[i];y=truth[i]
        sums.append([np.abs(z['candidate_ppm'][i]-y)[m].sum(),np.abs(z['frozen_ppm'][i]-y)[m].sum(),m.sum()])
    sums=np.array(sums);rng=np.random.default_rng(42);draw=rng.integers(0,len(days),(10000,len(days)));b=sums[draw].sum(axis=1);diff=(b[:,1]-b[:,0])/b[:,2]
    result={'status':'passed','run_manifest_sha256':digest(run/'manifest.json'),'outputs_checked':sorted(r['outputs']),
            'origins':len(t),'observed_labels':int(mask.sum()),'recomputed':metrics,
            'day_block_bootstrap':{'resamples':10000,'seed':42,'blocks':len(days),'candidate_mae_improvement_ppm_95_percentile_interval':np.quantile(diff,[.025,.975]).tolist(),
                                   'limitation':'Descriptive resampling of few observed days, not a guarantee of future performance; includes partial days and correlated windows.'}}
    path=run/'validated-summary.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();inspect(a.run)
