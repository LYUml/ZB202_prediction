"""Frozen seed averaging and low-dimensional inner-validation forecast combination."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch
from torch.utils.data import DataLoader
from .ablation import feature_columns
from .baselines import ridge_design
from .benchmark import fold_origins, summarize
from .data import digest
from .evaluation import predict, scores
from .features import build_inputs
from .tcn import TCN
from .windows import Windows


def choose_weight(tcn, ridge, truth, mask, candidates):
    if not np.asarray(mask).any():
        raise ValueError('No observed inner-validation labels')
    if not candidates or any(not 0 <= a <= 1 for a in candidates):
        raise ValueError('Combination weights must be within [0, 1]')
    if not np.isfinite(tcn).all() or not np.isfinite(ridge).all():
        raise ValueError('Nonfinite component prediction')
    losses=[scores(a*tcn+(1-a)*ridge,truth,mask)['mae_ppm'] for a in candidates]
    index=int(np.argmin(losses))
    return float(candidates[index]),[{'tcn_weight':float(a),'mae_ppm':float(loss)} for a,loss in zip(candidates,losses)]


def combine(root, config_path):
    root=Path(root); protocol=json.loads(Path(config_path).read_text()); base_dir=root/protocol['base_run']
    base=json.loads((base_dir/'manifest.json').read_text()); cfg=base['model_config']
    if base['status']!='complete' or protocol['features']!='co2_only':
        raise ValueError('Expected a completed CO2-only reference experiment')
    processed=root/base['protocol']['processed']; pm=json.loads((processed/'manifest.json').read_text())
    if digest(processed/'aligned.npz')!=base['aligned_sha256'] or digest(processed/'manifest.json')!=base['processed_manifest_sha256']:
        raise ValueError('Processed source mismatch')
    loaded=dict(np.load(processed/'aligned.npz')); n=len(loaded['timestamps']); end=int(n*.7)
    data={k:v[:end] for k,v in loaded.items()}; del loaded
    names=pm['feature_names']; torch.set_num_threads(4)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-ensemble-'+uuid4().hex[:8]
    output=root/'artifacts/forecasting'/run_id; output.mkdir(parents=True)
    record={'run_id':run_id,'status':'running','protocol':protocol,'base_manifest_sha256':digest(base_dir/'manifest.json'),
            'aligned_sha256':base['aligned_sha256'],'code_sha256':{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'results':[], 'calibration':[], 'reference_hashes':{},
            'limits':['Development data reused; no new untouched test.',
                      'Same inner window selects base-model epochs and combination weights; possible validation overfitting.',
                      'Fixed seed average is one forecast, not mean of individual MAE.',
                      'One global scalar weight per fold; no per-sensor or horizon fitting.',
                      'No old-test evaluation or raw/forecast clipping.']}
    def save(): (output/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    def checked(name):
        if digest(base_dir/name)!=base['outputs'][name]: raise ValueError('Reference artifact mismatch: '+name)
        record['reference_hashes'][name]=base['outputs'][name]
        return base_dir/name
    save()
    for i,fractions in enumerate(base['protocol']['fold_fractions']):
        fold=f'rolling-{i+1}'; bounds=[int(n*f) for f in fractions]
        origins=fold_origins(data,names,cfg,bounds)
        x,current,y,norm=build_inputs(data,names,bounds[0],cfg['input_encoding']); x=x[:,feature_columns(names,'co2_only')]
        mean,scale=np.array(norm['target_mean']),np.array(norm['target_scale'])
        loaders={k:DataLoader(Windows(x,current,y,idx,cfg),batch_size=cfg['batch_size']) for k,idx in origins.items() if k!='train'}
        validation_preds=[]; outer_preds=[]
        actual=mask=None
        for seed in protocol['seeds']:
            stem=f'{fold}-co2_only-tcn-{seed}'
            ckpt=torch.load(checked(stem+'.pt'),map_location='cpu',weights_only=False)
            model=TCN(x.shape[1],5,cfg); model.load_state_dict(ckpt['state_dict'])
            vp,actual,mask,_=predict(model,loaders['validation'],mean,scale)
            validation_preds.append(vp)
            saved=np.load(checked(stem+'.npz'))['predictions_ppm']
            # Replaying the frozen outer prediction also validates feature order and saved weights.
            replay,*_=predict(model,loaders['test'],mean,scale)
            np.testing.assert_allclose(replay,saved,rtol=0,atol=0)
            outer_preds.append(saved)
        vp=np.mean(validation_preds,axis=0); outer=np.mean(outer_preds,axis=0)
        ridge=np.load(checked(f'{fold}-co2_only-ridge.npz'))
        idx=origins['validation']; design=ridge_design(x,idx,cfg['lookback_steps'])
        rp=(current[idx,None,:]+(design@ridge['coef']+ridge['intercept']).reshape(-1,cfg['horizon_steps'],5))*scale+mean
        weight,losses=choose_weight(vp,rp,actual,mask,protocol['tcn_weights'])
        record['calibration'].append({'fold':fold,'tcn_weight':weight,'inner_losses':losses,
                                      'inner_origin_count':len(idx),'best_weight_selected_before_outer_metrics':True})
        labels=np.load(checked(fold+'-labels.npz')); truth=labels['actual_ppm']; observed=labels['observed_mask']; baseline=labels['persistence_ppm']
        np.testing.assert_array_equal(data['timestamps'][origins['test']],labels['origins'])
        np.testing.assert_array_equal(data['targets'][origins['test'][:,None]+np.arange(1,cfg['horizon_steps']+1)],truth)
        for name,pred in [('seed_mean',outer),('half_ridge',.5*outer+.5*ridge['predictions_ppm']),
                          ('inner_weighted',weight*outer+(1-weight)*ridge['predictions_ppm'])]:
            record['results'].append({'fold':fold,'model':name,'metrics':summarize(pred,truth,observed,baseline)})
            np.savez_compressed(output/f'{fold}-{name}.npz',predictions_ppm=pred)
        save()
        print(f"{fold}: chosen TCN weight={weight}; seed mean MAE={record['results'][-3]['metrics']['mae_ppm']:.3f}; inner-weighted MAE={record['results'][-1]['metrics']['mae_ppm']:.3f}",flush=True)
    record['status']='complete';record['outputs']={p.name:digest(p) for p in output.iterdir() if p.name!='manifest.json'};save()
    # Reports are curated manually; machine-readable results stay with the run.
    manifest_path = output / 'manifest.json'
    print('Results:', manifest_path, flush=True)
    return output, manifest_path
