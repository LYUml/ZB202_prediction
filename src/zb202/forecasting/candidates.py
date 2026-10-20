"""Fixed development-only model expansion with saved source snapshots and checkpoints."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
import torch
from .ablation import feature_columns
from .benchmark import fit_neural,fold_origins,summarize
from .data import digest
from .features import build_inputs


def compare_candidates(root,config_path):
    root=Path(root);protocol=json.loads(Path(config_path).read_text());base_dir=root/protocol['base_run']
    base=json.loads((base_dir/'manifest.json').read_text())
    if base['status']!='complete' or protocol['features']!='co2_only': raise ValueError('Expected completed CO2 reference')
    cfg=dict(base['model_config'],epochs=protocol['epochs'],patience=protocol['patience'])
    processed=root/base['protocol']['processed'];pm=json.loads((processed/'manifest.json').read_text())
    if digest(processed/'aligned.npz')!=base['aligned_sha256'] or digest(processed/'manifest.json')!=base['processed_manifest_sha256']:
        raise ValueError('Processed source mismatch')
    if digest(root/cfg['snapshot']/'zb202_iot.csv.gz')!=base['raw_sha256']:raise ValueError('Raw source mismatch')
    loaded=dict(np.load(processed/'aligned.npz'));n=len(loaded['timestamps']);end=int(n*.7)
    data={k:v[:end] for k,v in loaded.items()};del loaded
    names=pm['feature_names'];torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-candidates-'+uuid4().hex[:8]
    output=root/'artifacts/forecasting'/run_id;output.mkdir(parents=True)
    record={'run_id':run_id,'status':'running','protocol':protocol,'model_config':cfg,
            'base_manifest_sha256':digest(base_dir/'manifest.json'),'raw_sha256':base['raw_sha256'],
            'aligned_sha256':base['aligned_sha256'],'results':[],'ensembles':[],'folds':[],
            'code_sha256':{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'limits':['Repeated development windows, not an untouched test.',
                      'Different input representations and capacities; report rather than conceal them.',
                      'Epochs selected on internal validation only; all seeds reported.',
                      'No original test evaluation or clipping of targets/predictions.']}
    def save(): (output/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    # Keep the exact executable source even when future goal turns change the workspace.
    for p in Path(__file__).parent.glob('*.py'):
        out=output/'source_snapshot/zb202/forecasting'/p.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
    save()
    cache={}
    for i,fracs in enumerate(base['protocol']['fold_fractions']):
        fold=f'rolling-{i+1}';bounds=[int(n*f) for f in fracs];origins=fold_origins(data,names,cfg,bounds)
        x,current,y,norm=build_inputs(data,names,bounds[0],cfg['input_encoding']);cols=feature_columns(names,'co2_only');x=x[:,cols]
        labels=np.load(base_dir/(fold+'-labels.npz'))
        if digest(base_dir/(fold+'-labels.npz'))!=base['outputs'][fold+'-labels.npz']:raise ValueError('Reference labels mismatch')
        ix=origins['test'];truth=data['targets'][ix[:,None]+np.arange(1,cfg['horizon_steps']+1)];mask=np.isfinite(truth)
        np.testing.assert_array_equal(truth,labels['actual_ppm']);np.testing.assert_array_equal(mask,labels['observed_mask'])
        np.testing.assert_array_equal(data['timestamps'][ix],labels['origins'])
        record['folds'].append({'name':fold,'boundaries':bounds,'origins':{k:len(v) for k,v in origins.items()},'normalization':norm})
        np.savez_compressed(output/(fold+'-labels.npz'),origins=labels['origins'],actual_ppm=truth,observed_mask=mask,persistence_ppm=labels['persistence_ppm'],target_bin_end_utc=labels['target_bin_end_utc'])
        cache[fold]=(x,current,y,norm,origins,truth,mask,labels['persistence_ppm'])
    save()
    for spec in protocol['models']:
        kind=spec['kind'];modelcfg=dict(cfg,**spec['overrides'])
        for fold,(x,current,y,norm,origins,truth,mask,baseline) in cache.items():
            predictions=[]
            for seed in protocol['seeds']:
                print(f'Start {kind} {fold} seed={seed}',flush=True);start=time.perf_counter()
                model,(pred,actual,observed,persisted),history=fit_neural(kind,x,current,y,origins,modelcfg,norm,seed)
                np.testing.assert_allclose(actual[mask],truth[mask],atol=.002);np.testing.assert_array_equal(observed,mask)
                np.testing.assert_allclose(persisted,baseline,atol=.002)
                entry={'fold':fold,'model':kind,'seed':seed,'config':modelcfg,'seconds':time.perf_counter()-start,
                       'parameters':sum(p.numel() for p in model.parameters()),'history':history,
                       'best_epoch':min(history,key=lambda h:h['inner_validation_mae_ppm'])['epoch'],
                       'metrics':summarize(pred,truth,mask,baseline)}
                record['results'].append(entry);predictions.append(pred)
                stem=f'{fold}-{kind}-{seed}'
                torch.save({'state_dict':model.state_dict(),'model':kind,'config':modelcfg,'normalization':norm,
                            'feature_columns':feature_columns(names,'co2_only'),'feature_names':names,
                            'input_features':x.shape[1],'seed':seed},output/(stem+'.pt'))
                np.savez_compressed(output/(stem+'.npz'),predictions_ppm=pred)
                print(f"Done {stem}: MAE={entry['metrics']['mae_ppm']:.3f}, best={entry['best_epoch']}, seconds={entry['seconds']:.1f}",flush=True);save()
            avg=np.mean(predictions,axis=0);record['ensembles'].append({'fold':fold,'model':kind,'metrics':summarize(avg,truth,mask,baseline)})
            np.savez_compressed(output/f'{fold}-{kind}-seed_mean.npz',predictions_ppm=avg);save()
    record['status']='complete';record['outputs']={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'};save()
    # Reports are curated manually; machine-readable results stay with the run.
    manifest_path = output / 'manifest.json'
    print('Results:', manifest_path, flush=True)
    return output, manifest_path
