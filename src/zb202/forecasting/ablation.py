"""Development-only feature additions and local independent TCN comparison."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
import torch
from .benchmark import fit_neural, fold_origins, summarize
from .data import SENSORS, digest, iso
from .features import build_inputs


def feature_columns(names, group):
    co2 = [names.index(s+'/co2_ppm') for s in SENSORS]
    conditions = {
        'environment': lambda n: n.endswith(('/temperature_c', '/relative_humidity_pct')),
        'occupancy': lambda n: n.endswith('/last_occupancy_event'),
        'door': lambda n: n.endswith('/last_magnet_event'),
        'power': lambda n: n.endswith('/active_power'),
        'co2_only': lambda n: False,
    }
    if group not in conditions:
        raise ValueError('Unknown feature group: '+group)
    cols = co2+[j for j, name in enumerate(names) if conditions[group](name)]
    n = len(names)
    return cols+[n+j for j in cols]+[2*n+j for j in cols]+list(range(3*n, 3*n+4))


def ablate(root, config_path):
    root = Path(root)
    protocol = json.loads(Path(config_path).read_text())
    base_dir = root/protocol['base_run']
    base = json.loads((base_dir/'manifest.json').read_text())
    if base['status'] != 'complete':
        raise ValueError('Reference run incomplete')
    cfg = dict(base['model_config'], epochs=protocol['maximum_epochs'], patience=protocol['patience'])
    if cfg != base['model_config']:
        raise ValueError('Screening must keep reference training budget/config unchanged')
    processed = root/base['protocol']['processed']
    pm = json.loads((processed/'manifest.json').read_text())
    if digest(processed/'aligned.npz') != base['aligned_sha256'] or digest(processed/'manifest.json') != base['processed_manifest_sha256']:
        raise ValueError('Reference processed source mismatch')
    if digest(root/cfg['snapshot']/'zb202_iot.csv.gz') != base['raw_sha256']:
        raise ValueError('Raw source mismatch')
    loaded = dict(np.load(processed/'aligned.npz'))
    n = len(loaded['timestamps'])
    # Discard old validation/test values and labels before fitting or assessment.
    development_end = int(n*.70)
    data = {k:v[:development_end] for k,v in loaded.items()}
    del loaded
    names = pm['feature_names']
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-ablation-'+uuid4().hex[:8]
    output = root/'artifacts/forecasting'/run_id
    output.mkdir(parents=True)
    record = {'run_id':run_id, 'status':'running', 'protocol':protocol, 'model_config':cfg,
              'base_manifest_sha256':digest(base_dir/'manifest.json'), 'processed':str(processed.relative_to(root)),
              'raw_sha256':base['raw_sha256'], 'aligned_sha256':base['aligned_sha256'],
              'code_sha256':{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'versions':{'numpy':np.__version__, 'torch':torch.__version__},
              'development_end_utc':iso(data['timestamps'][-1]),
              'results':[], 'folds':[], 'selection':None,
              'limits':['No new untouched test; existing development windows are reused.',
                        'Screening one seed is provisional; only inner validation ranks candidates.',
                        'Independent models change both cross-sensor information and parameter sharing.',
                        'Independent per-network width16 vs joint32; total size reported, not exactly matched.',
                        'No old-test inference, labels clipped, or predictions clipped.']}
    def save():
        (output/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    save()
    cache = {}
    for i, fractions in enumerate(base['protocol']['fold_fractions']):
        fold=f'rolling-{i+1}'; bounds=[int(n*f) for f in fractions]
        if bounds[-1]>development_end:
            raise ValueError('Fold reaches outside development period')
        origins=fold_origins(data,names,cfg,bounds)
        x,current,y,norm=build_inputs(data,names,bounds[0],cfg['input_encoding'])
        ix=origins['test']; future=ix[:,None]+np.arange(1,cfg['horizon_steps']+1)
        truth=data['targets'][future]; mask=np.isfinite(truth)
        mean,scale=np.array(norm['target_mean']),np.array(norm['target_scale'])
        persistence=np.repeat(current[ix,None,:],cfg['horizon_steps'],axis=1)*scale+mean
        ref_labels=np.load(base_dir/f'{fold}-labels.npz')
        np.testing.assert_array_equal(data['timestamps'][ix],ref_labels['origins'])
        np.testing.assert_array_equal(truth,ref_labels['actual_ppm'])
        np.testing.assert_array_equal(mask,ref_labels['observed_mask'])
        np.testing.assert_allclose(persistence,ref_labels['persistence_ppm'],atol=.002)
        record['folds'].append({'name':fold,'boundaries':bounds,'normalization':norm,
                               'origins':{k:len(v) for k,v in origins.items()}})
        np.savez_compressed(output/f'{fold}-labels.npz',origins=data['timestamps'][ix],
                            target_bin_end_utc=data['timestamps'][future],actual_ppm=truth,
                            observed_mask=mask,persistence_ppm=persistence)
        cache[fold]=(x,current,y,norm,origins,truth,mask,persistence)
    # Freeze all reference checksums and copy reference entries before any new fitting.
    for r in base['results']:
        if r['fold'].startswith('rolling-') and r['model'] in ('tcn','persistence','ridge') and r['features'] in ('co2_only','shared','all'):
            stem=f"{r['fold']}-{r['features']}-{r['model']}"+(f"-{r['seed']}" if r['seed'] is not None else '')
            files=[] if r['model']=='persistence' else [stem+'.npz']+([stem+'.pt'] if r['model']=='tcn' else [])
            for name in files:
                if digest(base_dir/name)!=base['outputs'][name]:
                    raise ValueError('Reference artifact mismatch: '+name)
            record['results'].append(dict(r,reused=True,reference_files={name:base['outputs'][name] for name in files}))
    save()
    def fit(fold, group, kind, seed, stage):
        x,current,y,norm,origins,truth,mask,persistence=cache[fold]
        columns=feature_columns(names,group); inputs=x[:,columns]
        config=dict(cfg)
        if kind=='independent_tcn': config['channels']=protocol['independent_channels']
        print(f'Start {stage} {fold} {group} {kind} seed={seed}',flush=True)
        started=time.perf_counter()
        model,(pred,actual,observed,baseline),history=fit_neural(kind,inputs,current,y,origins,config,norm,seed)
        np.testing.assert_allclose(actual[mask],truth[mask],atol=.002)
        np.testing.assert_array_equal(observed,mask)
        np.testing.assert_allclose(baseline,persistence,atol=.002)
        entry={'fold':fold,'features':group,'model':kind,'seed':seed,'stage':stage,'reused':False,
               'seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters()),
               'input_features':len(columns),'best_epoch':min(history,key=lambda v:v['inner_validation_mae_ppm'])['epoch'],
               'history':history,'metrics':summarize(pred,truth,mask,persistence)}
        record['results'].append(entry)
        stem=f'{fold}-{group}-{kind}-{seed}'
        np.savez_compressed(output/f'{stem}.npz',predictions_ppm=pred)
        torch.save({'state_dict':model.state_dict(),'model':kind,'config':config,'normalization':norm,
                    'feature_columns':columns,'feature_names':names,'seed':seed,
                    'target_names':SENSORS,'input_features':len(columns)},output/f'{stem}.pt')
        print(f"Done {stem}: inner={min(h['inner_validation_mae_ppm'] for h in history):.3f}, outer={entry['metrics']['mae_ppm']:.3f}, seconds={entry['seconds']:.1f}",flush=True)
        save()
    folds=[f['name'] for f in record['folds']]
    for group in protocol['groups']:
        for fold in folds: fit(fold,group,'tcn',protocol['screen_seed'],'screen')
    for fold in folds: fit(fold,'co2_only','independent_tcn',protocol['screen_seed'],'screen')
    def inner(kind, group):
        selected=[r for r in record['results'] if r['model']==kind and r['features']==group and r['seed']==protocol['screen_seed']]
        if len(selected)!=len(folds): raise ValueError('Incomplete screening cells')
        return float(np.mean([min(h['inner_validation_mae_ppm'] for h in r['history']) for r in selected]))
    ranking={group:inner('tcn',group) for group in ['co2_only']+protocol['groups']}
    winner=min(ranking,key=ranking.get)
    local_score=inner('independent_tcn','co2_only')
    record['selection']={'ranking_inner_mae':ranking,'feature_winner':winner,
                         'independent_inner_mae':local_score,
                         'confirm_independent':local_score<ranking['co2_only'],
                         'rule':'Mean of best inner MAE across folds; never outer development MAE. Strict comparison; ties retain co2_only.'}
    save()
    if winner!='co2_only':
        for seed in protocol['confirmation_seeds']:
            for fold in folds: fit(fold,winner,'tcn',seed,'confirm')
    if record['selection']['confirm_independent']:
        for seed in protocol['confirmation_seeds']:
            for fold in folds: fit(fold,'co2_only','independent_tcn',seed,'confirm')
    record['status']='complete'
    record['outputs']={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    save()
    # Reports are curated manually; machine-readable results stay with the run.
    manifest_path = output / 'manifest.json'
    print('Results:', manifest_path, flush=True)
    return output, manifest_path
