"""Extend only epoch-capped CO2 fits; reuse genuinely early-stopped fits exactly."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
import torch
from .ablation import feature_columns
from .benchmark import fit_neural, fold_origins, summarize
from .data import digest
from .features import build_inputs


def hit_epoch_cap(entry, limit, patience):
    history=entry['history']
    if not history or len(history)>limit:
        raise ValueError('Invalid reference training history')
    best=min(history,key=lambda h:h['inner_validation_mae_ppm'])['epoch']
    return len(history)==limit and len(history)-best<patience


def refine(root,config_path):
    root=Path(root); protocol=json.loads(Path(config_path).read_text()); base_dir=root/protocol['base_run']
    base=json.loads((base_dir/'manifest.json').read_text()); cfg=dict(base['model_config'])
    if base['status']!='complete' or protocol['patience']!=cfg['patience'] or protocol['maximum_epochs']<=cfg['epochs']:
        raise ValueError('Expected only an increased epoch cap with unchanged early stopping')
    if protocol['model']!='tcn' or protocol['features']!='co2_only': raise ValueError('Only the fixed CO2 TCN is eligible')
    old_limit=cfg['epochs']; cfg['epochs']=protocol['maximum_epochs']
    source=[r for r in base['results'] if r['fold'].startswith('rolling-') and r['features']=='co2_only' and r['model']=='tcn' and r['seed'] in protocol['seeds']]
    if len(source)!=3*len(protocol['seeds']): raise ValueError('Incomplete reference fits')
    processed=root/base['protocol']['processed']; pm=json.loads((processed/'manifest.json').read_text())
    if digest(processed/'aligned.npz')!=base['aligned_sha256'] or digest(processed/'manifest.json')!=base['processed_manifest_sha256']:
        raise ValueError('Processed source mismatch')
    loaded=dict(np.load(processed/'aligned.npz')); n=len(loaded['timestamps']); end=int(n*.7)
    data={k:v[:end] for k,v in loaded.items()};del loaded
    names=pm['feature_names'];torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-refinement-'+uuid4().hex[:8]
    output=root/'artifacts/forecasting'/run_id;output.mkdir(parents=True)
    record={'run_id':run_id,'status':'running','protocol':protocol,'model_config':cfg,
            'base_manifest_sha256':digest(base_dir/'manifest.json'),'aligned_sha256':base['aligned_sha256'],
            'code_sha256':{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'planned_refits':[{'fold':r['fold'],'seed':r['seed']} for r in source if hit_epoch_cap(r,old_limit,cfg['patience'])],
            'results':[],'ensembles':[],'reference_hashes':{},
            'limits':['Reused development windows, not a new untouched test.',
                      'Only epoch cap increased; checkpoint selection still inner MAE.',
                      'Repeated internal validation also has selection risk.',
                      'No old-test evaluation, labels removed or predictions clipped.']}
    def save(): (output/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    def checked(name):
        if digest(base_dir/name)!=base['outputs'][name]:raise ValueError('Reference mismatch: '+name)
        record['reference_hashes'][name]=base['outputs'][name]
        return base_dir/name
    save()  # Refit identities and all parameters are fixed before reading assessment metrics.
    for i,fractions in enumerate(base['protocol']['fold_fractions']):
        fold=f'rolling-{i+1}';bounds=[int(n*f) for f in fractions];origins=fold_origins(data,names,cfg,bounds)
        x,current,y,norm=build_inputs(data,names,bounds[0],cfg['input_encoding']);x=x[:,feature_columns(names,'co2_only')]
        labels=np.load(checked(fold+'-labels.npz'));truth=labels['actual_ppm'];mask=labels['observed_mask'];baseline=labels['persistence_ppm']
        np.testing.assert_array_equal(labels['origins'],data['timestamps'][origins['test']])
        np.testing.assert_array_equal(truth,data['targets'][origins['test'][:,None]+np.arange(1,cfg['horizon_steps']+1)])
        predictions=[]
        for seed in protocol['seeds']:
            r=next(r for r in source if r['fold']==fold and r['seed']==seed)
            stem=f'{fold}-co2_only-tcn-{seed}'; checked(stem+'.pt');old=np.load(checked(stem+'.npz'))['predictions_ppm']
            if hit_epoch_cap(r,old_limit,cfg['patience']):
                print(f'Start refit {fold} seed={seed}, cap={cfg["epochs"]}',flush=True);started=time.perf_counter()
                model,(pred,actual,observed,persisted),history=fit_neural('tcn',x,current,y,origins,cfg,norm,seed)
                if history[:len(r['history'])]!=r['history']:
                    raise ValueError('Training prefix changed; early-stopped checkpoint reuse would be invalid')
                np.testing.assert_allclose(actual[mask],truth[mask],atol=.002);np.testing.assert_array_equal(observed,mask)
                np.testing.assert_allclose(persisted,baseline,atol=.002)
                entry={'fold':fold,'seed':seed,'reused':False,'history':history,'original_prefix_exact':True,
                       'best_epoch':min(history,key=lambda h:h['inner_validation_mae_ppm'])['epoch'],
                       'seconds':time.perf_counter()-started,'metrics':summarize(pred,truth,mask,baseline),
                       'reference_mae_ppm':r['metrics']['mae_ppm'],'reference_inner_mae_ppm':min(h['inner_validation_mae_ppm'] for h in r['history'])}
                torch.save({'state_dict':model.state_dict(),'model':'tcn','config':cfg,'normalization':norm,
                            'feature_columns':feature_columns(names,'co2_only'),'input_features':x.shape[1],'seed':seed},output/(stem+'.pt'))
                print(f"Done {fold} {seed}: epochs={len(history)}, best={entry['best_epoch']}, inner={min(h['inner_validation_mae_ppm'] for h in history):.3f}, outer={entry['metrics']['mae_ppm']:.3f}",flush=True)
            else:
                pred=old
                entry={'fold':fold,'seed':seed,'reused':True,'history':r['history'],'best_epoch':r['best_epoch'],
                       'seconds':0.,'metrics':summarize(pred,truth,mask,baseline),'reference_mae_ppm':r['metrics']['mae_ppm']}
            predictions.append(pred);record['results'].append(entry)
            np.savez_compressed(output/(stem+'.npz'),predictions_ppm=pred);save()
        pred=np.mean(predictions,axis=0)
        record['ensembles'].append({'fold':fold,'model':'fixed_seed_mean','metrics':summarize(pred,truth,mask,baseline)})
        np.savez_compressed(output/(fold+'-seed_mean.npz'),predictions_ppm=pred);save()
    record['status']='complete';record['outputs']={p.name:digest(p) for p in output.iterdir() if p.name!='manifest.json'};save()
    report=root/'reports/forecasting'/f'{run_id}.md'
    lines=['# 仅延长触及轮数上限的TCN','',f'Run: `{run_id}`；固定24轮上限、patience=3，只有原12轮触顶且尚未早停的模型重新拟合。','',
           '7个已早停模型复用原权重/预测，2个模型按相同种子从头重训；新训练前12轮内部验证历史要求与原运行逐项精确一致，之后只按内部验证选择checkpoint。无旧测试评估。','',
           '| 折 | 种子 | 复用 | 原开发MAE | 新开发MAE | 最佳epoch | 训练epoch数 |','| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in record['results']:
        lines.append(f"| {r['fold']} | {r['seed']} | {r['reused']} | {r['reference_mae_ppm']:.3f} | {r['metrics']['mae_ppm']:.3f} | {r['best_epoch']} | {len(r['history'])} |")
    lines+=['',f"三折三种子单模型MAE均值：{np.mean([r['metrics']['mae_ppm'] for r in record['results']]):.3f} ppm。",'',
            f"固定三种子平均预测的三折MAE均值：{np.mean([r['metrics']['mae_ppm'] for r in record['ensembles']]):.3f} ppm。",'',
            '不挑选开发段表现最好的epoch或种子，结果可能在个别开发折变差。内部验证更低也不保证外部泛化。开发窗口已多次使用，仍需新增时间数据确认。', '',
            f'来源哈希、完整指标、历史及预测/新权重在 `artifacts/forecasting/{run_id}/`；已早停权重来源见base_run。']
    report.write_text('\n'.join(lines)+'\n');print('Report:',report,flush=True)
    return output,report
