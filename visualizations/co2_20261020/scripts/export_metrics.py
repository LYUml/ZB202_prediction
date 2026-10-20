"""从已冻结NPZ重算指标，保存CSV与来源记录。"""
from _figbase import *
import csv,hashlib
out=BASE/'data';out.mkdir(exist_ok=True)
rows=[]
for name,v in [('TCN',e),('persistence',be)]:
 for scope,a in [('overall',v)]+[(s,v[:,:,j]) for j,s in enumerate(sensors)]+[(f'horizon_{(h+1)*5}min',v[:,h,:]) for h in range(12)]:
  a=a[np.isfinite(a)];rows.append(dict(model=name,scope=scope,n_observed_label_items=len(a),mae_ppm=float(np.mean(abs(a))),rmse_ppm=float(np.sqrt(np.mean(a*a))),bias_ppm=float(a.mean()),within_10ppm_fraction=float(np.mean(abs(a)<=10))))
with (out/'metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (out/'horizon_sensor_mae.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['sensor','horizon_minutes','n_observed_label_items','tcn_mae_ppm','persistence_mae_ppm'])
 for j,s in enumerate(sensors):
  for h in range(12):w.writerow([s,(h+1)*5,int(m[:,h,j].sum()),np.nanmean(abs(e[:,h,j])),np.nanmean(abs(be[:,h,j]))])
# One target time per row. These are real five-minute target-bin means, never filled labels.
with (out/'real_co2_confirmation.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['target_time_utc8']+sensors)
 for i,t in enumerate(dates(z['target_bin_end_utc'][:,0])):w.writerow([t.isoformat()]+[float(y[i,0,j]) if m[i,0,j] else '' for j in range(5)])
provenance=dict(source=str(RUN.relative_to(ROOT)),predictions_sha256=hashlib.sha256((RUN/'predictions.npz').read_bytes()).hexdigest(),origin_range_utc8=[d.isoformat() for d in dates(z['origins'][[0,-1]])],observed_label_items=int(m.sum()),overlapping_origins=len(e),metrics_reproduced=True,model_mae_ppm=float(np.nanmean(abs(e))),baseline_mae_ppm=float(np.nanmean(abs(be))),mae_improvement_fraction=float(1-np.nanmean(abs(e))/np.nanmean(abs(be))),timezone='UTC+8',scope='Frozen later-time confirmation; repeated labels across horizons are not independent samples; no model fitting or synthetic data',python=sys.executable)
(out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
print('指标重算、数据哈希与来源记录已保存。')
