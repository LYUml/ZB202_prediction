"""冻结确认期数据；图由真实NPZ生成，尺寸见PAPER_PLAN。"""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone,timedelta
import numpy as np
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from _utils.plot_utils import setup_style,save_fig,PALETTE,COLORS,_lighten
setup_style()
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
RUN=ROOT/'artifacts/forecasting/20261020T092656Z-confirmation-25df244e'; meta=json.loads((RUN/'manifest.json').read_text()); f=RUN/'predictions.npz'
assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['outputs']['predictions.npz']
z=np.load(f); y=z['actual_ppm']; p=z['predictions_ppm']; b=z['persistence_ppm']; m=z['observed_mask'].astype(bool)&np.isfinite(y)
e=np.where(m,p-y,np.nan); be=np.where(m,b-y,np.nan); sensors=list(meta['metrics']['by_sensor']); tz=timezone(timedelta(hours=8))
assert abs(np.nanmean(abs(e))-meta['metrics']['mae_ppm'])<1e-9
assert abs(np.nanmean(abs(be))-meta['persistence_metrics']['mae_ppm'])<1e-9
assert np.isfinite(p[m]).all() and np.isfinite(b[m]).all()
def dates(t):return [datetime.fromtimestamp(float(v),tz) for v in t]
def axis(ax,t):
 ax.set_xlim(t[0],t[-1]); ax.xaxis.set_major_locator(md.AutoDateLocator(minticks=4,maxticks=6,tz=tz));ax.xaxis.set_major_formatter(md.DateFormatter('%m-%d\n%H:%M',tz=tz));ax.set_xlabel('实测目标时间（UTC+8）')
def legend(ax,n=2):ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=n,frameon=False)
def draw(name):
 fig,ax=plt.subplots(figsize=(6,3.8))
 if name=='co2_trend':
  ts=z['target_bin_end_utc'][:,0];full=np.arange(ts.min(),ts.max()+300,300);v=np.full((len(full),5),np.nan);v[((ts-full[0])/300).astype(int)]=np.where(m[:,0],y[:,0],np.nan);t=dates(full)
  for j,s in enumerate(sensors):ax.plot(t,v[:,j],color=PALETTE[j],lw=1.1,ls=['-','--','-.',':','-'][j],label=s)
  axis(ax,t);ax.set_ylabel('真实 CO₂ 浓度（ppm）');legend(ax,3)
 elif name=='prediction_overlay':
  # Fixed clock window, chosen for readability rather than model performance.
  h=5;j=0;ts=z['target_bin_end_utc'][:,h];local=dates(ts)
  ds=np.array([v.strftime('%Y-%m-%d') for v in local]);days=sorted(set(ds))[1:-1];day=days[len(days)//2]
  ix=(ds==day)&np.array([12<=v.hour<15 for v in local]);t=dates(ts[ix])
  # Reindex missing origins so neither concentration nor error lines bridge gaps.
  full=np.arange(ts[ix].min(),ts[ix].max()+300,300);t=dates(full);at=((ts[ix]-full[0])/300).astype(int)
  vals=[]
  for v in [y,p,b]:
   a=np.full(len(full),np.nan);a[at]=np.where(m[ix,h,j],v[ix,h,j],np.nan);vals.append(a)
  plt.close(fig)
  fig,(ax,axerr)=plt.subplots(2,1,figsize=(6,4.7),sharex=True,gridspec_kw={'height_ratios':[2,1],'hspace':.14})
  for v,label,c,ls,mk in [(vals[0],'实测',COLORS['text'],'-','o'),(vals[1],'TCN 提前30分钟预测',PALETTE[0],'-','s'),(vals[2],'持续值预测',PALETTE[1],'--','^')]:
   ax.plot(t,v,label=label,color=c,ls=ls,lw=1.4,marker=mk,ms=3,markeredgewidth=.4,markeredgecolor='white')
  ax.set_ylabel('CO₂ 浓度（ppm）');legend(ax,2)
  for v,label,c,ls,mk in [(vals[1]-vals[0],'TCN',PALETTE[0],'-','s'),(vals[2]-vals[0],'持续值',PALETTE[1],'--','^')]:
   axerr.plot(t,v,color=c,ls=ls,lw=1.2,marker=mk,ms=2.7)
  axerr.axhline(0,color=COLORS['ref_line'],ls=':',lw=1)
  lim=np.nanmax(np.abs(np.array([vals[1]-vals[0],vals[2]-vals[0]])))*1.12
  axerr.set_ylim(-lim,lim);axerr.set_ylabel('预测 − 实测\n（ppm）')
  axerr.set_xlim(t[0],t[-1]);axerr.xaxis.set_major_locator(md.MinuteLocator(byminute=[0,30],tz=tz));axerr.xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=tz))
  axerr.set_xlabel(f'{day} 目标时间（UTC+8） · {sensors[j]}')
  (BASE/'example_selection.json').write_text(json.dumps(dict(day_utc8=day,sensor=sensors[j],horizon_minutes=30,clock_window_utc8='12:00–15:00 (end excluded)',rule='middle complete calendar day; first schema sensor; fixed noon window, no error-based selection; all five-minute observations retained'),ensure_ascii=False,indent=2))
 elif name=='horizon_error':
  x=np.arange(1,13)*5
  for v,label,c,mk in [(e,'TCN 三模型平均',PALETTE[0],'o'),(be,'持续值预测',PALETTE[1],'s')]:ax.plot(x,np.nanmean(abs(v),axis=(0,2)),color=c,marker=mk,lw=1.6,ms=4,markeredgecolor='white',label=label)
  ax.set_xticks([5,10,20,30,40,50,60]);ax.set_xlabel('提前预测多少分钟');ax.set_ylabel('平均绝对误差 MAE（ppm，越低越好）');ax.set_ylim(bottom=0);legend(ax)
 elif name=='sensor_error':
  x=np.arange(5)
  for i,(v,label) in enumerate([(e,'TCN 三模型平均'),(be,'持续值预测')]):
   bars=ax.bar(x+(i-.5)*.36,np.nanmean(abs(v),axis=(0,1)),.36,color=_lighten(PALETTE[i],.4),edgecolor=PALETTE[i],lw=1.2,label=label);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
  ax.set_xticks(x,sensors);ax.set_ylim(0,np.nanmax(np.nanmean(abs(be),axis=(0,1)))*1.2);ax.set_ylabel('平均绝对误差 MAE（ppm）');legend(ax)
 elif name=='error_heatmap':
  v=np.nanmean(abs(e),axis=0).T;cmap=LinearSegmentedColormap.from_list('theme',[_lighten(PALETTE[0],.94),PALETTE[0]]);im=ax.imshow(v,cmap=cmap,aspect='auto',vmin=0,vmax=v.max());ax.set_xticks(np.arange(12),np.arange(1,13)*5);ax.set_yticks(np.arange(5),sensors);ax.set_xlabel('提前预测多少分钟');ax.set_ylabel('传感器点位')
  for j in range(5):
   for h in range(12):ax.text(h,j,f'{v[j,h]:.1f}',ha='center',va='center',fontsize=7,color='white' if v[j,h]>.6*v.max() else COLORS['text'])
  fig.colorbar(im,ax=ax,pad=.03,label='TCN MAE（ppm）')
 elif name=='residual_distribution':
  bins=np.linspace(min(np.nanmin(e),np.nanmin(be)),max(np.nanmax(e),np.nanmax(be)),55)
  for i,(v,label) in enumerate([(e,'TCN 三模型平均'),(be,'持续值预测')]):ax.hist(v[np.isfinite(v)],bins=bins,density=True,histtype='stepfilled',alpha=.3,color=_lighten(PALETTE[i],.4),edgecolor=PALETTE[i],lw=1.1,label=label)
  ax.axvline(0,color=COLORS['ref_line'],ls='--',lw=1);ax.set_xlabel('预测 − 实测（ppm；左边偏低，右边偏高）');ax.set_ylabel('概率密度（1/ppm）');legend(ax)
 elif name=='prediction_scatter':
  plt.close(fig);fig,ax=plt.subplots(figsize=(5,4.9));h=11
  for j,s in enumerate(sensors):
   q=m[:,h,j];ax.scatter(y[q,h,j],p[q,h,j],s=7,color=PALETTE[j],alpha=.35,label=s,rasterized=True)
  lo=min(np.nanmin(y[:,h]),p[:,h].min());hi=max(np.nanmax(y[:,h]),p[:,h].max());pad=(hi-lo)*.04;ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color=COLORS['ref_line'],ls='--',lw=1,label='完全准确参考线');ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='真实浓度（ppm）',ylabel='提前60分钟预测浓度（ppm）');ax.set_aspect('equal');legend(ax)
 else:raise ValueError(name)
 fig.tight_layout(pad=.9)
 save_fig(fig,str(BASE/'figures'/f'fig_{name}.pdf'));fig.savefig(BASE/'figures'/f'fig_{name}.png',dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
