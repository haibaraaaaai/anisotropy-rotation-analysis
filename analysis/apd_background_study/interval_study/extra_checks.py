from pathlib import Path
import numpy as np,json
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent
# Load function definitions without re-running the study.
s=(R/'study.py').read_text();ns={'__file__':str(R/'study.py')};exec(s[:s.index('baseguide=')],ns)
xy,geom,guide,anchors,bin_cycles=[ns[x] for x in ['xy','geom','guide','anchors','bin_cycles']]
z=np.load(R/'averaged_channels.npz');c=z['channels_per_cycle'];mean=xy(c.mean(axis=0));n=len(c)
groups=np.array_split(np.arange(n),8)
inter=[np.arange(j,n,8) for j in range(8)]
res=dict(chronological_geometric_rms=[geom(xy(c[g].mean(axis=0)),mean) for g in groups],
 interleaved_geometric_rms=[geom(xy(c[g].mean(axis=0)),mean) for g in inter])
# A geometric check of whether the outer loop was accidentally skipped/doubled.
raw=ns['raw'];p=guide(raw,41);b=z['interval_bounds'];wind=[]
for a,e in zip(b[:-1],b[1:]):
 q=p[a:e]-[-.3,0];q=np.vstack([q,q[0]])
 t=np.unwrap(np.arctan2(q[:,1],q[:,0]));wind.append(int(np.rint((t[-1]-t[0])/(2*np.pi))))
res['outer_loop_winding_counts']={str(w):int(sum(np.array(wind)==w)) for w in set(wind)}
# Same number of cycles per mean, but chronological vs interleaved groups.
# This distinguishes stable ensemble mean from stable path in each time block.
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(11,5),layout='constrained')
for ax,gs,title in [(axs[0],groups,'Eight chronological groups'),(axs[1],inter,'Eight interleaved groups')]:
 for g in gs:
  q=xy(c[g].mean(axis=0));ax.plot(q[:,0],q[:,1],lw=1)
 ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel='X',ylabel='Y',title=title)
fig.suptitle('38 cycles per curve in both panels; no time warping')
fig.savefig(R/'chronological_vs_interleaved.png',dpi=160);plt.close(fig)
# Explore additional separated windows, keeping this same gate and fixed gains.
from nptdms import TdmsFile
res['separated_windows']=[]
fig,axs=plt.subplots(1,5,figsize=(18,4),layout='constrained')
import sys
sys.path.insert(0,str(R.parent))
from study_paths import tdms_path
with TdmsFile.open(tdms_path()) as f:
 ch=f.groups()[0].channels()
 for ax,start in zip(axs,[0,10,20,30,39]):
  a=round(start/4e-6);e=a+250000
  rr=np.array([t[a:e] for t in ch],float)[[3,0,1,2]];gg=guide(rr,41);bb=anchors(gg)
  if len(bb)>1:
   cc,_,_=bin_cycles(rr,gg,bb);mm=xy(cc.mean(axis=0))
   ax.plot(mm[:,0],mm[:,1],lw=1.5)
   parts=np.array_split(np.arange(len(cc)),4)
   for ids in parts:
    pp=xy(cc[ids].mean(axis=0));ax.plot(pp[:,0],pp[:,1],lw=.6,alpha=.45)
   res['separated_windows'].append(dict(start_s=start,n_cycles=len(cc),
     geometric_rms_to_30s=geom(mm,mean),quarter_mean_geometric_rms=[geom(xy(cc[ids].mean(axis=0)),mm) for ids in parts]))
   ax.set_title(f'{start}–{start+1} s; {len(cc)} cycles')
  else:ax.set_title(f'{start}–{start+1} s: gate failed')
  ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel='X',ylabel='Y')
fig.suptitle('Separated windows: bold = whole-second mean; faint = four chronological subgroup means')
fig.savefig(R/'separated_windows.png',dpi=150);plt.close(fig)
(R/'extra_metrics.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
