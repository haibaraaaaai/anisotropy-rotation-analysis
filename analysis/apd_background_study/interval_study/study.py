"""Empirical trajectory repeatability; no cone/background/interference fitting.
Run from this directory; outputs are local. Channel order: 0,90,45,135.
"""
from pathlib import Path
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(ROOT.parent))
from study_paths import tdms_path
from nptdms import TdmsFile
with TdmsFile.open(tdms_path()) as _tdms:
    raw=np.array([c[7500000:7750000] for c in _tdms.groups()[0].channels()],float)[[3,0,1,2]]
N=128;DT=4e-6

def xy(c):
 return np.stack([(c[...,0,:]-c[...,1,:])/(c[...,0,:]+c[...,1,:]),(c[...,2,:]-c[...,3,:])/(c[...,2,:]+c[...,3,:])],axis=-1)
def guide(c,w):return xy(savgol_filter(c,w,3,axis=-1))
def anchors(p,threshold=-.5):
 x,y=p.T;out=[];armed=False
 for i in range(1,len(x)):
  if x[i]>.3:armed=True
  if armed and x[i]<threshold<=x[i-1] and y[i]<-.3:out.append(i);armed=False
 return np.array(out)
def bin_cycles(raw,g,bounds):
 vals=[];gs=[];counts=[]
 for a,b in zip(bounds[:-1],bounds[1:]):
  pp=g[a:b];ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))]
  edges=np.r_[0,np.searchsorted(ss,np.linspace(0,ss[-1],N+1)[1:-1]),len(pp)]
  for k in range(1,N):edges[k]=np.clip(edges[k],edges[k-1]+1,len(pp)-(N-k))
  centers=(edges[:-1]+edges[1:]-1)/2
  vals.append(np.column_stack([raw[:,a+i:a+j].mean(axis=1) for i,j in zip(edges[:-1],edges[1:])]))
  gs.append(np.column_stack([np.interp(centers,np.arange(len(pp)),pp[:,j]) for j in range(2)]))
  counts.append(np.diff(edges))
 return np.array(vals),np.array(gs),np.array(counts)
def align_path(ref,p,band=12,penalty=.0005):
 # Monotone point correspondence; steps 0,1,2, fixed endpoints. Band prevents
 # large phase shifts/branch changes. Unwarped averages remain the control.
 n=len(ref);D=np.full((n,n),np.inf);back=np.full((n,n),-1,dtype=int)
 D[0,0]=np.sum((ref[0]-p[0])**2)
 for i in range(1,n):
  js=np.arange(max(0,i-band),min(n,i+band+1))
  for step in (1,0,2):
   prev=js-step;valid=prev>=0;j=js[valid];v=prev[valid]
   cost=D[i-1,v]+np.sum((p[j]-ref[i])**2,axis=1)+penalty*(step!=1)
   improve=cost<D[i,j];D[i,j[improve]]=cost[improve];back[i,j[improve]]=v[improve]
 j=n-1;path=[j]
 for i in range(n-1,0,-1):j=back[i,j];path.append(j)
 return np.array(path[::-1])
def rms(a,b):return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1))))
def geom(a,b):
 # Symmetric nearest-curve distance ignores timing; complement, not replace,
 # the ordered metric. Dense interpolation avoids coarse-grid artefacts.
 def dense(p):
  t=np.arange(len(p));u=np.linspace(0,len(p),4096,endpoint=False)
  return np.column_stack([np.interp(u,np.r_[t,len(p)],np.r_[p[:,j],p[0,j]]) for j in range(2)])
 a,b=dense(a),dense(b)
 return float(np.sqrt((np.mean(cKDTree(a).query(b)[0]**2)+np.mean(cKDTree(b).query(a)[0]**2))/2))
def normal_rms(p,ref):
 t=np.roll(ref,-1,axis=0)-np.roll(ref,1,axis=0);v=np.linalg.norm(t,axis=1)
 n=np.column_stack([-t[:,1],t[:,0]])/np.maximum(v[:,None],1e-10)
 return float(np.sqrt(np.mean(np.sum((p-ref)*n,axis=-1)**2)))
def axxy(ax):ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel='X',ylabel='Y')

baseguide=guide(raw,41);bounds=anchors(baseguide)
channels,guides,counts=bin_cycles(raw,baseguide,bounds)
k=len(channels);train=np.arange(k)%2==0
# Reference from even cycles only, never an optical/cone template.
ref=xy(channels[train].mean(axis=0))
paths=np.array([align_path(ref,g) for g in guides])
aligned=np.array([c[:,p] for c,p in zip(channels,paths)])
p=xy(channels);pa=xy(aligned)
mean=xy(channels.mean(axis=0));ma=xy(aligned.mean(axis=0))
groups=np.array_split(np.arange(k),8)
unwarped_groups=np.array([xy(channels[g].mean(axis=0)) for g in groups])
aligned_groups=np.array([xy(aligned[g].mean(axis=0)) for g in groups])
metrics=dict(n_cycles=k,gate=dict(x=-.5,y_below=-.3,rearm_x=.3),
 time_range_s=(30+bounds[[0,-1]]*DT).tolist(),duration_ms_quantiles=np.percentile(np.diff(bounds)*DT*1000,[0,25,50,75,100]).tolist(),
 raw_coordinate_results=dict(single_to_mean_ordered_rms_median=float(np.median([rms(a,mean) for a in p])),
 single_to_mean_normal_rms_median=float(np.median([normal_rms(a,mean) for a in p])),
 even_odd_mean_rms=rms(xy(channels[train].mean(axis=0)),xy(channels[~train].mean(axis=0))),
 even_odd_mean_geometric_rms=geom(xy(channels[train].mean(axis=0)),xy(channels[~train].mean(axis=0))),
 aligned_even_odd_mean_rms=rms(xy(aligned[train].mean(axis=0)),xy(aligned[~train].mean(axis=0))),
 unwarped_vs_aligned_mean_rms=rms(mean,ma),unwarped_vs_aligned_mean_geometric_rms=geom(mean,ma)),
 group_intervals_s=[[float(30+bounds[g[0]]*DT),float(30+bounds[g[-1]+1]*DT)] for g in groups],
 group_to_mean_rms=[rms(g,mean) for g in unwarped_groups],
 group_to_mean_geometric_rms=[geom(g,mean) for g in unwarped_groups],
 aligned_group_to_mean_rms=[rms(g,ma) for g in aligned_groups],
 alignment=dict(max_band_bins=12,step_penalty=.0005,
  band_touch_fraction=float(np.mean(np.abs(paths-np.arange(N))==12)),
  median_rms_shift_bins=float(np.median(np.sqrt(np.mean((paths-np.arange(N))**2,axis=1)))),
  fraction_repeated_source_bins=float(np.mean(np.diff(paths,axis=1)==0))))

# Change guide smoothing while keeping exactly the same interval boundaries.
variants={};sens={}
for w in (21,41,61,81):
 c,g,ct=bin_cycles(raw,guide(raw,w),bounds);m=xy(c.mean(axis=0));variants[w]=m
 sens[w]=dict(mean_geometric_rms=geom(m,mean),mean_ordered_rms=rms(m,mean),max_y=float(m[:,1].max()),
             independently_detected_crossings=len(anchors(guide(raw,w))))
metrics['smoothing_sensitivity']=sens
# Alternate seed templates from nonoverlapping chronological quarters.
seed_sensitivity=[]
for inds in [np.arange(0,k//4),np.arange(3*k//4,k)]:
 rr=xy(channels[inds].mean(axis=0));pp=np.array([align_path(rr,g) for g in guides])
 aa=np.array([c[:,q] for c,q in zip(channels,pp)]);mm=xy(aa.mean(axis=0))
 seed_sensitivity.append(dict(geometric_rms_to_base=geom(mm,ma),ordered_rms_to_base=rms(mm,ma)))
metrics['alignment_seed_sensitivity']=seed_sensitivity

# Fixed historical gain and T; never re-estimate per interval.
nb=json.loads((ROOT.parent/'prefit-pipeline/anisotropy_rotation_processing.ipynb').read_text())
ns={'np':np};exec(''.join(nb['cells'][6]['source']),ns)
a=np.array(json.loads((ROOT.parent/'prefit-pipeline/analysis/pupil_comparison/report/baseline.json').read_text())['inverse_gains_stack_90_45_135_0'])[[3,0,1,2]]
cal=ns['T_Icor_Matrix']()@np.diag(a)
cc=np.einsum('ij,kjl->kil',cal,channels);ac=np.einsum('ij,kjl->kil',cal,aligned)
cm=xy(cc.mean(axis=0));am=xy(ac.mean(axis=0))
metrics['fixed_calibration']=dict(gains_0_90_45_135=a.tolist(),
 even_odd_mean_rms=rms(xy(cc[train].mean(axis=0)),xy(cc[~train].mean(axis=0))),
 even_odd_mean_geometric_rms=geom(xy(cc[train].mean(axis=0)),xy(cc[~train].mean(axis=0))),
 individual_normal_rms_median=float(np.median([normal_rms(q,cm) for q in xy(cc)])),
 group_to_mean_geometric_rms=[geom(xy(cc[g].mean(axis=0)),cm) for g in groups])
# Brightness across intervals is checked separately; it was not used in alignment.
metrics['group_channel_means_0_90_45_135']=[np.mean(channels[g],axis=(0,2)).reshape(-1).tolist() for g in groups]

fig,axs=plt.subplots(2,4,figsize=(15,8),layout='constrained')
for j,(ax,g) in enumerate(zip(axs.ravel(),groups)):
 for idx in g[:5]:ax.plot(p[idx,:,0],p[idx,:,1],alpha=.18,lw=.7,color='grey')
 q=unwarped_groups[j];ax.plot(q[:,0],q[:,1],lw=1.6,color='tab:blue',label=f'{len(g)}-cycle mean')
 ax.plot(mean[:,0],mean[:,1],lw=1,color='black',ls='--',label='all-cycle mean')
 axxy(ax);lo,hi=metrics['group_intervals_s'][j];ax.set_title(f'{lo:.3f}–{hi:.3f} s')
axs[0,0].legend(fontsize=7);fig.suptitle('Independent chronological groups — no time warping, no physical fit')
fig.savefig(ROOT/'interval_groups.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for idx in range(0,k,max(1,k//12)):axs[0].plot(p[idx,:,0],p[idx,:,1],alpha=.35,lw=.7)
axs[0].set_title('12 separate complete trajectories')
for mask,label in [(train,'Even cycles'),(~train,'Odd cycles')]:
 q=xy(channels[mask].mean(axis=0));axs[1].plot(q[:,0],q[:,1],label=label,lw=1.7)
axs[1].set_title('Independent interleaved averages');axs[1].legend()
axs[2].plot(mean[:,0],mean[:,1],label='Arc-length bins, no warping',lw=1.8)
axs[2].plot(ma[:,0],ma[:,1],label='Constrained alignment',lw=1.3)
axs[2].set_title('Does alignment force a different shape?');axs[2].legend(fontsize=8)
for ax in axs:axxy(ax)
fig.suptitle(f'Raw-channel coordinates; all {k} detected complete cycles retained')
fig.savefig(ROOT/'repeatability.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for w,q in variants.items():axs[0].plot(q[:,0],q[:,1],label=f'{w} sample guide')
axs[0].set_title('Smoothing sensitivity');axs[0].legend(fontsize=8)
for g in groups:
 q=xy(cc[g].mean(axis=0));axs[1].plot(q[:,0],q[:,1],alpha=.7,lw=1)
axs[1].plot(cm[:,0],cm[:,1],'k--',lw=1);axs[1].set_title('Fixed gain + T; eight group means')
old=np.load(ROOT.parent/'prefit-pipeline/analysis/pupil_comparison/report/fit_channels.npz')['raw_fit']
qo=xy(old);axs[2].plot(qo[:,0],qo[:,1],label='Old ridge-selected reference')
axs[2].plot(mean[:,0],mean[:,1],label='New all-cycle channel average')
axs[2].set_title('Raw-channel reference comparison');axs[2].legend(fontsize=8)
for ax in axs:axxy(ax)
fig.savefig(ROOT/'sensitivity_and_calibration.png',dpi=150);plt.close(fig)
(ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
np.savez_compressed(ROOT/'averaged_channels.npz',channels_per_cycle=channels,aligned_channels=aligned,
 mean_channels=channels.mean(axis=0),mean_corrected_channels=cc.mean(axis=0),
 interval_bounds=bounds,group_means=np.array([channels[g].mean(axis=0) for g in groups]),
 alignment_paths=paths,counts=counts,calibration=cal)
print(json.dumps(metrics,indent=2))
