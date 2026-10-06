"""Audit the recorded 30–31 s reference without running any background/cone fit.

Run from repository root. Reuses archived fit outputs only to replot the previous
comparison at fixed axes. New trajectory comparisons are in raw-channel space.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nptdms import TdmsFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from trajectory_preprocessing import ordered_channel_bins, channel_xy, anisotropy_axes
from pupil_model import xy, cone

out=ROOT/'analysis/pupil_comparison/prefit_report'
out.mkdir(exist_ok=True)
archive=ROOT/'analysis/pupil_comparison/report'
import os
with TdmsFile.open(Path(os.environ.get('APD_TDMS_PATH',str(ROOT.parents[2]/'files3_first40s.tdms')))) as tdms:
    ch=tdms.groups()[0].channels()
    dt=float(ch[0].properties['wf_increment'])
    offset=float(ch[0].properties.get('wf_start_offset',0))
    i0=round((30-offset)/dt);i1=round((31-offset)/dt)
    raw=np.array([c[i0:i1] for c in ch],float)[[3,0,1,2]]
raw_xy=channel_xy(raw)
saved=np.load(archive/'fit_channels.npz')
old=channel_xy(saved['raw_fit'])
old_idx=saved['matched_indices']
new=ordered_channel_bins(raw,0,700,90,41)
# Independent display of local channel averages throughout the full second.
block=16; n=raw.shape[1]//block
pooled=channel_xy(raw[:,:n*block].reshape(4,n,block).mean(axis=2))
old_smooth=savgol_filter(raw_xy[:700],201,3,axis=0,mode='wrap')
metrics=dict(interval_samples=[0,700],interval_duration_s=700*dt,
    legacy_selected_max_y=float(old[:,1].max()),
    legacy_match_time_range_s=(old_idx[[np.argmin(old_idx),np.argmax(old_idx)]]*dt).tolist(),
    legacy_time_backsteps=int(np.sum(np.diff(old_idx)<0)),
    legacy_201_wrap_max_y=float(old_smooth[:,1].max()),
    new_selected_max_y=float(new['xy'][:,1].max()),
    new_min_max_bin_counts=[int(new['counts'].min()),int(new['counts'].max())],
    endpoint_gap=new['endpoint_gap'],open_arc_length=new['open_arc_length'],
    all_700_samples_used_once=bool(new['counts'].sum()==700),
    smoothing_sensitivity={})
for w in (21,41,81,201):
    r=ordered_channel_bins(raw,0,700,90,w)
    metrics['smoothing_sensitivity'][w]=dict(max_y=float(r['xy'][:,1].max()),
                                          endpoint_gap=r['endpoint_gap'])
fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for ax in axs:
    ax.scatter(pooled[:,0],pooled[:,1],s=1,c='grey',alpha=.1,rasterized=True)
    anisotropy_axes(ax);ax.set(xlabel='X',ylabel='Y')
axs[0].plot(old[:,0],old[:,1],color='tab:red',lw=.7)
axs[0].scatter(old[:,0],old[:,1],s=10,c=np.arange(90),cmap='hsv')
axs[0].set_title('Legacy 90 points: whole-cloud nearest matches')
axs[1].scatter(raw_xy[:700,0],raw_xy[:700,1],s=5,c=np.arange(700),cmap='hsv',alpha=.35)
axs[1].plot(old_smooth[:,0],old_smooth[:,1],c='black',lw=1,label='legacy 201-sample wrap filter')
axs[1].set_title('Selected [0,700) interval');axs[1].legend(fontsize=7)
axs[2].plot(new['xy'][:,0],new['xy'][:,1],c='grey',lw=.7)
axs[2].scatter(new['xy'][:,0],new['xy'][:,1],s=13,c=new['centers'],cmap='hsv')
axs[2].set_title('90 contiguous channel means, in time order')
fig.suptitle('Raw-channel coordinates; grey = 16-sample channel means over 30–31 s. No new fit.')
fig.savefig(out/'trajectory_comparison.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(10,6),layout='constrained')
t=np.arange(700)*dt*1e3
axs[0].plot(t,raw_xy[:700,1],color='grey',alpha=.4,lw=.6,label='raw Y')
axs[0].plot(t,old_smooth[:,1],label='legacy 201-sample wrap filter')
axs[0].plot(t,new['guide'][:,1],label='41-sample channel guide')
axs[0].scatter(new['centers']*dt*1e3,new['xy'][:,1],s=10,label='means of original channels')
axs[0].set(xlabel='Time within interval (ms)',ylabel='Y',ylim=(-1,1));axs[0].legend(fontsize=8)
axs[1].plot(np.arange(90),old_idx*dt*1e3,'.-',label='legacy selected sample times')
axs[1].plot(np.arange(90),new['centers']*dt*1e3,'.-',label='new bin center times')
axs[1].set(xlabel='Ordered output index',ylabel='Time from 30 s (ms)');axs[1].legend(fontsize=8)
fig.savefig(out/'ordering_and_smoothing.png',dpi=150);plt.close(fig)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')

# Replot archived fits, with identical parameters and data: no optimization.
data=saved['channels'];models=dict(np.load(archive/'pupil_response_matrices.npz'))
results=json.loads((archive/'real_data_refits.json').read_text())
fig,axs=plt.subplots(1,4,figsize=(16,4),layout='constrained')
for ax,key in zip(axs,['fourkas_038','annulus_0.38','mask_0','annulus_0.39']):
    p=np.array(results[key]['best']['parameters']);dc,fa,fb=p[3:]
    bg=dc/2*np.array([1+fa,1-fa,1+fb,1-fb])
    pp=channel_xy(data-bg[:,None]);mm=xy(models[key],cone(*p[:3],n=3000))
    ax.scatter(pp[:,0],pp[:,1],c=np.arange(len(pp)),cmap='hsv',s=12,label='archived 90 points')
    ax.plot(mm[:,0],mm[:,1],'k-',lw=1,label='archived fitted cone')
    anisotropy_axes(ax);ax.set(xlabel='X',ylabel='Y',title=key+'\ncost %.5g'%results[key]['best']['cost'])
axs[0].legend(fontsize=7)
fig.savefig(archive/'real_data_refits.png',dpi=160);plt.close(fig)

# The other archived XY comparison also gets the same fixed frame.
base=json.loads((archive/'baseline.json').read_text())
geoms=[(15,20,40),(40,20,25),(70,20,30),
       tuple(base[k] for k in ['Theta_axis_fitted','Phi_axis_fitted','Lambda_fitted'])]
fig,axs=plt.subplots(1,4,figsize=(16,4),layout='constrained')
for ax,g in zip(axs,geoms):
    for key,label in [('mask_0','Measured hole + interface'),('annulus_0.38','Annulus .38 + interface'),
                      ('fourkas_038','Historical homogeneous annulus .38')]:
        p=xy(models[key],cone(*g));ax.plot(p[:,0],p[:,1],label=label,lw=1.5)
    anisotropy_axes(ax);ax.set(xlabel='X',ylabel='Y',title='Axis %.1f°, %.1f°; cone %.1f°'%g)
axs[0].legend(fontsize=7)
fig.savefig(archive/'cone_shapes.png',dpi=160);plt.close(fig)
print(json.dumps(metrics,indent=2))
