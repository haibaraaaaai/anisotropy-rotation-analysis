import sys,json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));from models import ABC,xy,cone,mechanical_phase
shared=json.load(open(R.parent/'shared_field/fits.json'));fixed=json.load(open(R.parent/'fixed_excitation/fits.json'));rad=json.load(open(R.parent/'radial_field/fits.json'));free=json.load(open(R.parent/'mask_feasibility/free_fits.json'))
allres={};fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
for i,(key,path) in enumerate([('30s','fitted_predictions.npz'),('10s','interval10/averages.npz')]):
 z=np.load(R.parent/path);dat=z['train'];t=dat.sum(0);v=xy(dat);rr=np.linalg.norm(v,axis=1);s2=ABC[0]*rr/(ABC[2]-ABC[1]*rr);valid=(s2>=0)&(s2<=1);th=np.full(128,np.nan);th[valid]=np.rad2deg(np.arcsin(np.sqrt(s2[valid])));ph=.5*np.arctan2(v[:,1],v[:,0]);models={'Direct (no BG)':(th,ph,0,0)}
 def add(label,p,bg,di=-1):
  d=cone(p[:3],mechanical_phase(p,128,16,di));theta=np.rad2deg(np.arccos(np.abs(d[:,2])));phi=np.arctan2(d[:,1],d[:,0]);models[label]=(theta,phi,bg/t.max(),bg/t.min())
 p=np.array(fixed[key]['bg']['best']['p']);add('BG only, fixed excitation',p,p[19]*t.min())
 p=np.array(rad[key]['radial16']['best']['p']);add('3-mode field, separate',p,p[19]*t.min())
 for cap,label in [(2.,'5-mode shared, relaxed'),(.1,'5-mode shared, 10% cap')]:
  entry=shared[f'm5_cap{cap}']['best'];p=np.array(entry['p']);bg=entry['stats'][key]['background_fraction_max']*t.max();add(label,p[i*20:i*20+20],bg)
 p=np.array(free[key]['best']['p']);add('Flexible overlap, 10% witness',p,.1*t.max())
 oldpath='fits.json' if key=='30s' else 'interval10/fits.json';old=json.load(open(R.parent/oldpath));p=np.array(old['field']['best']['p']);add('Earlier field, free brightness',p,p[19]*t.min(),old['field']['best'].get('direction',-1))
 res={}
 colors={'Direct (no BG)':'#333333','BG only, fixed excitation':'#dd8b18','5-mode shared, relaxed':'#00866a','5-mode shared, 10% cap':'#bd3350','Flexible overlap, 10% witness':'#8961b1'}
 for label,(theta,phi,bmax,bmin) in models.items():
  dp=np.rad2deg(.5*np.angle(np.exp(2j*(phi-ph))))
  res[label]={'theta_range':[float(np.nanmin(theta)),float(np.nanmax(theta))],'theta_rms_vs_direct':float(np.sqrt(np.nanmean((theta-th)**2))),'theta_max_diff_vs_direct':float(np.nanmax(abs(theta-th))),'phi_rms_mod180_vs_direct':float(np.sqrt(np.nanmean(dp**2))),'bg_over_max':float(bmax),'bg_over_min':float(bmin),'theta':theta.tolist(),'phi_mod180':np.rad2deg(phi%np.pi).tolist()}
  if label not in ['Earlier field, free brightness','3-mode field, separate']:
   style='--' if 'Flexible' in label else '-';axes[i,0].plot(np.arange(128)/128,theta,style,label=label,lw=1.8,color=colors[label])
   if label!='Direct (no BG)':
    dp_plot=dp.copy();dp_plot[1:][np.abs(np.diff(dp))>90]=np.nan
    axes[i,1].plot(np.arange(128)/128,dp_plot,style,label=label,lw=1.8,color=colors[label])
 res['_direct_invalid_bins']=int((~valid).sum());allres[key]=res
 axes[i,0].set(title=f'{"30–31" if i==0 else "10–11"} s: folded polar angle',ylim=(0,90),ylabel='θ (degrees)',xlabel='Ordered trajectory fraction');axes[i,1].set(title='Azimuth difference from direct inversion',ylim=(-90,90),ylabel='Δφ modulo 180° (degrees)',xlabel='Ordered trajectory fraction')
 for ax in axes[i]:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Angle sensitivity to background assumptions — existing fits only\nDirect inversion uses measured anisotropy; fitted angles come from the imposed cone',fontsize=14)
fig.savefig(R.parent.parent/'deliverables/angle_model_sensitivity.png',dpi=160)
(R/'angles.json').write_text(json.dumps(allres,indent=2))
for key,res in allres.items():
 print(key)
 for label,x in res.items():
  if isinstance(x,dict):print(label,{k:v for k,v in x.items() if k not in ['theta','phi_mod180']})
