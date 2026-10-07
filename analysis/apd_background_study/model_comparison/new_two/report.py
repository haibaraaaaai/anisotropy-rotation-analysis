from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as r
R=Path(__file__).resolve().parent;v=json.load(open(R/'fits.json'));old=json.load(open(R.parent/'fixed_excitation/fits.json'));M=R.parent/'meeting_2026_10_06';oldrec=json.load(open(M/'comparison.json'))['records'];base={(q['id'],q['interval']):q for q in oldrec};out={};rows=[]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,2,figsize=(10,8.5),layout='constrained');fig2,bxs=plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
for i,k in enumerate(r.KEYS):
 out[k]={};tm=r.TR[i].sum(0).min();mx=r.TR[i].sum(0).max()
 for j,model in enumerate(['square','richard4']):
  best=v[k][model]['best'];p=np.array(best['p']);y,s,c,b,d=r.predict(p,model,tm,True);assert np.all(y>0)
  th=np.rad2deg(np.arccos(abs(d[:,2])));ph=np.rad2deg(np.arctan2(d[:,1],d[:,0])%np.pi)
  rec=dict(p=p.tolist(),scores=best['test'],theta=th.tolist(),phi_mod180=ph.tolist(),pred=y.tolist(),signal=s.tolist(),cross=c.tolist(),background=b.tolist(),bg_percent_max=100*b.sum()/mx,train=r.TR[i].tolist(),test=r.TE[i].tolist(),Tmax=mx,success=best['success'])
  if model=='square':rec.update(fa=float(p[21]),fb=float(p[22]),disk_radius=float(np.hypot(p[21],p[22])))
  else:
   recovered=((np.sqrt(p[20:,None]**2+4*y)-p[20:,None])/2)**2;assert np.max(abs(recovered-s))<1e-12
   rec.update(C_normalized=p[20:].tolist(),C_original_units=(p[20:]*np.sqrt(r.SCALE)).tolist(),minimum_coherent_bg_percent_if_interference=float(100*np.sum(p[20:]**2/4)/mx),cross_percent_max=[float(100*c.sum(0).min()/mx),float(100*c.sum(0).max()/mx)])
  out[k][model]=rec
  ax=axs[j,i];prior='bg' if model=='square' else 'richard';label='Square additive BG' if model=='square' else 'Richard: four C coefficients';bb=base[prior,k]
  ax.plot(*r.xy(r.TR[i]).T,color='#b5bdc5',lw=2,label='Training average');ax.scatter(*r.xy(r.TE[i]).T,s=10,color='#24384a',label='Held-out average');ax.plot(*r.xy(np.array(bb['pred'])).T,'--',color='#bf8c2d',lw=1.5,label='Previous Stokes BG' if j==0 else 'Previous additive + sqrt');ax.plot(*r.xy(y).T,color='#b64a45',lw=1.8,label='New fit');ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel='X',ylabel='Y',title=f'{"30–31" if i==0 else "10–11"} s · {label}\nHeld-out XY RMS {best["test"]["xy_rms"]:.4f}');ax.grid(alpha=.18)
  if i==0:ax.legend(fontsize=7,loc='lower left')
  rows.append([k,model,best['test']['xy_rms'],best['test']['channel_rms'],best['test']['total_rms'],rec['bg_percent_max'] if model=='square' else None,float(th.min()),float(th.max()),best['success']])
 # Richard total balance and new angle traces.
 t=np.arange(128)/128;rec=out[k]['richard4'];a=bxs[0,i]
 for yy,label,color in [(np.array(rec['signal']).sum(0),'Rod alone','#354d83'),(np.array(rec['cross']).sum(0),'Effective correction','#a34c86'),(np.array(rec['pred']).sum(0),'Predicted total','#118b8f')]:a.plot(t,yy/mx,label=label,color=color)
 a.scatter(t,r.TE[i].sum(0)/mx,s=6,color='#222',label='Held-out total');a.set(title=('30–31 s' if i==0 else '10–11 s')+' · Richard power balance',ylabel='Intensity / measured Tmax',xlabel='Ordered trajectory fraction');a.grid(alpha=.2)
 a=bxs[1,i]
 for yy,label,color,ls in [(base['bg',k]['theta'],'Stokes additive','#bf8c2d','--'),(out[k]['square']['theta'],'Square additive','#354d83','-'),(out[k]['richard4']['theta'],'Richard four-C','#b64a45','-'),(base['richard',k]['theta'],'Additive + sqrt extension','#118b8f',':')]:a.plot(t,yy,label=label,color=color,ls=ls)
 a.set(ylim=(0,90),ylabel='Folded θ (degrees)',xlabel='Ordered trajectory fraction');a.grid(alpha=.2)
for ax in bxs[:,0]:ax.legend(fontsize=7,ncol=2)
fig.suptitle('Two new model tests · same trajectory, optics, excitation and held-out cycles',fontsize=13);fig.savefig(R/'two_new_fits.png',dpi=180);fig.savefig(R/'two_new_fits.svg');fig2.savefig(R/'power_angles.png',dpi=180);plt.close('all')
(R/'predictions.json').write_text(json.dumps(out,indent=2));
with (R/'summary.csv').open('w') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['interval','model','test_xy_rms','test_channel_rms','test_total_rms','bg_percent_max','theta_min','theta_max','optimizer_success']);w.writerows(rows)
print(json.dumps({k:{m:{a:z[a] for a in z if a not in ['p','theta','phi_mod180','pred','signal','cross','background','train','test']} for m,z in d.items()} for k,d in out.items()},indent=2))
