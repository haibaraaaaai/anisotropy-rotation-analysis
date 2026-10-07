from pathlib import Path
import sys,json,csv,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run as f
from models import ABC,cone,xy
R=Path(__file__).resolve().parent;v=json.load(open(f.R/'fits.json'))['refined'];P=np.array(v['p']);stats=[]
labels=['Background only','Richard four-C','General stationary field'];colors=['#167d9a','#9256a6','#ce6436']
def frame(g):
 th,ph,lam=g;k=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]);e1=np.array([-np.cos(th)*np.cos(ph),-np.cos(th)*np.sin(ph),np.sin(th)]);return np.column_stack([e1,np.cross(k,e1),k])
def invert(dat,ref):
 a=xy(dat);rad=np.linalg.norm(a,axis=1);A,B,C=ABC;s2=A*rad/(C-B*rad)
 ok=np.all(np.isfinite(dat),axis=0)&np.all(dat>=0,axis=0)&np.isfinite(s2)&(s2>=0)&(s2<=1)
 out=np.full((len(rad),3),np.nan);ids=np.where(ok)[0];ph=.5*np.arctan2(a[ids,1],a[ids,0]);s=np.sqrt(s2[ids]);z=np.sqrt(1-s2[ids]);base=np.column_stack([s*np.cos(ph),s*np.sin(ph),z]);cand=base[:,None,:]*np.array([[1,1,1],[-1,-1,1],[1,1,-1],[-1,-1,-1]])[None,:,:];choice=np.argmax(np.einsum('nkj,nj->nk',cand,ref[ids]),axis=1);out[ids]=cand[np.arange(len(ids)),choice]
 return out,ok,rad
for row,model in enumerate(['square','richard_product','stationary']):
 fig=plt.figure(figsize=(10.4,10));gs=fig.add_gridspec(2,2,hspace=.32,wspace=.23);spherefig=plt.figure(figsize=(10.4,10));sg=spherefig.add_gridspec(2,2,height_ratios=[1.2,.8],hspace=.28,wspace=.25)
 for col,key in enumerate(f.KEYS):
  tr=f.TR[col];te=f.TE[col];u=f.U[col];orig=f.r.Z[key+'_train']/f.Z['scale'];tm=orig.sum(0).min();mx=orig.sum(0).max()
  if model=='stationary':
   local=P[5*col:5*col+5];pred,s,c,bg,d,mn=f.predict(local,P[10:],v['directions'][col],u,8192,True);curve=f.dense(local,P[10:],8192)[2]
  else:
   b=f.BASE[key]['uniform'][model]['best'];local=np.array(b['p']);pred,s,c,bg,d,mn=f.r.forward(local,model,tm,b['direction'],u,8192,True);curve=f.r.forward(local,model,tm,b['direction'],np.linspace(0,1,4097),8192)
  # Conditional residual diagnostic: subtract the fitted correction at fixed correspondence.
  corrected=te-bg[:,None]-c
  if model=='richard_product':
   C=local[5:]/np.sqrt(local[3]);amp=2*te/(np.sqrt(C[:,None]**2+4*te)+C[:,None]);corrected=amp**2
   assert np.max(abs(corrected+C[:,None]*np.sqrt(corrected)-te))<1e-10
  obs,ok,rad=invert(corrected,d);F=frame(local[:3]);dc=d@F;oc=obs@F;lam=local[2]
  # Cross-track cone-angle error and phase difference, in the cone-aligned frame.
  cross=np.degrees(np.arctan2(oc[:,2]-np.cos(lam),np.linalg.norm(oc[:,:2],axis=1)));phase=np.arctan2(oc[:,1],oc[:,0])-np.arctan2(dc[:,1],dc[:,0]);phase=np.arctan2(np.sin(phase),np.cos(phase));along=np.degrees(np.sin(lam)*phase)
  geo=np.degrees(np.arccos(np.clip(np.sum(obs*d,axis=1),-1,1)));pair=(corrected[0]+corrected[1]-corrected[2]-corrected[3])/corrected.sum(0)
  rec=dict(model=model,interval=key,valid=int(ok.sum()),invalid=int((~ok).sum()),negative_channels=int(np.any(corrected<0,axis=0).sum()),over_rmax=int((rad>ABC[2]/sum(ABC[:2])).sum()),geodesic_rms_valid=float(np.sqrt(np.nanmean(geo**2))),cross_rms_valid=float(np.sqrt(np.nanmean(cross**2))),along_rms_valid=float(np.sqrt(np.nanmean(along**2))),pair_imbalance_rms=float(np.sqrt(np.mean(pair**2))),bg_percent=None if model=='richard_product' else float(100*bg.sum()/mx),heldout_channel_rms=float(np.sqrt(np.mean(((pred-te)/te.sum(0))**2))),cone_half_angle_deg=float(np.degrees(lam)))
  stats.append(rec)
  np.savez(R/(model+'_'+key+'_sphere.npz'),corrected_channels=corrected,valid=ok,observed_directions=obs,matched_directions=d,cone_frame=F,arc_fraction=u,cross_deg=cross,along_deg=along,phase_deg=np.degrees(phase),geodesic_deg=geo,pair_imbalance=pair)
  diagnostic=s+(te-pred);da=xy(diagnostic);_,valid_diag,_=invert(diagnostic,d)
  ax=fig.add_subplot(gs[0,col]);ax.plot(*xy(curve).T,color=colors[row],lw=1.6,label='Fitted measured curve');ax.scatter(*xy(te).T,s=10,color='#172d3c',label='Held-out mean')
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f'{key} · 1. Measured space');ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=7,loc='lower left')
  ax=fig.add_subplot(gs[1,col])
  if model=='stationary':signal_curve=f.dense(local,P[10:],8192)[3]
  else:signal_curve=f.r.forward(local,model,tm,b['direction'],np.linspace(0,1,4097),8192,True)[1]
  ax.plot(*xy(signal_curve).T,color=colors[row],lw=1.8,label='S_fit (signal only)')
  closed=np.vstack([da,da[0]]);ax.plot(*closed.T,color='#398849',lw=1.1,label='S_fit + residual');ax.scatter(*da[~valid_diag].T,s=13,color='#d32030',marker='x',linewidths=.8,label='Outside inversion domain',zorder=5)
  aa=np.linspace(0,2*np.pi,300);limit=ABC[2]/sum(ABC[:2]);ax.plot(limit*np.cos(aa),limit*np.sin(aa),ls=':',color='.65',lw=.7,label='Signal-only radial limit')
  outside=(np.abs(da)>1).any(axis=1);rec['diagnostic_invalid']=int((~valid_diag).sum());rec['diagnostic_off_axes']=int(outside.sum())
  if outside.any():
   edge=da[outside]/np.maximum(1,np.max(abs(da[outside]),axis=1))[:,None]*.98;ax.scatter(*edge.T,s=19,color='#d32030',marker='^',zorder=6,label='Beyond displayed axes');ax.text(.02,.97,f'{outside.sum()} points beyond axes',transform=ax.transAxes,va='top',fontsize=7,color='#b02030')
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f'{"30–31" if key=="30s" else "10–11"} s · 2. Signal space');ax.set_aspect('equal');ax.grid(alpha=.15)
  if col==0:ax.legend(fontsize=6.5,loc='lower left')
  ax=spherefig.add_subplot(sg[0,col],projection='3d');aa=np.linspace(0,2*np.pi,37);bb=np.linspace(0,np.pi,19);ax.plot_wireframe(np.outer(np.cos(aa),np.sin(bb)),np.outer(np.sin(aa),np.sin(bb)),np.outer(np.ones_like(aa),np.cos(bb)),color='#b3bdc8',alpha=.2,lw=.4,rstride=3,cstride=3)
  # Plot laboratory coordinates; residuals use cone coordinates.
  cc=cone(local[:3],np.linspace(0,2*np.pi,361));ax.plot(*cc.T,color=colors[row],lw=2,label='Fitted cone circle')
  az=np.linspace(0,2*np.pi,65);rr=np.array([0,np.sin(lam)]);xx=np.outer(rr,np.cos(az));yy=np.outer(rr,np.sin(az));disk=np.stack([xx,yy,np.full_like(xx,np.cos(lam))],axis=-1)@F.T;ax.plot_surface(disk[:,:,0],disk[:,:,1],disk[:,:,2],color=colors[row],alpha=.09,shade=False)
  ax.plot(*np.stack([np.zeros(3),F[:,2]]).T,color=colors[row],ls='--',lw=1,label='Cone axis');ax.scatter(*obs[ok].T,color='#172d3c',s=10,depthshade=False,label='Corrected held-out points')
  for j in range(0,128,4):
   if ok[j]:ax.plot(*np.stack([d[j],obs[j]]).T,color='#888888',lw=.6,alpha=.8)
  ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel="x (lab)",ylabel="y (lab)",zlabel="z (optical axis)",title=f'Unit sphere · {ok.sum()}/128 points invertible');ax.set_box_aspect([1,1,1]);ax.view_init(elev=22,azim=-55);ax.tick_params(labelsize=7)
  if col==0:ax.legend(fontsize=6,loc='lower left')
  ax=spherefig.add_subplot(sg[1,col]);ax.axhline(0,color='.6',lw=.8);ax.plot(u,cross,color='#167d9a',lw=1,marker='.',ms=2,label='Circle-centre elevation: β');ax.plot(u,along,color='#b1543b',lw=1,marker='.',ms=2,label='Around circle: sin(λ) Δχ');ax.set(xlim=(0,1),ylim=(-90,90),xlabel='Ordered reference arc fraction',ylabel='Angular residual (degrees)',title='Residuals in fitted-cone frame');ax.grid(alpha=.15)
  if (~ok).any():ax.scatter(u[~ok],np.full((~ok).sum(),-84),marker='x',color='#bf2020',s=14,label='No physical inversion')
  if col==0:ax.legend(fontsize=6.5,loc='upper left')
  rec['residuals_outside_plot']=int(np.sum(ok&((abs(cross)>90)|(abs(along)>90))))
 fig.suptitle(labels[row]+' · fixed correspondence, held-out diagnostics',fontsize=15,y=.98);fig.savefig(R/(model+'.png'),dpi=170,bbox_inches='tight');buf=io.BytesIO();fig.savefig(buf,format='pdf',bbox_inches='tight');tmp=R/(model+'_new.pdf');tmp.write_bytes(buf.getvalue());tmp.replace(R/(model+'.pdf'));plt.close(fig)
 spherefig.suptitle(labels[row]+' · sphere and cone-frame residuals',fontsize=15);buf=io.BytesIO();spherefig.savefig(buf,format='pdf',bbox_inches='tight');tmp=R/(model+'_sphere_new.pdf');tmp.write_bytes(buf.getvalue());tmp.replace(R/(model+'_sphere.pdf'));plt.close(spherefig)
(R/'sphere_metrics.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
