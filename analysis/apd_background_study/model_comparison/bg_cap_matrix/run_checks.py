import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'radial_field'))
from run import fit,pred,cone,mechanical_phase,q_channels,ABC,ANALYZERS,GAMMA,KAPPA,stokes_background

def components(p,tm,n=128):
 d=cone(p[:3],mechanical_phase(p,n,16,-1));a0=np.exp(p[-1]);f,pol,ang,share,eta,rel,common,rf,rp=p[19:28];bt=f*tm
 b=np.sqrt(bt*share*(1-rf)/2)*np.exp(1j*common)*np.array([np.cos(eta),np.sin(eta)*np.exp(1j*rel)])
 br=np.sqrt(bt*share*rf/2)*np.exp(1j*rp);bj=ANALYZERS@b;ex=d[:,0]+1j*d[:,1]
 signal=q_channels(d)*(a0*a0*np.abs(ex)**2)[None,:]
 cu=2*a0*np.real(GAMMA*(ANALYZERS@d[:,:2].T)*np.conj(bj[:,None])*ex[None,:])
 cr=np.broadcast_to(2*a0*np.real(KAPPA*d[:,2]*np.conj(br)*ex),(4,n))
 bu=np.abs(bj)**2;brad=np.ones(4)*abs(br)**2/2;bi=stokes_background(bt*(1-share),pol,ang)
 assert np.allclose(signal+cu+cr+(bu+brad+bi)[:,None],pred(p,n,tm))
 return dict(signal=signal.sum(0),interference=(cu+cr).sum(0),uniform_cross=cu.sum(0),radial_cross=cr.sum(0),uniform_bg=float(bu.sum()),radial_bg=float(brad.sum()),incoherent_bg=float(bi.sum()),theta=np.rad2deg(np.arccos(np.abs(d[:,2]))))

def describe(p,train,test):
 tm=train.sum(0).min();c=components(p,tm);s=c['signal'];it=c['interference'];bg=c['uniform_bg']+c['radial_bg']+c['incoherent_bg'];t=train.sum(0);mx=t.max();i=int(np.argmin(t));pm=s+it+bg
 return dict(bg_fraction_max=float(bg/mx),bg_fraction_min=float(bg/tm),bg_composition={k:float(c[k]/bg) for k in ['uniform_bg','radial_bg','incoherent_bg']},
  ranges_relative_observed_max={k:[float(np.min(c[k])/mx),float(np.max(c[k])/mx)] for k in ['signal','interference']},
  interference_over_signal_range=[float(np.min(it/s)),float(np.max(it/s))],fraction_bins_interference_negative=float(np.mean(it<0)),fraction_bins_bg_gt_signal=float(np.mean(bg>s)),
  theta_range=[float(c['theta'].min()),float(c['theta'].max())],signal_min_max_ratio=float(s.min()/s.max()),
  dimmest_observed_bin=dict(index=i,theta=float(c['theta'][i]),signal_over_signal_max=float(s[i]/s.max()),signal=float(s[i]/tm),background=float(bg/tm),interference=float(it[i]/tm),prediction=float(pm[i]/tm)),
  pair_imbalance_rms=float(np.sqrt(np.mean(((test[0]+test[1]-test[2]-test[3])/test.sum(0))**2))))
if __name__=='__main__':
 prev=json.loads((R.parent/'radial_field/fits.json').read_text());nb=json.loads((R.parent.parent/'prefit-pipeline/anisotropy_rotation_processing.ipynb').read_text());src=''.join(nb['cells'][6]['source']);src=src[:src.index('# ── Gain fit')]
 gains=np.array(json.loads((R.parent.parent/'prefit-pipeline/analysis/pupil_comparison/report/baseline.json').read_text())['inverse_gains_stack_90_45_135_0'])[[3,0,1,2]]
 res={};rng=np.random.default_rng(519)
 for key,path in [('30s','interval_study/averaged_channels.npz'),('10s','model_comparison/interval10/averages.npz')]:
  z=np.load(R.parent.parent/path);raw=z['channels_per_cycle'];res[key]={};pb=np.array(prev[key]['radial16']['best']['p'])
  for sign in ['plus','minus']:
   ns={'np':np};exec(src if sign=='plus' else src.replace('beta  = 0.009','beta  = -0.009'),ns);cal=ns['T_Icor_Matrix']()@np.diag(gains)
   if sign=='plus':assert np.allclose(cal,z['calibration'])
   allc=np.einsum('ij,kjl->kil',cal,raw);train=allc[::2].mean(0);test=allc[1::2].mean(0)
   np.savez_compressed(R/f'{key}_{sign}_data.npz',train=train,test=test,calibration=cal)
   for limited in [False,True]:
    label=sign+('_cap10' if limited else '_wide');cap=.1*train.sum(0).max()/train.sum(0).min() if limited else 2.
    starts=[pb]
    for j in range(13):
     p=pb.copy();p[:3]+=rng.normal(0,.25,3);p[3]+=rng.normal(0,.2);p[4:19]+=rng.normal(0,.2,15);p[19]=rng.uniform(.02,cap*.9);p[20]=rng.uniform(.1,.95);p[21]=rng.uniform(-np.pi,np.pi)
     p[22:28]=[rng.uniform(.1,.95),rng.uniform(.1,1.4),rng.uniform(-np.pi,np.pi),rng.uniform(-np.pi,np.pi),rng.uniform(.01,.7),rng.uniform(-np.pi,np.pi)]
     starts.append(p)
    result=fit(starts,train,test,cap=cap);result['decomposition']=describe(np.array(result['best']['p']),train,test);res[key][label]=result
    print(key,label,result['best']['test'],result['decomposition'],flush=True)
    (R/'results.json').write_text(json.dumps(res,indent=2))
