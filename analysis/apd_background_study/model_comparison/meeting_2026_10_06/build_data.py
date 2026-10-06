"""Reconstruct saved predictions and report metrics; performs no optimization."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
R=Path(__file__).resolve().parent; M=R.parent
sys.path.insert(0,str(M/'three_tests'))
import run as r
from height import add_height
from models import xy,cone,mechanical_phase,ABC,stokes_background
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
fx=load('fixed',M/'fixed_excitation/run.py');rad=load('rad',M/'radial_field/run.py')
fixed=json.load(open(M/'fixed_excitation/fits.json'));shared=json.load(open(M/'shared_field/fits.json'));three=json.load(open(M/'three_tests/fits.json'));radfit=json.load(open(M/'radial_field/fits.json'))
records=[]
def add(id,label,k,y,s,c,b,inc,d,best,nparam,scope,source,notes=''):
 i=r.KEYS.index(k);tr,te=r.TR[i],r.TE[i];mx=tr.sum(0).max();mn=tr.sum(0).min()
 assert np.max(abs(y-(s+c+b[:,None])))<1e-9
 assert abs(b.sum()-(b-inc).sum()-inc.sum())<1e-9
 th=np.rad2deg(np.arccos(np.abs(d[:,2])));ph=np.rad2deg(np.arctan2(d[:,1],d[:,0])%np.pi)
 scores={}
 for name,dat in [('train',tr),('test',te)]:
  scores[name]={'xy_rms':float(np.sqrt(np.mean(np.sum((xy(y)-xy(dat))**2,axis=1)))),'channel_rms':float(np.sqrt(np.mean(((y-dat)/dat.sum(0))**2))),'total_rms':float(np.sqrt(np.mean(((y.sum(0)-dat.sum(0))/dat.sum(0))**2)))}
 reference=best.get('stats',{}).get(k,{}).get('test',best.get('test',{})).get('xy_rms')
 if reference is not None:assert abs(reference-scores['test']['xy_rms'])<1e-9,(id,k,reference,scores)
 rec=dict(id=id,label=label,interval=k,scope=scope,n_parameters=nparam,source=source,notes=notes,success=best.get('success'),nfev=best.get('nfev'),scores=scores,bg_max=100*b.sum()/mx,bg_min=100*b.sum()/mn,coherent_max=100*(b-inc).sum()/mx,residual_max=100*inc.sum()/mx,coherent_share=100*(b-inc).sum()/b.sum(),theta_min=th.min(),theta_max=th.max(),phi_mod180_min=ph.min(),phi_mod180_max=ph.max(),theta=th.tolist(),phi_mod180=ph.tolist(),pred=y.tolist(),signal=s.tolist(),cross=c.tolist(),background=b.tolist(),noninterfering=inc.tolist(),bg_channels_max=(100*b/mx).tolist(),train=tr.tolist(),test=te.tolist())
 if id=='richard':rec.update(coherent_max=None,residual_max=None,coherent_share=None,notes='Effective channel coefficients do not identify coherent/noninterfering composition.')
 records.append(rec)
for i,k in enumerate(r.KEYS):
 tr=r.prev.DATA[i]['train'];tm=tr.sum(0).min()
 for m,label in [('bg','Background only'),('richard','Richard effective intensity'),('field','Uniform field (2 modes)')]:
  best=fixed[k][m]['best'];p=np.array(best['p']);y,parts=fx.fixed_predict(p,m,128,tm);s,c,b=parts['signal'],parts['cross'],parts['background'];d=cone(p[:3],mechanical_phase(p,128,16,-1));inc=b if m!='field' else stokes_background(p[19]*tm*(1-p[22]),p[20],p[21])
  add(m,label,k,y/r.prev.SCALE,s/r.prev.SCALE,c/r.prev.SCALE,b/r.prev.SCALE,inc/r.prev.SCALE,d,best,len(p),'separate intervals','fixed_excitation/fits.json')
 best=radfit[k]['radial16']['best'];p=np.array(best['p']);y=rad.pred(p,128,tm);d=cone(p[:3],mechanical_phase(p,128,16,-1));a=np.exp(p[-1]);s=rad.q_channels(d)*(a*a*np.sum(d[:,:2]**2,axis=1))[None,:];p0=p.copy();p0[-1]=-100;b=rad.pred(p0,128,tm)[:,0];inc=stokes_background(p[19]*tm*(1-p[22]),p[20],p[21]);c=y-s-b[:,None]
 add('radial','Uniform + radial (3 modes)',k,y/r.prev.SCALE,s/r.prev.SCALE,c/r.prev.SCALE,b/r.prev.SCALE,inc/r.prev.SCALE,d,best,len(p),'separate intervals','radial_field/fits.json')
for key,v in shared.items():
 m=int(key[1]);best=v['best'];p=np.array(best['p']);_,inc=r.prev.bgparts(p[40:],m)
 for i,k in enumerate(r.KEYS):
  s,c,b,d=r.prev.predict(p[i*20:i*20+20],p[40:],m,True);y=s+c+b[:,None]
  add(key,f'{m}-mode shared / '+('10% cap' if key.endswith('0.1') else 'relaxed'),k,y,s,c,b,inc,d,best,len(p),'shared field','shared_field/fits.json')
cache={}
for key,v in three.items():
 best=v['best'];p=np.array(best['p']);motion=v['motion'];pure=v['pure']
 if key.startswith('edge'):
  width=float(key.split('_w')[1].split('_')[0]);tag=('edge',width)
  if tag not in cache:cache[tag]=r.make_basis('edge',width)
  opt=cache[tag];label=f'Edge w={width:g} / '+('10% cap' if v['cap']==.1 else 'relaxed')
 else:
  tag=('general',motion)
  if tag not in cache:cache[tag]=add_height(r.make_basis('general')) if motion else r.make_basis('general')
  opt=cache[tag]
  label=(f'Height R≤{v["rmax_um"]*1000:g} nm / 10%' if motion else 'General stationary / '+(f'{v["cap"]*100:g}% cap' if v['cap']<1 else 'relaxed'))
 off=44 if motion else 40;h=p[off:];bf,inc=r.field(h,opt,pure);b=inc+np.einsum('m,imn,n->i',bf.conj(),opt['g'],bf).real
 for i,k in enumerate(r.KEYS):
  local=p[i*20:i*20+20];chi=mechanical_phase(local,128,16,-1);d=cone(local[:3],chi);height=p[40+2*i:42+2*i] if motion else None;y=r.predict(local,h,opt,pure,height)
  exc2=1 if not motion else 1/(1+(height[0]*np.sin(local[0])*np.cos(chi+height[1])/2)**2)
  s=np.exp(2*local[-1])*np.sum(d[:,:2]**2,axis=1)[None,:]*exc2*np.einsum('tj,ijk,tk->it',d,opt['q'],d);c=y-s-b[:,None]
  add(key,label,k,y,s,c,b,inc,d,best,len(p),'shared field','three_tests/fits.json')
  if motion:records[-1].update(radius_nm=1000*height[0],height_amplitude_nm=1000*height[0]*np.sin(local[0]))
# Direct inversion is a descriptive baseline, not a fitted cone.
direct={}
for i,k in enumerate(r.KEYS):
 v=xy(r.TR[i]);rr=np.linalg.norm(v,axis=1);s2=ABC[0]*rr/(ABC[2]-ABC[1]*rr);assert np.all((s2>=0)&(s2<=1));th=np.rad2deg(np.arcsin(np.sqrt(s2)));ph=np.rad2deg(.5*np.arctan2(v[:,1],v[:,0])%np.pi)
 direct[k]=dict(theta=th.tolist(),phi_mod180=ph.tolist(),theta_min=th.min(),theta_max=th.max())
 for rec in records:
  if rec['interval']!=k:continue
  dp=(np.array(rec['phi_mod180'])-ph+90)%180-90;dt=np.array(rec['theta'])-th
  rec.update(theta_rms_vs_direct=float(np.sqrt(np.mean(dt*dt))),phi_rms_vs_direct=float(np.sqrt(np.mean(dp*dp))))
(R/'comparison.json').write_text(json.dumps(dict(records=records,direct=direct),indent=2))
import csv
cols=['id','label','interval','scope','n_parameters','success','nfev','bg_max','bg_min','coherent_max','residual_max','coherent_share','theta_min','theta_max','theta_rms_vs_direct','phi_rms_vs_direct']
with (R/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,lineterminator="\n",fieldnames=cols+['test_xy_rms','test_channel_rms','test_total_rms']);w.writeheader()
 for z in records:w.writerow({**{c:z[c] for c in cols},**{'test_'+a:b for a,b in z['scores']['test'].items()}})
print('Verified',len(records),'model/window predictions against saved scores.')
for z in records:
 if z['interval']=='30s':print(z['id'],round(z['scores']['test']['xy_rms'],4),round(z['bg_max'],2),None if z['coherent_share'] is None else round(z['coherent_share'],1),round(z['theta_min'],1),round(z['theta_max'],1),'converged',z['success'])
