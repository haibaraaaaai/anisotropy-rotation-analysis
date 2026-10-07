"""Construct uniform final-reference arc and raw provisional-plane phase references.
The provisional circle sets bin boundaries only; all fitted values are measured channels.
"""
from pathlib import Path
import importlib.util,sys,json,os
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import isotonic_regression
from nptdms import TdmsFile
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));from models import xy,ABC
sp=importlib.util.spec_from_file_location('corr',R.parent/'corrected_arc/run.py');corr=importlib.util.module_from_spec(sp);sp.loader.exec_module(corr)
Z=corr.Z;A,B,C=ABC;rmax=C/(A+B);N=128

def directions(channels):
 pp=xy(channels);rr=np.linalg.norm(pp,axis=1);valid=(channels>0).all(0)&(rr<=rmax)&(rr>=0)
 if not valid.all():return None,rr,valid
 th=np.arcsin(np.sqrt(A*rr/(C-B*rr)));ph=np.unwrap(np.arctan2(pp[:,1],pp[:,0]))/2
 return np.column_stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]),rr,valid

def main():
 out={};diag={};cal=np.load(R.parent.parent/'interval_study/averaged_channels.npz')['calibration']
 for key,start in [('30s',30),('10s',10)]:
  train=Z[key+'_train'];test=Z[key+'_test'];u=corr.fractions(train,128)
  # Invert dense final-reference arc -> original bin-index coordinate. Same channel interpolation weights for test.
  nn=128*128;tt=np.arange(nn+1)/128;dense=np.array([np.interp(tt,np.arange(129),np.r_[row,row[0]]) for row in train]);pp=xy(dense);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))];arc/=arc[-1]
  target=np.arange(128)/128;idx=np.interp(target,arc,tt)
  for label,dat in [('train',train),('test',test)]:out[key+'_uniform_'+label]=np.array([np.interp(idx,np.arange(129),np.r_[row,row[0]]) for row in dat])
  out[key+'_uniform_index']=idx
  # Weighted original data isolates objective-weight change without interpolation.
  out[key+'_weighted_train']=train;out[key+'_weighted_test']=test
  du=np.diff(np.r_[u,1]);w=(du+np.roll(du,1))/2;out[key+'_weighted_weights']=w/w.mean()
  with TdmsFile.open(Path(os.environ.get('APD_TDMS_PATH',str(R.parents[3]/'files3_first40s.tdms')))) as f:raw=np.array([c[start*250000:(start+1)*250000] for c in f.groups()[0].channels()],float)[[3,0,1,2]]
  channels=cal@raw;guide=savgol_filter(channels,41,3,axis=1);bounds=Z[key+'_bounds'];cycles=[];guide_points=[];valid_cycle_ids=[]
  # Sample each training cycle uniformly in time to fit ONE frozen guide plane, equal cycle weight.
  rawrr=np.linalg.norm(xy(channels),axis=1);rawbad=(channels<=0).any(0)|(rawrr>rmax)
  for j,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
   d,rr,valid=directions(guide[:,a:b])
   if d is None:continue
   cycles.append((j,a,b,d));valid_cycle_ids.append(j)
   if j%2==0:guide_points.append(d[np.linspace(0,len(d)-1,128).astype(int)])
  cloud=np.concatenate(guide_points);mean=cloud.mean(0);val,vec=np.linalg.eigh((cloud-mean).T@(cloud-mean)/len(cloud));axis=vec[:,0];offset=float(mean@axis)
  if offset<0:axis=-axis;offset=-offset
  e1=cloud[0]-(cloud[0]@axis)*axis;e1/=np.linalg.norm(e1);e2=np.cross(axis,e1)
  means=[];ids=[];spans=[];back=[];changes=[];rawphasecoords=[]
  for j,a,b,d in cycles:
   phase=np.unwrap(np.arctan2(d@e2,d@e1));span=phase[-1]-phase[0]
   if abs(span)<1e-6:continue
   phase=(phase-phase[0])/span
   # Monotone guide only, preserves acquisition order. Retain correction size as diagnostic.
   isotonic=np.asarray(isotonic_regression(phase,increasing=True).x);isotonic=(isotonic-isotonic[0])/(isotonic[-1]-isotonic[0])
   edges=np.r_[0,np.searchsorted(isotonic,np.arange(1,128)/128),b-a]
   for k in range(1,128):edges[k]=np.clip(edges[k],edges[k-1]+1,b-a-(128-k))
   means.append(np.column_stack([channels[:,a+i:a+l].mean(1) for i,l in zip(edges[:-1],edges[1:])]))
   ids.append(j);spans.append(span/(2*np.pi));back.append(float(-np.minimum(np.diff(phase),0).sum()));changes.append(float(np.max(abs(isotonic-phase))));rawphasecoords.append(np.interp(np.linspace(0,1,128),np.linspace(0,1,len(phase)),phase))
  means=np.array(means);ids=np.array(ids);trainmask=ids%2==0
  out[key+'_phase_train']=means[trainmask].mean(0);out[key+'_phase_test']=means[~trainmask].mean(0);out[key+'_phase_cycles']=means;out[key+'_phase_ids']=ids;out[key+'_phase_guide_samples']=np.array(rawphasecoords)
  # Circle guide contains actual uncorrected inversion samples, NOT fitted-circle replacement points.
  out[key+'_guide_cloud']=cloud;out[key+'_guide_axis']=axis;out[key+'_guide_offset']=offset
  def quant(x):return np.percentile(x,[0,50,95,100]).tolist()
  # Closed-lift issue is measured rather than hidden: raw anisotropy may wind once, so phi changes pi.
  d,rr,valid=directions(train);pp=xy(train);psi=np.unwrap(np.r_[np.arctan2(pp[:,1],pp[:,0]),np.arctan2(pp[0,1],pp[0,0])]);wind=int(np.rint((psi[-1]-psi[0])/(2*np.pi)))
  diag[key]=dict(raw_invalid_fraction=float(rawbad.mean()),guide_cycles_valid=len(cycles),cycles_total=len(bounds)-1,used_cycles=len(ids),train_cycles=int(trainmask.sum()),test_cycles=int((~trainmask).sum()),rmax=rmax,training_mean_r_range=[float(np.linalg.norm(pp,axis=1).min()),float(np.linalg.norm(pp,axis=1).max())],training_mean_winding=wind,guide_plane_half_angle_deg=float(np.degrees(np.arccos(offset))),guide_plane_rms=float(np.sqrt(np.mean((cloud@axis-offset)**2))),guide_phase_span_turns_quantiles=quant(spans),guide_backtracking_fraction_quantiles=quant(back),isotonic_max_fraction_change_quantiles=quant(changes),uniform_reference_step_cv=float(np.std(np.linalg.norm(np.roll(xy(out[key+'_uniform_train']),-1,axis=0)-xy(out[key+'_uniform_train']),axis=1))/np.mean(np.linalg.norm(np.roll(xy(out[key+'_uniform_train']),-1,axis=0)-xy(out[key+'_uniform_train']),axis=1))))
  print(key,diag[key],flush=True)
 out['scale']=Z['scale'];np.savez_compressed(R/'references.npz',**out);(R/'preparation_checks.json').write_text(json.dumps(diag,indent=2))
if __name__=='__main__':main()
