"""Behavioral checks for time-ordered references. No parameter fitting."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functools import partial

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from trajectory_preprocessing import ordered_channel_bins, channel_xy, anisotropy_axes
from pupil_model import analytic_q, fourkas_abc, intensities, directions

# Two exactly overlapping turns, with different brightness on the second visit.
n=360;p=np.arange(n)*2*np.pi/n
one=np.array([1+.6*np.cos(p),1-.6*np.cos(p),1+.6*np.sin(p),1-.6*np.sin(p)])
c=np.concatenate([one,2*one],axis=1)
r=ordered_channel_bins(c,0,2*n,90,21)
assert np.all(np.diff(r['centers'])>0)
assert r['bounds'][0,0]==0 and r['bounds'][-1,1]==2*n
assert np.all(r['bounds'][1:,0]==r['bounds'][:-1,1])
assert np.all(r['counts']>0) and r['counts'].sum()==2*n
phase=np.unwrap(np.arctan2(r['xy'][:,1],r['xy'][:,0]))
assert (phase[-1]-phase[0])/(2*np.pi)>1.9
pair_sum=r['channels'][0]+r['channels'][1]
assert np.allclose(pair_sum[r['bounds'][:,1]<=n],2)
assert np.allclose(pair_sum[r['bounds'][:,0]>=n],4)

# A self-crossing trace with unequal speed retains traversal order.
u=np.linspace(0,1,1400,endpoint=False);phase=2*np.pi*(u+.12*np.sin(2*np.pi*u))
x=.6*np.sin(phase);y=.4*np.sin(2*phase)
c=np.array([1+x,1-x,1+y,1-y])
r=ordered_channel_bins(c,0,len(u),90,41)
truth=np.column_stack([np.interp(r['centers'],np.arange(len(u)),x),
                       np.interp(r['centers'],np.arange(len(u)),y)])
assert np.max(np.linalg.norm(r['xy']-truth,axis=1))<.015
assert r['xy'][:,0].max()>.59 and r['xy'][:,0].min()<-.59

# Noiseless dipole data with a varying lab theta, under a fixed cone.
from pupil_model import cone
d=cone(15,20,40,n=700)
c=intensities(analytic_q(fourkas_abc()),d).T
r=ordered_channel_bins(c,0,700,90,41)
truth=channel_xy(c)
at_centers=np.column_stack([np.interp(r['centers'],np.arange(700),truth[:,j]) for j in range(2)])
assert np.max(np.linalg.norm(r['xy']-at_centers,axis=1))<.015

# Invalid denominators stay missing; they cannot become a fake crossing at (0,0).
assert np.isnan(channel_xy(np.zeros((4,2)))).all()

# Run actual notebook pre-fit cells on a synthetic time series, with manual bounds.
# Confirm that neither SCMS nor automatic cycle detection is called in the default.
nb=json.loads((ROOT/'anisotropy_rotation_processing.ipynb').read_text())
ns={'np':np,'plt':plt,'partial':partial,'minimize':minimize,
    'ordered_channel_bins':ordered_channel_bins,'anisotropy_axes':anisotropy_axes}
exec(''.join(nb['cells'][6]['source']),ns)
original_t=ns['T_Icor_Matrix']().copy()
ns.update(c0_raw=c[0],c90_raw=c[1],c45_raw=c[2],c135_raw=c[3],
          time_s=np.arange(700)*4e-6,time_inc=4e-6,time_off=0.,datasize=700,
          fit_start_s=0.,fit_end_s=None,dec_correction=1,PREFIT_METHOD='time_ordered',
          PREFIT_GUIDE_WINDOW=41,CYC_MANUAL_OVERRIDE=True,CYC_MANUAL_START=0,CYC_MANUAL_END=700,
          use_tmatrix=True,a_manual=[1.,1.,1.,1.])
exec(''.join(nb['cells'][13]['source']),ns)
def forbidden(*args,**kwargs):raise AssertionError('Legacy extraction was called')
ns['_ob_detect_cycle']=forbidden;ns['_ob_scms']=forbidden
for i in [15,17,19,21]:exec(''.join(nb['cells'][i]['source']),ns)
assert ns['tcor_fit_sub'].shape==(4,90)
np.testing.assert_allclose(ns['T_Icor_Matrix'](),original_t)
for flag in [False,True]:
    ns['use_tmatrix']=flag
    exec(''.join(nb['cells'][21]['source']),ns)
    if not flag:np.testing.assert_allclose(ns['matcor'],np.eye(4))
    else:assert not np.allclose(ns['matcor'],np.eye(4))
# Automatic gains must see identity if the optical matrix is disabled.
def gain_stub(a,b,c,d,m):
    np.testing.assert_allclose(m,np.eye(4))
    return type('Result',(),{'x':np.ones(3)})()
ns.update(a_manual=None,use_tmatrix=False,find_best_coeff_using_mat=gain_stub)
exec(''.join(nb['cells'][21]['source']),ns)
for cell in nb['cells']:
    if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')
plt.close('all')
print('PASS: two overlapping turns retain separate brightness and chronology; crossing/variable-speed and dipole curves preserved; invalid sums; actual pre-fit cells; T toggle and gain-matrix selection.')
