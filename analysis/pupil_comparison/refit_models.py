"""Controlled real-data pupil refits; these are sensitivity tests, not angle validation.

Reuse the historical fit's selected points, gains, matrix, bounds and arc-length
metric. Unlike the historical objective, require all 90 points to remain valid;
this prevents a model from improving by discarding samples. Use joint DE rather
than the historical four alternating steps, with a second seed for comparison.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pupil_model import *

out=Path('analysis/pupil_comparison/results')
base=json.loads((out/'baseline.json').read_text())
data=np.load(out/'fit_channels.npz')['channels']
models=dict(np.load(out/'pupil_response_matrices.npz'))
upper=max(0.,min(np.min(data[0])+np.min(data[1]),np.min(data[2])+np.min(data[3])))
bounds=[(0,90),(-180,180),(0,90),(0,upper),(-1,1),(-1,1)]
def background(p):
    dc,fa,fb=p[3:]
    return dc/2*np.array([1+fa,1-fa,1+fb,1-fb])
def arclength(p,n=360):
    p=np.roll(p,-np.argmin(p[:,0]),axis=0)
    p=np.concatenate([p,p[:1]])
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if s[-1]<1e-12:return np.repeat(p[:1],n,axis=0)
    s/=s[-1];f=np.arange(n)/n
    return np.column_stack([np.interp(f,s,p[:,i]) for i in range(2)])
def points(p,channels=data):
    adj=channels-background(p)[:,None]
    if np.any(adj<=0):return None
    return np.column_stack([(adj[0]-adj[1])/(adj[0]+adj[1]),(adj[2]-adj[3])/(adj[2]+adj[3])])
def cost(p,q):
    dd=points(p)
    if dd is None:return 1e6
    dd=arclength(dd);mm=xy(q,cone(*p[:3]))
    return min(np.mean(np.sum((dd-arclength(mm))**2,axis=1)),
               np.mean(np.sum((dd-arclength(mm[::-1]))**2,axis=1)))

# Same measured, time-ordered points for each geometric holdout diagnostic.
val=np.load(out/'validation_channels.npz');raw=val['raw'];tt=val['time']
import nbformat
nb=nbformat.read('anisotropy_rotation_processing.ipynb',as_version=4)
namespace={'np':np};exec(nb.cells[6].source,namespace)
gain=np.array(base['inverse_gains_stack_90_45_135_0'])[[3,0,1,2]]
cal=namespace['T_Icor_Matrix']()@(gain[:,None]*raw)

results={};fits={}
for key in ['fourkas_038','annulus_0.38','mask_0','annulus_0.39']:
    q=models[key];runs=[]
    for seed in [42,91]:
        r=differential_evolution(cost,bounds,args=(q,),seed=seed,maxiter=300,popsize=18,
                                 tol=1e-7,polish=False,workers=1)
        # Preserve DE if a local search is worse; nonsmooth anchors defeat gradients.
        pol=minimize(cost,r.x,args=(q,),method='Powell',bounds=bounds,options={'maxiter':300,'ftol':1e-9,'xtol':1e-7})
        best=pol if pol.fun<r.fun else r
        run=dict(seed=seed,parameters=best.x.tolist(),cost=float(best.fun),
                 DE_success=bool(r.success),DE_message=str(r.message),DE_iterations=int(r.nit),
                 polish_success=bool(pol.success),background=background(best.x).tolist())
        runs.append(run);print(key,run,flush=True)
    best=min(runs,key=lambda r:r['cost']);p=np.array(best['parameters']);fits[key]=p
    adj=cal-background(p)[:,None];valid=np.all(adj>0,axis=0)
    pp=np.column_stack([(adj[0]-adj[1])/(adj[0]+adj[1]),(adj[2]-adj[3])/(adj[2]+adj[3])])
    template=xy(q,cone(*p[:3],n=5000));dist=cKDTree(template).query(pp)[0]
    metrics={}
    for label,m in [('fit_window',(tt>=30)&(tt<31)),('held_out',(tt<30)|(tt>=31))]:
        use=m&valid;metrics[label]=dict(valid_fraction=float(valid[m].mean()),
              median_distance=float(np.median(dist[use])),rms_distance=float(np.sqrt(np.mean(dist[use]**2))))
    dd=cone(*p[:3],n=5000)
    folded=np.rad2deg(np.arccos(np.abs(dd[:,2])))
    results[key]=dict(best=best,runs=runs,geometric_holdout=metrics,
                     folded_theta_range_deg=[float(folded.min()),float(folded.max())],
                     background_stokes_radius=float(np.hypot(p[4],p[5])))
    (out/'real_data_refits.json').write_text(json.dumps(results,indent=2)+'\n')

fig,axs=plt.subplots(1,4,figsize=(16,4),layout='constrained')
for ax,(key,p) in zip(axs,fits.items()):
    pp=points(p);mm=xy(models[key],cone(*p[:3],n=3000))
    ax.scatter(pp[:,0],pp[:,1],c=np.arange(len(pp)),cmap='hsv',s=12,label='same 90 selected points')
    ax.plot(mm[:,0],mm[:,1],'k-',lw=1,label='fitted cone')
    ax.set(aspect='equal',xlabel='X',ylabel='Y',title=key+'\ncost %.5g'%results[key]['best']['cost'])
axs[0].legend(fontsize=7);fig.savefig(out/'real_data_refits.png',dpi=160);plt.close(fig)

# At fixed historical geometry, compare optical changes to residual scale.
p0=np.array([base[k] for k in ['Theta_axis_fitted','Phi_axis_fitted','Lambda_fitted','dc_fitted','fa_fitted','fb_fitted']])
fixed={k:cost(p0,q) for k,q in models.items() if k in results}
(out/'fixed_parameter_costs.json').write_text(json.dumps(fixed,indent=2)+'\n')
