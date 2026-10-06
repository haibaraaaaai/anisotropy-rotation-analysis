"""Controlled pupil/weighting comparisons. Run from repo root with --mask PATH."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pupil_model import *

ap=argparse.ArgumentParser();ap.add_argument('--mask',required=True);args=ap.parse_args()
out=Path('analysis/pupil_comparison/results');out.mkdir(parents=True,exist_ok=True)
mask,meta=load_mask(args.mask)
pix=np.argwhere(mask==meta['blocked_value'])
qscale=meta['mask_scale_mm_per_px']/meta['outer_radius_mm']*1.3
req=np.sqrt(len(pix)/np.pi)*qscale
theta,phi=np.meshgrid(np.arange(5.,86.,1.),np.arange(0.,360.,5.),indexing='ij')
d=directions(theta,phi)
models={};checks={}
for radius in [240,480]:
    print('BFP grid radius',radius,flush=True)
    x,y,fields,area=basis(radius=radius)
    f=fields();rho=np.hypot(x,y)
    qs={f'annulus_{na}':response(f,rho>=na,area) for na in [0.38,0.39,req]}
    for rot in [0,45,90,135]:
        qs[f'mask_{rot}']=response(f,~mask_blocked(x,y,args.mask,rot),area)
    qs['legacy_weight_mask']=response(fields(legacy=True),~mask_blocked(x,y,args.mask),area)
    qs['legacy_weight_annulus']=response(fields(legacy=True),rho>=0.38,area)
    qs['fourkas_038']=analytic_q(fourkas_abc(.38))
    qs['fourkas_039']=analytic_q(fourkas_abc(.39))
    # Independent homogeneous-medium limit: numerical solid-angle quadrature
    # must reproduce the analytic annular Fourkas anisotropy.
    xh,yh,fh,ah=basis(radius=radius,no=1.33)
    qh=response(fh(),np.hypot(xh,yh)>=.38,ah)
    checks[f'homogeneous_max_xy_error_R{radius}']=float(np.max(np.linalg.norm(xy(qh,d)-xy(qs['fourkas_038'],d),axis=-1)))
    checks[f'pair_sum_max_relative_R{radius}']=float(max(np.max(np.abs(q[0]+q[1]-q[2]-q[3]))/np.max(np.abs(q)) for q in qs.values()))
    if models:
        checks['grid_convergence_max_xy_change']={k:float(np.max(np.linalg.norm(xy(q,d)-xy(models[k],d),axis=-1))) for k,q in qs.items()}
    models=qs

summaries={};maps={}
comparisons=[('hole_shape_equal_area','mask_0',f'annulus_{req}'),
 ('hole_shape_038','mask_0','annulus_0.38'),('hole_shape_039','mask_0','annulus_0.39'),
 ('radius_038_vs_039','annulus_0.38','annulus_0.39'),
 ('interface_vs_notebook','annulus_0.38','fourkas_038'),
 ('combined_vs_notebook','mask_0','fourkas_038'),
 ('weighting_only_annulus','annulus_0.38','legacy_weight_annulus')]
for label,truth,assumed in comparisons:
    pt=xy(models[truth],d);pa=xy(models[assumed],d)
    tr,pr,u=invert_radial(pt,abc_from_q(models[assumed]))
    te=tr-theta;pe=(pr-phi+90)%180-90;delta=np.linalg.norm(pt-pa,axis=-1)
    def stats(a):return dict(rms=float(np.sqrt(np.nanmean(a*a))),max_abs=float(np.nanmax(np.abs(a))),median_abs=float(np.nanmedian(np.abs(a))))
    summaries[label]=dict(delta_xy=stats(delta),theta_error_deg=stats(te),phi_error_deg=stats(pe),
                          invalid_theta_fraction=float(np.mean(~np.isfinite(tr))),
                          theta_range_deg=[5,85],phi_range_deg=[0,355])
    maps[label]=(delta,te,pe,u)

# Dependence on pupil registration to analyser axes, with the same interface.
registration={}
for rot in [0,45,90,135]:
    tr,pr,u=invert_radial(xy(models[f'mask_{rot}'],d),abc_from_q(models['annulus_0.38']))
    registration[rot]=dict(theta_rms_deg=float(np.sqrt(np.nanmean((tr-theta)**2))),
                           theta_max_deg=float(np.nanmax(np.abs(tr-theta))))

result=dict(mask_source='haibaraaaaai/thesis@7576b50:analysis/image_mirror_binary.tif',
            metadata=meta,mask_equivalent_inner_NA=req,
            mask_centroid_offset_NA=((pix.mean(axis=0)-meta['mask_center_xy'])*qscale).tolist(),
            assumptions='Propagating dipole far field; real Fresnel amplitudes; flat water/oil interface; uniform Cartesian BFP power integration; no background/noise; exact mask registration not experimentally confirmed.',
            checks=checks,comparisons=summaries,registration=registration)
(out/'pupil_summary.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez(out/'pupil_response_matrices.npz',**models)

fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
extent=[-meta['mask_center_xy'][0]*qscale,(mask.shape[0]-meta['mask_center_xy'][0])*qscale,
        -meta['mask_center_xy'][1]*qscale,(mask.shape[1]-meta['mask_center_xy'][1])*qscale]
ax[0,0].imshow(mask.T,origin='lower',extent=extent,cmap='Greys',vmin=0,vmax=255)
for rr,col in [(.38,'tab:blue'),(.39,'tab:orange')]:ax[0,0].add_patch(plt.Circle((0,0),rr,fill=False,color=col,label=f'NA {rr}'))
ax[0,0].set(xlabel='NA x',ylabel='NA y',title='Measured blocked pupil region');ax[0,0].legend()
for a,label in zip([ax[0,1],ax[0,2],ax[1,0],ax[1,1],ax[1,2]],
                   ['hole_shape_038','radius_038_vs_039','interface_vs_notebook','combined_vs_notebook','weighting_only_annulus']):
    te=maps[label][1];lim=max(np.nanmax(np.abs(te)),.01)
    im=a.pcolormesh(phi,theta,te,cmap='RdBu_r',vmin=-lim,vmax=lim,shading='auto')
    a.set(xlabel='Input azimuth (deg)',ylabel='Input polar angle (deg)',title=label.replace('_',' '))
    fig.colorbar(im,ax=a,label='Recovered minus true theta (deg)')
fig.suptitle('Noiseless controlled comparisons; invalid inversions left blank')
fig.savefig(out/'pupil_bias_maps.png',dpi=160);plt.close(fig)

# Compare shape on representative fixed cones, including the baseline fit if present.
geoms=[(15,20,40),(40,20,25),(70,20,30)]
if (out/'baseline.json').exists():
    b=json.loads((out/'baseline.json').read_text());geoms.append(tuple(b[k] for k in ['Theta_axis_fitted','Phi_axis_fitted','Lambda_fitted']))
fig,axs=plt.subplots(1,len(geoms),figsize=(4*len(geoms),4),layout='constrained');orbit_metrics=[]
for a,g in zip(axs,geoms):
    dd=cone(*g);p0=xy(models['mask_0'],dd)
    metrics={'geometry_deg':g}
    for key,label in [('mask_0','Measured hole + interface'),('annulus_0.38','Annulus .38 + interface'),('fourkas_038','Notebook annulus .38')]:
        pp=xy(models[key],dd);a.plot(pp[:,0],pp[:,1],label=label,lw=1.5)
        metrics[key+'_rms_xy']=float(np.sqrt(np.mean(np.sum((pp-p0)**2,axis=1))))
    orbit_metrics.append(metrics);a.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel='X',ylabel='Y',title='Axis %.1f°, %.1f°; cone %.1f°'%g)
axs[0].legend(fontsize=7);fig.savefig(out/'cone_shapes.png',dpi=160);plt.close(fig)
(out/'orbit_metrics.json').write_text(json.dumps(orbit_metrics,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
