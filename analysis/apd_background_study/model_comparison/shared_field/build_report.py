import run as r
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import base64,ast
R=r.R;D=r.ROOT/'deliverables';shared=json.loads((R/'fits.json').read_text());ctrl=json.loads((R/'controls.json').read_text())
plt.rcParams.update({'font.size':11})
fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
for i,k in enumerate(r.KEYS):
 data=r.TE[i];pts=r.xy(data)
 for j,cap in enumerate([2.,.1]):
  ax=axs[i,j];ax.scatter(*pts.T,s=12,color='.5',label='Held-out cycle average')
  for m,color in [(3,'#eb794b'),(5,'#007ea8')]:
   p=np.array(shared[f'm{m}_cap{cap}']['best']['p']);y=r.predict(p[20*i:20*i+20],p[40:],m)
   ax.plot(*r.xy(y).T,color=color,lw=2,label=f'Shared {m}-mode field')
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f'{"30–31" if i==0 else "10–11"} s: '+('background cap relaxed' if j==0 else 'background ≤10%'))
  ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
 ax=axs[i,2];t=np.arange(128)/128
 ax.plot(t,data.sum(0),color='.4',lw=2,label='Held-out total')
 for cap,style in [(2.,'-'),(.1,'--')]:
  p=np.array(shared[f'm5_cap{cap}']['best']['p']);y=r.predict(p[20*i:20*i+20],p[40:],5)
  ax.plot(t,y.sum(0),style,color='#007ea8' if cap==2 else '#bd3b2c',label='5-mode: '+('cap relaxed' if cap==2 else '10% cap'))
 ax.set(xlabel='Ordered trajectory fraction',ylabel='Total intensity / common reference maximum',title='Total-intensity constraint');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('One stationary background field shared across both intervals\nSeparate cone, ordered phase map and constant brightness scale per interval',fontsize=15)
png=D/'shared_stationary_field_comparison.png';fig.savefig(png,dpi=165);plt.close(fig)
# Compact table suitable for source notes and notebook.
rows=[]
for m,cap in [(3,2.),(5,2.),(3,.1),(5,.1)]:
 b=shared[f'm{m}_cap{cap}']['best'];st=b['stats'];rows.append(f'| {m} modes, {"relaxed" if cap==2 else "10%"} | {st["30s"]["test"]["xy_rms"]:.4f} | {st["10s"]["test"]["xy_rms"]:.4f} | {100*st["30s"]["background_fraction_max"]:.2f}% / {100*st["10s"]["background_fraction_max"]:.2f}% |')
table='| Shared field | Held-out XY RMS, 30–31 s | Held-out XY RMS, 10–11 s | BG/max total, 30–31 s / 10–11 s |\n|---|---:|---:|---:|\n'+'\n'.join(rows)
transfer=[]
for cap in [2.,.1]:
 c=ctrl[f'm5_cap{cap}'];transfer.append(f'- Cap {cap}: independent fields XY RMS {c["30s"]["test_xy_rms"]:.4f} / {c["10s"]["test_xy_rms"]:.4f}; field from 10–11 s frozen for 30–31 s gives {c["10s_to_30s"]["test_xy_rms"]:.4f}; reverse gives {c["30s_to_10s"]["test_xy_rms"]:.4f}. Target cone, phase map and a0 are refitted on target training cycles; target held-out cycles are used only for scoring.')
notes=f'''# Shared stationary fields — 5 October 2026

The matrix and gains remain unchanged. This test follows the previous mask and freely varying overlap experiments. It tests whether one fixed coherent field and one fixed noninterfering background can explain both intervals. It does not assume the independently optimized overlaps from the last experiment describe a physical field.

## Observation model and assumptions

At pupil position u and analyzer j:

E_s,j(u,chi)=a0 (d_x+i d_y) [d_x F_x,j(u)+d_y F_y,j(u)+d_z F_z,j(u)].

E_b,j(u)=sum_m b_m g_m,j(u), with the same complex b_m at every trajectory point AND both intervals.

I_j=integral |E_s,j+E_b,j|² du + B_inc,j.

F are the collected dipole fields with water–oil Fresnel transmission and corrected BFP weighting. Circular annulus NA .38–1.3, water n=1.33, oil n=1.51. No amplitude/phase is adjusted independently per trajectory point. Excitation magnitude remains a0 sin(theta); complex factor d_x+i d_y also retains excitation phase. a0 is real, positive and constant within each interval. Background is shared in absolute corrected-intensity units, not independently renormalized to each interval.

Three-mode basis: uniform x and y pupil fields plus collected z-dipole (radial) field. Five-mode basis adds the two collected transverse-dipole patterns, orthogonalized against the existing modes. These are additional spatial shapes, not two new time-varying functions or claims that these are the actual sources. All modes are vector fields projected onto the four analyzers, so channel overlaps and backgrounds are physically linked. Background modes are orthonormal in full vector pupil power; their summed four-channel coherent power is 2 sum|b_m|². Noninterfering background is a positive Stokes intensity with equal pair sums, shared across intervals. A component outside the signal-overlap subspace is observationally indistinguishable from incoherent intensity; do not identify that split with source location.

Each interval has 20 parameters: cone geometry (3), ordered phase mapping (16), a0 (1). The three-mode shared background has 9 parameters; the five-mode version 13. Total 49 versus 53 jointly fitted parameters for 1024 training channel values (128 bins ×4×2). The phase mapping describes nonuniform timing/order, not background variation. Counting averaged bins as independent observations for formal statistical tests would not be justified.

The 10% cap is imposed relative to the smaller of the two training maximum totals, ensuring it holds in both windows. The relaxed cap is 200% of that reference and is inactive in the reported solutions. Independent-window models use the same reference/cap for comparability.

## Results

{table}

The additional two modes modestly improve the fit but do not enable a good fit at 10% background. The relaxed shared five-mode result uses approximately 55% background. Its coherent share is {shared['m5_cap2.0']['best']['p'][41]:.4f}; the rest is the fitted noninterfering component. This is a model decomposition, not a measured separation. The fitted noninterfering polarization lies at its upper bound in these solutions, indicating continuing model pressure.

## Independent-field and transfer controls

{chr(10).join(transfer)}

Sharing one field produces only a modest penalty relative to fitting it separately. This reduces concern that the earlier agreement required completely different fields in each window; it does not prove the physical field is stationary. Cone and brightness freedom can absorb some differences. A fully frozen model predicting the other interval was not tested, because the trajectory and brightness may differ.

## Validation and limits

- Retained previous alternate-cycle training/held-out averages (304 cycles at 30–31 s; 217 at 10–11 s). No new cycle selection, calibration fit or matrix changes.
- Residual objective: channel error divided by measured total at each bin, identical weighting across model comparisons. Table reports XY distance RMS, not the minimized objective. Held-out values are cycle-averaged curve repeatability, not independent orientation ground truth.
- Multiple initializations, including geometries from the prior small-background feasible constructions. Repeated runs reach the reported minima; no global optimum certificate.
- Numerical three-mode prediction reproduces the previous analytic model within 1.2e-5 of the reference total. Sum of mode Gram matrices equals 2I to 7e-14. Pair-sum balance checked at about 1e-14.
- Pupil grid radius 240 versus 400 changes predicted channel intensity by less than 4e-5 of reference maximum total. Smaller than observed residuals.
- No exact angles established. Fixed cone, dipole emission, circular excitation, calibrated channels and stationary geometry of signal fields are assumptions.
- Five modes are a physically consistent restricted family, not all possible static spatial fields or detector-path backgrounds. Failure at 10% does not prove small background impossible.
- Synthetic angle-recovery robustness and shared geometry across intervals have not been tested here.

A useful next theoretical step is to express the most general stationary field through its overlaps with the dipole basis and its positive-semidefinite Gram constraints. That could distinguish limitations of these selected modes from a broader stationary-field limitation before adding more physical sources. Independent fixed/constant-theta control traces remain needed for experimental discrimination.
'''
(R/'notes.md').write_text(notes)
# Self-contained notebook: cache integrals and data, predictions reproducible without workspace imports.
src=(R/'run.py').read_text();tree=ast.parse(src);funcs=[]
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ['spherical','invsphere','bgparts','predict','residual','bounds','stats']:
  funcs.append(ast.get_source_segment(src,node))
models=(R.parent/'models.py').read_text();mtree=ast.parse(models)
basefunc=[ast.get_source_segment(models,node) for node in mtree.body if isinstance(node,ast.FunctionDef) and node.name in ['cone','mechanical_phase','stokes_background','xy']]
cache={'shared':shared,'controls':ctrl,'scale':r.SCALE,'bref':r.BREF,'train':[v.tolist() for v in r.TR],'test':[v.tolist() for v in r.TE],'optics':{str(k):[v.tolist() for v in val] for k,val in r.OPT.items()}}
def md(s):return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)}
cells=[md(notes),{'cell_type':'markdown','metadata':{},'source':['![Shared stationary fields](attachment:comparison.png)'],'attachments':{'comparison.png':{'image/png':base64.b64encode(png.read_bytes()).decode()}}},md('## Reproduce predictions from cached pupil integrals\nRequires NumPy, SciPy and Matplotlib. Arrays use one common intensity normalization. No raw TDMS or repository required.'),code('import json, numpy as np\nfrom scipy.special import softmax\nfrom scipy.optimize import least_squares\nimport matplotlib.pyplot as plt\nCACHE=json.loads('+repr(json.dumps(cache))+')\nKEYS=["30s","10s"]\nTR=[np.array(v) for v in CACHE["train"]]\nTE=[np.array(v) for v in CACHE["test"]]\nSCALE=CACHE["scale"]; BREF=CACHE["bref"]\nOPT={int(k):tuple(np.array(a) for a in v) for k,v in CACHE["optics"].items()}\n'+ '\n\n'.join(basefunc+funcs)),code('for key,item in CACHE["shared"].items():\n    m=int(key[1]); p=np.array(item["best"]["p"])\n    result=stats(p,m)\n    for window in KEYS:\n        assert abs(result[window]["test"]["xy_rms"]-item["best"]["stats"][window]["test"]["xy_rms"])<1e-10\n    print(key, [round(result[k]["test"]["xy_rms"],5) for k in KEYS])'),code('fig,axes=plt.subplots(1,2,figsize=(10,5))\nfor i,ax in enumerate(axes):\n    ax.scatter(*xy(TE[i]).T,s=10,color="gray",label="Held-out")\n    for cap in [2.,.1]:\n        p=np.array(CACHE["shared"][f"m5_cap{cap}"]["best"]["p"])\n        ax.plot(*xy(predict(p[20*i:20*i+20],p[40:],5)).T,label=f"Cap {cap}")\n    ax.set(xlim=(-1,1),ylim=(-1,1),xlabel="X",ylabel="Y",title=KEYS[i]);ax.set_aspect("equal");ax.legend()\nplt.show()'),md('## Optional local optimization\nThis refines a cached fit; it is not a fresh global search. Full multi-start and transfer-control source code is embedded below for provenance.'),code('RUN_REFINEMENT=False\nif RUN_REFINEMENT:\n    m=5;cap=.1;p=np.array(CACHE["shared"]["m5_cap0.1"]["best"]["p"])\n    lo,hi=bounds(m,cap)\n    fit=least_squares(residual,np.clip(p,lo+1e-9,hi-1e-9),args=(m,),bounds=(lo,hi),max_nfev=650,x_scale="jac")\n    print(stats(fit.x,m))'),code('ORIGINAL_FIT_SOURCE='+repr(src)+'\nORIGINAL_TRANSFER_SOURCE='+repr((R/'controls.py').read_text()))]
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for i,c in enumerate(cells):c['id']=f'shared-{i}'
(D/'shared_stationary_field_test.ipynb').write_text(json.dumps(nb,indent=1))
print(table);print('\n'.join(transfer))
