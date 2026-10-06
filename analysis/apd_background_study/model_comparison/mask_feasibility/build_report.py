from pathlib import Path
import sys,json,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;ROOT=R.parent.parent
sys.path.insert(0,str(R));from free_run import signal,envelopes
from mask_run import predict as mask_predict,response
sys.path.insert(0,str(R.parent/'radial_field'));from run import pred
sys.path.insert(0,str(R.parent));from models import *
mr=json.loads((R/'mask_fits.json').read_text());fr=json.loads((R/'free_fits.json').read_text());old=json.loads((R.parent/'radial_field/fits.json').read_text());op=np.load(R/'mask_optics.npz');data={};computed={}
fig,axs=plt.subplots(2,3,figsize=(13.5,8.6),layout='constrained');fig2,ax2=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for row,(key,path,label) in enumerate([('30s','fitted_predictions.npz','30–31 s'),('10s','interval10/averages.npz','10–11 s')]):
 z=np.load(R.parent/path);tr=z['train'];te=z['test'];tm=tr.sum(0).min();scale=fr[key]['scale'];target=tr/scale;data[key]={'train':tr.tolist(),'test':te.tolist()};x=np.arange(128)/128
 ax=axs[row,0];ax.plot(*xy(te).T,'.',color='.5',ms=3,label='Held-out data')
 for k,col,ls in [('None','#d45339','-'),('0','#237ba8','--')]:
  opts=tuple(op[f'{k}_{j}'] for j in range(3));y=mask_predict(np.array(mr[key][k]['p']),128,tm,opts);ax.plot(*xy(y).T,color=col,ls=ls,lw=1.5,label='Annulus .38' if k=='None' else 'Measured mask, 0°')
 ax.set(title=label+': mask refit');ax.legend(fontsize=8,loc='lower left')
 p=np.array(fr[key]['best']['p']);s=signal(p);bmin=p[20:];b=np.full(4,.025);c=target-s-b[:,None];rho=c/(2*np.sqrt(s*b[:,None]));assert np.max(np.abs(rho))<=1+1e-7
 # This is exact construction, not held-out prediction.
 witness=s+b[:,None]+2*np.sqrt(s*b[:,None])*rho;assert np.max(abs(witness-target))<1e-12
 ax=axs[row,1];ax.plot(*xy(s).T,color='#438cad',label='Rod-only cone');ax.plot(*xy(target).T,'.',color='.5',ms=3,label='Training data');ax.plot(*xy(witness).T,color='#d45339',lw=1,label='Constructed match (10% BG)');ax.set(title=label+': flexible-overlap construction');ax.legend(fontsize=8,loc='lower left')
 for ax in axs[row,:2]:ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y');ax.set_aspect('equal');ax.grid(alpha=.2)
 ax=axs[row,2]
 for j,ch in enumerate(['0°','90°','45°','135°']):ax.plot(x,rho[j],label=ch,lw=1.1)
 ax.axhline(1,c='.5',ls=':');ax.axhline(-1,c='.5',ls=':');ax.set(ylim=(-1.1,1.1),xlabel='Ordered trajectory fraction',ylabel='Required overlap ρⱼ',title=label+': overlap at 10% background');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
 # Keep previous cone/background fixed and ask what additional interference is required.
 po=np.array(old[key]['radial16']['best']['p']);yold=pred(po,128,tm)/scale;delta=target-yold
 for j,ch in enumerate(['0°','90°','45°','135°']):ax2[row,0].plot(x,delta[j],label=ch)
 ax2[row,0].set(title=label+': extra term at previous large-BG fit',xlabel='Ordered trajectory fraction',ylabel='ΔIⱼ / maximum total');ax2[row,0].legend(fontsize=8,ncol=2);ax2[row,0].grid(alpha=.2)
 ax2[row,1].plot(x,s.sum(0),label='Rod alone');ax2[row,1].plot(x,c.sum(0),label='Required signed interference');ax2[row,1].axhline(.1,label='Constant background',color='green',ls='--');ax2[row,1].plot(x,target.sum(0),'k',label='Measured total');ax2[row,1].set(title=label+': 10%-BG construction',xlabel='Ordered trajectory fraction',ylabel='Intensity / maximum measured total');ax2[row,1].legend(fontsize=8);ax2[row,1].grid(alpha=.2)
 computed[key]=dict(feasible_budget=fr[key]['best']['bg_fraction_max'],max_abs_rho_10=float(np.max(abs(rho))),bg_channels_10=b.tolist(),extra_term_max_perchannel=float(abs(delta).max()),extra_total_range=[float(delta.sum(0).min()),float(delta.sum(0).max())])
fig.suptitle('Measured mask versus a deliberately flexible interference test\nThe constructed match allows a different overlap at every point and channel; it is NOT a stationary-field fit.',fontsize=13)
fig.savefig(ROOT/'deliverables/mask_and_flexible_interference.png',dpi=170)
fig2.suptitle('What the required correction looks like — diagnostic constructions, not identified physical fields',fontsize=12);fig2.savefig(R/'required_corrections.png',dpi=150)
# Grid convergence at fixed parameters, avoiding another optimization confound.
q480=response(0,radius=480);q320=tuple(op[f'0_{j}'] for j in range(3));p=np.array(mr['30s']['0']['p']);tm=np.array(data['30s']['train']).sum(0).min();conv=float(np.max(abs(mask_predict(p,128,tm,q480)-mask_predict(p,128,tm,q320))))
print('grid max channel difference',conv);print(json.dumps(computed,indent=2));(R/'computed.json').write_text(json.dumps(dict(results=computed,mask_grid_max_channel_difference=conv),indent=2))
notes='''# Mask, background feasibility, and source review — 5 October 2026

## Scope and fixed choices
Current positive-beta matrix and fixed gains retained. Same 30–31 s / 10–11 s averages and alternate-cycle split as preceding tests. Fixed cone, one a0 per interval with a=a0 sin(theta), 16 monotone mechanical-phase mapping parameters. The physical fit has uniform plus axial-dipole/radial pupil background modes and incoherent background. No claim of calibrated angles or global optimization.

## Measured mask
Used thesis_sources/analysis/image_mirror_binary.tif, previously obtained from thesis commit 7576b50. Used TIFF metadata axis ordering, center and physical scale, rather than resizing to NA_in=.38. Reference annulus remains .38–1.3. Mask rotations 0,45,90,135 degrees bracket uncertain registration; none is an experimentally established alignment. Correct Fresnel and BFP weighting used. Numerical integration recomputed signal response, background intensity Gram matrices, and signal/background overlaps together. The radial mode is orthogonalized to uniform modes over the accepted pupil; this spans the same three-mode space and keeps the power budget normalized. Per-channel mode cross terms are retained for the asymmetric mask.

Annular numerical forward calculation agrees with the previous analytic implementation to maximum absolute channel difference 4.7e-6 at the test parameters. At fitted mask parameters, grid-radius 320 vs 480 differs by the value recorded below. Mask refits change held-out XY RMS by less than about 0.0002, leaving backgrounds near 52%/55% of maximum total. This mask does not resolve the mismatch. This tests an amplitude mask at the pupil; it does not test defocus, scattering from hole edges, a shifted current alignment or finite detector clipping.

## Flexible interference: what was allowed
For each channel and ordered point, set I=S+B+C and require |C| <= 2 sqrt(S B). B is constant per channel; four B values are independent. S is still generated by one fixed cone with circular-excitation amplitude a0 sin(theta). C, however, is allowed to change independently at every point/channel. This is a deliberately relaxed feasibility problem, not an alternative stationary physical field model. It permits four independent analyzer-path backgrounds and arbitrary orientation-dependent phase/spatial overlap. Shared-pupil, common-polarization and temporal/optical field consistency are NOT enforced.

The scalar interference bound gives (sqrt(I)-sqrt(S))^2 <= B <= (sqrt(I)+sqrt(S))^2. For an entire trace, take the maximum lower bound and minimum upper bound per channel. We optimized cone, brightness and phase mapping to find feasible budgets; the minima found are not certified global minima. Feasible constructions need only about 2.7% total background relative to maximum training total in both intervals. Using an equal background of 2.5% of maximum total in each channel (10% overall) still satisfies every bound (max |rho| about .71/.68). The plotted exact match is constructed algebraically using rho=(I-S-B)/(2 sqrt(S B)); it is not predictive validation and would overfit if promoted to a fitted correction. The fitted cone theta range is about 31–41 degrees here, very different from the large-background physical-model fits. This demonstrates angle ambiguity, not a recovered-angle result.

The same cone needs about 2.8–2.9% background to satisfy the scalar bounds for the held-out average when overlaps are allowed to change again. That is a repeatability/feasibility observation, not held-out prediction. At the original large-background cone, retain S and background and replace the interference by C_required=I-S-B: the additional channel term is simply the previous residual. It is plotted separately. All required per-channel overlaps remain within their Cauchy bound (about .94/.95 at maximum magnitude), but no single stationary field producing them has been demonstrated.

Conclusion: 10% background is NOT ruled out by scalar interference energetics or the fixed-cone/excitation law. It failed in the previous restricted stationary-field model. Arbitrary corrections can reproduce the curve, but do not identify physical sources or angles.

## Bearing paper and SI: useful constraints, with limits
Sources: supplied manuscript.pdf, Methods Cell selection (printed pp 10–11), Results first orientation discussion; supplied supplementary-information (1).pdf, Methods S1–S3 and Notes S1–S4.

- The bearing paper selects near-optical-axis rotation with overlapping loops, explicitly fewer than 10% of rods, avoiding near-vertical/low-intensity portions. Its motor state/manipulation and selected geometry are not representative constraints for ordinary BFM traces. Do not transfer phi≈motor angle, a narrow theta range, or biological diffusion conclusions to this dataset.
- It already lists hook conformation/axis motion, imperfect polarization optics, cell scattering or circular dichroism, and deviations from pure dipole emission as possible causes of nonideal trajectories. Background alone is not the only allowed explanation.
- SI S1 describes a drilled mirror close to an inaccessible objective BFP, residual scattering from hole walls, circular-polarization adjustment with a rotating analyzer, and detector alignment by maximizing four APD signals. It specifically warns about angularly varying mirror retardance when light is not collimated. The present pupil-mask test is narrower than these real optical effects.
- SI p5 gives relative transmission/extinction measurements after Berek adjustment. These constrain historical optical quality but are not a current calibration or a background measurement. Gains fitted by pair-sum balance can absorb transmission errors (Note S1).
- SI S3 / Fig S3: fixed rods, laser-power steps, shot noise at high frequencies and low-frequency pink drift; phi noise about 1 to 0.4 degrees over 1 Hz–125 kHz as summed APD voltage rises about 1 to 12 V. This directly motivates PSD/Allan analysis of the user's fixed-rod files, not treating flat plotted plateaus as zero noise. It does not bound coherent background.
- SI S2 explicitly uses an empirical effective NA and clipping and does not claim high-accuracy absolute theta. S4 simulations of mask/Fresnel are historical calculations; their accuracy is not assumed given the identified BFP-weighting issue. The present corrected calculation independently tests mask effects.
- SI S5 manual rotation validates azimuth; re-centering/re-focusing errors remain. It is not a known-theta calibration. Heating estimates/power tests depend on those rods, powers and geometry; no universal no-heating assumption transferred.

## Polarcam draft and supporting documents: retained separately
Sources: Polarcam Manuscript V2 2026-09-08 RB.docx, Results setup/validation and Methods; SI outline.docx sections 1–3; RB Figs V2.docx.

- The draft distinguishes widefield camera and focused APD illumination. It attributes substantial camera background to objective/optical reflections and changes detection compensation from Berek to waveplates. Same four-channel algebra does not mean same calibration/background as historical APDs, or even between the camera and APD branches of Polarcam.
- SI section 1 describes camera background phase varying spatially, axial modulation to suppress the cross term, then lateral translation/minimum projection to measure the stationary intensity background. These are useful source-separation principles, not APD background estimates for our data. Sample-associated scattering need not be removed by the same procedure.
- Draft numerical background fractions and several validation numbers are placeholders. The main text/SI use different fringe displacement/timing descriptions. The SI itself describes residual modulation at 0.6 ms and better cancellation at 1.2 ms; the figure notes question exposure averaging. Do not treat exact cancellation as established for every exposure.
- SI annular collection mapping uses .39–1.3 and n-specific coefficients; this is not justification to change historical APD .38. Keep coefficient conventions separate. Its 'negligible' Fresnel/jagged-hole statement is a draft claim, not a calibration of our experiment.
- Statistical theta validation uses freely tumbling rods in glycerol (main draft: four rods about 2 micrometers from glass); it does not supply exact angles or a background bound for attached APD rods. Stationary-rod precision likewise does not establish absolute angular accuracy.
- RB Figs is a working figure/comment document, useful for proposed tests and unresolved exposure/power questions, not independent numerical evidence.

## Recommended interpretation
Measured mask geometry is a low-priority explanation for the current mismatch. The physically discriminating next step is to constrain the field family (and excitation/dipole/calibration assumptions), not to declare a large background necessary or fit an arbitrary function. Shared-field tests across intervals and fixed/constant-theta controls can constrain this. The current large-background solution is one model-dependent explanation.
'''
(R/'review_notes.md').write_text(notes)
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s,outputs=None):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':outputs or []})
md(notes)
for path in [ROOT/'deliverables/mask_and_flexible_interference.png',R/'required_corrections.png']:
 code('# Saved figure', [{'output_type':'display_data','data':{'image/png':base64.b64encode(path.read_bytes()).decode(),'text/plain':['Scientific comparison']},'metadata':{}}])
code((R.parent/'models.py').read_text())
code('data='+repr(data)+'\nmask_results='+repr(mr)+'\nfree_results='+repr(fr)+'\ncomputed='+repr(computed)+'\n')
code('''# Reconstruct the relaxed 10% witness. This is NOT a stationary field fit.
for key,v in free_results.items():
    target=np.array(data[key]['train'])/v['scale']
    p=np.array(v['best']['p'])
    d=cone(p[:3],mechanical_phase(p,128,16,-1))
    S=np.exp(2*p[19])*np.sum(d[:,:2]**2,axis=1)[None,:]*q_channels(d)
    B=np.full(4,.025)
    C=target-S-B[:,None]
    rho=C/(2*np.sqrt(S*B[:,None]))
    assert np.max(np.abs(rho)) <= 1+1e-7
    assert np.allclose(S+B[:,None]+C,target)
    print(key, 'BG budget',B.sum(),'largest overlap',np.max(np.abs(rho)))
''')
for i,c in enumerate(cells):
 c['id']=f'mask-feasibility-{i}'
 if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':cells}
(ROOT/'deliverables/mask_feasibility_and_source_review.ipynb').write_text(json.dumps(nb,indent=1))
