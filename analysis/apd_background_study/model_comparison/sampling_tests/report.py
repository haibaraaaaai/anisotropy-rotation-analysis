from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import isotonic_regression
import fit as f
R=f.R;r=f.r;z=f.Z;v=json.load(open(R/'fits.json'));prep=json.load(open(R/'preparation_checks.json'));rows=[];diagnostic={}
LABEL={'none':'No background','square':'Additive background','richard_product':'Richard four-C','complex':'Additive + bounded interference'}
COLOR={'none':'#999999','square':'#287c9a','richard_product':'#9874aa','complex':'#b53f41'}
fig,axs=plt.subplots(2,2,figsize=(11,10),layout='constrained')
for col,key in enumerate(f.KEYS):
 tr0=r.Z[key+'_train']/z['scale'];te0=r.Z[key+'_test']/z['scale'];tm=tr0.sum(0).min();u0=r.fractions(tr0,64);w0=z[key+'_weighted_weights'];mx=tr0.sum(0).max()
 for method in f.METHODS:
  tr=z[key+'_'+method+'_train']/z['scale'];te=z[key+'_'+method+'_test']/z['scale'];u=r.fractions(tr,64)
  for model in f.MODELS:
   b=v[key][method][model]['best'];p=np.array(b['p']);pred,s,c,bg,d,mn=r.forward(p,model,tm,b['direction'],u,8192,True);th=np.rad2deg(np.arccos(abs(d[:,2])))
   common=None
   if method!='phase':
    # Same first reference point for weighted/uniform constructions; no extra alignment.
    yp=r.forward(p,model,tm,b['direction'],u0,8192);common=float(np.sqrt(np.sum(((yp-te0)/te0.sum(0))**2*w0[None,:])/(4*w0.sum())))
   rows.append(dict(interval=key,reference=method,model=model,reference_valid_for_primary_comparison=method!='phase',heldout_objective_rms=b['test']['objective_channel_rms'],common_arc_weighted_heldout_rms=common,heldout_xy_rms=b['test']['xy_rms'],bg_percent_original_Tmax=100*bg.sum()/mx if model!='richard_product' else None,theta_min=float(th.min()),theta_max=float(th.max()),alpha=float(p[3]) if model=='richard_product' else float(np.exp(2*p[3])),success=b['success']))
   if method=='uniform':axs[0,col].plot(*r.xy(pred).T,color=COLOR[model],lw=1.7,ls='--' if model=='none' else '-',label=LABEL[model])
  if method=='uniform':axs[0,col].scatter(*r.xy(te).T,s=12,color='#223340',label='Held-out reference')
 for method,label,color,ls in [('weighted','Original averaged track','#666666','--'),('uniform','Uniform final-arc reference','#287c9a','-'),('phase','Provisional phase bins — rejected','#b53f41','-')]:axs[1,col].plot(*r.xy(z[key+'_'+method+'_train']).T,color=color,ls=ls,lw=1.8,label=label)
 for row in [0,1]:
  ax=axs[row,col];ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y');ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=7,loc='lower left')
 axs[0,col].set_title(key+' · uniform final-arc reference fits');axs[1,col].set_title(key+' · effect of constructing the reference')
 # Per-cycle plane check: separate fits do not resolve unstable phase guide.
 cloud=z[key+'_guide_cloud'].reshape(-1,128,3);changes=[]
 for d in cloud:
  _,vv=np.linalg.eigh((d-d.mean(0)).T@(d-d.mean(0)));axis=vv[:,0];e=d[0]-(d[0]@axis)*axis;e/=np.linalg.norm(e);e2=np.cross(axis,e);a=np.unwrap(np.arctan2(d@e2,d@e));a=(a-a[0])/(a[-1]-a[0]);b=isotonic_regression(a).x;b=(b-b[0])/(b[-1]-b[0]);changes.append(float(np.max(abs(a-b))))
 diagnostic[key]=dict(per_cycle_plane_phase_adjustment_median=float(np.median(changes)),per_cycle_plane_phase_adjustment_p95=float(np.percentile(changes,95)))
fig.savefig(R/'sampling_comparison.png',dpi=180);plt.close(fig)
with (R/'summary.csv').open('w') as ff:w=csv.DictWriter(ff,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(R/'additional_guide_checks.json').write_text(json.dumps(diagnostic,indent=2))
# Check nested primary training fits and common objective denominators.
for key in f.KEYS:
 for method in ['weighted','uniform']:
  a=v[key][method];assert a['complex']['best']['cost']<=a['square']['best']['cost']+1e-7
  assert all(a[m]['best']['test']['min_pred']>0 for m in f.MODELS)
tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm]{geometry}\usepackage{fontspec,amsmath,graphicx,booktabs}
\setmainfont{Latin Modern Roman}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
{\LARGE Reference distribution tests}\par 6 October 2026

\textbf{Result.} Uniform arc resampling of the final averaged reference and arc weighting of the original points give similar physical fits. The simple direct-inversion sphere-circle guide does not provide a reliable monotone phase coordinate here: making it monotone largely removes the inner loop from the averaged reference. This is a reference-construction failure, not evidence that a physical background model fits better.

\includegraphics[width=\linewidth,height=.82\textheight,keepaspectratio]{sampling_comparison.png}
\newpage
{\Large Primary comparison: weighting versus resampling}

Two tests isolate the uneven-density concern. Weighted: preserve all original intensities and use trapezoidal arc weights. Uniform: interpolate the four channels together to 128 equally spaced locations along the final training-average arc; apply the identical training-derived interpolation map to the held-out average. Simulation is always matched using actual final-reference arc coordinates, one cyclic offset and both directions. No flexible phase warp.

\[
J=\sum_k w_k\sum_i\left[\frac{I_{i,k}^{\rm pred}-I_{i,k}^{\rm obs}}{T_k^{\rm obs}}\right]^2,
\quad w_k\propto\frac{\Delta s_{k-1}+\Delta s_k}{2}.
\]
For the uniform reference, bin weights are equal. To compare the two fitted models fairly, the table below evaluates both on the \emph{same original held-out intensities with the same arc weights}. Their starting reference point is identical; no extra held-out alignment is fitted.

\begin{tabular}{llrrrr}\toprule
Window & Model & Weighted fit & Uniform fit & Uniform BG/\(T_{\max}\) & Uniform \(\theta\)\\\midrule
'''
for key in f.KEYS:
 for model in f.MODELS:
  a=next(x for x in rows if x['interval']==key and x['model']==model and x['reference']=='weighted');b=next(x for x in rows if x['interval']==key and x['model']==model and x['reference']=='uniform');label={'none':'None','square':'Additive','richard_product':'Richard four-C','complex':'Additive + interference'}[model];bg='---' if b['bg_percent_original_Tmax'] is None else f"{b['bg_percent_original_Tmax']:.1f}\\%"
  tex+=f"{key} & {label} & {a['common_arc_weighted_heldout_rms']:.5f} & {b['common_arc_weighted_heldout_rms']:.5f} & {bg} & {b['theta_min']:.1f}--{b['theta_max']:.1f}$^\\circ$ \\\\\n"
tex+=r'''\bottomrule\end{tabular}

Both preserve the measured two-loop trajectory. The residual mismatch remains after removing uneven-density weighting. This supports using the simpler uniform-final-arc reference for presentation and retaining arc weighting as a control. It does not prove the physical models or angles correct.

All four models retain fixed excitation with one brightness scale, original optics/calibration, and the same background budget in original intensity units. The additive budget is held fixed at 95\% of the original training minimum across reference methods, rather than changing the budget when resampling changes a sampled minimum. Physical curves are not used to replace measured points. Mean-of-channel values are retained before calculating anisotropy.

Richard uses the stable equivalent parameterization \(I_i=\alpha f_i+K_i\sqrt{f_i}\), with \(\alpha=a_0^2\), \(K_i=a_0C_i\). Under these new arc-weight objectives the selected alpha values are finite (about 0.0192/0.00511 for uniform 30/10 s). The previously reported zero-brightness tendency therefore depends on weighting; the structural degeneracy of the unconstrained model has not been removed.

Nine base starts per model/window/reference cover both directions and four offsets, plus nested-model starts. Selected primary fits converge and have positive predicted channels. Local optimality is not a global guarantee. Uniformity refers to integrated arc on the reference interpolant; adjacent straight chord lengths need not be identical at curved sections.
\newpage
{\Large Direct-inversion sphere-circle phase guide}

Raw calibrated channels from 30--31 s and 10--11 s were checked: \textbf{zero samples outside the nominal inversion domain} in either 250,000-sample window. The user's expectation about the range is supported on these data. A 41-sample channel smoother constructs the guide only; it does not replace the intensities being averaged.

For each ordered cycle, invert guide anisotropy to folded theta and unwrapped phi. Fit one provisional least-squares plane using equally weighted training cycles (128 time-decimated guide directions each). Project directions into that plane's basis and unwrap their polar angle. This circle/plane sets a coordinate only; neither its cone geometry nor its projected points enter the physical fit as measured data.

The coordinate reverses substantially within individual traversals. Normalizing the first-to-last angle span to one traversal does not remove these reversals. A monotone least-squares (isotonic) guide was tried as a deliberately flagged sensitivity, while preserving acquisition order. Raw channel means in the resulting contiguous bins were then averaged with equal cycle weight. No cycles were removed (304 total, 152/152 split; 217 total, 109/108 split).

\begin{tabular}{lrr}\toprule
Guide diagnostic & 30--31 s & 10--11 s\\\midrule
Median maximum monotone adjustment & 15.0\% & 21.7\%\\
95th percentile maximum adjustment & 28.3\% & 39.1\%\\
Median adjustment with separate plane per cycle & 15.4\% & 22.4\%\\\bottomrule
\end{tabular}

These are fractions of each normalized provisional traversal, not angular measurement errors. The median raw plane-angle span is only about 0.53--0.54 turns per detected track. The uncorrected anisotropy winds once around the origin, so its half-angle inversion does not yield a closed directed upper-hemisphere trajectory. This is a limitation of this simple uncorrected guide, not an assertion that phase guidance can never work.

The guide is not one-to-one through the inner loop. Flattening its reversals concentrates different trajectory sections into a small phase region; averaging then largely erases the inner loop (bottom row of figure). This occurs even though every raw intensity sample is retained in some bin and no intensity is replaced by a circle prediction.

\textbf{Disposition.} Fits on the phase-binned references were completed as diagnostics, saved in fits.json/summary.csv, and explicitly excluded from the primary comparison. They should not be interpreted as corrected rod angles or improved fits. The 10 s complex phase-reference run also exhausted its refinement budget; no conclusion relies on its optimum. A separate best plane per cycle did not cure the guide-coordinate instability. No extra fitted phase flexibility or new inverse-circle constraints were introduced.

Files: prepare.py (raw reference construction), fit.py (24 model/reference/window fits), report.py, references.npz, preparation\_checks.json, additional\_guide\_checks.json, fits.json and summary.csv. Reproduce using numpy/scipy/matplotlib/nptdms; XeLaTeX builds this report. No commit or push.
\end{document}
'''
(R/'report.tex').write_text(tex)
(R/'README.md').write_text('''# Sampling/weighting tests (6 October 2026)

Three references: (1) original measured bins with arc-length quadrature weights; (2) final-training-track equal-arc interpolation of all four channels, same map applied to test; (3) raw-data direct-inversion plane-angle guide with isotonic monotonicity imposed, diagnostic only. Third reference FAILS: large angle backtracking and 15/22% median maximum monotone adjustment; averaged inner loop is lost. No raw intensity replacement by projected directions, no data outside nominal range in either 250k window, all cycles retained. Individual-cycle plane check also fails. Phase fits are retained solely as diagnostic outputs; do not promote their recovered angles.

All fits use corrected final-reference arc matching, one cyclic offset, both directions, fixed optics, excitation and background budget. Richard product parameterization permits alpha=0 closure but selected new fits are finite. No fitted correspondence flexibility. Original equal-bin cost and new uniform-arc cost are different; compare both new methods on common original held-out data with arc weights (report.py computes this). Phase reference excluded from common score because its guide already fails feature preservation. Shift-invariant or nearest-point matching is not used in the fit.

prepare.py requires raw APD file/nptdms, honors APD_TDMS_PATH. fit.py uses cached references and previous fits as starts, report.py writes figures/CSV/LaTeX; compile report.tex in this directory. Old cached arc_matching data provide original gate boundaries. All samples 250kHz, windows30–31 and10–11. No push.
''')
print('Common held-out scores:')
for a in rows:
 if a['reference']!='phase':print(a['interval'],a['reference'],a['model'],a['common_arc_weighted_heldout_rms'])
