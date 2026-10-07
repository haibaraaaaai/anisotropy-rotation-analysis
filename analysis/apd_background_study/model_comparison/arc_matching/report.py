from pathlib import Path
import json,csv,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as r
R=Path(__file__).resolve().parent;v=json.load(open(R/'fits.json'));z=np.load(R/'inputs.npz');LABEL={'none':'No background','square':'Additive background','richard4':'Richard four-C'};COL={'arc':'#bb493e','flex':'#287c9a'}
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
fig,axs=plt.subplots(3,2,figsize=(10,12),layout='constrained');fig2,bxs=plt.subplots(3,2,figsize=(11,9),layout='constrained');rows=[];preds={}
for col,key in enumerate(r.KEYS):
 tr=z[key+'_train']/z['scale'];te=z[key+'_test']/z['scale'];tm=tr.sum(0).min();mx=tr.sum(0).max();preds[key]={}
 for row,model in enumerate(r.MODELS):
  ax=axs[row,col];ax.plot(*r.xy(tr).T,color='.65',lw=1,label='Training average');ax.scatter(*r.xy(te).T,s=8,color='#283340',label='Held-out average')
  for kind in ['arc','flex']:
   b=v[key][kind+'_'+model]['best'];p=np.array(b['p']);y,s,c,bg,d,mn=r.forward(p,model,tm,kind,b['direction'],8192,parts=True);th=np.rad2deg(np.arccos(abs(d[:,2])))
   preds[key][kind+'_'+model]=dict(pred=y.tolist(),signal=s.tolist(),correction=c.tolist(),background=bg.tolist(),theta=th.tolist(),p=p.tolist(),direction=b['direction'])
   ax.plot(*r.xy(y).T,color=COL[kind],lw=1.7,ls='-' if kind=='arc' else '--',label='Equal arc + offset' if kind=='arc' else 'Flexible phase (16)')
   ax.scatter(*r.xy(y)[::16].T,s=17,color=COL[kind],marker='x')
   bxs[row,col].plot((np.arange(128)+.5)/128,th,color=COL[kind],ls='-' if kind=='arc' else '--',label=kind)
   sc=b['test'];rows.append(dict(interval=key,model=model,matching=kind,channel_rms=sc['channel_rms'],xy_rms=sc['xy_rms'],total_rms=sc['total_rms'],bg_percent_Tmax=100*bg.sum()/mx if model!='richard4' else None,theta_min=sc['theta'][0],theta_max=sc['theta'][1],success=b['success'],direction=b['direction']))
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f'{"30–31" if key=="30s" else "10–11"} s · {LABEL[model]}');ax.set_aspect('equal');ax.grid(alpha=.15)
  if row==0:ax.legend(fontsize=8,loc='lower left')
  bxs[row,col].set(ylim=(0,90),ylabel='Folded θ (degrees)',xlabel='Observed ordered-track fraction',title=key+' · '+LABEL[model]);bxs[row,col].grid(alpha=.2)
  if row==0:bxs[row,col].legend()
fig.suptitle('Same calibrated bins and intensity cost; only correspondence changes',fontsize=13);fig.savefig(R/'arc_comparison.png',dpi=180);fig.savefig(R/'arc_comparison.svg');fig2.savefig(R/'angles.png',dpi=180);plt.close('all')
with (R/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'predictions.json').write_text(json.dumps(preds,indent=2))
# Regression checks: positivity and nested zero-background training baseline.
for key in r.KEYS:
 for kind in ['arc','flex']:
  base=v[key][kind+'_none']['best']['cost']
  for model in r.MODELS:
   best=v[key][kind+'_'+model]['best'];assert best['success'] and best['train']['min_dense_intensity']>0
   assert best['cost']<=base+1e-10
# Known model simulation: reverse traversal and cyclic shifts preserve full cycle, not nearest neighbours.
p=np.array([.7,.3,.5,-.8,.17,.25,.2,-.3]);tm=.4
for direction in [-1,1]:
 y=r.forward(p,'square',tm,direction=direction,density=8192);q=p.copy();q[4]+=1
 np.testing.assert_allclose(y,r.forward(q,'square',tm,direction=direction,density=8192),atol=1e-12)
 assert np.isfinite(y).all() and (y>0).all()
summary='''# Ordered arc-length matching comparison — 6 October 2026

Rebuilt 30–31 s / 10–11 s averages from raw recording: 304 / 217 complete traversals, same original gate boundaries. Smoothed calibrated-channel X/Y supplies arc-length bin edges; actual unsmoothed calibrated channels are averaged, equal weight per traversal. Alternating-cycle training/held-out split. No dynamic time warping, no independent nearest-neighbour assignment.

Three models: none, original equal-pair-sum additive (3 parameters, BG cap 95% of minimum training total), exact Richard four constant signed C_i and no additive term. One brightness a0 per interval with circular excitation. Same NA .38–1.3, Fresnel/BFP coefficients, historical gains and T. Matching: dense full predicted measured anisotropy curve, cumulative ordered arc length, one fractional cyclic offset, both directions searched. 5/8/9 parameters; flexible controls 20/23/24. No plotting in objective. Fits use 2048 segments, checked at 4096 and 8192.

Objective sum [(predicted channel - observed channel)/observed total]^2, no noise weighting change. Additional invalid-candidate penalty enforces positive predicted channels; selected predictions all strictly positive and penalty zero. Arc: 20/21/21 starts per window; flexible: 9/10/10, including arc-derived start. All selected runs converge and recurring starts agree; global optimality is not certified. Zero-background seeds included in expanded models; training costs obey nesting.

Findings: arc matching gives consistently larger held-out channel and XY errors than flexible matching. Both corrections beat the zero-background baseline under both correspondences. Additive arc matching improves total-intensity error relative to flexible despite poorer full-channel/XY agreement, illustrating the trade-off. Angle estimates also change. This is a correspondence-sensitivity result, not proof of more accurate physical angles for either method.

Within-bin check uses the empirical distribution of fractional calibrated arc positions from raw sample timing, with equal cycle weights and 32 quantile nodes per bin. Refit using those averaged predictions; this is conditional on the smoothed guide as the latent arc coordinate, not exact independent orientation truth. Largest pointwise theta change <0.024 degrees; channel RMS changes <0.000012. Increasing curve density 2048 to 8192 changes predicted intensities by at most ~1.02e-6 of measured Tmax. Neither explains correspondence differences. Theta is folded 0–90, evaluated at bin centres; not per-bin independently inverted truth.

Files: run.py (preparation/fits/checks), report.py, inputs.npz, fits.json, predictions.json, summary.csv, plots, report.tex/PDF. Existing results retained, no commit or push. Reproduce with numpy scipy matplotlib nptdms and XeLaTeX for PDF; run.py then report.py; compile report.tex in this directory. Optional APD raw path is currently repo files3_first40s.tdms.
'''
(R/'README.md').write_text(summary)
tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm]{geometry}
\usepackage{fontspec,amsmath,graphicx,booktabs,array}
\setmainfont{Latin Modern Roman}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
{\LARGE Ordered arc-length matching}\hfill 6 October 2026

Three models, two intervals; fixed optics and intensity objective. Equal-arc matching uses one cyclic offset and tests both directions, preserving the order through both loops. Flexible controls retain the 16 phase parameters. Both are fitted anew to the same calibrated bins.

\includegraphics[width=\linewidth,height=.84\textheight,keepaspectratio]{arc_comparison.png}
\newpage
{\Large Quantitative comparison}

Errors below are held-out scores. Channel RMS is dimensionless: each channel residual is divided by the observed total intensity at that bin. XY RMS is an additional diagnostic, not the fitted objective.

\begin{tabular}{llrrrr}\toprule
Window & Model / matching & Channel RMS & XY RMS & BG/\(T_{\max}\) & \(\theta\) range\\\midrule
'''
for a in rows:
 bg='---' if a['bg_percent_Tmax'] is None else f"{a['bg_percent_Tmax']:.1f}\\%"
 tex+=f"{a['interval']} & {LABEL[a['model']]} / {a['matching']} & {a['channel_rms']:.5f} & {a['xy_rms']:.5f} & {bg} & {a['theta_min']:.1f}--{a['theta_max']:.1f}$^\\circ$ \\\\\n"
tex+=r'''\bottomrule\end{tabular}

Richard's model does not infer physical background power; a dash does not mean zero background. Its effective correction is \(C_i\sqrt{S_i}\), with one constant signed \(C_i\) per channel per interval.

\textbf{Result.} Both background hypotheses improve substantially over no background. Equal-arc correspondence gives poorer held-out four-channel and XY agreement than flexible phase on both intervals. Additive equal-arc fits do give smaller total-intensity residuals, so the difference includes a trade-off between intensity balance and anisotropy shape. Neither correspondence is validated against known rod angles.

\textbf{Sampling controls.} Original gate crossings were retained (304 and 217 complete traversals). Arc bins were rebuilt using calibrated channel ratios. All observed bin intensities are means of unsmoothed calibrated samples. Alternating traversals form training and held-out averages, with equal weight per traversal. The flexible results remain close to the preceding raw-guide-bin results.

\textbf{Numerical and averaging checks.} Increasing simulated curve resolution from 2048 to 8192 segments changes predicted channel intensities by no more than \(1.02\times10^{-6}T_{\max}\) in these selected arc fits. Averaging predictions over the empirical within-bin dwell distributions and refitting changes held-out channel RMS by less than 0.000012 and pointwise folded theta by less than \(0.024^\circ\). Thus neither midpoint approximation nor numerical resolution explains the larger correspondence difference. Dwell coordinates come from the smoothed measured guide, not independent true orientations.

\textbf{Optimization checks.} Arc fits used 20--21 starts per model/window, both directions. Flexible controls used 9--10 starts, including an arc-derived start. All selected runs converged, selected full arc curves have positive channel intensities, and multiple starts found the reported minima. These are local-search results, not certified global minima. Expanded models were seeded with the fitted zero-background solution and obey the nested training-cost bound.

\newpage
{\Large Recovered angles and model definition}

\includegraphics[width=\linewidth]{angles.png}

For \(q_i=A_F+B_F\sin^2\theta+C_F\sin^2\theta\cos2(\phi-\psi_i)\),
\[
S_i=a_0^2\sin^2\theta\,q_i,
\qquad I_i^{\rm pred}=\begin{cases}S_i&\text{none},\\ S_i+B_i&\text{additive},\\S_i+C_i\sqrt{S_i}&\text{Richard}.
\end{cases}
\]
\[
B_i=\frac{B_{\rm total}}4(1+f_a,1-f_a,1+f_b,1-f_b)_i,
\quad f_a,f_b\in[-1,1],\quad B_{\rm total}\leq0.95\min T_{\rm train}.
\]
Equal-arc matching evaluates this full model densely, forms measured-model X/Y, integrates its arc length, and interpolates the four intensities at ordered fractions plus one cyclic offset. No independent nearest-point matches are made. Cone geometry, brightness and correction parameters are jointly fitted using
\[
J=\sum_{k,i}\left(\frac{I_{i,k}^{\rm pred}-I_{i,k}^{\rm obs}}{T_k^{\rm obs}}\right)^2.
\]
Parameters: 5/8/9 for equal arc; 20/23/24 for flexible phase. Angles here follow each fitted cone; they are not independent angle measurements. Folding near 90 degrees can create cusps in the displayed traces.
\end{document}
'''
(R/'report.tex').write_text(tex)
print('Verified selected fits, nested baseline, cyclic periodicity; wrote figures and report source.')
