from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as r
R=r.R;v=json.load(open(R/'fits.json'));strict=json.load(open(R/'strict_fits.json'));trap=json.load(open(R/'collapse_check.json'));old=r.OLD
fig,axs=plt.subplots(1,2,figsize=(11,5.5),layout='constrained');rows=[]
for col,key in enumerate(r.KEYS):
 tr=r.Z[key+'_train']/r.Z['scale'];te=r.Z[key+'_test']/r.Z['scale'];tm=tr.sum(0).min();a=axs[col];a.scatter(*r.xy(te).T,s=10,color='#263440',label='Held-out average')
 for model,label,color,ls in [('square','Additive background','#bb8054','--'),('richard4','Richard four-C','#7178a0',':'),('complex','Additive + bounded interference','#b53743','-')]:
  if model=='complex':b=v[key]['arc_complex']['best'];y=np.array(b['pred']);sc=b['test_channel_rms'];xy=b['test_xy_rms'];bg=b['bg_percent_Tmax'];theta=b['theta_range']
  else:
   b=old[key]['arc_'+model]['best'];y,_,_,bb,d,_=r.arc.forward(np.array(b['p']),model,tm,direction=b['direction'],parts=True);sc=b['test']['channel_rms'];xy=b['test']['xy_rms'];bg=100*bb.sum()/tr.sum(0).max() if model=='square' else None;theta=b['test']['theta']
  a.plot(*r.xy(y).T,color=color,ls=ls,lw=1.8,label=label);rows.append(dict(interval=key,model=model,channel_rms=sc,xy_rms=xy,bg_percent_Tmax=bg,theta_min=min(theta),theta_max=max(theta)))
 a.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=key+' · ordered equal-arc fit');a.set_aspect('equal');a.grid(alpha=.2)
 a.legend(fontsize=8,loc='lower left')
fig.savefig(R/'complex_arc.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for key,col in [('30s','#b53743'),('10s','#287c9a')]:
 t=trap[key];xx=[x['multiplier'] for x in t];axs[0].loglog(xx,[x['circle_deg'] for x in t],'o-',color=col,label=key);axs[1].loglog(xx,[x['half_angle'] for x in t],'o-',color=col,label=key)
axs[0].set(xlabel='Multiplier on negative C coefficients',ylabel='Held-out circle deviation (degrees)',title='Apparently better circle agreement');axs[1].set(xlabel='Multiplier on negative C coefficients',ylabel='Fitted cone half-angle (degrees)',title='But the recovered trajectory collapses')
for a in axs:a.grid(alpha=.2);a.legend()
fig.savefig(R/'collapse.png',dpi=180);plt.close(fig)
with (R/'forward_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}\usepackage{fontspec,amsmath,graphicx,booktabs}
\setmainfont{Latin Modern Roman}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
{\LARGE More complex background and inverse-circle tests}\par 6 October 2026

\textbf{Main findings.} Adding bounded interference modestly improves the equal-arc intensity fit but leaves the inner-loop discrepancy. Direct signal inversion and circle fitting is computationally possible, and a known synthetic cone is recovered to numerical precision. Circle residual alone, however, admits a collapse/cancellation solution. A constrained inverse reconstruction remains an exploratory branch-specific test, not a validated replacement.

\includegraphics[width=\linewidth]{complex_arc.png}

\begin{tabular}{llrrrr}\toprule
Window & Equal-arc model & Channel RMS & XY RMS & BG/\(T_{\max}\) & \(\theta\) range\\\midrule
'''
for a in rows:
 label={'square':'Additive','richard4':'Richard four-C','complex':'Additive + interference'}[a['model']];bg='---' if a['bg_percent_Tmax'] is None else f"{a['bg_percent_Tmax']:.1f}\\%"
 tex+=f"{a['interval']} & {label} & {a['channel_rms']:.5f} & {a['xy_rms']:.5f} & {bg} & {a['theta_min']:.1f}--{a['theta_max']:.1f}$^\\circ$ \\\\\n"
tex+=r'''\bottomrule\end{tabular}

The complex model is a separate channel-effective hypothesis, not Richard's exact four-C model:
\[
I_i=S_i+B_i+2\rho_i\sqrt{B_iS_i},\qquad |\rho_i|\leq1.
\]
It fits three additive parameters (equal pair sums, independent asymmetries) and four constant signed overlaps. The bound prevents a cross term larger than the scalar coherent-field limit; it does not prove realization by a single common optical background field. Background cap remains 95\% of minimum training total. Cone geometry, one brightness and one cyclic offset add five parameters: 12 in total. No flexible phase mapping is fitted. Thirty-two starts across both directions; selected arc fits converge. Calibration, NA .38--1.3, fixed excitation, 128 calibrated bins and relative channel-intensity objective are unchanged.
\newpage
{\Large Inversion and the best circle without explicit cone parameters}

For a candidate background, let \(C_i=2\rho_i\sqrt{B_i}\) (or Richard's independently fitted coefficient). On the unique positive-root domain \(I_i-B_i>0\),
\[
S_i=\left[\frac{\sqrt{C_i^2+4(I_i-B_i)}-C_i}{2}\right]^2.
\]
Compute corrected X/Y, then use
\[
\sin^2\theta=\frac{A_Fr}{C_F-B_Fr},\qquad \phi=\tfrac12\operatorname{unwrap}\!\left[\operatorname{atan2}(Y,X)\right].
\]
A small circle has plane equation \(\mathbf n\cdot\mathbf d=c\), with \(\|\mathbf n\|=1\). For recovered directions \(\mathbf d_k\), subtract their mean and form the covariance. Its smallest-eigenvalue eigenvector gives the least-squares plane normal; \(c=\mathbf n\cdot\overline{\mathbf d}\). This minimizes squared plane distance. It is an algebraic circle fit, not an exact angular-distance minimizer. Angular deviations are reported separately.

Thus the cone is still implicit in the circle: axis \(\mathbf n\), half-angle \(\arccos c\). These quantities are recalculated from each candidate correction, not independent optimizer parameters. No per-point phase variables are optimized and no point order is changed.

\textbf{Synthetic verification.} A simulated cone with half-angle 0.22 radians, fixed brightness and known additive/interference correction was inverted. Maximum signal error \(1.11\times10^{-16}\), maximum plane error \(3.33\times10^{-16}\), reconstructed channel RMS \(2.70\times10^{-16}\). This verifies the implementation on a valid non-crossing branch, not identifiability on real data.

\textbf{Observed failure of circle-only fitting.} Richard's free signed coefficients run toward large negative values, producing a large recovered signal almost cancelled by interference. The recovered direction changes very little. At 30--31 s, a fitted example has a cone half-angle of only \(0.652^\circ\), corrected theta 36.33--37.88 degrees, and angular circle RMS \(0.139^\circ\). Multiplying all four negative coefficients by eight, without further optimization, reduces half-angle to \(0.0105^\circ\), theta span to about \(0.025^\circ\), and circle error to \(0.00224^\circ\). This is not evidence of accurate angles.

\includegraphics[width=\linewidth]{collapse.png}

The corrected pair sums disagree by roughly 11\% of corrected total at 30 s (15\% at 10 s), despite the near-perfect circle. A common-brightness optical signal cannot explain those recovered four channels. This illustrates why circle agreement must retain the information discarded by the two anisotropy ratios. The initial numerical coefficient box was [-5,5]; the scaling test shows that this box merely truncates the collapse, not that it resolves it.
\newpage
{\Large Retaining intensity information in the inverse approach}

A second pilot profiles the plane, projects each recovered direction onto its circle, evaluates the full four-channel model with one fitted brightness scale, and minimizes the original relative-intensity residual. Phase locations are inferred from each observed four-channel vector, not independently adjusted parameters.

\begin{tabular}{llrrrr}\toprule
Window & Inverse reconstruction & Channel RMS & BG/\(T_{\max}\) & Recovered \(\theta\) & Circle RMS\\\midrule
'''
for key in r.KEYS:
 for model in ['square','richard4','complex']:
  b=strict[key]['reconstruct_'+model]['best']['result'];a=b['test'];bg='---' if b['bg_percent_Tmax'] is None else f"{b['bg_percent_Tmax']:.1f}\\%";label={'square':'Additive','richard4':'Richard four-C','complex':'Additive + interference'}[model]
  tex+=f"{key} & {label} & {a['channel_rms']:.5f} & {bg} & {a['theta_range'][0]:.1f}--{a['theta_range'][1]:.1f}$^\\circ$ & {a['angular_circle_rms_deg']:.2f}$^\\circ$ \\\\\n"
tex+=r'''\bottomrule\end{tabular}

\textbf{Interpretation limits.} These are conditional held-out reconstruction scores: background, plane and brightness are trained on training cycles, but each held-out vector supplies its own recovered direction and hence phase location. They are not the same prediction test as evaluating a fixed curve at predetermined held-out bin indices. The 30 s complex result retains the nested additive solution because the searches did not improve its feasible training cost; it is not a separately converged complex optimum.

The strict pilot requires nonnegative corrected roots, anisotropy below the Fourkas maximum, and a closed continuous lift in one hemisphere. It retains acquisition order; no nearest-point permutation is allowed. The inferred phase is checked for backtracking; selected reconstruction examples show none or only small fluctuations. Hemisphere-crossing branches are not globally searched. These searches do not prove that a more complete inverse treatment cannot work.

The first unconstrained trials included nonclosing lifted trajectories and out-of-domain points. They are retained as diagnostic failures, not accepted angle corrections. Some circle-only complex solutions were feasible on training data but exceeded the physical anisotropy range on held-out data; these cannot be interpreted by clipping theta. Strict reconstructed examples above have valid held-out inversions.

\textbf{Why branch handling matters.} Fourkas inversion determines folded theta and phi modulo pi. A collection of individually admissible points is not automatically a continuous closed physical trajectory. A cone crossing the equator also folds when represented only on the upper hemisphere. Eliminating explicit cone parameters does not eliminate these geometric decisions.

\textbf{Recommended interpretation.} Keep the equal-arc fits as the controlled comparison. The bounded interference extension tested here does not resolve the shape mismatch. The inverse route is useful as a diagnostic, but a raw spherical-circle objective is demonstrably insufficient: it must enforce four-channel optical consistency, brightness and branch continuity, or explicitly incorporate their measurement uncertainty. Neither route establishes absolute angle truth from these data.

Scripts, caches and numerical details are in model\_comparison/inverse\_circle/. No commit or push.
\end{document}
'''
(R/'report.tex').write_text(tex)
(R/'README.md').write_text('''# Complex equal-arc and profiled inverse-circle tests (6 October 2026)

See report.pdf / report.tex for scientific assumptions and limitations. run.py fits complex arc and initial inverse trials; strict.py runs a limited feasible closed-hemisphere inverse sensitivity. fits.json includes invalid exploratory trials explicitly flagged; strict_fits.json retains feasible starts if optimization worsens them. These are not interchangeable with validated angle estimates. report.py builds plots and report source. Compile with XeLaTeX.

Branch restriction: positive Im-B unique amplitude root; theta restricted to a continuous upper-hemisphere lift, checked for closed topology. Not a complete branch/equator-crossing search. First-stage soft penalties allow small invalidities and nonclosing results; do not use them as accepted recovered angles. Strict reconstruction stage enforces training domain; held-out inversions checked separately. Some strict circle-only complex results fail held-out domain.

Inverse background-only geometric objective profiles a least-squares plane by covariance eigenvector, not exact angular geodesic fitting. Intensities discarded by ratios are diagnosed. Inverse reconstruction profiles plane, projects directions radially within plane to its circle, fits one brightness and correction parameters to four relative-channel residuals. Held-out conditional reconstruction uses held-out directions, unlike fixed-bin forward prediction. Richard inverse test bounds coefficients to [-5,5] only as a numerical search box; negative-C collapse hits box, and scaling proof demonstrates unbounded degeneration without it. No physical BG power inferred from Richard. More complex model uses 3 B coefficients + 4 bounded overlaps, not a spatially arbitrary field and not guaranteed a common optical realization.

Reproduction needs arc_matching/inputs.npz and fits.json plus models.py, numpy scipy matplotlib. Feasible start caches saved as *_validstarts.npy from seeded uniform searches; strict.py uses these. Synthetic known-cone validation saved in synthetic_validation.json. collapse_check.json records direct coefficient-scaling counterexample. No raw data or private papers copied here. No push.
''')
print('Wrote figures and three-page LaTeX report')
