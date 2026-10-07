from pathlib import Path
import json,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as r
R=r.R;v=json.load(open(R/'fits.json'));rows=[]
for w,a in v.items():
 b=a['best']
 for k in r.f.KEYS:rows.append(dict(weight=w,window=k,success=b['success'],bg=b['scores'][k]['bg_percent'],train=b['scores'][k]['train'],test=b['scores'][k]['test']))
fig,axs=plt.subplots(3,2,figsize=(10,14),layout='constrained')
for row,(title,b) in enumerate([('Gain corrected, no domain penalty',v['0']['best']),('Gain corrected, penalty weight 10',v['10']['best']),('Gain corrected, alternative local solution',v['0']['runs'][1])]):
 for col,k in enumerate(r.f.KEYS):
  sc=b['scores'][k];s=np.array(sc['signal']);pred=np.array(sc['pred']);obs=s+r.TE[col]-pred;p=np.array(b['p']);curve=r.f.dense(p[5*col:5*col+5],p[10:],8192)[3];a=r.f.xy(obs);rad=np.linalg.norm(a,axis=1);bad=(obs<0).any(0)|(rad>r.rmax);ax=axs[row,col]
  ax.plot(*r.f.xy(curve).T,color='#ce6436',lw=1.8,label='Fitted signal');ax.plot(*np.vstack([a,a[0]]).T,color='#398849',lw=1.2,label='Signal + held-out residual');ax.scatter(*a[bad].T,color='#d32030',marker='x',s=16,label='Outside inversion domain')
  outside=(abs(a)>1).any(1)
  if outside.any():ax.scatter(*(a[outside]/np.max(abs(a[outside]),axis=1)[:,None]*.98).T,color='#d32030',marker='^',s=20)
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=title+f'\n{k}: {bad.sum()}/128 invalid');ax.set_aspect('equal');ax.grid(alpha=.15)
  if col==0:ax.legend(fontsize=7,loc='lower left')
for ext in ['png','pdf']:
 buf=io.BytesIO();fig.savefig(buf,format=ext,dpi=160);(R/('comparison.'+ext)).write_bytes(buf.getvalue())
tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=16mm]{geometry}\usepackage{fontspec,amsmath,graphicx,booktabs}
\setmainfont{Latin Modern Roman}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
{\Large Gain calibration followed by soft recovery-domain penalties}\par 6 October 2026

One relative gain correction was estimated from pooled training-average bins, before the unchanged inverse transmission matrix. Multipliers on historical inverse gains, in $(0,90,45,135)$ order: $(0.992729,1.005313,1.009114,0.992952)$. Their product is one to fix the overall scale. Separate-window estimates differ by at most about 0.37 percentage points. This is pair-balance calibration conditional on the common-field optical model, not independent detector calibration; offsets and matrix errors were not fitted.

Held-out measured pair-imbalance RMS changes from 0.375\% to 0.302\% (30--31 s), and 0.320\% to 0.303\% (10--11 s). Gain therefore explains only part of the imbalance. Existing accepted cycles and original bins are retained; the new calibration is applied linearly to those channel means, then uniform final-arc references and held-out interpolation weights are rebuilt from training data. Historical gain corrections are not applied twice.

Signal, coherent background, noninterfering Stokes background and cross term each obey pair balance by construction in this model. Gains are frozen before fitting. No extra pair-sum penalty is needed to impose those model identities; data imbalance remains part of the four-channel residual. Richard's four independent coefficients would require a separate restricted parameterization or explicit pair penalty and were not refitted here.

For each training point, $\widehat S=I^{train}-B-L(d^{fit})$. The objective is
\[
J=\sum_{ij}(\varepsilon_{ij}/T_j^{train})^2
+w\sum_j[\max(0,\widehat r_j/r_{max}-1)]^2
+w\sum_{ij}[\min(0,\widehat S_{ij})/T_j^{train}]^2.
\]
No signal-relative residual weighting is used. In the radial penalty only, pair denominators have a numerical floor $10^{-4}T_j^{train}$ to avoid division by zero; negative channels are penalized separately. Physical inversion and reported invalid counts use unmodified corrected channels. These soft squared-hinge penalties allow finite exceedance. Weight values are sensitivity choices, not experimentally calibrated tolerances.

High- and lower-background starts were checked. Penalized selected fits use 8192 curve intervals; all selected fits converged. The same single stationary field is shared across windows. No parameter selection used held-out outcomes.

\begin{tabular}{llrrrr}\toprule
Weight & Window & Train invalid & Test invalid & Test RMS & Max test $r-r_{max}$\\\midrule
'''
for a in rows:
 tex+=f"{a['weight']} & {a['window']} & {a['train']['invalid']} & {a['test']['invalid']} & {a['test']['channel_rms']:.5f} & {a['test']['max_radial_excess']:.4f} \\\\\n"
tex+=r'''\bottomrule\end{tabular}

Invalid includes negative channels and radial exceedance, even a tiny exceedance. Max excess is floored at zero. Counts alone do not measure violation size. The penalty is applied to training data only; it cannot guarantee physically invertible held-out corrections. These are conditional fitted-cross-term subtraction diagnostics, not independent field inversions or absolute angle validation.
\newpage
\includegraphics[width=\linewidth,height=.94\textheight,keepaspectratio]{comparison.pdf}
\end{document}
'''
(R/'report.tex').write_text(tex);(R/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
