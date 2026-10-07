from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as r
R=r.R;v=json.load(open(R/'fits.json'));rv=json.load(open(R/'richard_product_fits.json'));checks=json.load(open(R/'checks.json'));rows=[];preds={}
LABEL={'none':'No background','square':'Additive background','richard4':'Richard four-C','complex':'Additive + bounded interference'};COLOR={'none':'#999999','square':'#287c9a','richard4':'#9b79a5','complex':'#bb423e'}
fig,axs=plt.subplots(1,2,figsize=(11,5.6),layout='constrained');fig2,bxs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for col,key in enumerate(r.KEYS):
 tr=r.Z[key+'_train']/r.Z['scale'];te=r.Z[key+'_test']/r.Z['scale'];tm=tr.sum(0).min();u=r.fractions(tr);preds[key]={};ax=axs[col]
 ax.scatter(*r.xy(te).T,color='#223440',s=12,label='Held-out average')
 for model in r.MODELS:
  b=rv[key]['best'] if model=='richard4' else v[key][model]['best'];variant='richard_product' if model=='richard4' else model
  p=np.array(b['p']);yp,s,c,bg,d,mn=r.forward(p,variant,tm,b['direction'],u,8192,True);sc=b['test'];th=np.rad2deg(np.arccos(abs(d[:,2])))
  label=LABEL[model]+(' (limiting regime)' if model=='richard4' and p[3]<1e-8 else '')
  ax.plot(*r.xy(yp).T,color=COLOR[model],lw=1.6,ls='--' if model=='none' else '-',label=label)
  if model=='complex':
   for k in range(0,128,8):
    pp=r.xy(yp)[k];qq=r.xy(te)[k];ax.plot([pp[0],qq[0]],[pp[1],qq[1]],color=COLOR[model],alpha=.35,lw=.7)
  bgpct=None if model=='richard4' else 100*bg.sum()/tr.sum(0).max()
  row=dict(interval=key,model=model,channel_rms=sc['channel_rms'],xy_rms=sc['xy_rms'],total_rms=sc['total_rms'],bg_percent_Tmax=bgpct,theta_min=float(th.min()),theta_max=float(th.max()),limiting=bool(model=='richard4' and p[3]<1e-8));rows.append(row)
  preds[key][model]=dict(pred=yp.tolist(),signal=s.tolist(),cross=c.tolist(),background=bg.tolist(),theta=th.tolist(),reference_u=u.tolist(),**row)
 ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=('30–31 s' if key=='30s' else '10–11 s'));ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=7,loc='lower left')
 bxs[col].plot(np.arange(128),u,label='Actual training-reference arc');bxs[col].plot(np.arange(128),np.arange(128)/128,'--',label='Incorrect uniform index');bxs[col].set(xlabel='Original bin index',ylabel='Cumulative arc fraction',title=key);bxs[col].legend(fontsize=8);bxs[col].grid(alpha=.2)
fig.suptitle('Corrected correspondence: actual final-reference arc positions',fontsize=13);fig.savefig(R/'corrected_fits.png',dpi=180);fig2.savefig(R/'reference_coordinates.png',dpi=180);plt.close('all')
(R/'predictions.json').write_text(json.dumps(preds,indent=2))
with (R/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}\usepackage{fontspec,amsmath,graphicx,booktabs}
\setmainfont{Latin Modern Roman}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
{\LARGE Corrected final-reference correspondence}\par 6 October 2026

\textbf{Correction to earlier reports.} The final channel-averaged trajectory was not uniformly spaced in anisotropy, although its constituent cycles had equal-guide-arc bins. Matching that final reference at uniform index fractions was incorrect. This report supersedes the previous equal-arc forward-fit comparisons, including the forward part of the complex/inverse-circle report. The inverse-circle collapse counterexample is independent of this error.

\includegraphics[width=\linewidth]{corrected_fits.png}

Observed points are intentionally unchanged and therefore remain nonuniformly spaced. The model is now evaluated at their actual training-reference arc fractions. Thin red segments join selected corresponding observed and predicted points; no nearest-neighbour matching is used.

\begin{tabular}{llrrrr}\toprule
Window & Model & Channel RMS & XY RMS & BG/\(T_{\max}\) & \(\theta\) range\\\midrule
'''
for a in rows:
 label={'none':'None','square':'Additive','richard4':'Richard four-C','complex':'Additive + interference'}[a['model']]+('$^*$' if a['limiting'] else '')
 bg='---' if a['bg_percent_Tmax'] is None else f"{a['bg_percent_Tmax']:.1f}\\%"
 tex+=f"{a['interval']} & {label} & {a['channel_rms']:.5f} & {a['xy_rms']:.5f} & {bg} & {a['theta_min']:.1f}--{a['theta_max']:.1f}$^\\circ$ \\\\\n"
tex+=r'''\bottomrule\end{tabular}

$^*$Richard's 10 s row is a limiting regime with vanishing signal brightness and growing coefficients; it is not a stable finite-coefficient correction. No physical background power is inferred from Richard's approximation. All angular ranges are conditional model outputs, not independently validated angles.
\newpage
{\Large What changed, and how it was checked}

For each final training track, interpolate the four intensities together along the closed, ordered list of bins. Calculate X/Y from this channel interpolant, integrate its arc length, and record the fraction \(u_k\) at each original bin. A candidate full measured-model curve is then evaluated at fractions \((\delta\pm u_k)\bmod1\), with one fitted cyclic offset and both directions tested.

The held-out track uses exactly the training-derived \(u_k\), offset, direction and fitted physical parameters. It is not independently reparameterized. This preserves every original observed channel value and the original equal-bin intensity weighting. It is not a new uniformly reweighted reference; uniform resampling would additionally change the objective weights.

\includegraphics[width=\linewidth]{reference_coordinates.png}

The intensity objective remains
\[
J=\sum_{k,i}\left[\frac{I_{i,k}^{\rm pred}-I_{i,k}^{\rm obs}}{T_k^{\rm obs}}\right]^2.
\]
Three cone parameters, one brightness scale and one starting offset are fitted, plus correction parameters. No 16-parameter phase deformation. Optics, fixed gains/T, NA 0.38--1.3 and cycle selection remain unchanged (304/217 traversals). Training and test averages use alternate traversals.

\textbf{Numerical check.} A deliberately nonuniformly sampled synthetic model curve produces channel RMS 0.11364 when incorrectly matched by uniform indices. The corrected coordinate method reduces this to \(1.54\times10^{-5}\), with the remaining difference due to finite reference interpolation. Increasing reference interpolation subdivisions from 32 to 128 changes fractions by less than \(10^{-11}\). The selected finite non-Richard fits are refined at 8192 simulated segments; comparison at 16384 changes intensities by less than \(4\times10^{-8}T_{\max}\).

\textbf{Search and comparisons.} Each ordinary model/window used 18--19 coarse starts; the complex model used 24, across both directions, including nested zero/additive seeds and varied geometry. Selected none/additive/complex fits converge and beat their nested training baselines. Local search is not a global optimality certificate. Earlier flexible-phase controls use the same unchanged observations and still have lower held-out channel errors for the three original model families. That does not establish better physical angles. The coordinate repair changes recovered geometry materially even where the score changes modestly.

\newpage
{\Large Brightness constraint and the remaining Richard degeneracy}

For the fixed circular-excitation model,
\[
S_i=a_0^2\sin^2\theta\,q_i(\theta,\phi),\qquad
\sum_iS_i=4a_0^2\sin^2\theta(A_F+B_F\sin^2\theta).
\]
A sphere-circle inversion can use this total-intensity law at the recovered/projected orientations, plus the channel pair-sum condition. It must not choose an independent brightness per point. Time order supports phi unwrapping but does not by itself determine theta reflection branches at an equator crossing; those choices require continuity and physical-consistency checks. This report repairs correspondence only; it does not add new inverse-fit constraints.

Richard's model has another potential degeneracy even in a forward cone fit. Write \(S_i=a_0^2 f_i\), then
\[
I_i=a_0^2f_i+(a_0C_i)\sqrt{f_i}=\alpha f_i+K_i\sqrt{f_i},
\qquad \alpha=a_0^2,\quad K_i=a_0C_i.
\]
For positive alpha this is exactly the same model, with no additional degrees of freedom. The optimizer can reduce alpha toward zero while keeping K finite, which sends C to infinity. The original 10 s coefficient fit exhausted its refinement budget along this direction. Reparameterizing and searching nine starts converges to \(\alpha\approx6.7\times10^{-15}\), confirming the limiting tendency. Its predicted curve is well defined as a limit, but its separated signal/interference amplitudes are not a stable correction. The earlier numerical lower bound on brightness merely truncates this direction.

For 30 s the selected reparameterized solution has finite \(\alpha\approx0.01063\). Its physical adequacy still requires independent constraints. The bounded additive-plus-interference model instead uses
\[
I_i=S_i+B_i+2\rho_i\sqrt{B_iS_i},\quad |\rho_i|\leq1,
\quad B_{\rm total}\leq0.95\min T_{\rm train}.
\]
A finite background budget bounds its effective coefficients. These channel-wise bounds are necessary physical restrictions but do not establish realization by one common optical field.

\textbf{Interpretation.} The correspondence bug is repaired. The bounded-interference model now gives the smallest errors among these four corrected fixed-correspondence fits, but substantial mismatch remains. The result supports testing intensity/physical-budget constraints in the inverse-circle approach next, without claiming that current fitted angles are established or that flexible matching is physically superior.

Scripts, input-coordinate arrays, fit trials, predictions, resolution checks and Richard limiting analysis are saved in model\_comparison/corrected\_arc/. No commit or push.
\end{document}
'''
(R/'report.tex').write_text(tex)
(R/'README.md').write_text('''# Corrected final-reference arc coordinates (6 October 2026)

Supersedes equal-arc forward interpretations in arc_matching and inverse_circle. Original bin intensities unchanged. Coordinates obtained from a dense closed ordered four-channel interpolant of the FINAL training average, computing ratios after interpolation. Simulation samples actual reference fractions plus one cyclic offset, both directions. Same train-derived fractions used for held-out bins, not independently aligned. Objective still equally weights original bins; this is distinct from uniformly resampling/reweighting the final curve.

run.py fits none, additive, exact Richard, bounded additive+interference; richard_limit.py profiles exact Richard using alpha=a0^2,K=a0*C, with alpha=0 admitted only as a closure diagnostic. Richard10 finite parameter refinement does not converge and approaches limiting regime; never report as a well-determined finite-C fit. report.py uses stabilized product fits, labels limiting curve explicitly. Four-channel brightness law remains fixed excitation with one a0 per interval.

Reproduce run.py then richard_limit.py then report.py, compile report.tex with XeLaTeX from this folder. numpy scipy matplotlib required; cached arc_matching/inputs.npz supplies training/held-out data and historical fit files supply starts. Checks: reference interpolation convergence, simulation convergence, synthetic known nonuniform correspondence. Scripts do not push.
''')
print('Report figures and tables generated.')
