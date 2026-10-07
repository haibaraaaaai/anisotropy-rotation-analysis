from pathlib import Path
import json
R=Path(__file__).resolve().parent;v=json.load(open(R/'predictions.json'));old=json.load(open(R.parent/'meeting_2026_10_06/comparison.json'))['records'];base={(q['id'],q['interval']):q for q in old}
rows=[]
for k,label in [('30s','30--31 s'),('10s','10--11 s')]:
 for model,name in [('bg','Stokes additive'),('square','Notebook-square additive'),('richard','Additive + sqrt extension'),('richard4','Richard four-$C$')]:
  if model in v[k]:
   q=v[k][model];st=q['scores'];th=st['theta_range'];bg='---' if model=='richard4' else f'{q["bg_percent_max"]:.2f}'
  else:
   q=base[model,k];st=q['scores']['test'];th=[q['theta_min'],q['theta_max']];bg=f'{q["bg_max"]:.2f}'
  rows.append(f'{label} & {name} & {st["xy_rms"]:.4f} & {100*st["channel_rms"]:.2f} & {bg} & {th[0]:.1f}--{th[1]:.1f} '+r'\\')
tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{fontspec,unicode-math,amsmath,graphicx,booktabs,array,xcolor,fancyhdr}
\setmainfont{Latin Modern Roman}\setmathfont{Latin Modern Math}
\setlength{\parindent}{0pt}\setlength{\parskip}{8pt}
\pagestyle{fancy}\fancyhf{}\fancyfoot[L]{\small APD model comparison --- 6 October 2026}\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\begin{document}
{\LARGE\bfseries Two new tests: notebook background and Richard's four coefficients}\par
\textbf{Result.} The notebook's square background domain returns essentially the same optimum as the Stokes disk. Richard's exact four-coefficient approximation fits less well on held-out anisotropy and channel residuals. These tests use the same two intervals, calibrated channels, annulus, excitation law and ordered trajectory as the preceding comparisons.

\textbf{Test 1: relaxing only the background polarization constraint.}
\[
(B_0,B_{90},B_{45},B_{135})=\frac{B}{4}(1+f_a,1-f_a,1+f_b,1-f_b),
\qquad |f_a|,|f_b|\leq1.
\]
The extra Stokes restriction $f_a^2+f_b^2\leq1$ is removed. The total-background cap is kept at the same $0.95T_{\min}$ as the previous fixed-excitation additive fit, to isolate the domain change. This tests the original notebook's background family, not its old shape-only optimizer or preprocessing.

\textbf{Test 2: Richard's quoted approximation, without an additive term.}
\[
I_{m,j}=S_j+C_j\sqrt{S_j},\qquad C_j\in\mathbb R.
\]
Four coefficients are freely fitted, including either sign, with no imposed finite coefficient cap. Both tests retain one constant $a_0$ per interval and the prescribed circular-excitation dependence of $S_j$. The new parameters are 23 per interval for the square additive model and 24 for Richard's model.

{\small\begin{tabular}{llrrrr}\toprule
Window & Model & XY RMS & Channel RMS (\%) & BG/Tmax (\%) & $\theta$ (degrees)\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}}

Scores are held-out. Channel residuals are normalized by measured total at each bin. Background is not inferred by Richard's four-$C$ approximation: the dash is not a claim of zero physical background. Angles are conditional on the fitted cone, not confidence intervals or known-angle accuracy.

\clearpage
{\Large\bfseries Fit overlays}\par
\includegraphics[width=\linewidth,height=.82\textheight,keepaspectratio]{two_new_fits.png}

The square-background prediction lies on top of the old Stokes prediction. The lower row distinguishes Richard's new four-$C$ fit from the previously fitted additive-plus-square-root extension. Every anisotropy axis spans $[-1,1]$.

\clearpage
{\Large\bfseries Power balance and angle consequences}\par
\includegraphics[width=\linewidth]{power_angles.png}

\textbf{Square versus disk.} The new optima have $(f_a,f_b)=(0.172,-0.405)$ and $(0.137,-0.452)$ for 30--31 s and 10--11 s. Their radii in the $(f_a,f_b)$ plane are $0.440$ and $0.472$, both well below one. The Stokes boundary was therefore not limiting these fixed-excitation additive fits. Background remains $31.74\%$ / $34.62\%$ of maximum total and the angle ranges are unchanged to numerical tolerance.

\textbf{Richard four-$C$.} All four selected coefficients are positive in both windows. The effective correction adds $35.2$--$56.3\%$ of maximum total at 30--31 s and $42.0$--$63.9\%$ at 10--11 s. These are signed correction terms evaluated along the fitted path, not constant background powers. The fitted $\theta$ ranges are $36.6$--$72.0^\circ$ and $40.2$--$71.4^\circ$.

Its held-out total-intensity RMS is slightly lower than additive-only (5.67\% versus 5.81\%, and 5.79\% versus 6.66\%), but its XY and channel RMS are worse. This is a tradeoff across observables, not improvement of the whole fit.

\clearpage
{\Large\bfseries Interpretation and checks}\par
\textbf{Can the fitted four-$C$ correction be interpreted literally as interference?}
For a given fitted rod signal, a physical coherent background obeys
\[
C_j=2\sqrt{B_{\mathrm{coh},j}}\,\rho_j,
\qquad |\rho_j|\leq1,
\qquad B_{\mathrm{coh},j}\geq C_j^2/4.
\]
If the selected coefficients are interpreted this way, the necessary summed coherent background is at least $23.4\%$ / $37.4\%$ of maximum measured total. Those are conditional bounds for this fitted decomposition, not estimates of the true background or proof that a common stationary field exists. They show that the omitted background-only power would not be negligible in that interpretation. These input traces had no independently measured optical-background subtraction.

Richard's equation remains a useful phenomenological approximation, particularly if background power has already been removed or is demonstrably negligible. The present results do not validate those assumptions for this recording. A coefficient acting as an effective correction may also absorb a different physical mismatch.

\textbf{Coefficient convention.} Normalize all four intensities by the common reference $T_{\mathrm{ref}}$, the larger of the two training maxima. Then $C'_j=C_j/\sqrt{T_{\mathrm{ref}}}$. In channel order $(0,90,45,135)$, the fitted coefficients are
\[
\mathbf C'_{30\mathrm{s}}=(0.5100,0.4381,0.3282,0.6096),
\]
\[
\mathbf C'_{10\mathrm{s}}=(0.6354,0.5467,0.3968,0.7986).
\]
The coefficients were fitted separately in the two intervals; a shared-coefficient hypothesis was not tested here.

\textbf{Numerical checks.} Ten starting points were used for each model/window (40 optimizations). All selected solutions terminated normally and predict positive channels at all 128 bins. Square-background starts return the same objective within numerical tolerance. The four-$C$ search has a worse local solution among the 30 s starts; the selected minimum is reported, not a certified global optimum. Brightness is not at its imposed numerical bounds. Direct quadratic inversion of each selected four-$C$ prediction recovers its modeled signal to numerical precision. No bins were dropped and no pointwise brightness freedom was introduced.

\textbf{What changes in the discussion.} The Stokes restriction does not explain the current additive-model mismatch. The exact four-$C$ proposal has now been tested and is not the better description under the present fixed-cone/excitation assumptions. This does not rule out effective interference models under other conditions or establish the true angles. This supplement supersedes the earlier note that Richard's exact model had not been run. No repository commit or push was made.
\end{document}
'''
(R/'two_new_tests.tex').write_text(tex)
(R/'README.md').write_text('''# Two matched tests — 6 October 2026

Run `python run.py` for cached multistart fits (completed results are skipped); `python report.py` to reproduce plots/numerical summaries; `python build_pdf.py` and `xelatex two_new_tests.tex` to build the PDF.

Square additive: same 95%-of-minimum-total cap as preceding Stokes baseline, but independent fa/fb bounds [-1,1]. Fixed excitation retained: this is not the historical shape-only objective. Exact Richard: four unrestricted real C_j, no constant B term; positivity checked on selected predictions. Same fixed gains/matrix, .38–1.3 optics, ordered 128-bin trajectories, train/test split, constant a0 and cone/phase parameters as baseline. Ten starts per model/window, all runs saved. No new full instrument calibration or angle truth.

No background power is inferred by the four-C model. Zero background in its arithmetic decomposition means omitted term, not physically absent background. Report includes the conditional Cauchy lower bound if coefficients are interpreted as physical interference with the fitted signal. No push authorized at this stage.
''')
