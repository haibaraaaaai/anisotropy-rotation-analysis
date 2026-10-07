from pathlib import Path
import json,sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import run as f
R=Path(__file__).resolve().parent;stats=json.load(open(R/'sphere_metrics.json'));old=json.load(open(f.R/'predictions.json'))
t=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=16mm]{geometry}\usepackage{fontspec,amsmath,amssymb,graphicx,booktabs,array}
\setmainfont{Latin Modern Roman}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
{\LARGE APD background fits: processing and sphere residuals}\par
Draft for discussion with Richard · 6 October 2026

\section*{1. Data processing: uniform final-arc references}
\textbf{Aim.} Compare physical models against the same ordered two-loop trajectory, with roughly equal weight per unit anisotropy arc. The measured reference is constructed before any background fit. No fitted cone or direct-inversion sphere circle determines its shape.

\textbf{Calibration and optics.} APD voltages are reordered from $(90,45,135,0)$ to $(0,90,45,135)$. Historical channel gains precede the current inverse transmission matrix, which is unchanged. The forward signal uses NA$_{\rm in}=0.38$, NA$_{\rm out}=1.3$, water/oil indices 1.33/1.51, Fresnel transmission and corrected BFP mapping.

\textbf{Cycles and averaging.} Retain the existing complete-cycle detection and ordered 128-bin channel representation. Alternate accepted cycles form training and held-out groups: 152/152 for 30--31 s and 109/108 for 10--11 s. Average corresponding \emph{four-channel intensities} across cycles in each group. The resulting final-average anisotropy points are not generally equally spaced, even if each cycle was binned by equal guide arc.

\textbf{Redistribute the final training average.} Interpolate linearly between consecutive four-channel mean vectors, closing last to first. Let $\mathbf I(t)$ be this interpolant, with original-bin coordinate $0\leq t\leq128$. Calculate
\[
X(t)=\frac{I_0(t)-I_{90}(t)}{I_0(t)+I_{90}(t)},\qquad
Y(t)=\frac{I_{45}(t)-I_{135}(t)}{I_{45}(t)+I_{135}(t)}.
\]
Integrate cumulative distance along the \emph{ordered} path, counting both loops and every repeated pass:
\[
s(t)=\frac{\int_0^t\sqrt{X'(v)^2+Y'(v)^2}\,dv}
{\int_0^{128}\sqrt{X'(v)^2+Y'(v)^2}\,dv},\qquad
 t_j=s^{-1}(j/128),\quad j=0,\ldots,127.
\]
Numerically use 128 subdivisions per original segment. The new training reference is $\mathbf I(t_j)$, retaining all four intensities. Apply the \emph{same} original-bin interpolation indices and weights to the held-out average. Held-out points are not independently made uniform or realigned. Equal arc does not require equal chords around sharp bends. Resampling correlates adjacent points and adds no independent information.

\textbf{Fit with frozen observations.} Simulate a dense mechanical revolution for a proposed cone, brightness and background. Compute its predicted \emph{measured} intensities, including interference, then its own anisotropy arc. Sample those intensities at
\[
s_j^{\rm model}=(\delta+\epsilon u_j)\bmod1,\qquad \epsilon=\pm1.
\]
Here $u_j$ are the actual arc coordinates of the final resampled training reference (recomputed to remove small discretization differences). One cyclic offset $\delta$ is fitted per window and both directions are tested. There is no nearest-point matching or fitted phase warp.
\[
J=\sum_j\sum_{i=1}^4\left[\frac{I_{i,j}^{\rm pred}-I_{i,j}^{\rm train}}{T_j^{\rm train}}\right]^2,
\qquad T_j=\sum_i I_{i,j}.
\]
Uniform bins remove the main anisotropy-density weighting; the normalization still weights relative intensity error. This is not uniform mechanical phase or a noise-calibrated likelihood. Held-out cycles assess repeatability with frozen parameters, not independent absolute-angle accuracy. The following diagnostics reuse the existing fits; no sphere residual was optimized.
\newpage
\section*{2. Signal, background models and orientation diagnostics}
For rod direction $\mathbf d=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$ and analyser angle $\psi_i$,
\[
Q_i(\mathbf d)=A_F+B_F\sin^2\theta+C_F\sin^2\theta\cos2(\phi-\psi_i),
\]
\[
S_i(\mathbf d)=a_0^2\sin^2\theta\,Q_i(\mathbf d),\qquad
T_S=4a_0^2\sin^2\theta(A_F+B_F\sin^2\theta).
\]
One constant $a_0$ per window; the $\sin^2\theta$ factor is circular-excitation intensity, not an independently fitted per-point brightness. $A_F=0.890806$, $B_F=0.109194$, $C_F=0.927781$ in the adopted normalization.

\textbf{2A. Background only (8 parameters/window).}
\[
I_i=S_i+B_i,\qquad
\mathbf B=\frac{B_{\rm tot}}4(1+f_x,1-f_x,1+f_y,1-f_y),\quad |f_x|,|f_y|\leq1.
\]
Three cone parameters, $a_0$, cyclic offset, and three background parameters. Equal background pair sums, nonnegative channels, no Stokes-disk restriction. $T=T_S+B_{\rm tot}$, with constant $B_{\rm tot}\leq0.95\min T_{\rm train,original}$. Sphere diagnostic: $\widehat S_i=I_i^{\rm heldout}-B_i$.

\textbf{2B. Richard exact four-C (9 parameters/window).}
\[
I_i=S_i+C_i\sqrt{S_i},\qquad T=T_S+\sum_i C_i\sqrt{S_i}.
\]
Three cone parameters, $a_0$, cyclic offset and four constant $C_i$. There is no additive background power in this approximation. Its correction varies with orientation through $S_i$. Numerical parameters are $\alpha=a_0^2$, $K_i=a_0C_i$; selected $\alpha>0$, but the zero-brightness/divergent-$C$ limit remains structurally possible. For the sphere diagnostic, invert each \emph{measured} channel exactly:
\[
\widehat S_i=\left[\frac{-C_i+\sqrt{C_i^2+4I_i^{\rm heldout}}}{2}\right]^2.
\]
Measured channels are positive in these references, giving one positive amplitude root. Do not subtract a cross term evaluated at the predicted signal for this model.

\textbf{2C. General stationary field (33 parameters jointly).}
\[
E_{s,i}(\mathbf d)=a_0(d_x+\mathrm i d_y)\,\mathcal L_i\mathbf d,
\]
\[
I_i=S_i+L_i(\mathbf d)+B_i,\quad
L_i=2\operatorname{Re}\langle E_{s,i},E_{b,i}\rangle,\quad
B_i=\|E_{b,i}\|^2+N_i,
\]
\[
T=T_S+\sum_i L_i(\mathbf d)+\sum_i\bigl(\|E_{b,i}\|^2+N_i\bigr).
\]
Ten complex transverse pupil modes describe one fixed common field shared by both windows; $N_i$ is a constant noninterfering positive-Stokes background. There are 23 shared background parameters and five cone/brightness/offset parameters per window. $B_i$ stays constant; $L_i$ changes through the predicted rod field, not free per-point coefficients. The shared total background is capped at 95\% of the stricter original minimum intensity. Generality assumes fixed-position ideal-dipole common-pupil optics.

Sphere diagnostic uses $\widehat S_{i,j}=I_{i,j}^{\rm heldout}-B_i-L_i(\mathbf d_j^{\rm fit})$. This is a \emph{conditional subtraction at the matched fitted orientation}, not a self-consistent inverse of the full field model. It shows how its intensity residual is amplified by signal-only inversion. The three panels therefore compare explicitly stated correction diagnostics, not identical inverse estimators.
\newpage
{\Large 2A. Background only}\par
Four-channel fits and corrected held-out points; no new fit to sphere coordinates.

\includegraphics[width=\linewidth,height=.88\textheight,keepaspectratio]{square.pdf}
\newpage
{\Large Sphere and cone-frame residuals}\par

\includegraphics[width=\linewidth,height=.88\textheight,keepaspectratio]{square_sphere.pdf}
\newpage
{\Large 2B. Richard exact four-C}\par
Sphere points use the exact channelwise quadratic inverse with the fitted constants.

\includegraphics[width=\linewidth,height=.88\textheight,keepaspectratio]{richard_product.pdf}
\newpage
{\Large Sphere and cone-frame residuals}\par

\includegraphics[width=\linewidth,height=.88\textheight,keepaspectratio]{richard_product_sphere.pdf}
\newpage
{\Large 2C. General stationary field}\par
Shared field across both windows. Sphere points use conditional fitted cross-term subtraction.

\includegraphics[width=\linewidth,height=.88\textheight,keepaspectratio]{stationary.pdf}
\newpage
{\Large Sphere and cone-frame residuals}\par

\includegraphics[width=\linewidth,height=.88\textheight,keepaspectratio]{stationary_sphere.pdf}
\newpage
{\Large Reading the unit-sphere and cone-frame residual plots}

From corrected channels, calculate $(\widehat X,\widehat Y)$, $r=\sqrt{\widehat X^2+\widehat Y^2}$, then
\[
\sin^2\widehat\theta=\frac{A_F r}{C_F-B_F r},\qquad
\widehat\phi=\tfrac12\operatorname{atan2}(\widehat Y,\widehat X).
\]
Only nonnegative corrected channels and $0\leq\sin^2\widehat\theta\leq1$ are plotted. Points outside this domain are marked with red crosses along the bottom residual panel, not clipped onto the equator. This necessary domain test does not enforce pair-sum balance or the full brightness law.

\textbf{Anisotropy overlays.} Each model has separate measured-space (held-out mean versus fitted measured curve) and signal-space plots ($S^{\rm fit}+\varepsilon$ versus $S^{\rm fit}$). Green is calculated from $S^{\rm fit}+\varepsilon$, with $\varepsilon=I^{\rm heldout}-I^{\rm pred}$, for all three models. Red crosses flag negative corrected channels or an anisotropy outside the signal-only inversion domain. These coordinates are not physically clipped. Triangles at the plot boundary indicate points beyond the fixed $[-1,1]$ axes; they are display indicators, not reconstructed positions. The dotted circle marks $r_{\max}$. For Richard, this green frozen-cross-term diagnostic differs from the exact quadratic inverse used in its sphere panel; do not equate the two sets of points.

\textbf{Branch choice.} For each point, enumerate the four combinations $\widehat\phi$ or $\widehat\phi+\pi$, and $\widehat\theta$ or $\pi-\widehat\theta$. Choose the unit vector maximizing its dot product with the \emph{already matched} fitted cone direction. This changes only a measurement-degenerate branch; it does not move the point onto the cone or choose a new correspondence. It is a cone-conditioned convention, not independent recovery of direction or proof of continuous branch tracking.

\textbf{Laboratory sphere plots.} All sphere coordinates retain the laboratory frame, with optical axis $z$. The fitted cone tilt is preserved; a dashed line shows its axis $\mathbf k$. The circle centre is $\mathbf c=\cos\lambda\,\mathbf k$, and the faint disk lies in its plane. Grey connectors show every fourth valid matched residual.

\textbf{Cone-frame residuals.} Rotate only for calculating residual coordinates, so $z'$ follows $\mathbf k$. Draw the vector from the fitted circle centre to the corrected point, $\mathbf v_j=\widehat{\mathbf d}_j-\mathbf c$. Its signed elevation above the circle plane is
\[
\beta_j=\operatorname{atan2}\left(\mathbf v_j\cdot\mathbf k,
\|\mathbf v_j-(\mathbf v_j\cdot\mathbf k)\mathbf k\|\right)
=\operatorname{atan2}\left(\widehat d'_{z,j}-\cos\lambda,
\sqrt{\widehat d_{x,j}'^2+\widehat d_{y,j}'^2}\right).
\]
Blue is $\beta$: zero on the fitted circle, positive on the cone-axis side of its plane. It is an angle measured from the \emph{circle centre}, not lab $\theta$ or a polar-angle residual measured from the sphere centre.
\[
\Delta\chi_j=\operatorname{wrap}_{[-\pi,\pi)}
\left[\operatorname{atan2}(\widehat d'_{y,j},\widehat d'_{x,j})-
\operatorname{atan2}(d'^{\rm fit}_{y,j},d'^{\rm fit}_{x,j})\right].
\]
Orange is $\sin\lambda\,\Delta\chi$, the signed displacement around the fitted circle on a unit sphere. It retains the original matched correspondence. Horizontal position is ordered reference arc, not time or uniformly spaced mechanical phase. For small departures, $\beta\simeq-(\widehat\alpha-\lambda)$, where $\widehat\alpha$ is polar angle from the cone axis, and squared geodesic error is approximately $\beta^2+\sin^2\lambda(\Delta\chi)^2$. For large deviations these approximations do not hold.

\begin{tabular}{llrrrr}\toprule
Model & Window & Invalid/128 & Elevation RMS & Along RMS & Geodesic RMS\\\midrule
'''
for a in stats:
 label={'square':'Background only','richard_product':'Richard','stationary':'Stationary'}[a['model']]
 t+=f"{label} & {a['interval']} & {a['invalid']} & {a['cross_rms_valid']:.1f}$^\\circ$ & {a['along_rms_valid']:.1f}$^\\circ$ & {a['geodesic_rms_valid']:.1f}$^\\circ$ \\\\\n"
t+=r'''\bottomrule\end{tabular}

\textbf{Interpretation.} RMS values above include valid points only, with different valid subsets; they must not be used alone to rank models. Even the stationary fit leaves 21/23 points outside the signal-only inversion domain. Its close measured-anisotropy fit does not imply small or everywhere-defined corrected-angle errors. Near the equator the inversion is poorly conditioned; near low signal, subtraction magnifies residuals. Pair-sum mismatch is also retained rather than silently repaired.
\newpage
{\Large Fit composition and limits of this draft}

The following ranges describe the \emph{forward fitted} signal and cross term at the 128 matched positions, normalized by original training maximum total intensity. They are not recovered experimental component measurements.

\begin{tabular}{llrrrr}\toprule
Model & Window & BG/\(T_{\max}\) & Signal/\(T_{\max}\) & Cross/\(T_{\max}\) & Fitted folded \(\theta\)\\\midrule
'''
for a in stats:
 key=a['interval'];model=a['model'];o=old[key+'_'+model];mx=(f.r.Z[key+'_train']/f.Z['scale']).sum(0).max();s=100*np.array(o['signal']).sum(0)/mx;c=100*np.array(o['cross']).sum(0)/mx;th=o['theta'];label={'square':'BG only','richard_product':'Richard','stationary':'Stationary'}[model];bg='---' if a['bg_percent'] is None else f"{a['bg_percent']:.1f}\\%"
 t+=f"{label} & {key} & {bg} & {s.min():.1f}--{s.max():.1f}\\% & {c.min():.1f}--{c.max():.1f}\\% & {min(th):.1f}--{max(th):.1f}$^\\circ$ \\\\\n"
t+=r'''\bottomrule\end{tabular}

Background-only does not identify a physical coherent/noncoherent composition: it includes no interference term. Richard's model specifies a signed correction, not an additive background power, so no physical BG percentage is inferred from it. For the selected stationary field, essentially 100\% of represented background power is assigned to the coherent component, with negligible fitted $N_i$. This is conditional on this model and optimum; it is not a source measurement.

\begin{tabular}{llrr}\toprule
Model & Window & Held-out channel RMS & Corrected pair-imbalance RMS\\\midrule
'''
for a in stats:
 label={'square':'Background only','richard_product':'Richard','stationary':'Stationary'}[a['model']];t+=f"{label} & {a['interval']} & {a['heldout_channel_rms']:.5f} & {100*a['pair_imbalance_rms']:.2f}\\% \\\\\n"
t+=r'''\bottomrule\end{tabular}

Channel RMS is $\sqrt{\operatorname{mean}_{i,j}[(I^{pred}_{i,j}-I^{heldout}_{i,j})/T^{heldout}_j]^2}$. Pair imbalance is $(\widehat S_0+\widehat S_{90}-\widehat S_{45}-\widehat S_{135})/\sum_i\widehat S_i$, using all bins, including those not invertible. A unit-sphere plot alone discards this additional inconsistency.

\textbf{What the new plots establish.} The stationary field gives much better forward agreement under fixed ordered correspondence, but residuals can become substantial or nonphysical after correction. The unit-sphere test exposes that amplification; it does not independently validate the fitted cone. Reference-conditioned branch selection chooses the visually closest permitted branch and therefore gives an optimistic branch-resolved comparison.

\textbf{What has not changed.} No model was refitted for this draft. The selected stationary optimization converged, but multiple starts do not certify a global optimum. Background is near its cap: 54.84\% of the smaller original maximum versus an allowed 55.69\%. Another local solution around 47\% background has narrower recovered angles and slightly poorer fit. Absolute angles and physical background composition remain unvalidated.

\textbf{Numerical consistency.} The stationary forward calculation uses cached pupil quadrature; sphere inversion uses the analytic annular coefficients above. The tested unit-signal difference is below $7.7\times10^{-5}$, much smaller than current residuals. Doubling forward curve density changes predicted channels by less than $5\times10^{-8}T_{\max}$. Invalid samples are retained in diagnostics, not hidden by clipping or selection. The sphere residuals are conditional diagnostics, not uncertainty estimates or a new fit objective.

Scripts and arrays: draft/build.py (sphere/branch/residual calculations), draft/document.py (this LaTeX report), draft/sphere\_metrics.json and six sphere NPZ files. Original fits and decompositions remain in the parent directory. No commit or push.
\end{document}
'''
(R/'report.tex').write_text(t)
