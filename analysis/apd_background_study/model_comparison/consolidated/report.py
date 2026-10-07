from pathlib import Path
import sys,json,io,csv,os
import numpy as np
from sphere_branches import recover
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util
sp=importlib.util.spec_from_file_location('consolidated_runner',Path(__file__).with_name('run.py'));a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
R=a.R;OUT=R/'figures';OUT.mkdir(exist_ok=True)
DATA={}
for name in ['simple','fields','axial']:
 if (R/(name+'.json')).exists():DATA.update(json.load(open(R/(name+'.json'))))
ORDER=[m for m in a.LABEL if m in DATA]
PARAM={'none':10,'square':16,'richard_product':18,'complex':24,'uniform':17,'three':19,'five':23,'general':33,'edge':25,'axial':37}
NAMES={'none':'No background','square':'Additive background','richard_product':'Richard four-C','complex':'Additive plus bounded interference','uniform':'Uniform pupil field','three':'Three-mode field','five':'Five-mode field','general':'General stationary field','edge':'Hole-edge field','axial':'General field with axial motion'}
DESC={
'none':r'''\[I_i=S_i,\qquad T=T_S.\]
Fit only the cone (three parameters), brightness $a_0$ and cyclic offset for each window. This is the reference for whether adding a correction actually improves the measured intensities. There is no background parameter to cap.''',
'square':r'''\[I_i=S_i+B_i,\qquad T=T_S+B_{tot}.\]
\[\mathbf B=\frac{B_{tot}}4(1+f_x,1-f_x,1+f_y,1-f_y),\quad |f_x|,|f_y|\leq1.\]
Fit five cone/brightness/offset parameters plus $B_{tot},f_x,f_y$ separately per window. The background is constant, nonnegative and has equal pair sums. No Stokes-disk restriction is imposed, matching the recovered notebook form. Its coherent versus noninterfering composition is not identifiable because no cross term is modelled.''',
'richard_product':r'''\[I_i=S_i+C_i\sqrt{S_i},\qquad T=T_S+\sum_iC_i\sqrt{S_i}.\]
Fit five cone/brightness/offset parameters plus four constant signed $C_i$ separately per window. This is Richard's exact proposed approximation, without an additive $B_i$. Numerical variables are $\alpha=a_0^2$ and $K_i=a_0C_i$, so $I_i=\alpha f_i+K_i\sqrt{f_i}$. A background-power cap is \emph{not defined}; imposing one on $C_i$ would change the model. The limit $\alpha\to0$, $C_i\to\infty$ can preserve the product term.

For uniform comparison the green curves and penalties here use frozen fitted cross-term subtraction $I^{obs}-C_i\sqrt{S_i^{fit}}$. This differs from the exact quadratic inverse used in an earlier sphere report. Four independent $C_i$ do not guarantee interference pair balance or a common physical background field.''',
'complex':r'''\[I_i=S_i+B_i+2\rho_i\sqrt{B_iS_i},\qquad |\rho_i|\leq1.\]
\[T=T_S+B_{tot}+2\sum_i\rho_i\sqrt{B_iS_i}.\]
Use the same three-parameter nonnegative equal-pair-sum additive background as above, plus four constant overlap coefficients $\rho_i$. Fit these seven correction parameters and five cone/brightness/offset parameters per window. Each channel obeys a scalar interference bound. This alone does not establish a realizable common pupil field or interference pair balance. $B_i$ is the represented available background power; its coherent/noninterfering split is not separately inferred.''',
'uniform':r'''\[E_b=b_1u_x+b_2u_y,\qquad I_i=S_i+L_i+B_i.\]
The two normalized pupil modes have uniform scalar amplitude and orthogonal transverse polarization. Fit two complex field coefficients and a noninterfering Stokes background, shared across windows. Equivalent background coordinates are total power, coherent fraction, noninterfering polarization degree/angle and three real angles specifying the normalized complex field. Seven shared background parameters plus ten local parameters. This restricts spatial overlap strongly.''',
'three':r'''\[E_b=b_1u_x+b_2u_y+b_3u_{d_z},\qquad I_i=S_i+L_i+B_i.\]
Add a dipole-derived longitudinal-orientation pupil field to the two uniform modes, orthonormalized under the detector integral. Three complex coefficients plus noninterfering Stokes background: nine shared background parameters and ten local parameters. This is the earlier radial/dipole-overlap family expressed as a common stationary field.''',
'five':r'''\[E_b=\sum_{m=1}^{5}b_mu_m,\qquad I_i=S_i+L_i+B_i.\]
Modes comprise two uniform-polarization fields and three dipole-derived vector fields, orthonormalized. Five complex coefficients plus noninterfering Stokes background: thirteen shared background parameters and ten local parameters. Spatial shape, polarization and phase are fixed throughout both windows; only the signal direction changes.''',
'general':r'''\[E_b=\sum_{m=1}^{10}b_mu_m,\qquad I_i=S_i+L_i+B_i.\]
Ten complex transverse pupil modes span the scalar dipole-component overlaps in this fixed-position, common-pupil model. Twenty-three shared background parameters and ten local parameters. This does not cover arbitrary moving-source fields or independent downstream analyser-path backgrounds. Uniform modes can contain a component orthogonal to this overlap span, so different basis families are not automatically identical in their background-power representation.''',
'edge':r'''\[E_b=\sum_{m=1}^{6}b_mu_m,\qquad
u_m\propto e^{-(\rho-0.38)^2/(2w^2)}\{1,\cos\varphi,\sin\varphi\},\quad w=0.10\ \mathrm{NA}.\]
Each scalar shape is used in two transverse polarizations and orthonormalized. Fit six complex coefficients plus a noninterfering Stokes background (fifteen shared plus ten local parameters). The width is fixed, not fitted. This draft adds the noninterfering component to the earlier pure edge-field surrogate. It tests a restricted pupil pattern, not a full model of multiple reflection or hole-edge scattering.''',
'axial':r'''\[z(\chi)=R_h\sin\beta_{axis}\cos(\chi+\eta),\qquad 0\leq R_h\leq0.30\ \mu\mathrm m.\]
Use the general field with an orientation-linked axial trajectory, fitting $R_h,\eta$ per window in addition to the 33 stationary parameters. The 300 nm orbit-radius bound is a sensitivity assumption, not a measured hook length. The mean plane is fixed at the assumed excitation waist.
\[F(z)=[1+(z/z_R)^2]^{-1/2},\quad z_R=2\ \mu\mathrm m,\quad
S_i=a_0^2\sin^2\theta F(z)^2Q_i.\]
The pupil overlap carries the phase factor\[\exp\{\mathrm i(2\pi/\lambda_0)[n_w+\sqrt{n_w^2-\rho^2}]z\},\qquad \lambda_0=633\ \mathrm{nm}.\]The excitation also contributes the Gouy phase $-\arctan(z/z_R)$. Thus $L_i$ changes with both orientation and height. Background power stays constant. This retains the earlier bounded axial-phase/focus model; it is not a full finite-detector image-space defocus calculation.'''}

def sav(fig,name):
 if os.environ.get("REUSE_ALL_FIGURES") and (OUT/(name+".pdf")).exists():plt.close(fig);return
 if os.environ.get("SPHERES_ONLY") and not name.endswith("_sphere") and (OUT/(name+".pdf")).exists():plt.close(fig);return
 b=io.BytesIO();fig.savefig(b,format='pdf',bbox_inches='tight');(OUT/(name+'.pdf')).write_bytes(b.getvalue());plt.close(fig)
def titlecap(tag):return 'No BG-power cap' if tag=='uncapped' else tag.replace('pct','% of maximum')
def tableescape(x):return str(x).replace('_',r'\_').replace('%',r'\%')
localstats=json.load(open(R/'local_residual'/'representative.json'))
rows=[];sphereinfo={}
for model in ORDER:
 for tag,b in DATA[model]['cases'].items():
  fig,axs=plt.subplots(3,2,figsize=(10,12),layout='constrained')
  for i,k in enumerate(a.g.KEYS):
   sc=b['scores'][k];pred=np.array(sc['pred']);sig=np.array(sc['signal']);cross=np.array(sc['cross']);bg=np.array(sc['background']);dat=a.f.TE[i];u=a.f.U[i];mx=a.f.original[i].sum(0).max();d=np.array(sc['directions']);cor=dat-cross-bg[:,None];xy=a.g.xy(cor);rr=np.linalg.norm(xy,axis=1);bad=~np.isfinite(rr)|(rr>a.f.rmax+1e-8)|(cor<0).any(0)
   if model in a.MODES:
    a.setup(model,None if tag=='uncapped' else float(tag[:-3])/100);p=np.array(b['p'])
    if model=='axial':
     orig=a.f.U[i];a.f.U[i]=np.linspace(0,1,1025);parts=a.fieldpredict(model,p,b['directions'],i,8192);a.f.U[i]=orig;curve,csig=parts[:2]
    else:_,_,curve,csig,*_=a.g.dense(p[5*i:5*i+5],p[10:],4096)
   else:
    sol=b['solutions'][i];p=np.array(sol['p']);parts=a.g.r.forward(p,model,a.f.original[i].sum(0).min(),sol['direction'],np.linspace(0,1,1025),8192,True);curve,csig=parts[:2]
   ax=axs[0,i];ax.plot(*a.g.xy(curve).T,color='#ce6436',lw=1.8,label='Fitted measured curve');ax.scatter(*a.g.xy(dat).T,s=10,color='#172d3c',label='Held-out mean');ax.set_title(('30–31' if k=='30s' else '10–11')+' s · measured space')
   ax=axs[1,i];ax.plot(*a.g.xy(csig).T,color='#ce6436',lw=1.8,label='Fitted rod signal');ax.plot(*np.vstack([xy,xy[0]]).T,color='#398849',lw=1.1,label='Signal + held-out residual');ax.scatter(*xy[bad].T,s=15,marker='x',color='#c82432',label='Invalid corrected point');out=(abs(xy)>1).any(1)
   if out.any():ax.scatter(*(xy[out]/np.max(abs(xy[out]),axis=1)[:,None]*.98).T,s=18,marker='^',color='#c82432')
   tt=np.linspace(0,2*np.pi,300);ax.plot(a.f.rmax*np.cos(tt),a.f.rmax*np.sin(tt),':',color='.7',lw=.7);ax.set_title(f'Signal space · {bad.sum()}/128 invalid')
   for j in [0,1]:axs[j,i].set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y');axs[j,i].set_aspect('equal');axs[j,i].grid(alpha=.15);axs[j,i].legend(fontsize=6,loc='lower left')
   ax=axs[2,i];ax.plot(u,dat.sum(0)/mx,color='#172d3c',lw=1.1,label='Observed total');ax.plot(u,sig.sum(0)/mx,color='#ce6436',label='Rod signal');ax.plot(u,cross.sum(0)/mx,color='#7961a6',label='Signed interference');ax.axhline(bg.sum()/mx,color='#398849',label='Total BG')
   if 'incoherent' in sc:ax.axhline(np.sum(sc['incoherent'])/mx,color='#398849',ls=':',label='Noninterfering BG')
   ax.axhline(0,color='.65',lw=.6);ax.set(xlabel='Ordered reference arc fraction',ylabel='Fraction of original measured Tmax',xlim=(0,1));ax.grid(alpha=.15);ax.legend(fontsize=6,ncol=2,loc='best')
   rec=dict(model=model,case=tag,window=k,converged=b['success'],bg_percent=sc['test']['bg_percent'],measured_rms=sc['test']['channel_rms'],signal_rms=sc['test']['signal_rms'],train_invalid=sc['train']['invalid'],test_invalid=sc['test']['invalid'],max_r=sc['test']['max_r'],theta_min=sc['test']['theta_range'][0],theta_max=sc['test']['theta_range'][1],signal_min=100*sig.sum(0).min()/mx,signal_max=100*sig.sum(0).max()/mx,cross_min=100*cross.sum(0).min()/mx,cross_max=100*cross.sum(0).max()/mx,channel_bg_percent=(100*bg/mx).tolist(),noninterfering_percent=(100*np.sum(sc['incoherent'])/mx if 'incoherent' in sc else None));rows.append(rec)
  fig.suptitle(NAMES[model]+' · '+titlecap(tag)+(' · NOT CONVERGED' if not b['success'] else ''),fontsize=14);sav(fig,model+'_'+tag)
 # Representative sphere/residual: 10% for comparable display, otherwise unbudgeted baseline.
 tag='10pct' if '10pct' in DATA[model]['cases'] else 'uncapped';b=DATA[model]['cases'][tag];fig=plt.figure(figsize=(10,9));gs=fig.add_gridspec(2,2,height_ratios=[1.3,.8],hspace=.25,wspace=.25);summary=[]
 for i,k in enumerate(a.g.KEYS):
  sc=b['scores'][k];cor=a.f.TE[i]-np.array(sc['cross'])-np.array(sc['background'])[:,None];d=np.array(sc['directions']);obs,ok,branch_audit=recover(cor,d,a.f.U[i])
  geometry=(np.array(b['p'])[5*i:5*i+3] if model in a.MODES else np.array(b['solutions'][i]['p'])[:3]);th,ph,lam=geometry;axis=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]);e1=np.array([-np.cos(th)*np.cos(ph),-np.cos(th)*np.sin(ph),np.sin(th)]);F=np.column_stack([e1,np.cross(axis,e1),axis]);od=obs@F;dd=d@F;beta=np.degrees(np.arctan2(od[:,2]-np.cos(lam),np.linalg.norm(od[:,:2],axis=1)));delta=np.arctan2(od[:,1],od[:,0])-np.arctan2(dd[:,1],dd[:,0]);along=np.degrees(np.sin(lam)*np.arctan2(np.sin(delta),np.cos(delta)));geo=np.degrees(np.arccos(np.clip(np.sum(obs*d,axis=1),-1,1)))
  ax=fig.add_subplot(gs[0,i],projection='3d');tt=np.linspace(0,2*np.pi,37);uu=np.linspace(0,np.pi,19);ax.plot_wireframe(np.outer(np.cos(tt),np.sin(uu)),np.outer(np.sin(tt),np.sin(uu)),np.outer(np.ones_like(tt),np.cos(uu)),color='.7',alpha=.18,lw=.4,rstride=3,cstride=3);curve=a.g.cone(geometry,np.linspace(0,2*np.pi,361));ax.plot(*curve.T,color='#ce6436',lw=2);ax.scatter(*obs[ok].T,s=9,color='#172d3c');validids=np.flatnonzero(ok);ax.scatter(*obs[validids[0]],s=38,color='#159d82',marker='o',label='Start');ax.scatter(*obs[validids[-1]],s=45,color='#ad3792',marker='x',label='End');ax.legend(fontsize=7,loc='upper left');ax.plot(*np.stack([np.zeros(3),axis]).T,color='#ce6436',ls='--',lw=1);ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel='x',ylabel='y',zlabel='Optical z',title=k+f' · {ok.sum()}/128 valid');ax.set_zlabel('');ax.text2D(.94,.62,'Optical z',transform=ax.transAxes,rotation=90,fontsize=8);ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-1,0,1]);ax.set_box_aspect([1,1,1]);ax.view_init(22,-55)
  ax=fig.add_subplot(gs[1,i]);ax.plot(a.f.U[i],beta,color='#167d9a',label='Circle-centre elevation β');along_plot=along.copy();wraps=np.flatnonzero(np.abs(np.diff(delta-np.round(delta/(2*np.pi))*2*np.pi))>np.pi)+1;along_plot[wraps]=np.nan;ax.plot(a.f.U[i],along_plot,color='#b1543b',label='Around-circle displacement');ax.axhline(0,color='.6',lw=.7);ax.set(xlim=(0,1),ylim=(-90,90),xlabel='Ordered reference arc fraction',ylabel='Degrees');limit=max(90,30*np.ceil(np.nanmax(np.abs(along))/30));ax.set_ylim(-limit,limit);ax.grid(alpha=.15);ax.scatter(a.f.U[i][~ok],np.full((~ok).sum(),-85),marker='x',color='#c82432',s=12);ax.legend(fontsize=7)
  summary.append(dict(window=k,valid=int(ok.sum()),branch_audit=branch_audit,geo_rms=float(np.sqrt(np.nanmean(geo**2))) if ok.any() else None))
 fig.suptitle(NAMES[model]+' · '+titlecap(tag)+' · sphere/residual diagnostic',fontsize=14);sav(fig,model+'_sphere');sphereinfo[model]=dict(case=tag,stats=summary)
(R/'composition.json').write_text(json.dumps(rows,indent=2));(R/'sphere_summary.json').write_text(json.dumps(sphereinfo,indent=2))
with (R/'metrics.csv').open('w') as ff:
 w=csv.DictWriter(ff,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=15mm]{geometry}\usepackage{fontspec,amsmath,amssymb,graphicx,booktabs,longtable,hyperref}
\setmainfont{Latin Modern Roman}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\hypersetup{colorlinks=true,linkcolor=blue,pdftitle={APD consolidated model and background-cap comparison}}\setcounter{tocdepth}{1}
\begin{document}
{\LARGE APD background and interference models}\par
{\large Consolidated cap-sweep draft: exact and local angular diagnostics}\par 7 October 2026 · for discussion with Richard

This draft compares ten model families where available, using the same gain-corrected reference construction and intensity objective. Its purpose is to expose the trade-off between forward agreement, background budget and stability of recovered orientations. No model supplies independently validated rod angles.

\textbf{Reading order.} Each model has equations and an explicit list of fitted quantities, a cap summary, then one full page per tested budget. Each case page shows measured-space fits (top), corrected signal space (middle), and signal/background/interference composition (bottom), in 30--31 s and 10--11 s columns. A representative sphere/residual page follows. Tables show actual fitted powers, not merely their allowed caps.

\textbf{Scope.} The primary sweep includes no BG, additive BG, Richard four-C, additive plus bounded interference, uniform pupil, three-mode, five-mode, general stationary, edge-localized pupil and bounded axial-motion families. Earlier mask, unconstrained-overlap and inverse-circle tests are summarized as diagnostics rather than mixed into the cap ranking. Superseded reference-coordinate fits and free per-point brightness are not treated as current results.

\textbf{Uncapped means uncapped.} No finite upper bound is imposed on $B_{tot}$ in the new uncapped fits. Earlier ``broad cap'' figures instead imposed $B_{tot}<0.95\min T_{obs}$ and are not relabelled uncapped. Numerical bounds remain on geometry and brightness; finite local solutions do not prove global identifiability.

\textbf{Adaptive sweep.} Test 5\% and 10\% for every model with a defined additive background power. If neither produces radial exceedance or negative corrected channels in either training or held-out window, test 20\%,30\%,\ldots until the first tested failure. The first failing case is included. Stop also if the cap becomes inactive and reproduces an admissible uncapped solution. No-BG and Richard have no BG-power cap to vary. This is an exploratory path through local solutions, not a proof that every solution below a threshold is valid or every solution above it fails. Held-out data now help decide which budgets to explore; they are repeatability diagnostics, not untouched validation data.

\textbf{Failure definition.} A radial exceedance is $r>r_{sat}+10^{-8}$ with $r_{sat}=0.92778057$. Negative corrected channels are tracked as a separate failure. No physical clipping is performed. Plot triangles mark off-axis data; axes remain $[-1,1]$ in both anisotropy directions. Tiny numerical exceedances and large failures should be distinguished using the composition/metric files.

\tableofcontents
\newpage
\section{Processing, calibration and common objective}
Raw APD channels are reordered from $(90,45,135,0)$ to $(0,90,45,135)$. Gains act before the unchanged inverse transmission matrix. A single pooled training pair-balance adjustment multiplies the historical inverse gains by $(0.992729,1.005313,1.009114,0.992952)$, with product one. Gains are then frozen for every model and interval. Held-out pair imbalance remains about 0.30\% RMS; this is conditional pair-balance calibration, not independent proof of detector gains or absence of path-specific background.

Retain complete-cycle detection and original ordered channel bins: 304 cycles (152/152 alternating train/test) at 30--31 s and 217 (109/108) at 10--11 s. Apply the new calibration to cached four-channel means; average training and held-out groups separately. Rebuild uniform final-reference arc from training data. No current fit changes cycle selection.

Interpolate the four training channels together, including last-to-first, then calculate
\[
X=\frac{I_0-I_{90}}{I_0+I_{90}},\qquad Y=\frac{I_{45}-I_{135}}{I_{45}+I_{135}}.
\]
Integrate ordered XY distance on this interpolant and select 128 equal arc fractions. Carry all four interpolated intensities into the reference. Apply identical original-bin interpolation weights to held-out means, which are not independently re-uniformized. Both loops and repeated passes retain their order. This operation introduces correlated neighboring points, not new independent observations.

For each proposed physical model simulate a full mechanical revolution, form measured-model channels and anisotropy, then sample its own ordered arc at $(\delta+\epsilon u_j)\bmod1$. The measured points and training-derived coordinates $u_j$ remain fixed. One cyclic offset per window, no 16-parameter correspondence warp or independent nearest-point assignment. Current restarts retain historical traversal directions; these local searches are not exhaustive over every initialization.

Let $\varepsilon=I^{obs}-I^{pred}$, and evaluate the fitted correction at the matched cone direction:
\[
\widehat S=I^{obs}-B-L(d^{fit})=S^{fit}+\varepsilon.
\]
The fit objective, applied to training means only, is
\[
J=\sum_{ij}\left(\frac{\varepsilon_{ij}}{T_j^{obs}}\right)^2
+100\sum_j\left[\max\left(0,\frac{\widehat r_j-r_{sat}}{1-r_{sat}}\right)\right]^6
+10\sum_{ij}\left[\frac{\min(0,\widehat S_{ij})}{T_j^{obs}}\right]^2.
\]
The radial penalty is exactly zero inside $r_{sat}$ and 100 per point at $r=1$. It uses a pair-denominator guard $10^{-4}T^{obs}$ near zero to avoid numerical singularities; diagnostic ratios use unmodified channels. Negative predicted measured channels are additionally penalized in phenomenological intensity models. Signal-relative RMS is reported only, not minimized. No weights were tuned on the held-out score for each case.

The common-field optical families enforce pair balance for $S$, $B$ and $L$ separately by construction. Additive BG also has equal pair sums. Independent-channel Richard/overlap coefficients are explicit phenomenological exceptions and are not silently redefined as common-field models.

Physical field backgrounds are shared across intervals; simple intensity corrections are fitted per interval. Thus parameter sharing differs and model fit quality is not a complexity-adjusted statistical comparison. Field caps use the smaller original calibrated training $T_{max}$, satisfying the cap in both windows; simple models use each window's own $T_{max}$.
\newpage
\section{Signal and field composition}
A fixed cone is parameterized by its axis (two angles) and opening $\lambda$:
\[
d(\chi)=\cos\lambda\,k+\sin\lambda(\cos\chi\,e_1+\sin\chi\,e_2).
\]
\[
Q_i=A_F+B_F\sin^2\theta+C_F\sin^2\theta\cos2(\phi-\psi_i),
\qquad S_i=a_0^2\sin^2\theta Q_i.
\]
\[
T_S=4a_0^2\sin^2\theta(A_F+B_F\sin^2\theta).
\]
One fitted brightness $a_0$ per interval. The excitation factor $\sin^2\theta$ is not an independent per-point brightness. $A_F=0.890806$, $B_F=0.109194$, $C_F=0.927781$. NA0.38--1.3, water/oil1.33/1.51, Fresnel and corrected BFP mapping are held fixed. Physical field families use common numerical pupil quadrature; the analytic scalar formula differs slightly through quadrature, not through a changed optical model.

For every physical field family,
\[
E_{s,i}=a_0(d_x+\mathrm i d_y)\mathcal L_i d,\qquad E_b=\sum_m b_mu_m,
\]
\[
I_i=\underbrace{\|E_{s,i}\|^2}_{S_i}
+\underbrace{2\operatorname{Re}\langle E_{s,i},E_{b,i}\rangle}_{L_i}
+\underbrace{\|E_{b,i}\|^2+N_i}_{B_i},
\]
\[
N_i=\frac{N_{tot}}4[1+P\cos2(\psi_i-\gamma)],\quad0\leq P\leq1,
\qquad T=T_S+\sum_iL_i+B_{tot}.
\]
The detector norm/inner product integrates spatial intensity/overlap; fields are not summed across independent detector pixels before squaring. $B_i$ and $N_i$ are constant, while $S_i,L_i$ vary with orientation (and height in the axial family). The field parameterization uses total BG power, coherent fraction, $P,\gamma$, and $2m-1$ real angles for a normalized complex $m$-vector: $2m+3$ shared BG parameters. Coherent fractions are conditional fitted allocations, not measured source fractions.

Channel BG values in each model table are percentages of the original measured maximum, in $(0,90,45,135)$ order. Signal and cross-term ranges refer to sums over channels at matched reference positions. Their extrema need not occur at the same point, so range endpoints should not be added together.

\subsection*{Sphere and residual interpretation}
For corrected channels, $\sin^2\widehat\theta=A_Fr/(C_F-B_Fr)$ and $\widehat\phi=\tfrac12\arg(X+\mathrm iY)$. Negative channels or an out-of-range sine square are excluded from the sphere plot, counted and marked; never clipped. Choose among $\phi,\phi+\pi$ and $\theta,\pi-\theta$ jointly along the ordered cycle. Dynamic programming minimizes the sum of squared angular steps divided by their reference-arc spacing, without forcing the last point back onto the first branch. The matched cone only breaks ties between equally smooth branch sequences. At the seam, the initial anisotropy is lifted onto the branch nearest the final direction; its separation from the initial direction is reported as the oriented closure mismatch (the head--tail-equivalent rod mismatch is the smaller of this angle and its supplement). A continuous lift need not close: one winding of anisotropy about the origin changes $\phi$ by $\pi$. For example, the no-BG reconstruction cannot be made both continuous and closed here. This supersedes the earlier pointwise snapping diagnostic. No points are projected onto the cone or moved to close a gap. Invalid points remain excluded; continuity across an invalid interval is not measured. This continuity convention cannot establish the true polar-angle branch, especially near a turning point or equator crossing. Sphere residuals are recomputed using this revised convention.

Sphere plots retain laboratory tilt. In the cone frame, with circle centre $c=\cos\lambda\,k$, the elevation residual is
\[
\beta=\operatorname{atan2}(\widehat d'_z-\cos\lambda,\sqrt{\widehat d_x'^2+\widehat d_y'^2}).
\]
The second residual is $\sin\lambda\operatorname{wrap}(\widehat\chi-\chi^{fit})$, measuring displacement around the matched circle. These plots are conditional subtraction diagnostics; interference has not been re-solved self-consistently at each corrected direction. Valid-only angular RMS cannot rank models with different excluded subsets.
\newpage\subsection*{Local cone-referenced angular residuals}
This additional diagnostic tests a local approximation alongside exact inversion. At each matched fitted cone direction $d_j$, take the corrected held-out anisotropy minus the fitted signal anisotropy:
\[e_j=XY(I^{heldout}_j-B-L_j^{fit})-XY(S_j^{fit}).\]
The interference term is still evaluated at the fitted direction. This is a conditional diagnostic, not a self-consistent refit of interference.

Let $k$ be the fitted cone axis, $\lambda$ its half-angle, and define unit tangent directions
\[t_\perp=(\cos\lambda\,d-k)/\sin\lambda,\qquad t_\parallel=(k\times d)/\sin\lambda.\]
The first points towards increasing cone angle; the second follows increasing mechanical phase. Using the signal-only anisotropy map $F(d)$, calculate its local two-by-two Jacobian and solve
\[J_j=\left[\partial_{t_\perp}F\quad\partial_{t_\parallel}F\right],\qquad
\begin{pmatrix}\delta\lambda\\\delta s\end{pmatrix}=J_j^{-1}e_j,
\qquad \delta s\simeq\sin\lambda\,\delta\psi.\]
This uses anisotropy residuals, not a new four-channel intensity objective. It does not constrain total brightness or pair balance. Brightness factors cancel in signal anisotropy. The analytic annular Fourkas map is used consistently for the derivative; residuals are measured relative to the actual numerical fitted signal so quadrature differences do not create a spurious zero-residual shift.

To display the displacement on the unit sphere, use its tangent-plane exponential map:
\[v=\delta\lambda\,t_\perp+\delta s\,t_\parallel,\quad a=\|v\|,\qquad
 d_{local}=\cos a\,d+\frac{\sin a}{a}v.\]
This stays on the sphere but need not stay on the cone. No clipping, smoothing or residual shrinkage is applied. This is a first-order angular estimate, not an exact inverse. It can form a closed path because the reference geometry and residual are periodic; closure here is not independent evidence that the model is correct.

\textbf{Check the approximation.} Re-evaluate the nonlinear signal map and calculate
\[\eta_j=\frac{\|F(d_{local})-F(d_j)-e_j\|}{\max(\|e_j\|,10^{-8})}.\]
A small $\eta$ means the displayed local displacement reproduces the requested anisotropy change. Red points flag any of: angular step above $15^\circ$, $\eta>25\%$, Jacobian condition number above100, rank deficiency, or invalid corrected channels/radius. These are explicit exploratory caution thresholds, not confidence intervals. Exponential mapping of large steps can wrap around the sphere and is not physically trustworthy. The lower panel exposes this instead of hiding it.

\textbf{Comparison.} Grey points show the previous continuous exact inverse, where valid; blue/red points show the local approximation. Red points may be drawn even when exact inversion is invalid: these are extrapolations, not newly valid measurements. The middle panels show local tangent components, not the earlier circle-centre elevation $\beta$.


\newpage
\section{Sweep overview}
\subsection*{Discuss first if time is limited}
\textbf{1. General stationary field, 10\% cap (fit p.~\pageref{case:general:10pct}; local residual p.~\pageref{local:general}).} This is the main stationary-field candidate: corrected held-out points remain valid in both windows, and local angular RMS is2.91/3.48 degrees with 0/128 caution flags in each. Discuss whether the improvement warrants the 33 fitted parameters and what experiment could distinguish this field from other explanations. Compare 5\% and 10\% before the high-budget solutions; a good fit alone does not establish the actual background.

\textbf{2. General field with axial motion, 10\% cap (fit p.~\pageref{case:axial:10pct}; local residual p.~\pageref{local:axial}).} Forward agreement improves further, with local angular RMS 1.66/2.79 degrees and no caution flags. The key question is physical: is the fitted height-dependent phase plausible? Radius is bounded at 300 nm, the waist and 2-micron focus scale are assumed, and the 10--11s radius reaches its bound. This is a sensitivity result, not evidence that the rod follows the fitted height trajectory.

\textbf{3. No BG versus additive BG at 10\% (fits pp.~\pageref{case:none:uncapped},\pageref{case:square:10pct}; local residuals pp.~\pageref{local:none},\pageref{local:square}).} These anchor how much is achieved without interference. NoBG gives about 20 degrees local RMS and 62/63 flagged bins; additive 10\% gives 10.58/12.96degrees and 20/41 flags. The noBG exact inverse also fails closure. Discuss which residual features genuinely require interference, rather than judging only whether the displayed local path looks closed.

\textbf{4. Richard's exact four-$C_i$ approximation (fit p.~\pageref{case:richard_product:uncapped}; local residual p.~\pageref{local:richard_product}).} Include this to answer the original proposal directly. It has no additive BG-power cap; corrected held-out invalid counts are 3/14, and 107/111 local estimates trigger caution. Discuss whether the constant-coefficient approximation is adequate and its brightness/coefficient ambiguity. Do not interpret its unstable local angular RMS as a measured rotation.

\textbf{If there is extra time: five-mode10\% (p.~\pageref{case:five:10pct}).} It provides a lower-dimensional comparison with the general field: 23 versus 33 fitted parameters. Local RMS is 9.96/10.19degrees with 25/30 flags despite no pointwise invalid corrected channels. This illustrates why radius validity alone is insufficient.

These are discussion priorities, not a validated ranking of physical truth. Values are 30--31s /10--11s; all local flags use the explicit approximation checks in Section 2. No refits were performed for the local residual diagnostic.
\newpage
\begin{longtable}{p{52mm}p{35mm}p{70mm}}\toprule
Model & Tested finite caps & Stopping observation\\\midrule\endhead
'''
for m in ORDER:
 caps=', '.join(k.replace('pct','') for k in DATA[m]['cases'] if k!='uncapped') or 'Not applicable';tex+=NAMES[m]+' & '+caps+' & '+tableescape(DATA[m]['stop'] or 'Sweep still incomplete')+r'\\'+'\n'
tex+=r'''\bottomrule\end{longtable}
These observations apply to the selected local fit at each cap and either-window/train-or-test validity. They do not establish a unique optimal background or a sharp physical threshold. All uncapped solutions are reported as well. Nonconvergence is printed on case pages and in tables. Old results with different correspondence, gain or objectives are not substituted for missing new cases.
'''
for model in ORDER:
 tex+='\\newpage\n\\section{'+NAMES[model]+'}\n'+DESC[model]+'\n\\textbf{Total fitted continuous parameters across both windows: '+str(PARAM[model])+'.} The three pre-estimated gain adjustments are additional calibration parameters, held fixed here.\n'
 tex+=r'''\begin{longtable}{llrrrrrr}\toprule
Budget & Window & BG\% & Meas.RMS & Signal RMS & Bad(train/test) & Max $r$ & Conv.\\\midrule\endhead
'''
 for x in [q for q in rows if q['model']==model]:
  bg='---' if model=='richard_product' else f"{x['bg_percent']:.1f}";tex+=f"{tableescape(x['case'])} & {x['window']} & {bg} & {x['measured_rms']:.4f} & {x['signal_rms']:.4f} & {x['train_invalid']}/{x['test_invalid']} & {x['max_r']:.3f} & {'yes' if x['converged'] else 'NO'} "+r'\\'+'\n'
 tex+=r'''\bottomrule\end{longtable}
\begin{longtable}{llrrrr}\toprule
Budget & Window & Signal\% range & Cross\% range & Noninterf.BG\% & Fitted folded $\theta$\\\midrule\endhead
'''
 for x in [q for q in rows if q['model']==model]:
  inc='---' if x['noninterfering_percent'] is None else f"{x['noninterfering_percent']:.1f}";tex+=f"{tableescape(x['case'])} & {x['window']} & {x['signal_min']:.1f}--{x['signal_max']:.1f} & {x['cross_min']:.1f}--{x['cross_max']:.1f} & {inc} & {x['theta_min']:.1f}--{x['theta_max']:.1f}$^\\circ$ "+r'\\'+'\n'
 tex+=r'''\bottomrule\end{longtable}
\textbf{Per-channel constant background, percent of measured maximum:}
\begin{longtable}{llrrrr}\toprule
Budget & Window & $B_0$ & $B_{90}$ & $B_{45}$ & $B_{135}$\\\midrule\endhead
'''
 for x in [q for q in rows if q['model']==model]:tex+=tableescape(x['case'])+' & '+x['window']+' & '+' & '.join(f'{b:.2f}' for b in x['channel_bg_percent'])+r'\\'+'\n'
 tex+=r'\bottomrule\end{longtable}'+'\n'
 if model=='axial':
  tex+=r'\textbf{Fitted axial trajectory (sensitivity assumptions, not independent measurements):}\par'+'\n'+r'\begin{longtable}{llrrr}\toprule Budget & Window & $R_h$ (nm) & $|z|_{max}$ (nm) & $\eta$ (degrees)\\\midrule\endhead'+'\n'
  for tag,bb in DATA[model]['cases'].items():
   pp=np.array(bb['p'])
   for i,k in enumerate(a.g.KEYS):tex+=tableescape(tag)+' & '+k+' & '+f'{1000*pp[-4+2*i]:.1f} & {1000*pp[-4+2*i]*np.sin(pp[5*i]):.1f} & {np.degrees(pp[-3+2*i]):.1f}'+chr(92)*2+'\n'
  tex+=r'\bottomrule\end{longtable}'+'\n'
 if model=='richard_product':
  b=DATA[model]['cases']['uncapped'];tex+='\\textbf{Fitted effective coefficients (normalized intensity units):}\\par\n'
  for i,k in enumerate(a.g.KEYS):
   p=np.array(b['solutions'][i]['p']);alpha=p[3];kk=p[5:];tex+=k+': $\\alpha='+f'{alpha:.5g}'+r'$, $K_i=(' + ','.join(f'{v:.4g}' for v in kk)+r')$. '+('Finite $C_i=K_i/\\sqrt{\\alpha}$.' if alpha>1e-10 else 'Near-zero brightness limit: do not report stable finite $C_i$.')+'\\par\n'
 for tag in DATA[model]['cases']:
  tex+='\\newpage\n\\subsection{'+tableescape(titlecap(tag))+'}\\label{case:'+model+':'+tag+'}\n\\includegraphics[width=\\linewidth,height=.9\\textheight,keepaspectratio]{figures/'+model+'_'+tag+'.pdf}\n'
 tex+='\\newpage\n\\subsection{Representative sphere diagnostic: '+tableescape(titlecap(sphereinfo[model]['case']))+'}\n\\includegraphics[width=\\linewidth,height=.82\\textheight,keepaspectratio]{figures/'+model+'_sphere.pdf}\n'
 for x in sphereinfo[model]['stats']:
  bstat=x['branch_audit'];cl=f"{bstat.get('closure_mismatch_deg',0):.2f}";qual='oriented closure mismatch' if bstat.get('closure_assessed') else 'closure unresolved (excluded bins); oriented endpoint mismatch'
  tex+=x['window']+': '+str(x['valid'])+'/128 valid; matched geodesic RMS '+(f"{x['geo_rms']:.2f}$^\\circ$" if x['geo_rms'] is not None else 'undefined')+'; '+qual+' '+cl+r'$^\circ$.'+'\\par\n'


 tex+='\\newpage\n\\subsection{Local cone-referenced angular residual: '+tableescape(titlecap(sphereinfo[model]['case']))+'}\\label{local:'+model+'}\n\\includegraphics[width=\\linewidth,height=.86\\textheight,keepaspectratio]{local_residual/'+model+'.pdf}\n'
 for x in [v for v in localstats if v['model']==model]:
  tex+=x['window']+f": local angular RMS {x['linear_rms_deg']:.2f}"+r'$^\circ$; '+str(x['local_flagged'])+'/128 caution flags; median nonlinear mismatch ratio '+f"{x['nonlinear_relative_median']:.3f}"+'.\\par\n'

tex+=r'''\newpage\section{Earlier tests retained as context}
\textbf{Stokes-restricted additive background.} Earlier fits restricted $(f_x,f_y)$ to a disk, while the current notebook-style additive model permits independent contrasts in a square. Earlier optima were inside the disk and changed little on relaxation. The user requested the simpler square form for current comparisons; the disk version is not duplicated as a separate sweep.

\textbf{Measured aperture mask.} Previous calculations compared the measured mirror mask against the annulus; selected held-out XY RMS changed by less than about0.0002 in that earlier setup. This does not justify importing old fit coefficients into the current reference pipeline. Current sweep uses the user-selected NA0.38 annulus throughout.

\textbf{Hole-edge variants.} Earlier widths0.03/0.10/0.25NA were tested with pure coherent edge backgrounds. The current representative uses0.10NA with additional noninterfering Stokes power. A poor edge-surrogate fit does not exclude arbitrary multiple reflections or edge scattering.

\textbf{Arbitrary overlap construction.} Allowing overlap to vary freely at each point can match trajectories without identifying a stationary field. It is a feasibility witness, not a fitted physical source model, and is not ranked as a competing cap sweep.

\textbf{Inverse sphere-circle fitting.} Earlier circle-only inverse fits exhibited collapse/cancellation degeneracy: an increasingly large correction could map observations into an increasingly tiny circle. Those tests remain evidence that circle closeness alone is insufficient, not a validated angle-recovery method.

\textbf{Flexible phase correspondence.} Previous sixteen-parameter monotone phase fits are not mixed with the fixed ordered-arc models here. Some old disagreement came from nonuniform final reference coordinates; those comparisons were explicitly withdrawn and repaired. The present plots use the repaired final-reference construction.

\textbf{Axial motion and finite-aperture focus.} Earlier50/150/300nm bounds and a finite-detector defocus sensitivity were conditional simulations, not measurements of hook geometry. The current axial family, when present, uses the300nm orbital bound with full pupil-phase overlap and the stated simple excitation focus factor. The separate finite-detector calculation is not silently included.

\section{Limits and next discussion}
No current fit proves a unique background composition or correct absolute angles. Low BG caps can stabilize subtraction while leaving measured or signal-space mismatch. More modes can reduce residuals but also permit cancellation. Pair balance alone does not distinguish gain error from every downstream-background mechanism. Current gain adjustment is fixed and shared, but not independently calibrated.

Uncapped searches have no BG-power upper bound, but geometry, log brightness and axial motion retain stated numerical/physical bounds. Multiple local starts are not a global search. Failure at a cap can depend on which local solution minimizes the training objective. Adaptive use of held-out clipping to choose further caps makes the overall sweep exploratory.

All scripts, per-fit parameters, train/test diagnostics, channel decompositions and figure sources are retained locally under model\_comparison/consolidated/. Files: simple.json, fields.json, axial.json, composition.json, metrics.csv, sphere\_summary.json, run.py and report.py. This is a draft for revision; no commit or push.
\end{document}
'''
from notation import format_notation
(R/'report.tex').write_text(format_notation(tex))
print('Models',ORDER,'cases',sum(len(DATA[m]['cases']) for m in ORDER),'rows',len(rows))
