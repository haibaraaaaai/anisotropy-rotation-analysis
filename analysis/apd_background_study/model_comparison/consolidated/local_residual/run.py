from pathlib import Path
import importlib.util,sys,json,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from sphere_branches import recover
sp=importlib.util.spec_from_file_location('parent_runner',P/'run.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
DATA={}
for name in ['simple','fields','axial']:DATA.update(json.load(open(P/(name+'.json'))))
NAMES={'none':'No background','square':'Additive background','richard_product':'Richard four-C','complex':'Additive + bounded interference','uniform':'Uniform field','three':'Three-mode field','five':'Five-mode field','general':'General stationary field','edge':'Hole-edge field','axial':'General field + axial motion'}
def xy(v):return np.column_stack([(v[0]-v[1])/(v[0]+v[1]),(v[2]-v[3])/(v[2]+v[3])])
def forward(d):
 s=d[:,0]**2+d[:,1]**2;den=.8908059777413628+.10919402225863714*s
 return .9277805723143716*np.column_stack([d[:,0]**2-d[:,1]**2,2*d[:,0]*d[:,1]])/den[:,None]
def expmap(d,v):
 t=np.linalg.norm(v,axis=1);return np.cos(t)[:,None]*d+np.sinc(t/np.pi)[:,None]*v

def calculate(model,b,i,k):
 sc=b['scores'][k];d=np.array(sc['directions']);sig=np.array(sc['signal']);cor=a.f.TE[i]-np.array(sc['cross'])-np.array(sc['background'])[:,None]
 geometry=np.array(b['p'])[5*i:5*i+3] if model in a.MODES else np.array(b['solutions'][i]['p'])[:3]
 th,ph,lam=geometry;axis=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
 across=np.cos(lam)*d-axis;across/=np.linalg.norm(across,axis=1)[:,None]
 along=np.cross(axis,d);along/=np.linalg.norm(along,axis=1)[:,None]
 h=1e-5;J=np.stack([(forward(expmap(d,h*t))-forward(expmap(d,-h*t)))/(2*h) for t in [across,along]],axis=2)
 e=xy(cor)-xy(sig);sv=np.linalg.svd(J,compute_uv=False);rank=sv[:,-1]>1e-8;delta=np.full((128,2),np.nan);delta[rank]=np.linalg.solve(J[rank],e[rank,:,None])[:,:,0];v=delta[:,0,None]*across+delta[:,1,None]*along;local=expmap(d,v)
 nonlinear=forward(local)-forward(d)-e;rel=np.linalg.norm(nonlinear,axis=1)/np.maximum(np.linalg.norm(e,axis=1),1e-8)
 mag=np.linalg.norm(delta,axis=1);exact,valid,audit=recover(cor,d,a.f.U[i]);condition=sv[:,0]/sv[:,-1]
 flagged=(~valid)|(~rank)|(mag>np.deg2rad(15))|(rel>.25)|(condition>100)
 stat=dict(model=model,window=k,invalid=int((~valid).sum()),local_flagged=int(flagged.sum()),linear_rms_deg=float(np.degrees(np.sqrt(np.nanmean(mag**2)))),linear_max_deg=float(np.degrees(np.nanmax(mag))),nonlinear_relative_median=float(np.nanmedian(rel)),nonlinear_relative_max=float(np.nanmax(rel)),min_singular=float(sv[:,-1].min()),condition_max=float(condition.max()),analytic_quadrature_xy_offset=float(np.max(np.linalg.norm(forward(d)-xy(sig),axis=1))))
 assert np.nanmax(abs(np.linalg.norm(local,axis=1)-1))<1e-12
 assert np.nanmax(abs(np.einsum('nij,nj->ni',J,delta)-e))<1e-10
 return dict(d=d,local=local,exact=exact,valid=valid,flagged=flagged,delta=delta,rel=rel,stat=stat,geometry=geometry,J=J,e=e)
# Synthetic small tangent perturbation: decreasing displacement must approach exact recovery.
d=np.array([[.4,.3,np.sqrt(.75)]]);t=np.cross(d,np.array([[0.,0.,1.]]));t/=np.linalg.norm(t,axis=1)[:,None]
assert np.linalg.norm(expmap(d,np.zeros_like(d))-d)<1e-14
stats=[];rep=[]
for model in a.LABEL:
 tag='10pct' if '10pct' in DATA[model]['cases'] else 'uncapped'
 for cap,b in DATA[model]['cases'].items():
  for i,k in enumerate(a.g.KEYS):
   out=calculate(model,b,i,k);out['stat']['cap']=cap;stats.append(out['stat'])
 b=DATA[model]['cases'][tag];fig=plt.figure(figsize=(10,11));gs=fig.add_gridspec(3,2,height_ratios=[1.4,1,1],hspace=.35,wspace=.3)
 for i,k in enumerate(a.g.KEYS):
  o=calculate(model,b,i,k);rep.append(o['stat']);u=a.f.U[i];ax=fig.add_subplot(gs[0,i],projection='3d');tt=np.linspace(0,2*np.pi,31);uu=np.linspace(0,np.pi,17)
  ax.plot_wireframe(np.outer(np.cos(tt),np.sin(uu)),np.outer(np.sin(tt),np.sin(uu)),np.outer(np.ones_like(tt),np.cos(uu)),color='.7',alpha=.15,lw=.4,rstride=3,cstride=3)
  curve=a.g.cone(o['geometry'],np.linspace(0,2*np.pi,361));ax.plot(*curve.T,color='#ce6436',lw=1.7,label='Fitted cone');ax.scatter(*o['exact'][o['valid']].T,s=5,color='.6',alpha=.5,label='Exact inverse')
  ok=~o['flagged'];ax.scatter(*o['local'][ok].T,s=11,color='#087f9b',label='Local estimate');ax.scatter(*o['local'][~ok].T,s=14,color='#c32646',marker='x',label='Approximation caution');ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel='x',ylabel='y',title=k+f" | {o['stat']['local_flagged']}/128 flagged");ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-1,0,1]);ax.text2D(.96,.6,'Optical z',transform=ax.transAxes,rotation=90,fontsize=8);ax.set_box_aspect([1,1,1]);ax.view_init(22,-55);ax.legend(fontsize=6,loc='upper left')
  ax=fig.add_subplot(gs[1,i]);deg=np.degrees(o['delta']);ax.plot(u,deg[:,0],color='#167d9a',label='Across cone: delta lambda');ax.plot(u,deg[:,1],color='#b1543b',label='Along cone: sin(lambda) delta psi');ax.scatter(u[~ok],deg[~ok,0],color='#c32646',s=10,marker='x');ax.axhline(0,color='.6',lw=.7);ax.set(xlim=(0,1),xlabel='Ordered reference arc fraction',ylabel='Local angular displacement (deg)');ax.grid(alpha=.2);ax.legend(fontsize=7)
  ax=fig.add_subplot(gs[2,i]);ax.semilogy(u,np.maximum(o['rel'],1e-5),color='#45445d');ax.axhline(.25,color='#c32646',ls='--',label='25% caution threshold');ax.set(xlim=(0,1),xlabel='Ordered reference arc fraction',ylabel='Nonlinear mismatch / requested XY change');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle(NAMES[model]+' | '+('No BG-power cap' if tag=='uncapped' else tag.replace('pct','% cap'))+'\nLocal cone-referenced angular residual',fontsize=14)
 buf=io.BytesIO();fig.savefig(buf,format='pdf',bbox_inches='tight');(R/(model+'.pdf')).write_bytes(buf.getvalue());plt.close(fig)
(R/'metrics.json').write_text(json.dumps(stats,indent=2));(R/'representative.json').write_text(json.dumps(rep,indent=2))
tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=16mm]{geometry}\usepackage{fontspec,amsmath,graphicx,booktabs,longtable,hyperref}
\setmainfont{Latin Modern Roman}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\begin{document}
{\LARGE Local cone-referenced angular residuals}\par
7 October 2026. Diagnostic trial; all intensity fits and background parameters are unchanged.

This supplement tests the requested alternative to exact inversion. At each matched fitted cone direction $d_j$, take the corrected held-out anisotropy minus the fitted signal anisotropy:
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

All90 window/cap cases were evaluated numerically. Figures show the10\% case for each family where applicable, plus uncapped no-BG and Richard. The full original cap-sweep report is unchanged.
\newpage
\section*{Representative results}
\begin{longtable}{llrrrr}\toprule
Model & Window & RMS (deg) & Max (deg) & Flagged & Median $\eta$\\\midrule\endhead
'''
for x in rep:
 tex+=NAMES[x['model']]+f" & {x['window']} & {x['linear_rms_deg']:.1f} & {x['linear_max_deg']:.1f} & {x['local_flagged']}/128 & {x['nonlinear_relative_median']:.2f}"+r'\\'+'\n'
tex+=r'\bottomrule\end{longtable} The angular RMS includes flagged finite estimates; it is not an accuracy ranking. Singular directions have no usable local inverse. See each figure for where the linear approximation fails.'
for model in a.LABEL:tex+='\n'+r'\newpage\includegraphics[width=\linewidth,height=.96\textheight,keepaspectratio]{'+model+'.pdf}'
tex+='\n'+r'\end{document}';(R/'report.tex').write_text(tex)
for x in rep:print(x['model'],x['window'],'rms',round(x['linear_rms_deg'],2),'flags',x['local_flagged'],'eta median',round(x['nonlinear_relative_median'],3))
