import caps as c
import json,io,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=c.R;v=json.load(open(R/'fits.json'));rows=[]
for col,m in enumerate([5,10]):
 for row,tag in enumerate(['broad','10pct','5pct']):
  b=v[f'm{m}_{tag}']['best'];c.setup(m,v[f'm{m}_{tag}']['cap']);p=np.array(b['p'])
  for i,k in enumerate(c.g.KEYS):
   y,s,inter,bg,d,mn=c.g.predict(p[5*i:5*i+5],p[10:],b['directions'][i],c.f.U[i],8192,True);obs=s+c.f.TE[i]-y;t=c.f.TE[i].sum(0);rec=dict(model=m,cap=tag,window=k,converged=b['success'],bg_percent=b['scores'][k]['bg_percent'],measured_channel_rms=b['scores'][k]['test']['channel_rms'],invalid=b['scores'][k]['test']['invalid'],signal_normalized_rms=float(np.sqrt(np.mean(((y-c.f.TE[i])/s.sum(0))**2))),theta_min=float(np.degrees(np.arccos(abs(d[:,2]))).min()),theta_max=float(np.degrees(np.arccos(abs(d[:,2]))).max()));rows.append(rec)
for i,k in enumerate(c.g.KEYS):
 for kind in ['measured','signal']:
  fig,axs=plt.subplots(3,2,figsize=(10,13),layout='constrained')
  for col,m in enumerate([5,10]):
   for row,tag in enumerate(['broad','10pct','5pct']):
    b=v[f'm{m}_{tag}']['best'];c.setup(m,v[f'm{m}_{tag}']['cap']);p=np.array(b['p']);chi,arc,y,s,inter,bg,d=c.g.dense(p[5*i:5*i+5],p[10:],8192);sc=b['scores'][k];sig=np.array(sc['signal']);pred=np.array(sc['pred']);obs=sig+c.f.TE[i]-pred;ax=axs[row,col]
    if kind=='measured':
     ax.plot(*c.g.xy(y).T,color='#ce6436',lw=1.7,label='Fitted measured curve');ax.scatter(*c.g.xy(c.f.TE[i]).T,color='#172d3c',s=10,label='Held-out mean')
    else:
     a=c.g.xy(obs);bad=(obs<0).any(0)|(np.linalg.norm(a,axis=1)>c.f.rmax);ax.plot(*c.g.xy(s).T,color='#ce6436',lw=1.7,label='Fitted signal S_fit');ax.plot(*np.vstack([a,a[0]]).T,color='#398849',lw=1.2,label='S_fit + held-out residual');ax.scatter(*a[bad].T,marker='x',s=18,color='#d32030',label='Outside inversion domain')
     beyond=(abs(a)>1).any(1)
     if beyond.any():ax.scatter(*(a[beyond]/np.max(abs(a[beyond]),axis=1)[:,None]*.98).T,color='#d32030',marker='^',s=18)
     tt=np.linspace(0,2*np.pi,200);ax.plot(c.f.rmax*np.cos(tt),c.f.rmax*np.sin(tt),':',color='.7',lw=.7)
    label=('Five-mode' if m==5 else 'General stationary');caplabel={'broad':'Broad cap','10pct':'10% cap','5pct':'5% cap'}[tag];rec=next(a for a in rows if a['model']==m and a['cap']==tag and a['window']==k)
    ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f'{label} · {caplabel}\nBG {rec["bg_percent"]:.1f}%, invalid {rec["invalid"]}/128'+(' · not converged' if not b['success'] else ''));ax.set_aspect('equal');ax.grid(alpha=.15)
    if col==0:ax.legend(fontsize=6.5,loc='lower left')
  fig.suptitle(f'{"30–31" if k=="30s" else "10–11"} s · {kind} space · fixed gains and steep penalty',fontsize=14)
  for ext in ['png','pdf']:
   buf=io.BytesIO();fig.savefig(buf,format=ext,dpi=160);(R/(k+'_'+kind+'.'+ext)).write_bytes(buf.getvalue())
  plt.close(fig)
(R/'summary.json').write_text(json.dumps(rows,indent=2))
tex=r'''\documentclass[10pt,a4paper]{article}\usepackage[margin=16mm]{geometry}\usepackage{fontspec,amsmath,graphicx,booktabs}\setmainfont{Latin Modern Roman}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\begin{document}
{\Large Five-mode versus general stationary fields: background caps}\par 6 October 2026

Six joint two-window fits. Same pooled inverse-gain calibration, unchanged transmission matrix, uniform final-arc references, cyclic-offset correspondence and measured-total-normalized channel objective. No signal-relative weighting. One background field is shared across both windows; cone geometry, brightness and offset are window-specific.

Both families have a stationary coherent pupil field plus a noninterfering Stokes background. Five-mode uses the earlier two uniform-polarization and three dipole-derived vector modes (five complex coefficients); general uses ten complex transverse pupil modes spanning signal overlaps. Both are evaluated on the same optical quadrature and annular Fresnel/BFP model. Pair sums hold by construction. Different finite bases need not have literally nested background-power representations; comparison here is of the implemented physical families.

Background caps refer to the smaller of the two original, gain-corrected training maxima, so the shared background satisfies the stated cap in both windows. Broad cap retains the previous 95\% of the stricter original minimum total intensity (about 56\% of the smaller maximum). The actual fitted BG percentage is reported below, separately for each window.
\[
J=\sum_{ij}(\varepsilon_{ij}/T_j^{train})^2+
100\sum_j\left[\max\left(0,\frac{\widehat r_j-r_{max}}{1-r_{max}}\right)\right]^6+
10\sum_{ij}[\min(0,\widehat S_{ij})/T_j^{train}]^2.
\]
Here $\widehat S=I^{train}-B-L(d^{fit})$. The radial term has a numerical denominator guard near zero pair intensity; raw corrected intensities are used for diagnostics. Penalties use training data only. Multiple starts and 8192-point refinement are used; convergence is reported, not a global-optimum guarantee.

\begin{tabular}{llrrrrr}\toprule
Model/cap & Window & BG\% & Meas. RMS & Signal RMS & Invalid & Conv.\\\midrule
'''
for a in rows:
 tex+=f"{a['model']} / {a['cap']} & {a['window']} & {a['bg_percent']:.1f} & {a['measured_channel_rms']:.4f} & {a['signal_normalized_rms']:.4f} & {a['invalid']} & {'yes' if a['converged'] else 'no'} " + chr(92)*2 + chr(10)
tex+=r'''\bottomrule\end{tabular}

RMS values are four-channel residuals divided by measured total or fitted signal total respectively. Signal RMS is diagnostic only. Invalid counts include negative corrected channels or $r>r_{max}$. Signal-space curves show conditional fitted-interference subtraction, not self-consistent field inversion. Red triangles indicate points beyond fixed axes. A low cap may improve subtraction stability while worsening measured-space agreement; neither establishes correct rod angles. No commit or push.
'''
for k in ['30s','10s']:
 for kind in ['measured','signal']:tex+='\\newpage\n\\includegraphics[width=\\linewidth,height=.94\\textheight,keepaspectratio]{'+k+'_'+kind+'.pdf}\n'
tex+='\\end{document}\n';(R/'report.tex').write_text(tex)
print(json.dumps(rows,indent=2))
