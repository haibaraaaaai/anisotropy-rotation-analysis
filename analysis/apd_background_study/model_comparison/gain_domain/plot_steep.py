import run as f
import json,numpy as np,io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=f.R;b=json.load(open(R/'steep_fits.json'))['best'];p=np.array(b['p']);fig,axs=plt.subplots(3,2,figsize=(10,13),layout='constrained')
for col,k in enumerate(f.f.KEYS):
 i=col;sc=b['scores'][k];sig=np.array(sc['signal']);pred=np.array(sc['pred']);dense=f.f.dense(p[5*i:5*i+5],p[10:],8192);curve=dense[2];signal=dense[3]
 ax=axs[0,col];ax.plot(*f.f.xy(curve).T,color='#ce6436',lw=1.8,label='Fitted measured curve');ax.scatter(*f.f.xy(f.TE[i]).T,color='#172d3c',s=12,label='Held-out mean');ax.set_title(f'{"30–31" if k=="30s" else "10–11"} s · measured space');ax.legend(fontsize=7,loc='lower left')
 for row,(label,dat) in enumerate([('Training',f.TR[i]),('Held-out',f.TE[i])],1):
  cor=sig+dat-pred;a=f.f.xy(cor);rr=np.linalg.norm(a,axis=1);bad=(cor<0).any(0)|(rr>f.rmax);ax=axs[row,col];ax.plot(*f.f.xy(signal).T,color='#ce6436',lw=1.8,label='Fitted signal S_fit');ax.plot(*np.vstack([a,a[0]]).T,color='#398849',lw=1.2,label='S_fit + residual');ax.scatter(*a[bad].T,color='#d32030',marker='x',s=18,label='Outside inversion domain')
  tt=np.linspace(0,2*np.pi,300);ax.plot(f.rmax*np.cos(tt),f.rmax*np.sin(tt),':',color='.65',lw=.8)
  outside=(abs(a)>1).any(1)
  if outside.any():ax.scatter(*(a[outside]/np.max(abs(a[outside]),axis=1)[:,None]*.98).T,color='#d32030',marker='^',s=20,label='Beyond axes')
  ax.set_title(f'{label} signal space · {bad.sum()}/128 invalid\nMaximum r = {rr.max():.3f}');ax.legend(fontsize=7,loc='lower left')
 for row in range(3):axs[row,col].set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y');axs[row,col].set_aspect('equal');axs[row,col].grid(alpha=.15)
fig.suptitle('Steep radial penalty · fixed gains, no signal-relative weighting',fontsize=14)
for ext in ['png','pdf']:
 buf=io.BytesIO();fig.savefig(buf,format=ext,dpi=160);(R/('steep_comparison.'+ext)).write_bytes(buf.getvalue())
