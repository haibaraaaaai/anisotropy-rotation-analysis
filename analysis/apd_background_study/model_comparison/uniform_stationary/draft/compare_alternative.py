from pathlib import Path
import sys,json,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));import run as f
from models import xy,ABC
v=json.load(open(f.R/'fits.json'));fig,axs=plt.subplots(2,2,figsize=(10,10),layout='constrained')
for row,(label,b) in enumerate([('Previous selection (~55% BG)',v['refined']),('Alternative (~47% BG)',v['runs'][10])]):
 p=np.array(b['p'])
 for col,k in enumerate(f.KEYS):
  y,s,c,bg,d,mn=f.predict(p[col*5:col*5+5],p[10:],b['directions'][col],f.U[col],8192,True);a=xy(s+f.TE[col]-y);signal=f.dense(p[col*5:col*5+5],p[10:],8192)[3];rr=np.linalg.norm(a,axis=1);valid=(s+f.TE[col]-y>=0).all(0)&(rr<=ABC[2]/sum(ABC[:2]));ax=axs[row,col]
  ax.plot(*xy(signal).T,color='#ce6436',lw=2,label='Fitted signal S_fit');ax.plot(*np.vstack([a,a[0]]).T,color='#398849',lw=1.3,label='S_fit + measured residual');ax.scatter(*a[~valid].T,color='#d32030',marker='x',s=20,label='Outside inversion domain' if (~valid).any() else None)
  outside=(abs(a)>1).any(1)
  if outside.any():
   edge=a[outside]/np.max(abs(a[outside]),axis=1)[:,None]*.98;ax.scatter(*edge.T,color='#d32030',marker='^',s=20,label='Beyond axes')
  tt=np.linspace(0,2*np.pi,300);rm=ABC[2]/sum(ABC[:2]);ax.plot(rm*np.cos(tt),rm*np.sin(tt),':',color='.7',lw=.8)
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f'{label} · {"30–31" if k=="30s" else "10–11"} s\n{(~valid).sum()}/128 outside inversion domain');ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=7,loc='lower left')
for ext in ['png','pdf']:
 buf=io.BytesIO();fig.savefig(buf,format=ext,dpi=180);p=R/'alternative'/('comparison.'+ext);p.write_bytes(buf.getvalue())
plt.close(fig)
for k in f.KEYS:
 z=np.load(R/'alternative'/('stationary_'+k+'_sphere.npz'));print(k,'max beta/along',np.max(abs(z['cross_deg'])),np.max(abs(z['along_deg'])))
