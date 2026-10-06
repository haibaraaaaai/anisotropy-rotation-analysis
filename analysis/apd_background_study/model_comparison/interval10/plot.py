import sys,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R))
from models import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
checks=json.loads((R/'interval10/checks.json').read_text());old=json.loads((R/'fits.json').read_text());new=json.loads((R/'interval10/fits.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(12,8.7),layout='constrained')
labels={'bg':'Background only','richard':'Effective intensity','field':'Restricted field'}
for row,(label,z,r) in enumerate([('30–31 s',np.load(R/'fitted_predictions.npz'),old),('10–11 s',np.load(R/'interval10/averages.npz'),new)]):
 dat=z['test'];tm=z['train'].sum(0).min()
 for col,m in enumerate(labels):
  ax=axes[row,col];b=r[m]['best'];p=np.array(b['p']);di=b['direction'];gm=checks[label][m]['geometric_test_rms']
  if m=='field':
   if row==0:
    pred,_=predict(p,m,dat,tm,16,di);ax.plot(*xy(pred).T,'--',color='#5e7cb0',lw=1.3,label='Earlier local minimum')
   b=checks[label]['field_checks'][1];p=np.array(b['p']);gm=b['test_geometric_rms']
  pred,_=predict(p,m,dat,tm,16,di)
  ax.plot(*xy(dat).T,'.-',color='.5',ms=3,lw=.9,label='Held-out average')
  ax.plot(*xy(pred).T,color='#d74d34',lw=1.6,label='Model prediction')
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f'{label} · {labels[m]}\nGeometric curve RMS = {gm:.4f}')
  ax.set_aspect('equal');ax.grid(alpha=.2)
  if col==2:ax.legend(loc='lower left',fontsize=7)
fig.suptitle('Independent-cycle averages vs fitted predictions\nFixed gains, current T, NAᵢₙ = 0.38; field fits shown at the 95% background cap',fontsize=13)
fig.savefig(R.parent/'deliverables/two_interval_model_comparison.png',dpi=170)
