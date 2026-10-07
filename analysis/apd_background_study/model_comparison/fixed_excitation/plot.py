from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
res=json.loads((R/'fits.json').read_text());relaxed=json.loads((R.parent/'interval10/checks.json').read_text())
fig,axes=plt.subplots(2,4,figsize=(17,8.2),layout='constrained');summary={}
colors={'bg':'#4878a8','richard':'#b78525','field':'#d44b35'}
for row,(key,path,label) in enumerate([('30s','fitted_predictions.npz','30–31 s'),('10s','interval10/averages.npz','10–11 s')]):
 z=np.load(R.parent/path);data=z['test'];tm=z['train'].sum(0).min();summary[key]={}
 for col,(m,title) in enumerate([('bg','Background only'),('richard','Effective intensity'),('field','Restricted field')]):
  b=res[key][m]['best'] if m!='field' else res[key]['field_cap_checks']['2.0']['best'];p=np.array(b['p']);pred,d=fixed_predict(p,m,128,tm);summary[key][m]=b['test']
  ax=axes[row,col];ax.plot(*xy(data).T,'.-',color='.5',lw=.8,ms=2.5,label='Held-out data');ax.plot(*xy(pred).T,color=colors[m],lw=1.7,label='Fixed excitation')
  if m=='field':
   pr=np.array(relaxed[label]['field_checks'][1]['p']);old,_=predict(pr,m,data,tm,16,-1);ax.plot(*xy(old).T,'--',color='#75519a',lw=1.2,label='Earlier free brightness')
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f"{label}: {title}\nOrdered XY RMS {b['test']['xy_rms']:.3f}");ax.set_aspect('equal');ax.grid(alpha=.2)
  if col==2:ax.legend(fontsize=7,loc='lower left')
  x=np.arange(128)/128;axes[row,3].plot(x,pred.sum(0),color=colors[m],lw=1.4,label=title)
 axes[row,3].plot(x,data.sum(0),color='.3',lw=1.8,label='Held-out data');axes[row,3].set(xlabel='Ordered trajectory fraction (not time)',ylabel='Sum of corrected channels (a.u.)',title=f'{label}: total intensity');axes[row,3].grid(alpha=.2);axes[row,3].legend(fontsize=7)
fig.suptitle('Constant excitation scale: a = a₀ sin θ\nAll four channel intensities fitted; no pointwise brightness adjustment. Field background cap widened to 2× minimum total.',fontsize=13)
fig.savefig(R.parent.parent/'deliverables/fixed_excitation_comparison.png',dpi=170)
(R/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
