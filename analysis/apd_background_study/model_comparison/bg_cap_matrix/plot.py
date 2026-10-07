from run_checks import *
from run import xy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads((R/'results.json').read_text());fig,axes=plt.subplots(2,3,figsize=(14,8.6),layout='constrained')
summary={}
for row,key in enumerate(['30s','10s']):
 title='30–31 s' if key=='30s' else '10–11 s';z=np.load(R/f'{key}_plus_data.npz');tr=z['train'];te=z['test'];tm=tr.sum(0).min();mx=tr.sum(0).max();p=np.array(r[key]['plus_wide']['best']['p']);c=components(p,tm);x=np.arange(128)/128;ax=axes[row,0]
 ax.plot(x,te.sum(0)/mx,color='.2',lw=1.6,label='Measured total (held out)');ax.plot(x,c['signal']/mx,color='#267db2',label='Rod alone');ax.plot(x,c['interference']/mx,color='#c24c3b',label='Signed interference')
 ax.axhline((c['uniform_bg']+c['radial_bg'])/mx,color='#7c59a5',ls='--',label='Coherent background');ax.axhline(c['incoherent_bg']/mx,color='#39834c',ls=':',label='Noninterfering background');ax.axhline(0,color='.5',lw=.6)
 ax.set(xlabel='Ordered trajectory fraction (not time)',ylabel='Intensity / maximum training total',title=f'{title}: current-fit decomposition');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower left')
 ax=axes[row,1];ax.plot(*xy(te).T,'.-',color='.55',ms=2.5,lw=.7,label='Held-out data')
 for label,color,name in [('plus_wide','#d44c35','Current background fit'),('plus_cap10','#2d7ba5','Background ≤ 10% maximum')]:
  pp=np.array(r[key][label]['best']['p']);y=pred(pp,128,tm);ax.plot(*xy(y).T,color=color,lw=1.4,label=name)
 ax.set(title=f'{title}: hard background cap',xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower left')
 ax=axes[row,2]
 for sign,color in [('plus','#d44c35'),('minus','#2d7ba5')]:
  zz=np.load(R/f'{key}_{sign}_data.npz');tt=zz['test'];tmin=zz['train'].sum(0).min();pp=np.array(r[key][sign+'_wide']['best']['p']);y=pred(pp,128,tmin)
  ax.plot(*xy(tt).T,'.',color=color,ms=2.5,alpha=.5)
  ax.plot(*xy(y).T,color=color,lw=1.5,label=f"β {'+' if sign=='plus' else '−'}0.009: fit")
 ax.set(title=f'{title}: matrix sign (dots = respective data)',xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower left')
 summary[key]={}
 for name,v in r[key].items():
  summary[key][name]={'test':v['best']['test'],'decomposition':v['decomposition']};print(key,name,v['best']['test'],'BG/max',v['decomposition']['bg_fraction_max'])
fig.suptitle('Background magnitude and matrix-sign checks\nSame uniform + radial field model; a = a₀ sin θ; gains, extracted points and 16-parameter ordered mapping held consistent',fontsize=13)
fig.savefig(R.parent.parent/'deliverables/background_cap_matrix_checks.png',dpi=170)
(R/'summary.json').write_text(json.dumps(summary,indent=2))
