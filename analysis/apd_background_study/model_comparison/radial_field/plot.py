from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads((R/'fits.json').read_text());old=json.loads((R.parent/'fixed_excitation/fits.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(13,8.3),layout='constrained')
for row,(key,path,title) in enumerate([('30s','fitted_predictions.npz','30–31 s'),('10s','interval10/averages.npz','10–11 s')]):
 z=np.load(R.parent/path);test=z['test'];tm=z['train'].sum(0).min();base=old[key]['field_cap_checks']['2.0']['best'];new=r[key]['radial16']['best'];p=np.array(new['p'])
 y0=pred(np.array(base['p']),128,tm,radial=False);y1=pred(p,128,tm)
 for col,(y,label,err) in enumerate([(y0,'Uniform field',base['test']['xy_rms']),(y1,'Uniform + radial field',new['test']['xy_rms'])]):
  ax=axes[row,col];ax.plot(*xy(test).T,'.-',color='.5',lw=.8,ms=3,label='Held-out data');ax.plot(*xy(y).T,color='#d54b35',lw=1.6,label='Prediction')
  ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=f'{title}: {label}\nOrdered XY RMS = {err:.4f}');ax.set_aspect('equal');ax.grid(alpha=.2)
 axes[row,0].legend(fontsize=8,loc='lower left')
 ax=axes[row,2];x=np.arange(128)/128;ax.plot(x,test.sum(0),color='.4',lw=1.5,label='Held-out data');ax.plot(x,y0.sum(0),'--',color='#547aaa',label='Uniform field');ax.plot(x,y1.sum(0),color='#d54b35',label='+ radial field');ax.set(xlabel='Ordered trajectory fraction (not time)',ylabel='Total intensity (a.u.)',title=f'{title}: brightness prediction');ax.legend(fontsize=8);ax.grid(alpha=.2)
 print(key,'old',base['test'],'new',new['test'],'bg/min',p[19],'radial fraction coherent',p[26],'radial fraction totalbg',p[22]*p[26],'incoherent polarization',p[20])
 for name in ['uniform8','radial8','uniform32','radial32']:print(name,r[key][name]['best']['test']['xy_rms'])
fig.suptitle('One additional spatial background mode: two extra real parameters\na = a₀ sin θ enforced; fixed calibration; 16 monotone position-mapping parameters in both models',fontsize=13)
fig.savefig(R.parent.parent/'deliverables/radial_field_comparison.png',dpi=170)
