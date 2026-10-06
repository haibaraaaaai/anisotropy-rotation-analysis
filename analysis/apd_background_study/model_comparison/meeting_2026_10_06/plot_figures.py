from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;F=R/'figures';F.mkdir(exist_ok=True)
z=json.load(open(R/'comparison.json'));D={(q['id'],q['interval']):q for q in z['records']};KEYS=['30s','10s'];WIN=['30–31 s','10–11 s']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'white'})
def xy(c):
 c=np.asarray(c);return np.array([(c[0]-c[1])/(c[0]+c[1]),(c[2]-c[3])/(c[2]+c[3])])
def save(fig,name):
 for ext in ['png','svg']:
  path=F/f'{name}.{ext}';fig.savefig(path,dpi=190,bbox_inches='tight')
  if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
 plt.close(fig)
def trajectories(ids,name,title):
 fig,axs=plt.subplots(len(ids),2,figsize=(9,3.25*len(ids)),layout='constrained')
 for j,id in enumerate(ids):
  for i,k in enumerate(KEYS):
   q=D[id,k];ax=axs[j,i];tr,te,y=map(xy,[q['train'],q['test'],q['pred']]);ax.plot(*tr,color='#c5cbd2',lw=1.6,label='Training average');ax.scatter(*te,s=9,color='#24364b',label='Held-out average');ax.plot(*y,color='#cc573a',lw=2,label='Fitted prediction');ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel='X',ylabel='Y',title=f'{WIN[i]} · {q["label"]}\nBG {q["bg_max"]:.1f}%  |  held-out XY RMS {q["scores"]["test"]["xy_rms"]:.4f}');ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks([-1,-.5,0,.5,1]);ax.grid(alpha=.17)
   if j==0 and i==0:ax.legend(fontsize=8,loc='lower left')
 fig.suptitle(title,fontsize=14);save(fig,name)
trajectories(['bg','richard','m5_cap2.0'],'01_model_fits','Additive and effective models versus a shared coherent field')
trajectories(['general_verified_cap0.1','general_cap0.3','height_R0.3_cap0.1'],'02_low_background_fits','Lower-background alternatives: stationary field and bounded axial motion')
# Background composition + angle ranges (30s vs10s).
ids=['bg','richard','field','radial','m5_cap2.0','general_verified_cap0.1','general_cap0.2','general_cap0.3','general_verified_cap2.0','height_R0.3_cap0.1']
labels=['BG only','Richard effective','Uniform field','Uniform + radial','5-mode relaxed','General ≤10%','General ≤20%','General ≤30%','General relaxed','Height ≤300 nm / ≤10%']
fig,axs=plt.subplots(1,3,figsize=(14,6.3),layout='constrained',gridspec_kw={'width_ratios':[1.3,1,1]})
for j,id in enumerate(ids):
 q=D[id,'30s'];a=axs[0]
 if q['coherent_max'] is None:a.barh(j,q['bg_max'],color='#b4b4b4',hatch='//',height=.65)
 else:a.barh(j,q['coherent_max'],color='#168c94',height=.65);a.barh(j,q['residual_max'],left=q['coherent_max'],color='#dba248',height=.65)
 a.text(q['bg_max']+.7,j,f'{q["bg_max"]:.1f}%',va='center',fontsize=9)
 for i,k in enumerate(KEYS):
  q=D[id,k];axs[i+1].plot([q['theta_min'],q['theta_max']],[j,j],color='#354d83',lw=4,solid_capstyle='round');axs[i+1].text(92,j,f'{q["theta_min"]:.1f}–{q["theta_max"]:.1f}°',va='center',fontsize=9)
axs[0].set(yticks=range(len(ids)),yticklabels=labels,xlim=(0,68),title='Background composition · 30–31 s',xlabel='% of maximum measured total')
from matplotlib.patches import Patch
axs[0].legend(handles=[Patch(color='#168c94',label='Coherent field'),Patch(color='#dba248',label='Noninterfering'),Patch(facecolor='#b4b4b4',hatch='//',label='Split not identified')],fontsize=8,loc='lower right')
for i,a in enumerate(axs):
 a.invert_yaxis();a.grid(axis='x',alpha=.2)
 if i:a.set(yticks=range(len(ids)),yticklabels=[],xlim=(0,124),xticks=[0,30,60,90],xlabel='Folded θ (degrees)',title=f'Angle range · {WIN[i-1]}');k=KEYS[i-1];a.axvspan(z['direct'][k]['theta_min'],z['direct'][k]['theta_max'],color='#aab2bb',alpha=.22,zorder=-1)
fig.suptitle('Background assumptions change inferred polar angles substantially\nGrey bands: direct no-background inversion; ranges are not uncertainty intervals',fontsize=14);save(fig,'03_background_and_angles')
# Budget and height curves.
fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for i,k in enumerate(KEYS):
 ids=['general_verified_cap0.1','general_cap0.2','general_cap0.3','general_verified_cap2.0'];qs=[D[id,k] for id in ids];axs[0].plot([q['bg_max'] for q in qs],[q['scores']['test']['xy_rms'] for q in qs],'o-',label=WIN[i],color=['#354d83','#cb573c'][i]);base=D['m5_cap2.0',k];axs[0].scatter(base['bg_max'],base['scores']['test']['xy_rms'],marker='x',s=70,color=['#354d83','#cb573c'][i]);qs=[D['general_verified_cap0.1',k]]+[D[f'height_R{rr}_cap0.1',k] for rr in [.05,.15,.3]];axs[1].plot([0,50,150,300],[q['scores']['test']['xy_rms'] for q in qs],'o-',label=WIN[i],color=['#354d83','#cb573c'][i])
axs[0].set(xlabel='Fitted BG / maximum measured total (%)',ylabel='Held-out XY RMS',title='General stationary field\n× = earlier five-mode relaxed fit',ylim=(0,.065));axs[1].set(xlabel='Allowed centre-orbit radius R (nm)',ylabel='Held-out XY RMS',title='Axial-motion sensitivity at 10% BG\nR limits are scenarios, not measured bounds',ylim=(0,.065))
for a in axs:a.grid(alpha=.2);a.legend()
save(fig,'04_fit_tradeoffs')
# Decompositions in normalized detector power. Signed cross term is never a positive pie slice.
ids=['m5_cap2.0','general_cap0.3','height_R0.3_cap0.1'];fig,axs=plt.subplots(3,2,figsize=(11,8.5),layout='constrained');t=np.arange(128)/128
for j,id in enumerate(ids):
 for i,k in enumerate(KEYS):
  q=D[id,k];mx=np.asarray(q['train']).sum(0).max();a=axs[j,i];s=np.asarray(q['signal']).sum(0)/mx;c=np.asarray(q['cross']).sum(0)/mx;b=np.sum(q['background'])/mx;inc=np.sum(q['noninterfering'])/mx
  for y,label,color,ls in [(s,'Rod alone','#354d83','-'),(c,'Signed interference','#a94c89','-'),(np.full(128,b),'All background','#b47c27','--'),(np.asarray(q['pred']).sum(0)/mx,'Predicted total','#168c94','-')]:a.plot(t,y,label=label,color=color,ls=ls,lw=1.6)
  a.scatter(t,np.asarray(q['test']).sum(0)/mx,s=5,color='#252525',label='Held-out total');a.axhline(0,color='grey',lw=.6);a.set(title=f'{WIN[i]} · {q["label"]}',xlabel='Ordered trajectory fraction',ylabel='Intensity / measured Tmax');a.grid(alpha=.15)
axs[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Power balance along the curve: T = signal + signed interference + background',fontsize=13);save(fig,'05_power_decomposition')
# Angles vs selected trace bin, no geometry truth implied.
ids=['bg','m5_cap2.0','general_cap0.3','general_verified_cap2.0','height_R0.3_cap0.1'];colors=['#bd8a23','#168c94','#6755a4','#b84864','#3679b4'];fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for i,k in enumerate(KEYS):
 dr=z['direct'][k];axs[i,0].plot(t,dr['theta'],'k--',label='Direct, no BG',lw=1.6)
 for id,color in zip(ids,colors):
  q=D[id,k];axs[i,0].plot(t,q['theta'],color=color,label=q['label'],lw=1.5);dp=(np.array(q['phi_mod180'])-dr['phi_mod180']+90)%180-90;dp[1:][abs(np.diff(dp))>90]=np.nan;axs[i,1].plot(t,dp,color=color,lw=1.5)
 axs[i,0].set(ylim=(0,90),ylabel='Folded θ (degrees)',title=WIN[i]);axs[i,1].set(ylim=(-90,90),ylabel='Δφ modulo 180° (degrees)',title='Difference from direct inversion')
 for a in axs[i]:a.set_xlabel('Ordered trajectory fraction');a.grid(alpha=.2)
axs[0,0].legend(fontsize=7,ncol=2);fig.suptitle('Conditional angle reconstructions — no known-angle reference',fontsize=14);save(fig,'06_angle_traces')
# Constraint controls.
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for i,k in enumerate(KEYS):
 for cap,ls in [('0.1','-'),('2.0','--')]:
  qs=[D[f'edge_w{w}_cap{cap}',k] for w in [.03,.1,.25]];axs[0].plot([.03,.1,.25],[q['scores']['test']['xy_rms'] for q in qs],'o'+ls,color=colors[i],label=WIN[i]+(' ≤10% BG' if cap=='0.1' else ' relaxed BG'))
f=json.load(open(R.parent/'three_tests/focus_aperture.json'))['summary'];xs=[.3,.6,1.2,3.];axs[1].plot(xs,[f[str(x)]['theta_change_deg'] for x in xs],'o-',color='#354d83');axs[1].set(xlabel='Assumed object-space aperture radius (µm)',ylabel='Maximum |Δθ| relative to focus (degrees)',title='Pure defocus ±300 nm, finite collection\nSimulation scenarios; not APD calibration',xscale='log');axs[1].set_xticks(xs,labels=['0.3','0.6','1.2','3.0']);axs[0].set(xlabel='Edge-envelope width w (NA units)',ylabel='Held-out XY RMS',title='Localized hole-edge field family');axs[0].legend(fontsize=8)
for a in axs:a.grid(alpha=.2)
save(fig,'07_edge_and_focus')
print('Wrote 7 figure sets (PNG + editable SVG).')
