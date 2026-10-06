from fit_models import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
r=json.loads((R/'fits.json').read_text());sens=json.loads((R/'sensitivity.json').read_text())
labels={'bg':'Background only','richard':'Effective intensity','field':'Restricted field'}
fig,axes=plt.subplots(1,3,figsize=(12,4.4))
for ax,(m,v) in zip(axes,r.items()):
 b=v['best'];pred,_=predict(np.array(b['p']),m,test,TMIN,16,b['direction'])
 obs=xy(test);fit=xy(pred)
 ax.plot(obs[:,0],obs[:,1],'.-',color='.55',ms=3,lw=.8,label='Held-out average')
 ax.plot(fit[:,0],fit[:,1],color='#cf5237',lw=1.5,label='Prediction')
 ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X = (I₀ − I₉₀)/(I₀ + I₉₀)',ylabel='Y = (I₄₅ − I₁₃₅)/(I₄₅ + I₁₃₅)',title=f"{labels[m]}\nHeld-out XY RMS: {b['test']['xy_rms']:.4f}")
 ax.set_aspect('equal');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('30–31 s: common gain/T correction; backgrounds included in predictions',fontsize=12)
fig.tight_layout();fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
# Independent-cycle chronological transfer, with fixed registration and fixed fit parameters.
groups=[]
for ids in np.array_split(np.arange(len(allc)),8):
 dat=allc[ids].mean(0);entry={}
 for m,v in r.items():
  b=v['best'];entry[m]=stats(np.array(b['p']),m,b['direction'],data=dat)['xy_rms']
 groups.append(entry)
(R/'group_scores.json').write_text(json.dumps(groups,indent=2))
print('chronological group errors',groups)
# Check field overlap bound and exact background-only nesting.
rng=np.random.default_rng(918);d=rng.normal(size=(10000,3));d/=np.linalg.norm(d,axis=1)[:,None]
assert np.min(q_channels(d)-GAMMA**2*(ANALYZERS@d[:,:2].T)**2)>-1e-12
pb=np.array(r['bg']['best']['p']);di=r['bg']['best']['direction'];base=predict(pb,'bg',train,TMIN,16,di)[0]
assert np.allclose(base,predict(np.r_[pb,[0,0,0,0]],'richard',train,TMIN,16,di)[0])
assert np.allclose(base,predict(np.r_[pb,[0,.7,.4,.3]],'field',train,TMIN,16,di)[0])
md=r'''# Background and interference: controlled comparison

**Exploratory result:** the specified restricted field model fits this averaged trajectory substantially better than background only or the constant-coefficient effective-intensity version tested here. This is evidence about curve agreement, not recovery of true rod angles or proof of the interference mechanism.

## Coordinates and data

$X=(I_0-I_{90})/(I_0+I_{90})$, $Y=(I_{45}-I_{135})/(I_{45}+I_{135})$.
The earlier first-cycle maximum $Y\simeq0.54$ referred to locally averaged **raw** channel voltages. Figures here use gain- and T-corrected channels; their maxima are not directly interchangeable.

Source: `files3_first40s.tdms`, 30–31 s. The preceding repeatability study extracted 304 complete trajectories using the same crossing gate and 128 ordered, contiguous raw-channel averaging bins per trajectory. Guide smoothing determined boundaries; channel means use raw samples. Equal-cycle means avoid overweighting slow cycles. Alternate cycles give 152 training and 152 held-out cycles. No optional dynamic time warping is used. Both passes of the anisotropy trajectory remain ordered, and the fitted mechanical phase advances exactly one full revolution; this assumes the extracted full trajectory represents one mechanical revolution.

The data and calibration used here are embedded below; this notebook reruns the forward comparison without the 77 MB source file. It does not rerun raw cycle extraction.

## Common assumptions and objective

Current gains and T are fixed; NA_in=0.38, NA_out=1.3, water/oil indices 1.33/1.51. Signal coefficients include Fresnel transmission and the BFP field mapping sqrt(cos(theta_oil))/cos(theta_water). The normalized coefficients are A=0.89080598, B=0.10919402, C=0.92778057.

Each model has three cone geometry parameters, sixteen phase-registration parameters (initial phase plus fifteen positive-increment ratios), and three additive-background parameters. The latter specify total background, degree of linear polarization in [0,1], and polarization angle. Both interference alternatives add four parameters, giving 22 versus 26 joint fit parameters. A common brightness at each point is profiled from the measured total intensity: this is a comparison conditional on total intensity, not a prediction of total brightness. Held-out total intensity is likewise supplied, but cone/background/registration parameters are not refitted.

The objective is mean squared four-channel residual divided by each point's measured total intensity squared, with equal weight per ordered bin. Reported XY RMS is sqrt(mean((X_pred-X_obs)^2+(Y_pred-Y_obs)^2)); it is a secondary metric, in a common observed coordinate space. No points are dropped. This is not a calibrated noise likelihood.

## Precisely which models were tested

1. **Background only:** $I_j=S_j+B_j$, with a physically admissible common Stokes background.
2. **Effective intensity:** $I_j=S_j+B_j+c_j\sqrt{S_j}$, with four constant channel coefficients and $|c_j|\le2\sqrt{B_j}$. This is an explicit operational version of Richard's effective-intensity suggestion; it is not every possible intensity-level interference model. Independent channel coefficients need not preserve the ideal analyzer pair-sum equality.
3. **Restricted field:** one spatially uniform transverse complex field over the accepted BFP annulus, with two complex polarization amplitudes, plus the same admissible incoherent background. The cross term is computed from its overlap with the Fresnel/BFP dipole field. Circular-excitation orientation phase is retained; its magnitude is absorbed in brightness. This particular spatial mode is an assumption. It is not a general model of all scattered fields or all locations of background origin.

Total background is restricted below 95% of the smallest training total. This deliberately bounded domain prevents an arbitrarily large-background escape and gives a unique positive brightness root. **It is not a bound inferred from raw data under destructive interference.** Warm-start refits at 80% and 99% gave the same solutions; fitted fractions are 38%, 49%, 56%, respectively. Thus these particular solutions are not pressing against that cap. Other unconstrained solutions are not excluded by this study.

## Results and interpretation

| Model | Training XY RMS | Held-out XY RMS |
|---|---:|---:|
| Background only | 0.05752 | 0.06074 |
| Effective intensity | 0.05481 | 0.05823 |
| Restricted field | 0.02171 | 0.02772 |

The field model reduces held-out XY RMS by about 54% relative to background only. Increasing phase registration from 16 to 24 or 32 intervals preserves the ranking (field held-out RMS 0.0273/0.0272). Eight chronological group means also favor the field model, although those groups overlap the training set and are only a drift diagnostic.

There is still structured residual, and field-fit incoherent background polarization reaches its allowed maximum. Multiple local minima exist. Finite multistart optimization is not proof of a global optimum. Neither curve agreement nor same-recording holdout establishes correct angles, cone uniqueness, correct calibration, or the actual background field. Angle parameters should not yet be interpreted as calibrated measurements.

## Historical background-only control

Using the historical curve-matching objective on the old 90 selected points versus the new 128-point training average gives best costs 0.0076486 and 0.0076754: essentially unchanged. These costs are in the old corrected-anisotropy objective and must not be compared numerically with the new channel objective. Both historical fits exploit polarization components whose vector length exceeds one; enforcing a physical Stokes background on the new average gives 0.0087087. No negative-intensity points were silently discarded in this control.

Thus cycle extraction alone has not removed the old mismatch in this interval. It remains possible that calibration error, fixed-cone inadequacy, averaging bias, or a different field model contributes. Next useful checks are a separately selected stable interval and synthetic recovery/identifiability tests; the present comparisons do not prove true-angle recovery.
'''
cells=[]
def markdown(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s,outputs=None):cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=outputs or []))
markdown(md)
cells.append(dict(cell_type='code',metadata={},source=['# Saved comparison figure; recreated by the plotting cell below.'],execution_count=1,outputs=[dict(output_type='display_data',data={'image/png':base64.b64encode((R/'comparison.png').read_bytes()).decode(),'text/plain':['Comparison figure']},metadata={})]))
code((R/'models.py').read_text())
code('import json\nfrom scipy.optimize import least_squares\n'+f'train=np.array({train.tolist()!r})\ntest=np.array({test.tolist()!r})\nTMIN=float(train.sum(axis=0).min())\n'+ 'saved='+repr({m:v['best'] for m,v in r.items()})+'\n'+ 'sensitivity='+repr(sens)+'\n'+ 'historical_control='+repr(legacy)+'\n')
code('''# Rerun from saved best solutions. All parameters are trained only on train.
refits={}
for model,best in saved.items():
    lo,hi=bounds(model,16,.95)
    f=least_squares(residual,np.clip(best['p'],lo+1e-9,hi-1e-9),
        args=(model,train,TMIN,16,best['direction']),bounds=(lo,hi),
        max_nfev=650,ftol=2e-9,xtol=2e-9,gtol=2e-9,x_scale='jac')
    refits[model]=f.x
    for name,data in [('train',train),('held out',test)]:
        pred,_=predict(f.x,model,data,TMIN,16,best['direction'])
        error=np.sqrt(np.mean(np.sum((xy(pred)-xy(data))**2,axis=1)))
        print(model,name,round(error,6))
''')
code('''import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,(model,best) in zip(axes,saved.items()):
    pred,_=predict(refits[model],model,test,TMIN,16,best['direction'])
    ax.plot(*xy(test).T,'.-',color='.6',label='Held out')
    ax.plot(*xy(pred).T,lw=1.5,label='Prediction')
    ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',title=model)
    ax.set_aspect('equal');ax.legend();ax.grid(alpha=.2)
plt.tight_layout();plt.show()
''')
# Syntax validation of all code cells, then deterministic rerun cells without display dependency.
for c in cells:
 if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},cells=cells)
for i,c in enumerate(cells):c['id']=f'comparison-{i}'
Path('deliverables').mkdir(exist_ok=True)
Path('deliverables/background_interference_comparison.ipynb').write_text(json.dumps(nb,indent=1))
print('notebook written')
