from run import *
import importlib.util
spec=importlib.util.spec_from_file_location('fixed',R.parent/'fixed_excitation/run.py');fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
r=json.loads((R.parent/'fixed_excitation/fits.json').read_text());p=np.array(r['30s']['field_cap_checks']['2.0']['best']['p']);tm=np.load(R.parent/'fitted_predictions.npz')['train'].sum(0).min()
assert np.allclose(pred(p,128,tm,radial=False),fixed.fixed_predict(p,'field',128,tm)[0])
assert np.allclose(pred(np.r_[p[:-1],0,.5,p[-1]],128,tm),pred(p,128,tm,radial=False))
rng=np.random.default_rng(87);d=rng.normal(size=(10000,3));d/=np.linalg.norm(d,axis=1)[:,None]
# Projection on two orthogonal background spatial modes cannot exceed signal power.
assert np.min(q_channels(d)-GAMMA**2*(ANALYZERS@d[:,:2].T)**2-2*KAPPA**2*d[:,2][None,:]**2)>-1e-12
for j in range(30):
 pp=np.r_[p[:-1],rng.uniform(0,1),rng.uniform(-np.pi,np.pi),p[-1]];y=pred(pp,128,tm)
 assert y.min()>-1e-12
 assert np.allclose(y[0]+y[1],y[2]+y[3])
print('Passed: baseline equivalence, nested zero-mode limit, signal-overlap bound, positivity and analyzer pair-sum equality.')
