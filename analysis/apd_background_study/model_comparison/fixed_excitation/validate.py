from run import *
rng=np.random.default_rng(981)
p=np.r_[[.6,-1.1,.5,1.],np.zeros(15),[.1,.3,.2,.5,.6,.4,.2],np.log(.7)]
c,d=fixed_predict(p,'field',128,1.)
assert np.all(c>0)
assert np.allclose(d['signal'].sum(0),4*d['a0']**2*np.sin(np.deg2rad(d['theta']))**2*(ABC[0]+ABC[1]*np.sin(np.deg2rad(d['theta']))**2))
assert np.allclose(c[0]+c[1],c[2]+c[3])
old,details=predict(p[:-1],'field',c,1.,16,-1)
assert np.allclose(old,c,atol=1e-12)
lo,hi=bounds('field',16);lo=np.r_[lo,-8];hi=np.r_[hi,4]
start=p+rng.normal(0,.01,len(p))
f=least_squares(loss,start,args=('field',c,1.),bounds=(lo,hi),max_nfev=500,gtol=1e-12,ftol=1e-12,xtol=1e-12)
assert np.sqrt(np.mean(f.fun**2))<1e-8
print('Passed: positivity, total-signal law, pair sums, equivalence to old field equation at constrained amplitude, local noiseless recovery. No claim of global identifiability.')
