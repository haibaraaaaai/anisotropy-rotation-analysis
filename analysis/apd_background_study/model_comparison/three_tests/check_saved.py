import run as r
import numpy as np,json
from height import add_height
results=json.load(open(r.R/'fits.json'));count=0;cache={}
for key,v in results.items():
 if not v.get('complete'):raise RuntimeError('Incomplete fit '+key)
 if key.startswith('edge'):
  width=float(key.split('_w')[1].split('_')[0]);label=('edge',width)
  if label not in cache:cache[label]=r.make_basis('edge',width)
  opt=cache[label]
 else:
  label=('general',v['motion'])
  if label not in cache:
   cache[label]=r.make_basis('general')
   if v['motion']:cache[label]=add_height(cache[label])
  opt=cache[label]
 p=np.array(v['best']['p']);s=r.stats(p,opt,v['pure'],v['motion'])
 for k in r.KEYS:
  assert abs(s[k]['test']['xy_rms']-v['best']['stats'][k]['test']['xy_rms'])<1e-10,(key,k)
 count+=1
print('Reproduced scores for',count,'saved comparisons')
