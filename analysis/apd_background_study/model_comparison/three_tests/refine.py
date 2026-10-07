import run as r
import json,numpy as np
opt=r.make_basis('general');saved=json.load(open(r.R/'fits.json'));embed=json.load(open(r.R/'embedded_starts.json'))
for cap in [.1,2.]:
 starts=[np.array(saved[f'general_cap{cap}']['best']['p']),np.array(embed[str(cap)]['p'])]
 r.fit(f'general_verified_cap{cap}',opt,cap,starts=starts,nrandom=1)
saved=json.load(open(r.R/'fits.json'))
for cap in [.2,.3]:
 starts=[np.array(saved[f'general_verified_cap{c}']['best']['p']) for c in [.1,2.]]
 r.fit(f'general_cap{cap}',opt,cap,starts=starts,nrandom=1)
