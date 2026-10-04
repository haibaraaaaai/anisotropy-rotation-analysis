"""Additional 960-pixel-radius quadrature check. Run with --mask PATH."""
import argparse
import json
from pathlib import Path
import numpy as np
from pupil_model import *

ap=argparse.ArgumentParser();ap.add_argument('--mask',required=True);args=ap.parse_args()
out=Path('analysis/pupil_comparison/results')
x,y,f,a=basis(radius=960);rho=np.hypot(x,y);fields=f()
qs={k:response(fields,keep,a) for k,keep in [
    ('annulus_0.38',rho>=.38),('mask_0',~mask_blocked(x,y,args.mask))]}
th,ph=np.meshgrid(np.arange(5.,86.),np.arange(0.,360.,5.),indexing='ij')
d=directions(th,ph);old=dict(np.load(out/'pupil_response_matrices.npz'))
r={k:float(np.max(np.linalg.norm(xy(q,d)-xy(old[k],d),axis=-1))) for k,q in qs.items()}
tr,_,_=invert_radial(xy(qs['mask_0'],d),abc_from_q(qs['annulus_0.38']))
r['theta_RMS_deg_R960']=float(np.sqrt(np.mean((tr-th)**2)))
r['theta_max_deg_R960']=float(np.max(np.abs(tr-th)))
(out/'fine_grid_check.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
