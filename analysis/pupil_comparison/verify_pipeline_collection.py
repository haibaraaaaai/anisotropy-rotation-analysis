"""Run from repo root: python analysis/pupil_comparison/verify_pipeline_collection.py"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from collection_model import annular_interface_abc
from pupil_model import fourkas_abc, basis, response, analytic_q, xy, directions, invert_radial

# Independent analytic homogeneous limit, including the full pupil.
for inner in (0., .38, .39):
    got = np.array(annular_interface_abc(inner, n_immersion=1.33))
    expected = fourkas_abc(inner)
    expected /= expected[0]+expected[1]
    np.testing.assert_allclose(got, expected, atol=2e-13, rtol=2e-13)

abc = np.array(annular_interface_abc())
np.testing.assert_allclose(abc, annular_interface_abc(quadrature_order=256), atol=2e-13)
# Compare independent 2D Cartesian pupil integration, including s/p rotations.
x, y, fields, area = basis(radius=480)
q = response(fields(), np.hypot(x,y)>=.38, area)
t, p = np.meshgrid(np.linspace(5,85,17), np.arange(0,180,10), indexing='ij')
d = directions(t,p)
error = np.max(np.abs(xy(q,d)-xy(analytic_q(abc),d)))
assert error < 1e-4, error
theta, phi, u = invert_radial(xy(analytic_q(abc),d), abc)
np.testing.assert_allclose(theta,t,atol=2e-12)
np.testing.assert_allclose((phi-p+90)%180-90,0,atol=2e-12)

# Execute the actual notebook calibration cell and template with new defaults.
nb = json.loads((ROOT/'anisotropy_rotation_processing.ipynb').read_text())
ns = {'np':np}
exec(''.join(nb['cells'][6]['source']), ns)
exec(''.join(nb['cells'][23]['source']), ns)
np.testing.assert_allclose([ns['A'],ns['B'],ns['C']],abc)
assert ns['USE_FRESNEL_BFP'] and ns['NA_in']==.38
ax, ay = ns['fourkas_template'](40,20,25,*abc)
assert np.all(np.isfinite(ax)) and np.all(np.isfinite(ay))
for c in nb['cells']:
    if c['cell_type']=='code':
        compile(''.join(c['source']), '<notebook>', 'exec')
for bad in ({'na_in':1.4}, {'na_out':1.4}, {'n_sample':float('nan')}):
    try:
        annular_interface_abc(**bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
print(f'PASS: analytic limit, quadrature convergence, Cartesian pupil agreement '
      f'(max XY error {error:.3g}), angle round trip, notebook defaults/template and syntax.')
