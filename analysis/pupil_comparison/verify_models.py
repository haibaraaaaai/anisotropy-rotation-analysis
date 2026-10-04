"""Numerical checks for the optical comparison, independent of the data fit."""
import json
from pathlib import Path
import numpy as np
from pupil_model import *

out=Path('analysis/pupil_comparison/results')
r=json.loads((out/'pupil_summary.json').read_text())
assert r['checks']['homogeneous_max_xy_error_R480'] < 1e-4
assert r['checks']['pair_sum_max_relative_R480'] < 1e-10
q=dict(np.load(out/'pupil_response_matrices.npz'))
d=directions(np.array([10.,30.,70.]),np.array([0.,22.,105.]))
for name,m in q.items():
    assert np.allclose(intensities(m,d),intensities(m,-d))
    assert np.min(np.linalg.eigvalsh(m)) >= -1e-12
fine=json.loads((out/'fine_grid_check.json').read_text())
assert fine['mask_0'] < 5e-5
assert abs(fine['theta_RMS_deg_R960']-r['comparisons']['hole_shape_038']['theta_error_deg']['rms'])<.001
print('PASS: homogeneous Fourkas limit, pair sums, nonnegative intensity, d/-d symmetry and grid convergence.')
