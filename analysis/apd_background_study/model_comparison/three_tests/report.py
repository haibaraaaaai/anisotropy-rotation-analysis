from pathlib import Path
import json,zipfile,hashlib
R=Path(__file__).resolve().parent;ROOT=R.parent.parent;D=ROOT/'deliverables'
x=json.loads((R/'fits.json').read_text());focus=json.loads((R/'focus_aperture.json').read_text())
keys=['edge_w0.03_cap0.1','edge_w0.1_cap0.1','edge_w0.25_cap0.1','edge_w0.25_cap2.0','general_verified_cap0.1','general_cap0.2','general_cap0.3','general_verified_cap2.0','height_R0.05_cap0.1','height_R0.15_cap0.1','height_R0.3_cap0.1']
assert all(k in x and x[k]['complete'] for k in keys),'Analysis not finished'
summary={};rows=[]
for k in keys:
 v=x[k];b=v['best'];st=b['stats'];summary[k]={'cost':b['cost'],'success':b['success'],'nfev':b['nfev'],'stats':st}
 rows.append(f'| {k} | {st["30s"]["test"]["xy_rms"]:.4f} | {st["10s"]["test"]["xy_rms"]:.4f} | {100*st["30s"]["bg_fraction_max"]:.1f}% / {100*st["10s"]["bg_fraction_max"]:.1f}% |')
table='| Model | Held-out XY RMS, 30–31 s | 10–11 s | BG / maximum total, 30–31 s / 10–11 s |\n|---|---:|---:|---:|\n'+'\n'.join(rows)
heightrows=[]
for k in keys[-3:]:
 st=x[k]['best']['stats'];heightrows.append(f'- {k}: fitted radius {1000*st["30s"]["orbit_radius_um"]:.1f}/{1000*st["10s"]["orbit_radius_um"]:.1f} nm; axial amplitude {1000*st["30s"]["height_amplitude_um"]:.1f}/{1000*st["10s"]["height_amplitude_um"]:.1f} nm (30 s / 10 s).')
notes=f'''# Three bounded tests — 6 October 2026

User requested three analyses, then a pause before deciding meeting figures for Richard and subsequently committing/pushing accumulated updates. No meeting figures, commits or pushes were made in this step. This archive is a calculation record, not a presentation.

## Fixed baseline

Same calibrated four-channel data, current positive-beta inverse T and fixed historical gains. Annulus NA .38–1.3, n_water=1.33, n_oil=1.51, wavelength633nm. Correct Fresnel and BFP mapping. Alternate-cycle train/test averages:304 complete trajectories at30–31s and217 at10–11s;128 ordered bins. Background field and noninterfering intensity shared in absolute units between intervals. Separate cone geometry,16-point monotone phase map and constant a0 per interval. Excitation amplitude a0 sin(theta), with complex phase retained via d_x+i d_y. All background percentages below use MAXIMUM total measured intensity. Cap applied against smaller training maximum so it holds in both intervals. The relaxed cap is200%; its selected solutions are interior.

Residual objective: channel errors divided by measured total at each bin. Reporting XY RMS separately; it is not the optimized objective. Scores on held-out cycle averages assess trajectory repeatability, not known orientation accuracy. Train/test XY difference itself is .01559/.03973; this is context, not a formally estimated irreducible noise floor.

## 1. Constrained hole-edge field

Effective pupil-plane field: envelope exp[-(rho-.38)^2/(2 w^2)] on the accepted annulus, multiplied by1,cos(varphi),sin(varphi) and each of two transverse vector polarisations. Six complex coefficients, constant across the entire trajectory and both intervals. Three fixed widths w=.03,.10,.25 NA tested. All light in this test is one coherent field; no added incoherent intensity term. Background-background and background-signal cross terms are retained. Twelve real background parameters;40 interval geometry/timing/brightness parameters.

This is a deliberately low-order localized edge-field surrogate. It is NOT a Maxwell solution for the drilled wall: unknown sidewall topography/coating, mirror tilt, axial separation from true BFP and illumination on the wall prevent such a unique calculation. Broader envelopes test spreading but do not establish its source. A failure here excludes only these low-order localized patterns, not all hole scattering. No per-trajectory-point overlap is fitted.

## 2. General stationary common-pupil field

Under the current fixed-position dipole model, each signal channel has field

E_s,j(u,d)=a0(d_x+i d_y) sum_l d_l F_j,l(u).

Let V be the scalar function span of the x/y Cartesian components of the three collected dipole fields. Numerical rank is5 here. The complete background representation is two transverse components times V:10 complex coefficients. The remainder of any fixed background has each Cartesian component orthogonal to V. It has zero interference with every signal orientation in every analyzer and zero integrated cross terms with the represented field. Its remaining four linear-analyzer intensities are represented by a positive Stokes background (three observable real parameters). This proves completeness for the stated common-pupil, fixed-position, full-collection forward model;10 modes is a sufficient representation, not a claim of minimum parameter count or unique physical sources. Total shared background23 real parameters, plus40 interval parameters.

The represented field and residual intensity preserve positivity and a common background-power budget. This is stronger than independent channelwise Cauchy bounds at every point. Also, rod-background interference is quadratic in d under our circular-excitation law; on a fixed cone it contains mechanical-phase harmonics only through order2. Arbitrary pointwise overlaps need not satisfy that restriction. The rod-only intensity is quartic and can contain harmonics through order4.

Completeness does not remove nonconvexity of cone/phase/brightness fitting. Multi-start minima and feasible fits are reported, not global impossibility certificates or a proof that10% is the lowest feasible background. It does not encompass independent post-analyzer backgrounds, unknown calibration, changing excitation polarisation, or spatial motion. Adding more stationary pupil modes cannot enlarge the fixed-position ideal-dipole observation model beyond this representation.

## 3. Bounded axial motion and focus

The centre's orbit radius R is NOT set by the unit-vector cone or assumed equal to rod length. For a circular centre orbit about the fitted motor axis, the axial component is modelled as z(chi)=R sin(theta_axis) cos(chi+delta), with independent R and delta per interval. Tested R caps50,150,300nm are declared sensitivity scenarios, not measured biological bounds. This calculation retains only the axial motion component; lateral motion and flexing/noncircular centre orbits are not included.

The rod field gets collection phase exp[i(2pi/lambda) z sqrt(n_water^2-rho^2)] and incident plane-wave phase exp[i(2pi/lambda)n_water z]. A weak Gaussian axial amplitude with Rayleigh range2um and Gouy phase -atan(z/2um) is included, referenced to the mean plane at its waist. This is an assumed illumination scenario, not a measured beam calibration. Mean source height relative to coverslip and near-field changes in rod response are not inferred.

At10% total BG, refit the ten-mode stationary field plus bounded axial motion. Once the source moves, these ten modes are a restricted physical background family: the fixed-position completeness proof no longer applies. Height changes the overlaps deterministically, not through a freely chosen value per bin. Pupil overlap tables use125 axial positions and cubic interpolation; max checked interpolation error below6e-7.

Separately simulated pure defocus with finite identical centred circular detector apertures, WITHOUT BG or excitation modulation: fixed theta20,45,75deg, phi0,22.5,45deg, z=-300..300nm. Object-space aperture radii .3,.6,1.2,3um and full collection. These are acceptance scenarios, not APD calibration. Full-collection powers invariant at1.3e-13; therefore this explicitly tests clipping, not assigning arbitrary focus-dependent loss to an ideal detector.

## Results

{table}

### Fitted motion amplitudes

{chr(10).join(heightrows)}

### Finite-aperture sensitivity

{json.dumps(focus['summary'],indent=2)}

Aperture result values are maxima over the sampled orientations and defocus positions, not global bounds. Theta changes are direct-inversion changes relative to the same clipped detector at focus; absolute clipping bias is not calibrated away. Equal centred apertures are an assumption; channel offsets may worsen effects. Illumination amplitude variation, phase-induced interference and finite-aperture effects were not all combined into one full instrument fit.

## Interpretation and stopping point

- A narrow smooth hole-edge field is insufficient in this tested family. Broadening its support helps but weakens specificity to an edge origin.
- General stationary fields substantially improve low-background fits.20–30% backgrounds can give trajectory agreement comparable to the earlier five-mode55% result; the earlier55% value is not required by the data independently of field assumptions.
- Compare height scenarios with stationary10% baseline, not with the unconstrained pointwise construction. Any apparent gain demonstrates model sensitivity, not actual centre motion or reliable recovered theta. Boundary-hitting radii and different local minima must be retained in interpretation.
- Position/field/background remain confounded. Experimental centre-motion, focus/aperture or phase-perturbation measurements are needed to distinguish them. Do not continue adding unmeasured freedom solely for curve agreement.

## Reproduction and verification

Archive retains original directory structure. Requires Python with NumPy, SciPy and Pillow; no raw TDMS, network or GitHub required for these cached-average tests. `python model_comparison/three_tests/check_saved.py` recomputes reported predictions and verifies stored scores without reoptimizing. Main run.py, refine.py and height.py retain multistart optimization; completed entries are cached/skipped. focus_aperture.py recomputes finite-aperture simulation. validate_general.py verifies embedding of old models and random-field overlap completeness. grid_check.py compares coarser/finer pupil integrations using the SAME continuous field, accounting for mode-basis changes.

Grid errors recorded in grid_checks.json: approximately1.2e-4 reference-total units for general fields and7.2e-4 for the thinnest edge test, small versus residuals. Finite-aperture FFT acceptance scenarios are discrete simulations, not calibrated precision bounds. Original baseline and prior model sources are included as dependencies; this archive does not retroactively validate their experimental assumptions.

Primary optics context for axial/lateral pupil phase: https://pmc.ncbi.nlm.nih.gov/articles/PMC5810908/ (index-matched derivation there; this calculation uses water-side propagating wavevectors and existing water–oil Fresnel transmission). Actual drilling/alignment context remains the user's bearing SI S1. No new source claims about this sample's physical orbit radius are introduced.

## Deferred tasks recorded from user

1. Pause and discuss what general result figures to prepare for Richard.
2. Produce those agreed figures.
3. Then commit and push accumulated code/results/notes, excluding large raw data and transient intermediates as appropriate. No repository mutation performed during this step.
'''
(R/'README.md').write_text(notes);(R/'summary.json').write_text(json.dumps(summary,indent=2));print(table)
