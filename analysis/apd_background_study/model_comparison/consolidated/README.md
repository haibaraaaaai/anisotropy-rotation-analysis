# Consolidated APD background study — 7 October 2026

Start with `report.pdf`; Section 3 contains a short meeting discussion route.
The report contains 10 model families and 45 fit cases across 30–31 s and
10–11 s. The currently selected 40 s APD sample and historical calibration
remain the source; this is not Polarcam data.

## Authoritative outputs

- `simple.json`, `fields.json`, `axial.json`: selected parameters, convergence,
  per-window training/held-out diagnostics and four-channel decomposition.
- `metrics.csv`, `composition.json`: fitted powers, residuals and angle ranges.
- `sphere_branch_audit.json`: exact inversion continuity/domain audit for all
  90 window/case combinations. A closed XY curve need not have a closed lift.
- `local_residual/metrics.json`: local tangent residual and nonlinear checks
  for all 90 combinations. Figures use 10% caps where applicable.
- `report.py`, `notation.py`: main LaTeX/figure generation; `report.tex` is saved.

## Reproduce the report without refitting

From this directory, use Python with numpy, scipy and matplotlib and XeLaTeX:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python report.py
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

To regenerate local diagnostics first, run `python local_residual/run.py`.
`REUSE_ALL_FIGURES=1 python report.py` reuses existing main figures when only
text changes. All parent analysis directories are dependencies; do not copy
this directory alone. Cached cycle references allow report reconstruction
without reopening raw TDMS files. Optical caches and saved predictions are
retained. Python module imports may regenerate deterministic calibration or
optical cache files; run these scripts sequentially.

## Fits and refinement history

`run.py simple`, `run.py fields`, `run.py axial` perform the adaptive sweeps.
These can be expensive and overwrite result JSON files. `refine_edges.py`
and `refine_uncapped.py` document follow-up local optimization. The initial
Richard optimization failed; `richard_failed_initial.json` is retained for
audit only, not used as a valid comparison. Saved `simple.json` contains the
accepted no-background-seeded retry. Exact numerical optimization replay
may differ by library version; the saved selected results are authoritative.

## Fixed decisions

Gain correction precedes the unchanged inverse transmission matrix. One
pooled training-derived relative gain adjustment is applied to both windows.
Annulus NA 0.38–1.3; water/oil Fresnel and corrected BFP mapping. Uniform
final-training-arc reference, shared interpolation map for held-out cycles,
ordered correspondence with one offset/window; no 16-phase warp in current
fits. Brightness is one fitted constant per window, with explicit circular
excitation dependence, never a free pointwise brightness.

Finite caps start at 5/10%, then increments of 10% until the first invalid
corrected point on training OR held-out references. Invalid includes radial
exceedance and negative channels. No actual clipping is applied. Uncapped
means no finite additive-power ceiling. Richard's exact four-C approximation
and no-background baseline have no applicable background-power cap.

## Limits and unresolved questions

These are conditional local fits, not validated absolute rod orientations.
Held-out data guided adaptive exploration, so are repeatability checks rather
than untouched validation. Edge20%, axial5% and axial20% remain nonconverged.
Field models share background across windows; empirical intensity models
fit corrections separately. Their scores are not complexity-adjusted.

Uniform/3/5/general mode bases are modelling choices, not identified physical
sources. General is not a strict nesting of the uniform-containing five-mode
power representation. Extra noninterfering optical background remains Stokes
constrained; it is not necessarily unpolarized. Independent detector offsets
would require a different assumption. The standalone additive model permits
the looser equal-pair-sum square constraint.

Exact sphere inversion retains conditional corrected data. Local angular
residuals instead linearize around the cone and flag large steps, singularity,
invalid signal and nonlinear reproduction error. Local closure is not proof
of physical closure. Neither diagnostic re-solves interference at the recovered
orientation. Do not interpret huge local linear estimates as real rotations.

## Recommended experimental discussion

Prioritize a fixed-rod repeated bidirectional z-scan with four raw channels,
a stationary baseline and matched nearby rod-free measurements; then a
same-rotating-rod modulation off/on/off comparison with focus/collection
checks. Separately constrain electronic offsets and gain/mixing using suitable
calibration input. These are proposed experiments, not completed validations.
Pause expanding background flexibility until controlled measurements can
help distinguish explanations. Polarcam's observed intensity oscillation is
another task and is not evidence about this APD sample.
