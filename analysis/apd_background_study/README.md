# APD background/interference study — meeting milestone, 6 October 2026

This directory preserves the accumulated analysis and makes the cached-result comparison reproducible. It is a research record, not a validated production correction. Read START_HERE.md, CONVENTIONS.md and STATUS.md.

## Fast reproduction (no raw file, network or private mask)

From this directory, install requirements.txt and run:

```bash
python model_comparison/meeting_2026_10_06/build_data.py
python model_comparison/meeting_2026_10_06/plot_figures.py
python model_comparison/meeting_2026_10_06/build_report.py
```

These reconstruct 46 saved model/window predictions, check them against saved scores, and generate seven PNG/SVG figure sets and a PDF/DOCX report. No optimization occurs. PDF rendering uses DejaVu Sans; on Linux install fonts-dejavu-core if absent. On other systems set the font directory in build_report.py. Numeric comparisons and figure rendering do not require that font package.

Current compact table: model_comparison/meeting_2026_10_06/comparison.csv. Complete per-bin signal, signed cross term, background, theta, phi and observed/predicted channels: comparison.json. Figures: figures/. Report outputs are generated into output/ and distributed separately in the meeting pack.

Additional checks: `python model_comparison/three_tests/check_saved.py` and `python model_comparison/three_tests/validate_general.py`. The latter verifies the fixed-position common-pupil representation, not experimental truth. Avoid rerunning optimizer entrypoints just to view saved results; fixed_excitation/run.py and several historical scripts overwrite output files, and some modules execute work when imported.

## Map of the record

- prefit-pipeline/: audited notebook snapshot, optical pupil model and preprocessing verification. The top-level repository notebook is not replaced by this snapshot.
- interval_study/: complete ordered traversals, channel-before-ratio averaging, chronological/interleaved and registration sensitivity. Cached averages are included; no optional dynamic time warping is used in the fitting baseline.
- model_comparison/models.py and fit_models.py: initial free-brightness/effective model comparisons, superseded as physical brightness models but retained for provenance.
- fixed_excitation/: one constant a0 per interval and circular-excitation amplitude.
- radial_field/, shared_field/: restricted coherent mode families and common fields across windows.
- bg_cap_matrix/: low-budget and matrix-sign diagnostics; active matrix remains unchanged.
- mask_feasibility/: measured-mask and free-overlap diagnostics. Arbitrary overlap is not a stationary-field fit.
- angle_audit/: earlier angle-sensitivity audit.
- three_tests/: bounded edge-field, complete fixed-position stationary representation, axial-motion and aperture sensitivities; README has equations and numerical checks.
- meeting_2026_10_06/: current comparison and report generators.
- notebooks/: previous self-contained discussion notebooks, preserved as dated snapshots, with editable raw-data paths.

## Raw data and private mask

The repository already contains files3_first40s.tdms. Raw entrypoints use that file by default; set APD_TDMS_PATH to another local path if needed. No new raw recordings or source papers are added in this milestone. Recreating measured-mask optics requires APD_MASK_PATH pointing to the private thesis image_mirror_binary.tif (thesis commit7576b50). Cached mask results are included, so report reproduction does not need that source asset. The notebook snapshot's own data path can be edited directly. Older report-generating notebooks are snapshots, not the recommended current entrypoint.

Equation/optimization changes were not made while packaging. Raw-data path adapters and documentation were added. Historical mask source, papers and correspondence are intentionally not redistributed here. Local chronological data variation, imperfect calibration and nonlinear-fit ambiguities remain material. Meeting pack is a milestone, not the later Codex handoff script requested for a future step.
