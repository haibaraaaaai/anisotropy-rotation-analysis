# anisotropy-rotation-analysis

This repository is the active workspace for anisotropy orientation and rotation analysis.

Licensed under the MIT License. See `LICENSE`.

## Current notebooks

- `anisotropy_rotation_processing.ipynb` - Main processing workflow used for data analysis.
- `anisotropy_rotation_simulation.ipynb` - Forward model and geometry/signal simulation notebook.
- `anisotropy_rotation_theta_r_model.ipynb` - Theta/r deviation modeling and mechanism testing notebook.

## Current package and data

- `anisotropy_rotation_gui/` - Early GUI prototype derived from the processing workflow.
- `data/` - Input TDMS and related raw data files.
- `requirements.txt` - Python dependencies for notebook and analysis execution.

## Creating a shortened TDMS copy

Run `python trim_tdms_first60.py` to create
`data/Patricia/20250613/files3_first60s.tdms` without modifying the original.
You can also supply a source path, `--seconds` duration, and `--output` path.
For example, `python trim_tdms_first60.py files3_first60s.tdms --seconds 40 --output files3_first40s.tdms`
creates a 40-second test file (about 80 MB, below GitHub's 100 MiB file limit).
The script copies original
TDMS data segments byte-for-byte and retains later metadata-only segments,
preserving raw DAQmx samples, calibration properties, timestamps, and channel
names. It verifies the output sample counts and metadata and refuses
to overwrite existing files. Other inputs must have synchronous fixed-width
DAQmx channels and a segment boundary exactly at the requested duration; unsupported layouts
are rejected rather than rewritten.

## GUI status

The GUI prototype is currently on hold.

Why:
- The processing algorithms are still evolving quickly, so GUI behavior would churn frequently.
- Notebook iteration is currently faster and simpler for method development and diagnostics.

Plan:
- Keep the GUI code as a reference prototype.
- Resume GUI work after the core processing path stabilizes.

## Status

This repository is actively used for notebook-first analysis and method development.

## Current modeling direction

Decision log (2026-08-25): prioritize interference as the primary mechanism behind theta-r distortion in upcoming iterations.

Latest internal results indicate the dominant contributor to the observed theta-r curve distortion is interference, rather than pure background alone.

Near-term iteration focus:
- Prioritize interference-focused modeling and validation.
- Treat background-only tuning as secondary support work.

## APD background/interference study (6 October 2026)

The [study milestone](analysis/apd_background_study/README.md) contains the trajectory audit, saved fits, background/angle comparison figures, and reproducible meeting-report generators. It preserves the audited pre-fit notebook as a separate snapshot and leaves this repository’s main processing notebook unchanged.
