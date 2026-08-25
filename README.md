# anisotropy-rotation-analysis

This repository is the active workspace for anisotropy orientation and rotation analysis.

## Current notebooks

- `anisotropy_rotation_processing.ipynb` - Main processing workflow used for data analysis.
- `anisotropy_rotation_simulation.ipynb` - Forward model and geometry/signal simulation notebook.
- `anisotropy_rotation_theta_r_model.ipynb` - Theta/r deviation modeling and mechanism testing notebook.

## Current package and data

- `anisotropy_rotation_gui/` - Early GUI prototype derived from the processing workflow.
- `data/` - Input TDMS and related raw data files.
- `requirements.txt` - Python dependencies for notebook and analysis execution.

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
