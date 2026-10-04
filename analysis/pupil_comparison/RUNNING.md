# Pupil comparison workflow

Run from the repository root with numpy, scipy, matplotlib, Pillow, npTDMS and nbformat installed.

```bash
OPENBLAS_NUM_THREADS=1 python analysis/pupil_comparison/run_notebook.py
OPENBLAS_NUM_THREADS=1 python analysis/pupil_comparison/compare_pupils.py --mask /path/to/image_mirror_binary.tif
OPENBLAS_NUM_THREADS=1 python analysis/pupil_comparison/refit_models.py
OPENBLAS_NUM_THREADS=1 python analysis/pupil_comparison/check_fine_grid.py --mask /path/to/image_mirror_binary.tif
python analysis/pupil_comparison/verify_models.py
```

Supply the mask from your local thesis checkout. Generated files go to the ignored `results/` subdirectory. Recorded results are in `report/`; see README.md. The measured mask is supplied separately.

The restored notebook core runs through its speed overview. The archived step searches, optional interactive explorations and the later Allan comparison needing a separate fixed-rod TDMS are outside this core runner.

The comparison refits are separate from the historical notebook optimizer. They preserve all selected points instead of permitting a background-dependent positivity mask to discard samples. They are sensitivity diagnostics, not absolute-angle validation.
