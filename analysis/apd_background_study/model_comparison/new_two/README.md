# Two matched tests — 6 October 2026

Run `python run.py` for cached multistart fits (completed results are skipped); `python report.py` to reproduce plots/numerical summaries; `python build_pdf.py` and `xelatex two_new_tests.tex` to build the PDF.

Square additive: same 95%-of-minimum-total cap as preceding Stokes baseline, but independent fa/fb bounds [-1,1]. Fixed excitation retained: this is not the historical shape-only objective. Exact Richard: four unrestricted real C_j, no constant B term; positivity checked on selected predictions. Same fixed gains/matrix, .38–1.3 optics, ordered 128-bin trajectories, train/test split, constant a0 and cone/phase parameters as baseline. Ten starts per model/window, all runs saved. No new full instrument calibration or angle truth.

No background power is inferred by the four-C model. Zero background in its arithmetic decomposition means omitted term, not physically absent background. Report includes the conditional Cauchy lower bound if coefficients are interpreted as physical interference with the fitted signal. No push authorized at this stage.
