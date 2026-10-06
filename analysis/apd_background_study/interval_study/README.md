# Does the pre-fit trajectory repeat?

Local investigation, 5 October 2026. No repository changes or background/interference fits.

**Result:** the broad two-pass curve repeats, and averaging substantially reduces jitter.
The precise curve is not invariant across chronological intervals. Preserve that variability
instead of treating one smooth average as known ground truth.

- 30–31 s: 304 complete *detected traversals*, each with one clockwise outer-loop winding.
  A data-specific outer-loop crossing defines boundaries; this does not independently measure mechanical phase.
- Even/odd averages (152 cycles each) differ by about **0.008 geometric RMS** in raw X/Y.
- Eight consecutive 38-cycle averages differ from the full mean by **0.013–0.039**;
  eight interleaved groups of the same size differ by **0.009–0.017**. Chronological
  differences exceed those obtained by mixing cycles across the second. This is
  evidence of time-correlated variation, not a formal separation of detector noise,
  preprocessing error, background drift and rod motion.
- Constrained time alignment changes the overall mean by **0.026 geometric RMS**.
  It reduces registration error but must not be accepted just because curves agree better.
- Guide windows 21–81 samples change the unwarped mean by up to **0.015 geometric RMS**
  relative to the 41-sample default. Fine features smaller than this are not yet robust.
- Separated windows at 0, 10, 20, 30 and 39 s retain the broad two-pass structure.
  10–11 s is more consistent across its four subintervals than 30–31 s.

**Correction to the previous interpretation:** the selected first cycle reaching Y≈0.54
was not proof that the ensemble trajectory should reach that height. The 304-cycle
mean reaches Y≈0.39 without the old 201-sample periodic filter. That earlier single-cycle
comparison overstated the evidence for bias in the ensemble reference. The old filter
still demonstrably changes individual traces; neither old nor new reference is ground truth.

## What is averaged

The four original channel voltages are averaged within contiguous bins on each
observed cycle, then corresponding bins are averaged with **equal weight per cycle**.
X/Y are calculated afterwards. Cycles are not weighted by how long they take.
A light channel smoother sets bin boundaries only. Both repeated passes remain in
acquisition order. No cone or optical anisotropy template is imposed.

The optional alignment uses a reference from even cycles only, with monotone mappings,
fixed endpoints, a ±12-bin band and a step penalty. Some source bins are reused under
that mapping (about 22% of mapping steps repeat an index); those outputs are correlated.
It is a sensitivity diagnostic, not the default recommended final reference. The
unwarped and chronological-group comparisons are essential controls.

All comparisons use the same calibration. The archived gain estimates are held fixed;
T keeps the current numerical coefficients and /8 scale. Fresnel and pupil inversion
are not needed for this empirical repeatability test. No angle inference is performed.

## Interpretation and next step

Use a **time window containing several complete traversals**, review a single ordered
cycle as a branch-order reference, and check agreement across subwindows. This dataset
allows automatic boundaries using an isolated outer-loop crossing, avoiding hundreds
of manual cycle selections. The gate is dataset-specific and must be visually reviewed
on another sample. There is no universal one-track/two-track switch or assumed speed.

The old single-cycle binning is useful diagnostically but too noisy to promote directly
as the final fit reference. A candidate final fit should use multiple chronological
averages, retain between-window scatter, and demonstrate consistent conclusions across
registration choices. Do not infer that a good fit to the grand average fits every cycle.

## Run

Requires numpy, scipy, matplotlib, npTDMS and IPython. Put this notebook beside
`files3_first40s.tdms`, or edit `TDMS_PATH`. Saved outputs and plots are embedded below;
rerunning reads the raw data. The final cell also checks four other one-second windows.
The current numerical summary applies only to the supplied sample and settings.
