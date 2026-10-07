# APD background/interference study milestone — 6 October 2026

Read README.md, CONVENTIONS.md and STATUS.md first. The 3 October conventions retain historical provenance; the active values below supersede their pending-input statements.

Current baseline: APD voltage channels reorder 90,45,135,0 -> 0,90,45,135. Fixed historical gains before current inverse T; matrix unchanged. NA 0.38–1.3, water/oil 1.33/1.51, corrected Fresnel/BFP mapping. Fixed-cone comparisons use one a0 per interval with circular excitation a0*(dx+i*dy), 16 monotone phase parameters and three cone parameters. Background fits are conditional models, not validated absolute angle corrections.

The 40 s sample has now been analyzed, unlike the original October 3 baseline. Current report, figures, metrics and per-bin decompositions are in model_comparison/meeting_2026_10_06/. No future Codex handoff script has been prepared; discuss the next physical constraint before adding model freedom.

Latest correction (6 October, later follow-up): model_comparison/corrected_arc/ supersedes previous equal-arc forward-fit comparisons after fixing nonuniform final-reference coordinates. See latest STATUS.md entries before interpreting older meeting PDFs. Richard10 has a brightness/coefficient limiting degeneracy; it is not a stable recovered correction.

Latest sampling test: model_comparison/sampling_tests/ compares uniform final-reference arc interpolation against arc weighting of original bins; these agree closely. The simple uncorrected sphere-circle phase guide loses the inner loop after monotonicity enforcement and is explicitly rejected as a reference. Raw inversion range was valid. See current STATUS.md.

Latest three-family comparison: model_comparison/uniform_stationary/report.pdf. Uniform final-arc fixed references, one cyclic offset and direction per window, no 16-phase warp. General stationary field shared across two intervals; current results and caveats in STATUS.md.

Latest consolidated draft: model_comparison/consolidated/report.pdf (10 model families,45 fit cases). See associated run.py/report.py, simple.json/fields.json/axial.json, metrics.csv and latest STATUS entry. This uses fixed uniform-final-arc correspondence, not the historical16-phase scheme.

Wrap-up entry point (7 October): read model_comparison/consolidated/README.md
and report.pdf first. The report includes exact and local angular diagnostics
and a Section 3 discussion shortlist. Old 16-phase and earlier sphere plots
are historical, not current reference processing. Vector/scalar conventions
are explicit in the report and the latest CONVENTIONS.md entry.
