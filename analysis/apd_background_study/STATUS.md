# Status — 6 October 2026

Completed: pre-fit trajectory audit; ordered cycle/channel averaging; repeatability checks; old background-only and Richard-effective comparisons; fixed excitation, uniform/radial and shared field fits; background caps; matrix sign diagnostic (current matrix retained); measured-mask comparison; arbitrary-overlap feasibility diagnostic; angle audit; constrained hole-edge, general stationary-field and bounded axial-motion tests; finite-aperture defocus sensitivity. Meeting report reconstructs 46 model/window predictions without new optimization.

Key findings: earlier five-mode shared solution has ~55% BG/Tmax, ~92% represented coherent power. General stationary fields give comparable residuals at20–30% BG; at10% BG, bounded axial motion improves agreement. These models recover materially different theta ranges. General selected fields use essentially all budget as coherent power. No absolute angle truth, no unique physical source composition, no globally certified minimum BG.

Measured amplitude mask changes selected held-out XY RMS <~0.0002. The tested narrow edge surrogate fits poorly; this does not exclude all physical hole scattering. Axial caps50/150/300nm are sensitivity assumptions; selected300nm scenario has ~41/43nm axial amplitudes, not a measurement of hook geometry. Finite collection and calibration remain unresolved. Stop here pending discussion with Richard.

Numerical caveat: edge_w0.03_cap0.1 selected run reaches600 function evaluations with success=false. Other reported selected runs terminate normally; local-minimum and active-bound cautions remain. General stationary completeness holds only for fixed-position ideal-dipole common-pupil/full-collection model. It does not hold for moving source or independent post-analyzer backgrounds.

The root processing notebook is left at latest main; the audited pre-fit notebook is retained explicitly under prefit-pipeline/. Historical scripts and reports are preserved with dates; their earlier pending tasks and initial conclusions are superseded by this file and current meeting report.
