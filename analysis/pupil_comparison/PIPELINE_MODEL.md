# Collection and background model — 2026-10-05

Decision from the user: retain the current numerical T matrix; use an annulus
with inner NA 0.38 and outer NA 1.3; enable water–oil Fresnel transmission and
the corrected BFP mapping in the main processing notebook. Exact-mask modelling
remains a separate sensitivity calculation. The archived report has not changed.

## Implemented collection model

`collection_model.py` supplies axisymmetric A/B/C coefficients using analytical
azimuth integration and numerical radial quadrature. Defaults: n_water=1.33,
n_oil=1.51. On a Cartesian BFP grid the field factor is
sqrt(cos(alpha_oil))/cos(alpha_water), multiplying transmitted s/p fields before
their rotation into analyser coordinates. The intensity integral has the square
of this factor and area measure rho d(rho) d(azimuth). Fresnel, the objective
mapping and the annular acceptance are one forward collection calculation.

With A+B normalized to one, A=0.8908059777, B=0.1091940223,
C=0.9277805723. The arbitrary common scale belongs in the signal amplitude;
these coefficients do not calibrate absolute power. The shape fit and phase
templates use the notebook's existing A/B/C interface. Historical homogeneous
coefficients remain available with USE_FRESNEL_BFP=False. Notebook execution
outputs were cleared to avoid displaying old results under the new model.

Verification: `python analysis/pupil_comparison/verify_pipeline_collection.py`.
The equal-index limit agrees with analytic Fourkas coefficients to 2e-13;
128/256-point radial quadrature agrees to 2e-13; an independent Cartesian BFP
grid of radius 480 agrees within 4.01e-5 in X/Y. Synthetic inversion recovers
the input angles away from the poles/equator. Notebook calibration and template
execute; all code cells compile. The full data fit was not rerun for this change.

## Meaning of correction order

With voltage offsets o, detector gain D and the adopted optical transmission
matrix A_opt, corrected ideal-analyser intensities are
I_total = A_opt^-1 D^-1 (v-o). The historical implementation includes a common
factor 1/8 in its inverse T matrix; this is retained, so background amplitudes
must use the same intensity convention. Offsets are electronic, not coherent
optical fields. No new offset estimate is introduced by this change.

This intensity correction assumes that the total collected field is described
by the adopted instrument model. Light injected downstream of part of that
instrument does not automatically obey the same transfer; unknown retardance,
spatially varying transmission or unmeasured circular polarization can also
invalidate a four-linear-channel model. Retaining T is a working assumption,
not a new validation of it.

The hole has a null space: rejected light cannot be recovered by an inverse
intensity correction. Instead, infer the rod orientation using the transmitted
signal predicted by the complete collection model. The processing order is
offset/gain/T correction, background-plus-interference separation in collected
channel coordinates, then rod-only orientation inference using the joint
collection model. A joint forward fit is also possible and need not literally
perform these steps one at a time.

## Background and interference: framework for discussion, not implemented

At a common observation plane write E_s = L_s E_rod and E_b = sum_k L_k E_bk,
where every L_k contains only the optical elements that background path
traverses. Intensity is the spatial and exposure-time integral of
|E_s+E_b|^2, not the square of an integral over detector locations. Thus
I_j = S_j + B_j + C_j, with C_j=2 Re <E_s,j,E_b,j> and
|C_j| <= 2 sqrt(S_j B_j). An independent incoherent background adds to B_j
but not to the cross term. This inequality is necessary, not sufficient for
identifiability or avoidance of background/geometry degeneracy.

The multi-background expression supplied by the user also needs background–
background cross terms when those fields are mutually coherent. Alternatively,
combine their fields into E_b before calculating B and C. Individually bounded
overlap coefficients are not a substitute for joint physical consistency.

A common linear optical operator distributes over field addition:
L(E_s+E_b)=L E_s+L E_b. Four integrated intensities do not contain the angular,
polarization and phase information needed to apply Fresnel or the pupil after
that integration. Background terms may be moved between bookkeeping planes
only with the corresponding propagation of fields or full coherence matrices;
their intensity and overlap parameters do not remain unchanged.

Stage modulation suppresses a cross term if the weighted temporal average of
exp(i delta(t)) is small. It does not locate the background relative to the
water–oil interface. Common phase changes on co-moving fields cancel in their
relative phase, though different illumination/scattering directions can still
give different phase responses. For constant amplitudes and complete periods
of sinusoidal phase delta=delta0+m sin(omega t), the residual factor is J0(m),
not automatically zero. Exposure and APD averaging must be accounted for.

The reduced I_j=S_j+B_j+c_j sqrt(S_j) model is a candidate at the collection
output, with |c_j|<=2 sqrt(B_j), only when effective normalized overlap and
relative phase are sufficiently stable. It is not a sample-plane intensity
expression to propagate through the collection optics. No new interference
fit, background bound from these data, or absolute-angle validation is claimed.
