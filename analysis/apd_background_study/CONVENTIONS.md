# Measurement conventions and calibration

Updated 2026-10-03. Source labels refer to `SOURCE_NOTES.md`. These are documented conventions. The analysis notebook has been inspected (SOURCE_NOTES.md); experimental validity and instrument-specific calibration remain unverified.

## Angles and anisotropies

Theta is the polar angle away from the optical axis. Phi is the azimuth in the sample plane. The draft labels linear analyser angles relative to the vertical. Image-axis direction, handedness, and the viewing direction for clockwise versus counterclockwise still require an explicit experimental parity check.

Use physically labelled, appropriately corrected channel intensities:

$$X=\frac{I_0-I_{90}}{I_0+I_{90}},\qquad Y=\frac{I_{45}-I_{135}}{I_{45}+I_{135}}.$$

$$r=\sqrt{X^2+Y^2},\qquad \phi=\tfrac12\operatorname{atan2}(Y,X).$$

Use radians internally when deriving formulae; label degrees explicitly in experimental plots. This internal-unit choice is a recommendation, not a confirmed existing code convention. Check denominators and channel validity before calculating ratios. Negative corrected intensities can indicate noise or model failure; do not silently impose positivity without assessing the resulting bias.

In ideal matched measurements, the two analyser-pair sums represent the same total intensity. With real gains, apertures, offsets, or spatial sampling, raw pair sums need not match. Retain both sums and all four intensities during analysis instead of discarding them after obtaining X and Y.

## APD ordering and correction direction

Hugh's 14 September email, quoted in the TDMS thread, gives this acquisition mapping:

| Raw input | Physical analyser |
|---|---|
| ai0 | 90 degrees |
| ai1 | 45 degrees |
| ai2 | 135 degrees |
| ai3 | 0 degrees |

The correction matrix discussed on 15 September uses column-vector order `(I0, I90, I45, I135)`. Reorder acquisition channels explicitly before using it.

The bearing SI writes the forward model as `I_measured = D A I_true`, where A describes optical transmission and mixing, and D detector scaling. Its inverse is `I_true = A^-1 D^-1 I_measured`. Hugh confirmed that the matrix he sent is already the inverse correction: `I_actual = M I_measured`. Do not invert an inverse a second time. The actual numerical M was an attachment and has not been recovered here.

The user reported a beta-sign inconsistency in the bearing SI derivation on 14 September. Hugh agreed that either beta or the corresponding matrix signs must be changed consistently. In that discussion, the user's proposed beta was `(b-a)/2`, while the supplied bearing SI prints `(a-b)/2`. Treat this as a reported correction requiring the symbolic derivation and current code to be checked together. It does not establish that historical analyses were numerically wrong. Older and newer instrument matrices are not interchangeable without checking their calibration.

## Camera mosaic

Hugh explicitly confirmed on 16 September that the true pixel pattern is:

| | First column | Second column |
|---|---|---|
| First row | 90 degrees | 45 degrees |
| Second row | 135 degrees | 0 degrees |

The user had identified inconsistent 0/90 assignments across camera functions. Hugh planned to fix them; completion has not been verified. A pure 0/90 swap sends X to -X and leaves r, and thus a radial theta reconstruction, unchanged. It changes phi and parity. This statement assumes only a label swap; additional calibration or processing changes can have further effects. Cropping, ROI offsets, and array transformations can change mosaic parity, so verify actual indexing for each path.

## Theta mapping and coefficient names

For the camera SI convention:

$$\theta(r)=\arcsin\sqrt{\frac{A_c r}{B_c-C_c r}},\qquad r_{\max}=\frac{B_c}{A_c+C_c}.$$

Here `A_c=2 J3`, `B_c=J1-J2`, and `C_c=J1+J2-2 J3`. Subscript c distinguishes these scalar coefficients from optical matrix A and from the differently named scalar coefficients in the bearing SI. The square root is present in the Word equation structure and must survive conversion to text.

| Medium | Refractive index | A_c | B_c | C_c | r_max |
|---|---:|---:|---:|---:|---:|
| Water | 1.33 | 0.913 | 0.951 | 0.120 | 0.921 |
| Glycerol | 1.47 | 0.465 | 0.737 | 0.297 | 0.967 |

These rounded values are transcribed from the camera SI draft, not independently calibrated. The stated collection bounds are inner NA 0.39 and outer NA 1.3. Confirm the medium, geometry, formula implementation, and coefficients before use. Do not transfer the historical bearing paper's effective NA 1.28671 into this model automatically.

The camera SI clips r above its model limit before evaluating theta. Treat clipping as an explicit processing choice: record exceedance counts and retain unmodified r to investigate noise and model mismatch. The draft's q expression omits the refractive index in one printed equivalent expression; verify the angular-integral implementation instead of copying that line.

## Degeneracy and density

Four-channel reconstruction gives phi modulo pi and theta on a folded hemisphere. Phi -> phi+pi and theta -> pi-theta leave the ideal signal unchanged. Continuity or orbit assumptions choose branches but do not add independent measurements. Near the pole phi is poorly determined; near the equator theta inference is poorly conditioned. Distinguish these limitations from background bias.

A uniformly sampled orientation has marginal theta density proportional to sin(theta). Probability per steradian is instead uniform for an isotropic ensemble. Projection-space density, theta histograms, and density per solid angle require different Jacobians. An apparent peak must be assessed with the normalization and spot-selection procedure specified.

Use `r` only for anisotropy radius here. Use `E_s`, `E_b`, and phase `delta` for optical fields and relative phase; the camera SI instead uses r and b for field amplitudes and phi for interference phase. Do not conflate those symbols.

## Thesis scalar model and mechanical phase

Thesis §1.5 uses scalar coefficients A_F, B_F, C_F (subscript F added here to avoid optical-matrix and camera-coefficient ambiguity):

$$I_\psi=K[A_F+B_F\sin^2\theta+C_F\sin^2\theta\cos 2(\phi-\psi)].$$

K is a scale factor, not the measured sum of the four channels. In the ideal balanced model,

$$r=\frac{C_F\sin^2\theta}{A_F+B_F\sin^2\theta},\qquad \sin^2\theta=\frac{A_F r}{C_F-B_F r}.$$

This is the same functional structure as the camera convention, but coefficient values and normalisation must be checked before substitution. The phase of X+iY is 2 phi; it is not phi itself. Removing explicit azimuth denominators avoids those numerical singularities, but does not remove pole/equator conditioning or measurement degeneracies.

Use a distinct mechanical phase, e.g. chi, for motion around a fixed motor axis k:

$$\mathbf d(\chi)=\cos\lambda\,\mathbf k+\sin\lambda(\cos\chi\,\mathbf e_1+\sin\chi\,\mathbf e_2).$$

Here lambda is the cone half-angle, and e1,e2 span the plane perpendicular to k. A fixed cone generally has varying lab theta; fixed cone does not mean constant theta. Lab phi is the azimuth of d, and d(phi)/dt generally differs from d(chi)/dt for a tilted axis. Check angle branches and winding before interpreting revolution counts. In thesis §1.7 beta means motor-axis tilt, whereas in §1.6 beta is an optical transmission parameter; do not share one variable between them.

Thesis §1.6 and the notebook use APD inner NA about 0.38, whereas the camera SI uses about 0.39. Preserve these source-specific values pending calibration; they are not automatically a contradiction.

## Current report notation — 7 October 2026

Bold d, k and e1/e2 are real 3D orientation vectors. Bold E_b and U_m are
complex 2-component polarization fields over pupil position, not one number.
Their coefficients b_m are complex scalars. After an analyser, E_{b,i} and
E_{s,i} are scalar spatial fields; integrating gives scalar channel intensities.
The complete signed interference contribution is I_{int,i}; historical L_i
was the same quantity, while earlier aL expressions sometimes factored out
brightness. Bold four-channel I/B lists are not spatial vectors. Bold J and
e in local residual diagnostics mean a 2x2 Jacobian and a 2D XY residual.
Noninterfering and unpolarized are not synonyms; physical-field models retain
a positive-Stokes noninterfering component, while standalone additive BG has
the looser equal-pair-sum square domain. See consolidated/README.md.
