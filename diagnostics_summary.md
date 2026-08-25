# Background-fit diagnostic investigation — summary

Notebook: `Fourkas_processing_v3.ipynb`
Sample: `file1.tdms` (the "bad sample")
Date range: investigation conducted across Sections 9b–9n.

---

## 1. The original problem

The shape-only fit and the joint shape+gain fit (Sections 7–9a, plus the
extended fit in Section 9h "phase-matched joint") consistently produced
background shape parameters $f_a, f_b$ pegged at the box edge ±1, with
$\delta_a = f_a \cdot \delta_c$ and $\delta_b = f_b \cdot \delta_c$.
Pegging at the bound means the optimiser is reporting a corner of the box,
not an interior minimum — so the fit is not localising the true bg.

The forward model is

$$
c_k(\psi) \;=\; I_\text{tot}(\psi)\,f_k\!\big(\theta(\psi),\phi(\psi);\Theta_\text{ax},\Phi_\text{ax},\Lambda\big)
\;+\; b_k,
\qquad k\in\{0,90,45,135\},
$$

with $b_{45} = b_{135} \equiv \delta_a$, $b_0 = b_{90} \equiv \delta_b$
parameterised as $\delta_a = f_a\,\delta_c,\ \delta_b = f_b\,\delta_c,\
\delta_c$ free; $f_a, f_b\in[-1.5,+1.5]$, $\delta_c\in[0,\delta_c^{\max}]$.

We want to know **why** the bg parameters cannot be localised, and whether
this signals a degeneracy in the model, a calibration error, or genuine
physics outside the model.

---

## 2. Diagnostic cells in chronological order

### 9b. Repeat shape-only fit with tightened bounds and randomised seeds

**Goal.** Confirm $f_a,f_b$ pegging is robust across optimiser starts, not
an artefact of one bad seed.

**Result.** All seeds converge to $f_a=+1.5$ or $-1.5$ and $f_b=+1.5$ or
$-1.5$ (the ±box edge). The cost surface is extremely flat in the
$(f_a,f_b)$ direction once $\delta_c$ has been chosen — the optimiser slides
along the degenerate direction until it hits the box.

**Take-away.** The pegging is reproducible and reflects a flat valley in
the cost, not a numerical accident. → there is a real degeneracy.

### 9c. 1-D cost slices along $f_a, f_b, \delta_c$

**Result.** $\delta_c$ has a clean parabolic minimum near 0.04–0.05. $f_a$
and $f_b$ are essentially flat over the whole $[-1.5,1.5]$ range; the cost
varies by $<10^{-4}$ across the slice while shifting by 3 units of $f$.

**Analysis.** This is the visual confirmation of a **gain × bg
degeneracy**: per-channel multiplicative gain $a_k$ and per-channel bg
$b_k$ can compensate each other so that the observed $c_k$ is fit equally
well by infinitely many $(a_k, b_k)$ pairs along a 1-parameter family.

### 9d–9g. Profile-likelihood, conditioning, parameter freezing

**Result.** Profile likelihoods reproduce the slice picture; the
Hessian is nearly rank-deficient on the (gain, bg) block. Freezing $f_a,
f_b$ to specific values and refitting moves $\delta_c$ correspondingly to
preserve the data fit — direct evidence of a sliding minimum.

**Take-away.** The geometry parameters $(\Theta_\text{ax}, \Phi_\text{ax},
\Lambda)$ are well-determined; the gain and bg pieces are not.

### 9h. Phase-matched joint fit: full geometry + gain + bg
*(cell with `_phase` parameter suffix)*

**Goal.** Use a joint fit that sees not just the closed (a_x, a_y) shape
but the actual phase trajectory, hoping the time-ordering breaks the
degeneracy.

**Result.**
- Geometry recovered cleanly: $\Theta_\text{ax}=7.39°$, $\Phi_\text{ax}=
  32.54°$, $\Lambda=30.02°$.
- Gain: $a_{90}=0.982$, $a_{45}=1.293$, $a_{135}=\textbf{1.500 — pegged at
  upper bound}$.
- Bg: $\delta_c=0.0460$, $f_a=+0.722$, $f_b=+0.950$.

**Take-away.** Pegging migrated from $f_a,f_b$ to $a_{135}$ — same
degeneracy, different victim. The phase-matched cost did **not** break the
gain × bg degeneracy on this sample. The $a_{135}=1.50$ value is **not
physical** — it just happens to be where the optimiser landed on the
flat valley after running into the upper bound.

### 9i. $I_\text{sum} = c_0+c_{90}+c_{45}+c_{135}$ vs phase
*(intensity-sum bg estimator)*

**Goal.** A common back-of-envelope: $I_\text{sum}$ should be the
incoherent total scattered power $I_\text{tot}(\psi) + 2\,\delta_c$. If
$I_\text{tot}$ has a clear minimum vs $\psi$, then
$\min_\psi I_\text{sum}/2$ is an upper bound on $\delta_c$ — and possibly
a good estimator.

**Result.** $I_\text{sum}$ is broadly modulated with $\psi$, with a
visible minimum at the orientation where the rod is end-on. The minimum
gave $\delta_c \approx 0.05$ (consistent with the 9d slice estimate).

**Take-away.** $I_\text{sum}$ gives a reasonable scalar estimator for
$\delta_c$ alone, but it is silent on $\delta_a, \delta_b$ — those require
the pair-difference structure, not the pair-sum. (See draft response to
supervisor for full discussion of why pair-sum cannot give $\delta_a,
\delta_b$.)

### 9j. Per-pair linear regression for $(\delta_a, \delta_b, \delta_c)$

**Setup.** Algebraic identity: if the forward model is exact and gain is
correct, then
$$
(c_0 - c_{90}) - D\,\hat a_x \;=\; \delta_a - \delta_c\,\hat a_x,
\qquad
(c_{45} - c_{135}) - D\,\hat a_y \;=\; \delta_b - \delta_c\,\hat a_y,
$$
where $D = (c_0+c_{90}+c_{45}+c_{135})/2$ and $(\hat a_x, \hat a_y)$ is
the *pure-Fourkas* anisotropy at the fitted geometry. Linear regression
of LHS on $\hat a_x$ (resp. $\hat a_y$) should give intercept $=\delta_a$
($\delta_b$) and slope $=-\delta_c$. The two pairs must give the same
$\delta_c$ if the model is correct.

**Result (9h gain).**
$\delta_a = -0.0009$, $\delta_b = -0.0020$,
$\delta_c\!\big|_\text{pair-a} = +0.0164$,
$\delta_c\!\big|_\text{pair-b} = -0.0174$,
**difference $=+0.0339$**, residual rms $\approx 0.018$ on each pair.

The plot of LHS vs $\hat a_x$ shows a **closed loop, not a line** — the
data does not satisfy the linear regression model. The line fit is
LSQ-optimal but the residuals are large and structured.

**Take-away.** The Fourkas+gain+constant-bg model is missing something at
the level of $\sim$2% of the signal. The $\delta_c$ from pair-a and pair-b
disagree by 0.034, which is bigger than $\delta_c$ itself.

### 9k. Phase offset $\Delta k$ scan

**Goal.** Test whether a sub-bin phase misalignment between data and
model is responsible for the loop in 9j.

**Result.** Cost is monotone-rising on either side of $\Delta k=0$ across
$\pm 5$ bins. No misalignment.

**Take-away.** Rules out timing/phase mismatch as the source of the loop.

### 9l. Numerical-aperture (NA) scan

**Goal.** Test whether the assumed NA (and thus the calibration constants
$A,B,C$) is wrong, which would distort $(\hat a_x,\hat a_y)$ and create a
spurious loop.

**Result.** Scanned NA$_\text{in}\in[0,1.10]$ and NA$_\text{out}$. No
minimum within the physically reasonable range; the ellipse loop and the
$\delta_c$ disagreement persist across the entire scan, with $\delta_c(a)
- \delta_c(b) \approx 0.05$ at every NA tested.

**Take-away.** Rules out NA / $(A,B,C)$ calibration error.

### 9m. Pair-sum invariant test
$D_a(\psi) = c_0+c_{90}$ vs $D_b(\psi) = c_{45}+c_{135}$

**Goal.** Under the Fourkas model with correct per-channel gain, the two
pair-sums must be identical (both equal $I_\text{tot}+2\delta_c$). Any
inequality is a direct violation of the gain × T-matrix correctness.

**Result.** Std and max of $(D_a-D_b)$ relative to mean, plus first 4
harmonic amplitudes:

| Gain | std | max rel | h1 | h2 | h3 | h4 |
|---|---|---|---|---|---|---|
| 9h gain | 3.6% | 14.2% | 0.021 | **0.038** | 0.010 | 0.008 |
| Sec 7 gain | **1.8%** | **6.2%** | 0.004 | 0.004 | 0.008 | 0.009 |
| uniform (1,1,1) | 3.6% | 39.0% | 0.031 | 0.033 | 0.012 | 0.003 |

**Take-away (initial).** Sec 7 gain wins on pair-sum: residuals look like
flat noise (all 4 harmonics $\sim 0.5\%$). 9h gain produces a strong
**2nd-harmonic** residual — exactly the signature that explains the loop
in 9j. Initial conclusion was: *Sec 7 is the correct gain; 9h's pegged
$a_{135}=1.5$ was the artefact.*

### 9n. Re-run 9j regression at Sec 7 gain (top vs bottom row plot)

**Goal.** If Sec 7 gain restores pair-sum invariance, the 9j ellipse
should collapse to a line and the $\delta_c$ disagreement should vanish.

**Result.**

| | 9h gain | Sec 7 gain |
|---|---|---|
| $\delta_a$ | $-0.0009$ | $+0.0128$ |
| $\delta_b$ | $-0.0020$ | $+0.0675$ |
| $\delta_c(a)$ | $+0.0164$ | $+0.0139$ |
| $\delta_c(b)$ | $-0.0174$ | $-0.0231$ |
| $\delta_c(a)-\delta_c(b)$ | $+0.0339$ | $+0.0370$ |
| rms_a | 0.0176 | 0.0184 |
| rms_b | 0.0195 | 0.0209 |
| 1st harmonic LHS_a residual | 0.022 | 0.017 |

**The ellipse loop is essentially identical at both gains.** The
$\delta_c$ disagreement is *gain-invariant* at $\sim 0.034$. Sec 7 gain
just shifts the LHS_b panel up (because $a_{45}=1.29$ inflates
$c_{45}-c_{135}$); the *shape* of the residual structure is unchanged.

**Critical reinterpretation of 9m.** Pair-sum invariance ($D_a = D_b$) is
a **necessary** condition for the matcor + gain to be self-consistent, and
Sec 7 gain meets it. But it is **not sufficient** for the regression model
9j to be correct. Pair-sum sees only the symmetric combination
$c_0+c_{90}+c_{45}+c_{135}$; it is blind to whatever 2nd-harmonic structure
exists in the *differences*. The 2nd-harmonic in pair-differences is
gain-invariant (any per-channel rescaling preserves $\cos(2\psi)$ content)
and survives both gain choices.

---

## 3. Cumulative diagnosis

The bg-fit pegging is caused by a **gain × bg degeneracy** (confirmed in
9b–9g). Sections 9h–9n then attempted to break the degeneracy by joining
data sources (phase-matched fit), exploiting algebraic identities (9j
regression), and varying nuisance parameters (9k phase offset, 9l NA).

What we found:

1. **Geometry is well-determined.** Across every fit variant
   $(\Theta_\text{ax},\Phi_\text{ax},\Lambda)$ converge to consistent
   values within ~1° / ~3°. The geometry block of the cost is well-conditioned.

2. **Gain is partially determined.** Sec 7 gain
   (independent algorithm based on pair-sum equalisation) and 9h gain
   disagree on $a_{135}$ at the 25% level. 9m shows Sec 7 is the better
   choice for pair-sum invariance, so we believe Sec 7 over 9h, but Sec 7
   itself has no error bars and we have no independent check on its
   accuracy.

3. **Constant-bg model is incomplete.** 9j+9n establish that, even with
   correct geometry and a gain that satisfies pair-sum invariance, the
   pair-difference data $(c_0-c_{90})$ and $(c_{45}-c_{135})$ have a
   $\cos(2\psi)$ component that the model
   $b_k = \text{const}_k$ cannot account for. The amplitude of the
   missing 2nd-harmonic is $\sim 2\%$ of the signal — large enough to
   make $\delta_c$ disagree by ${\sim}0.03$ between pair-a and pair-b
   regressions, and large enough to drive the joint fit's bg parameter
   to the box edge.

4. **The 2nd-harmonic in pair-differences is gain-invariant.** Per-channel
   gain only changes the *amplitudes* of the cos(ψ), sin(ψ), cos(2ψ),
   sin(2ψ) basis components in each $c_k$ proportionally; it cannot
   create or destroy a 2nd-harmonic component in the difference. This
   means **no per-channel gain choice can make the 9j regression close
   the loop**.

---

## 4. Hypotheses for the missing 2nd-harmonic in pair-differences

In rough order of plausibility:

(H1) **Imperfect T-matrix `matcor`.** `matcor` is a 4×4 cross-coupling
correction calibrated separately. If it has residual error in the
off-diagonal entries (5–10% level is typical), pair-sum signal at
$\cos(2\psi)$ leaks into pair-differences. This is consistent with the
observation that 9m's pair-sum invariant is restored by Sec 7 gain
(diagonal correction) while 9j's loops persist (off-diagonal leakage
not addressed).

(H2) **Polar / axis wobble.** If the rod's polar axis $\Theta_\text{ax}$
oscillates synchronously with the rotation phase (e.g. precession from a
slightly off-axis tether), the measured $f_k(\theta(\psi),\phi(\psi))$
acquires harmonics beyond the pure Fourkas first harmonic.

(H3) **Detector / amplifier nonlinearity.** Any quadratic term in the
per-channel response $c_k = a_k \cdot s_k + b_k + \beta_k s_k^2$ converts
the strong $\cos(\psi)$ signal into a $\cos(2\psi)$ contamination of the
pair-difference. Channel-specific $\beta_k$ would also cause pair-sum to
fail; the fact that Sec 7 gain restores pair-sum to noise floor argues
against this — but Sec 7 gain only enforces the *mean* ratio, not the
$\cos(2\psi)$-coefficient ratio.

(H4) **Rotating background.** The "bg" might not be constant in time but
rotate in space — e.g. coherent stray light from a fixed feature near the
rod. Then $b_k(\psi)$ has $\cos(\psi)$ and $\cos(2\psi)$ components.
Plausible for low-SNR samples; would explain why this is the "bad"
sample.

(H5) **Saturation / R_SAT.** The hole-correction (R_SAT) is a nonlinear
remapping that creates harmonics. If R_SAT is mis-calibrated for this
sample (e.g. NA_eff slightly different in the lab beyond the 9l scan
because the optical system varies), the residual could be a 2nd-harmonic.

---

## 5. Recommended next steps (for tomorrow)

**Math first — re-derivation suggestions.**

(a) **Derive the leakage formula explicitly.** Take a perturbed T-matrix
$M = M_0 + \epsilon$ and compute
$\partial(c_0-c_{90})/\partial\epsilon_{ij}$ to find which off-diagonal
entries contribute to a $\cos(2\psi)$ signal in pair-difference. Then
express the observed harmonic amplitude as a function of those
perturbations. If the magnitude works out, H1 is confirmed and the right
fix is a 16-parameter (per-pair? per-element?) T-matrix refit using a
calibration sample with known anisotropy.

(b) **Derive the relation between $\delta_a, \delta_b, \delta_c$ and
the pair-sum / pair-difference DC components.** In the *constant-bg*
model the DC component of $D_a = c_0+c_{90}$ is exactly
$\langle I_\text{tot}\rangle + 2\delta_c$ (no $\delta_a, \delta_b$
appear), and the DC components of pair-differences are
$\delta_a - \delta_c\langle\hat a_x\rangle$ and
$\delta_b - \delta_c\langle\hat a_y\rangle$. So **$I_\text{sum}$ alone
gives $\delta_c$**, but not $\delta_a, \delta_b$. This is the core
algebraic point for the supervisor reply.

(c) **Test the wobble hypothesis (H2).** Add $\theta(\psi) =
\Theta_\text{ax} + \alpha\cos(\psi+\phi_0)$ to the forward model and
refit. If $\alpha\sim 1°$–3° absorbs the loop, H2 is the answer.

(d) **Diagnostic 9o (ready to add):** harmonic decomposition of
LHS_a, LHS_b, $D_a-D_b$. Report
$A_0, A_1\cos\psi+B_1\sin\psi, A_2\cos2\psi+B_2\sin2\psi,
A_3\cos3\psi+B_3\sin3\psi$ amplitudes. Confirms which harmonics carry the
unexplained variance and gives a clean number to track as we change the
model.

(e) **Calibration sample test.** If a calibration sample with no bg and
known anisotropy is available (a fluorescent dye with known
depolarisation, or a polariser-only test), run 9j on it. If the loop
disappears, it confirms the loop is sample-specific (H4) rather than
instrument-specific (H1, H3, H5).

**Practical workaround.** If full root-cause is not tractable in the
short term:

- Accept that single-rod bg is non-recoverable from this data; fix
  $\delta_a, \delta_b, \delta_c$ at population means from many rods, or
  from independent dark-frame / scattered-light reference measurements.
- Use Sec 7 gain (not 9h) for downstream geometry-conditional analyses
  since it satisfies pair-sum invariance.

---

## 6. Files / variables to inspect

- Notebook: `Fourkas_processing_v3.ipynb` cells 9b–9n (sections at lines
  ~2050 onward).
- Variables in kernel after a full run:
  - Sec 7 gain: `a = [a90, a45, a135]` (list).
  - 9h gain: `a90_phase, a45_phase, a135_phase`.
  - 9h geometry: `Theta_axis_phase, Phi_axis_phase, Lambda_phase`.
  - 9h bg: `dc_phase, fa_phase, fb_phase` → $\delta_a, \delta_b, \delta_c$.
  - 9j regression bg: `delta_a_regr, delta_b_regr, delta_c_a_regr,
    delta_c_b_regr`.
  - Merged subsampled cycle (40 pts): `c0_fit, c90_fit, c45_fit, c135_fit`.
  - Calibration: `A_cal, B_cal, C_cal, R_SAT, NA_in, NA_out, nw`.
- Plots saved by the user: `bad_sample_m.png` (9m output),
  `bad_sample_n.png` (9n output).
- Repo memory: `/memories/repo/fourkas_processing_v3_notes.md`.
