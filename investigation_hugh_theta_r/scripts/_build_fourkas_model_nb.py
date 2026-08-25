"""Builds fourkas_theta_r_model.ipynb (self-contained, documented notebook)."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# Fourkas $\theta(r)$ deviation model

**Purpose.** A single, self-contained notebook that assembles *every* physical
ingredient we have discussed for explaining why the measured orientation
anisotropy $r$ of tumbling gold nanorods does **not** follow the analytic Fourkas
$\theta(r)$ curve. All knobs live in one `Config` block at the top so you can
change one thing, re-run, and see the effect on $p(r)$ and on Hugh's
CDF-inversion $\theta(r)$.

The four ingredients, each in its own section:

| # | Ingredient | Physical knob(s) | What it does to $r$ |
|---|------------|------------------|---------------------|
| 1 | **Collection optics** | NA, medium index $n$, hole-mirror annulus $\mathrm{NA_{in}},\mathrm{NA_{out}}$ | sets Fourkas $A,B,C$ and the ceiling $r_{\mathrm{max}}$ |
| 2 | **Rod emission** (transverse plasmon modes) | resonance peaks/widths, amplitude ratio $\to\rho$ | bends $r(\theta)$, lowers $r_{\mathrm{max}}$, can flip sign |
| 3 | **Signal & background** | avg signal $\bar I_{\mathrm{max}}$, avg background $\bar b$ (counts), rod-to-rod & cross-channel spreads | turns the analytic pile-up **spike** into the observed **bell** |
| 4 | **Interference** | coherence $\eta$, structured per-rod phase $\delta$ | extra signal-scaled variance; a thin tail past $r_{\mathrm{max}}$ |

**How to read the results.** The analytic Fourkas prediction has a Jacobian
*pile-up* of $r$ near $r_{\mathrm{max}}$ (most random orientations land near the top),
which produces a sharp spike in $p(r)$. The data instead show a broad *bell*.
The question is which combination of the four ingredients reproduces the bell —
and the headline finding is that **background dominates** — both its level and its
rod-to-rod / channel-to-channel variation.

> ⚠️ **Status of the inputs (please sanity-check).** Several numbers are
> *provisional* and flagged inline in `Config` as `[PROVISIONAL]`, `[ASSUMED]`, or
> `[MODEL CHOICE]`:
>
> - **Optics:** `na_collect`, `na_in`, `na_out` and the laser wavelength `laser_nm`
>   all still need re-checking against the actual setup.
> - **Rod emission:** the peaks/widths are read off the vendor spec sheet
>   (Section 2); one linewidth is *assumed*, and `amp_ratio_T_over_L` is a modelling
>   choice.
> - **Interference (Section 4)** implements Richard's coherent-background model from
>   his emails; the phase structure is simplified, so still treat it as provisional.
""")

md(r"""## 0. Setup and the master `Config`

Everything you might want to change is here, grouped by ingredient. Intensities are
in **absolute camera counts**: one average peak signal `avg_signal_ct` (at
$\theta=90°$) and one average background `avg_bg_ct` per channel — more intuitive
than a background *fraction*. Values flagged `[PROVISIONAL]`/`[ASSUMED]`/
`[MODEL CHOICE]` still need confirming.
""")

code(r"""from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


@dataclass
class Config:
    # ── 1. COLLECTION OPTICS ───────────────────────────────────────────────
    na_collect: float = 1.3          # [PROVISIONAL] objective NA — re-check
    n_medium:   float = 1.33         # medium / immersion index (water)
    use_hole_correction: bool = True # hole-mirror annular back-focal-plane collection
    na_in:  float = 0.38             # [PROVISIONAL] inner NA (hole edge)   [v4 Section 10]
    na_out: float = 1.3              # [PROVISIONAL] outer NA (annulus rim) [v4 Section 10]

    # ── 2. ROD EMISSION: transverse plasmon modes -> rho ───────────────────
    #   rho = |alpha_T|^2 / |alpha_L|^2 at the laser wavelength. amp=0 => no transverse.
    #   Peaks/widths below are read off the C12-40-600 vendor spec sheet (see Section 2).
    laser_nm:    float = 633.0       # [PROVISIONAL] excitation = detection (backscatter) — re-check
    lambda_L_nm: float = 600.0       # longitudinal peak      [spec "Peak SPR Wave" = 600]
    lambda_T_nm: float = 520.0       # transverse peak        [spec "Peak LSPR Wave" = 520]
    fwhm_L_nm:   float = 60.0        # longitudinal linewidth [spec "SPR Linewidth 80%" = 60]
    fwhm_T_nm:   float = 60.0        # [ASSUMED] transverse linewidth (spec gives only one)
    amp_ratio_T_over_L: float = 0.1  # [MODEL CHOICE] |alpha_T|/|alpha_L|; 0 => no transverse
    #   Rod-to-rod LSPR spread (aspect-ratio polydispersity) => rod-to-rod rho spread.
    #   Each recording draws lambda_L uniformly in this range. [spec "Peak SPR" = 575-625]
    lambda_L_spread_nm: tuple = (575.0, 625.0)

    # ── 3. SIGNAL & BACKGROUND (absolute camera counts) ────────────────────
    #   What sets r is the ratio bg/signal, so we keep three independent lognormal
    #   spreads: rod-to-rod background, cross-channel background, rod-to-rod signal.
    #   Only the bg:signal RATIO matters, so round illustrative values are fine.
    avg_signal_ct:    float = 100.0  # round illustrative peak signal (measured raw ~636 ct)
    avg_bg_ct:        float = 15.0   # => bg:signal ~15% (measured ~11%: bg~67 / signal~636)
    sigma_bg_rod:     float = 0.46   # MEASURED rod-to-rod sigma(ln bg)
                                     #   [FOV single-frame spatial CV ~0.31 is a lower bound]
    sigma_bg_ch:      float = 0.06   # channel-to-channel bg spread within a rod
                                     #   [FOV systematic cross-channel offset ~6%]
    sigma_signal_rod: float = 0.52   # MEASURED rod-to-rod sigma(ln I_max)
    enforce_sum_invariance: bool = True

    # ── 4. INTERFERENCE (Richard's coherent-background model) ─────────────
    #   eta = 0 -> incoherent additive background; eta = 1 -> full interference.
    #   Per channel: I_j = I_rod,j + b_j + 2*eta*sqrt(I_rod,j*b_j)*cos(delta),
    #   with a shared per-rod background phase delta (structured, not random):
    eta:              float = 0.0    # coherence of the rod+background cross term
    phase_center_deg: float = 0.0    # bg phase at the chosen focus (0=max; 90=cancels)
    phase_spread_deg: float = 360.0  # per-rod phase spread (0=all in phase; 360=random)

    # ── SIMULATION ─────────────────────────────────────────────────────────
    n_total:  int = 1_000_000        # pooled single-orientation samples
    n_groups: int = 115              # number of "recordings" (rods)
    seed:     int = 20260726

cfg = Config()
print(f"Config ready. avg signal I_max = {cfg.avg_signal_ct:.0f} ct; "
      f"avg background = {cfg.avg_bg_ct:.0f} ct "
      f"(~{100*cfg.avg_bg_ct/cfg.avg_signal_ct:.0f}% of peak).")
""")

md(r"""## 1. Collection optics → Fourkas $A, B, C$

The Fourkas model writes each polarisation channel as

$$I_j = I_\mathrm{tot}\left(A + B\sin^2\theta + C\sin^2\theta\,m_j\right),\quad m_j\in\{\cos2\phi,\ \sin2\phi,\ -\cos2\phi,\ -\sin2\phi\}.$$

$A,B,C$ are pure **aperture integrals** set by the collection optics. Our
microscope uses a **hole-mirror**: a central cone $\mathrm{NA}<\mathrm{NA_{in}}$
is blocked and only the annulus $\mathrm{NA_{in}}\!\dots\!\mathrm{NA_{out}}$ is
collected. That changes $A,B,C$ (and hence the ceiling $r_{\mathrm{max}} = C/(A+B)$)
relative to the textbook full-cone formula.
""")

code(r"""def fourkas_ABC_fullcone(na, n):
    # Textbook full-cone Fourkas A,B,C.
    alpha = np.arcsin(na / n); ca = np.cos(alpha)
    A = 1/6 - 1/4*ca + 1/12*ca**3
    B = 1/8*ca - 1/8*ca**3
    C = 7/48 - 1/16*ca - 1/16*ca**2 - 1/48*ca**3
    return A, B, C

def fourkas_ABC_annular(na_in, na_out, n):
    # A,B,C for annular (hole-mirror) collection NA_in..NA_out (v4 Section 10).
    a_in, a_out = np.arcsin(min(na_in/n, 1.0)), np.arcsin(min(na_out/n, 1.0))
    J1 = quad(lambda b: (2*np.cos(b)**2 + 0.75*np.sin(b)**4)*np.sin(b), a_in, a_out)[0]
    J2 = quad(lambda b: 0.25*np.sin(b)**4*np.sin(b), a_in, a_out)[0]
    J3 = quad(lambda b: np.sin(b)**2*np.cos(b)**2*np.sin(b), a_in, a_out)[0]
    C = J1 - J2; A = 2.0*J3; B = J1 + J2 - 2.0*J3
    return A, B, C

def effective_ABC(cfg):
    if cfg.use_hole_correction:
        return fourkas_ABC_annular(cfg.na_in, cfg.na_out, cfg.n_medium)
    return fourkas_ABC_fullcone(cfg.na_collect, cfg.n_medium)

A, B, C = effective_ABC(cfg)
r_max0 = C / (A + B)
print(f"{'hole-corrected' if cfg.use_hole_correction else 'full-cone'} "
      f"A={A:.4f}  B={B:.4f}  C={C:.4f}   r_max(rho=0) = {r_max0:.4f}")
""")

md(r"""## 2. Rod emission: transverse plasmon modes ($\rho$)

A rod is uniaxial: $\alpha = \alpha_L\,(\hat u\otimes\hat u) + \alpha_T\,(I-\hat u\otimes\hat u)$
— one **longitudinal** mode along the long axis and two degenerate **transverse**
modes. Keeping only the longitudinal mode forces the signal to zero at
$\theta=0$; adding the transverse response changes that.

Under circular excitation the drives are
$\langle|E\!\cdot\!\hat u|^2\rangle = \tfrac12\sin^2\theta$ (longitudinal) and
$1-\tfrac12\sin^2\theta$ (transverse). With
$\rho \equiv |\alpha_T|^2/|\alpha_L|^2$ the detected channel is

$$C_j = \tfrac{s^2}{2}F_\mathrm{dip}(j) + \rho\left(1-\tfrac{s^2}{2}\right)F_{2T}(j),\quad F_{2T}=(3A+2B)-F_\mathrm{dip},\ \ s^2=\sin^2\theta.$$

Consequences: bright non-zero centre; $r(\theta)$ becomes **non-monotonic** (sign
flip at $s^{2*}=2\rho/(1+\rho)$); and the ceiling shrinks,
$r_{\mathrm{max}}(\rho)=C(1-\rho)/[(A+B)+\rho(2A+B)]$.

For elastic backscatter, excitation and emission share $\lambda$, so
$\rho(\lambda)=(\text{amp ratio})^2\,g_T(\lambda)/g_L(\lambda)$ with Lorentzian
line-shapes $g$. `amp_ratio_T_over_L = 0` disables transverse modes.

### Vendor spec sheet (Nanopartz **C12-40-600**) — source of the rod-emission numbers

| Diameter | Length | Aspect ratio | Peak SPR | Peak LSPR | SPR linewidth 80% | Peak SPR range | SPR molar ext. | LSPR molar ext. |
|---|---|---|---|---|---|---|---|---|
| 40 nm | 68 nm | 1.7 | 600 nm | 520 nm | 60 nm | 575–625 nm | 1.59e10 | 6.35e9 |

**What each `Config` value is taken from (and what is assumed):**

| `Config` field | Value | Provenance |
|---|---|---|
| `lambda_L_nm` | 600 nm | spec "Peak SPR Wave" (dominant **longitudinal** peak) |
| `lambda_T_nm` | 520 nm | spec "Peak LSPR Wave" (**transverse** peak) |
| `fwhm_L_nm` | 60 nm | spec "SPR Linewidth 80%" (of the dominant peak) |
| `fwhm_T_nm` | 60 nm | **assumed** equal to `fwhm_L_nm` — the sheet gives only one linewidth |
| `lambda_L_spread_nm` | 575–625 nm | spec "Peak SPR (nm)" batch range |
| `amp_ratio_T_over_L` | 0.1 (default) | **modelling choice**, not from the sheet (0 = transverse off). The molar-extinction columns (L 1.59e10 vs T 6.35e9, ratio $\approx0.4$) could motivate a value later. |

> **Naming.** By convention the **longitudinal** (long-axis) mode is at the
> *longer* wavelength (~600 nm here) and the **transverse** at ~520 nm; the
> vendor's "SPR/LSPR" column labels are mapped to that convention above. Please
> double-check this mapping — it is the one genuinely ambiguous point in the sheet.

**Rod-to-rod $\rho$ variation is already included** and needs no separate knob:
each recording draws its own longitudinal peak $\lambda_L$ uniformly from
`lambda_L_spread_nm` (575–625 nm), so each rod gets a different $\rho$ at the laser
line. That LSPR-position spread *is* the $\rho$-polydispersity. (You could add more
spread via per-rod `fwhm` or `amp_ratio`, but the peak-position spread dominates.)
""")

code(r"""def lorentz_sq(lam, lam0, fwhm):
    hw = 0.5 * fwhm
    return hw*hw / ((lam - lam0)**2 + hw*hw)

def rho_of_lambdaL(cfg, lambda_L_nm):
    # rho at the laser wavelength for a given longitudinal peak position.
    gL = lorentz_sq(cfg.laser_nm, lambda_L_nm, cfg.fwhm_L_nm)
    gT = lorentz_sq(cfg.laser_nm, cfg.lambda_T_nm, cfg.fwhm_T_nm)
    return (cfg.amp_ratio_T_over_L**2) * gT / gL

def r_max_of_rho(A, B, C, rho):
    return C * (1.0 - rho) / ((A + B) + rho * (2*A + B))

def channels_transverse(theta, phi, A, B, C, rho):
    # Detected 4 channels including transverse modes. rho scalar or per-point array.
    s2 = np.sin(theta)**2
    long_drive, trans_drive = s2/2.0, 1.0 - s2/2.0
    iso = 3.0*A + 2.0*B
    def chan(m):
        Fdip = A + B*s2 + C*s2*m
        return long_drive*Fdip + rho*trans_drive*(iso - Fdip)
    c2, s2p = np.cos(2*phi), np.sin(2*phi)
    return chan(c2), chan(s2p), chan(-c2), chan(-s2p)

rho_nominal = rho_of_lambdaL(cfg, cfg.lambda_L_nm)
print(f"amp ratio = {cfg.amp_ratio_T_over_L}  ->  rho @ {cfg.laser_nm:.0f} nm = {rho_nominal:.3f}")
print(f"r_max shrinks from {r_max0:.3f} (rho=0) to {r_max_of_rho(A,B,C,rho_nominal):.3f}")
""")

md(r"""## 3. Signal & background (absolute counts)

Instead of a background *fraction* we set one **average peak signal**
$\bar I_{\mathrm{max}}$ (counts at $\theta=90°$, averaged over rods) and one
**average background** $\bar b$ (counts per channel). What controls $r$ is the
**ratio** $b/I$, so a rod climbs the $p(r)$ bell either by having *more background*
or by being *dimmer* — so we keep three independent lognormal spreads:

1. **Background, rod-to-rod** (`sigma_bg_rod`): each rod sits at a different spot on
   the coverslip and sees a different local background. The provided FOV profiles
   show the background is **not** flat — block-to-block spatial $\mathrm{CV}\approx0.31$
   in a single uncorrected frame; measured **cross-recording** it is larger,
   $\approx0.46$ (the benchmark default). *This* spread is what smears the spike
   into the bell.
2. **Background, cross-channel** (`sigma_bg_ch`): the four channels within one rod
   differ. On the FOV the *systematic* channel offset is only ~6% (the benchmark
   default); the rest is local scatter.
3. **Signal, rod-to-rod** (`sigma_signal_rod`): rods differ in brightness
   $I_{\mathrm{max}}$. Measured $\approx0.52$ across the 40 nm tumbling rods.

We still impose **sum-invariance** ($b_0+b_{90}=b_{45}+b_{135}$), the constraint the
real 2×2 parity channels obey. Each channel signal is
$I_{\mathrm{max}}\times[\,$orientation shape from $A,B,C\,]$ (the $\sin^2\theta$ /
$A,B,C$ terms of Section 1–2).

**Benchmark (measured, 42 good+pending 40×65 nm rods).** Raw peak $I_{\max}\approx636$
ct; in-file background $\approx67$ ct; so net signal $\approx636-67=569$ ct and
$b/(b+\mathrm{signal})\approx11\%$ at the bright orientation — consistent with Hugh's
~10% uncorrected figure. (The FOV profile frames read ~32 ct but use a different
gain/exposure, so the two count scales must not be mixed.)
""")

code(r"""def two_layer_bg(cfg, rng):
    # (n_groups,4) background in counts [b0,b45,b90,b135]: rod-to-rod x cross-channel.
    level = cfg.avg_bg_ct * rng.lognormal(0.0, cfg.sigma_bg_rod, size=cfg.n_groups)
    b = level[:, None] * rng.lognormal(0.0, cfg.sigma_bg_ch, size=(cfg.n_groups, 4))
    if cfg.enforce_sum_invariance:
        sum0 = b[:, 0] + b[:, 2]; diff = b[:, 1] - b[:, 3]
        b[:, 1] = 0.5*(sum0 + diff); b[:, 3] = 0.5*(sum0 - diff)
        neg = (b[:, 1] < 0) | (b[:, 3] < 0)
        b[neg, 1] = 0.5*sum0[neg]; b[neg, 3] = 0.5*sum0[neg]
    return b

_bg = two_layer_bg(cfg, np.random.default_rng(cfg.seed))
print(f"background per channel (counts): mean {_bg.mean():.0f}, "
      f"rod-to-rod range {_bg.mean(1).min():.0f}..{_bg.mean(1).max():.0f}")
""")

md(r"""## 4. Interference (Richard's coherent-background model)

The background is a **coherent** field — scattered by the objective/relay optics,
not by the sample — so it *interferes* with the rod. Evidence: a rod's intensity is
periodic in stage height $z$, with maxima ~220 nm apart (a round trip of
$2\,n_{\mathrm{oil}}\,\Delta z \approx 660$ nm $\approx\lambda$). The rod is a point
source with **one** phase across channels; the background adds as a second field
with a **shared per-rod phase** $\delta$. Per channel the amplitudes add:

$$I_j = \left|\sqrt{I_{\mathrm{rod},j}} + \sqrt{b_j}\,e^{i\delta}\right|^2 = I_{\mathrm{rod},j} + b_j + 2\eta\sqrt{I_{\mathrm{rod},j}\,b_j}\cos\delta,$$

with coherence $\eta\in[0,1]$ ($\eta=0$ = incoherent Sections 1–3; $\eta=1$ = full
interference). The phase is **structured, not random**: $\delta$ is set by the focus
height (`phase_center_deg`) and varies across rods/FOV by `phase_spread_deg`
(a coverslip wedge → slow spatial variation; `phase_spread_deg=360` recovers a fully
random phase, `=0` puts every rod in phase).

The cross term scales as $\sqrt{I_{\mathrm{rod}}\,b}$ (not $b$) and, for random phase,
is zero-mean — so it perturbs the bright near-$r_{\mathrm{max}}$ orientations and can
nudge points past $r_{\mathrm{max}}$. **Testable:** the relative modulation
$\approx\sqrt{B/R}$ with $B=b^2$ measured from a rodless region.
""")

code(r"""def detect(rod, bg, delta, eta):
    # Rod + background per channel (Richard's coherent-background model).
    #   eta = 0  -> incoherent additive sum
    #   eta > 0  -> each channel amplitude interferes with the background, sharing
    #              one per-rod phase delta:
    #   I_j = I_rod,j + b_j + 2*eta*sqrt(I_rod,j * b_j) * cos(delta)
    C0, C45, C90, C135 = rod
    b0, b45, b90, b135 = bg
    if eta <= 0.0:
        return (C0 + b0, C45 + b45, C90 + b90, C135 + b135)
    cosd = eta * np.cos(delta)
    def mix(Cj, bj):
        return Cj + bj + 2.0 * np.sqrt(np.clip(Cj * bj, 0.0, None)) * cosd
    return (mix(C0, b0), mix(C45, b45), mix(C90, b90), mix(C135, b135))
""")

md(r"""## 5. Orientation sampling, anisotropy, and Hugh's $\theta(r)$

- **Isotropic tumbling**: $\cos\theta$ uniform, $\phi$ uniform (folded to a
  hemisphere).
- **Anisotropy**: $x=\frac{C_0-C_{90}}{C_0+C_{90}}$, $y=\frac{C_{45}-C_{135}}{C_{45}+C_{135}}$,
  $r=\sqrt{x^2+y^2}$.
- **Hugh's method**: pool $(x,y)$ from many recordings, balance the sample so
  every $\phi$ bin contributes equally, histogram $r$, and invert the CDF assuming
  $\cos\theta$ is uniform: $\theta(r)=\arccos(1-\mathrm{CDF}_r)$.
""")

code(r"""def sample_isotropic(N, rng):
    cos_theta = rng.uniform(-1.0, 1.0, N)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2*np.pi, N)
    return np.minimum(theta, np.pi - theta), phi

def xy_from_channels(C0, C45, C90, C135):
    x = (C0 - C90) / (C0 + C90)
    y = (C45 - C135) / (C45 + C135)
    return x, y, np.hypot(x, y)

def r_formula(theta, A, B, C):
    s2 = np.sin(theta)**2
    return C*s2 / (A + B*s2)

def _phi_deg(x, y):
    return np.degrees(np.mod(0.5*np.arctan2(y, x), np.pi))

def balanced_r_sample(x, y, rng, phi_bin_deg=5.0):
    r = np.hypot(x, y); phi = _phi_deg(x, y)
    edges = np.arange(0.0, 180.0 + phi_bin_deg, phi_bin_deg)
    bin_id = np.clip(np.digitize(phi, edges) - 1, 0, len(edges) - 2)
    idx = [np.flatnonzero(bin_id == i) for i in range(len(edges) - 1)]
    idx = [ib for ib in idx if ib.size > 0]
    n_per = max(1, min(ib.size for ib in idx) // 2)
    picks = [rng.choice(ib, size=n_per, replace=False) for ib in idx]
    return r[np.concatenate(picks)]

def _smooth(counts, sigma_bins=3.0):
    if sigma_bins <= 0: return counts.copy()
    radius = max(1, int(round(4*sigma_bins)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5*(xs/sigma_bins)**2); k /= k.sum()
    return np.convolve(counts, k, mode="same")

def hugh_theta_of_r(r_vals, bins=160, lo_pct=0.5, hi_pct=99.5, sigma_bins=3.0):
    r = np.asarray(r_vals, float); r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, lo_pct), np.percentile(r, hi_pct)
    counts, edges = np.histogram(r, bins=bins, range=(lo, hi), density=True)
    centers = 0.5*(edges[:-1] + edges[1:]); widths = np.diff(edges)
    sm = np.maximum(_smooth(counts, sigma_bins), 0.0)
    area = np.sum(sm*widths)
    if area > 0: sm /= area
    cdf = np.clip(np.cumsum(sm*widths), 0.0, 1.0)
    theta = np.degrees(np.arccos(np.clip(1.0 - cdf, 0.0, 1.0)))
    return centers, np.maximum.accumulate(theta)

# Empirical Hugh curves, HARD-CODED (downsampled to 60 pts) so the notebook needs
# no external CSVs. Read once from theta_r_curve_*_recording_bootstrap.csv.
# theta/lo/hi are in degrees; pr is the empirical p(r) implied by Hugh's relation
# CDF(r) = 1 - cos(theta(r))  =>  p(r) = d/dr[1 - cos(theta)].
_HUGH_HARDCODED = {
    "40x65nm": dict(
        r=[0.0440, 0.0580, 0.0720, 0.0860, 0.1013, 0.1153, 0.1293, 0.1433, 0.1573, 0.1713, 0.1867, 0.2007, 0.2147, 0.2287, 0.2427, 0.2567, 0.2707, 0.2861, 0.3001, 0.3141, 0.3281, 0.3421, 0.3561, 0.3715, 0.3855, 0.3995, 0.4135, 0.4275, 0.4415, 0.4554, 0.4708, 0.4848, 0.4988, 0.5128, 0.5268, 0.5408, 0.5548, 0.5702, 0.5842, 0.5982, 0.6122, 0.6262, 0.6402, 0.6556, 0.6696, 0.6836, 0.6976, 0.7116, 0.7256, 0.7396, 0.7550, 0.7690, 0.7830, 0.7970, 0.8110, 0.8249, 0.8403, 0.8543, 0.8683, 0.8823],
        theta=[2.66, 5.95, 8.68, 11.14, 13.65, 15.85, 17.98, 20.08, 22.13, 24.14, 26.34, 28.33, 30.30, 32.27, 34.21, 36.10, 37.92, 39.89, 41.67, 43.42, 45.13, 46.79, 48.41, 50.16, 51.71, 53.25, 54.77, 56.28, 57.77, 59.25, 60.88, 62.36, 63.83, 65.26, 66.66, 68.02, 69.35, 70.79, 72.07, 73.32, 74.54, 75.74, 76.93, 78.22, 79.38, 80.52, 81.62, 82.67, 83.66, 84.57, 85.47, 86.22, 86.89, 87.51, 88.07, 88.58, 89.07, 89.46, 89.79, 90.00],
        lo=[2.23, 5.43, 8.02, 10.39, 12.85, 15.01, 17.15, 19.25, 21.28, 23.28, 25.45, 27.36, 29.21, 31.08, 32.91, 34.81, 36.58, 38.55, 40.31, 42.01, 43.70, 45.36, 46.99, 48.73, 50.23, 51.79, 53.32, 54.83, 56.31, 57.79, 59.42, 60.97, 62.40, 63.88, 65.29, 66.70, 68.06, 69.51, 70.83, 72.14, 73.41, 74.64, 75.88, 77.23, 78.46, 79.62, 80.79, 81.89, 82.96, 83.92, 84.89, 85.69, 86.43, 87.12, 87.72, 88.29, 88.86, 89.33, 89.70, 89.96],
        hi=[3.43, 6.52, 9.14, 11.60, 14.21, 16.45, 18.63, 20.78, 22.89, 24.95, 27.21, 29.21, 31.23, 33.20, 35.16, 37.11, 39.02, 41.06, 42.90, 44.72, 46.47, 48.20, 49.85, 51.60, 53.19, 54.71, 56.27, 57.79, 59.24, 60.71, 62.27, 63.75, 65.21, 66.65, 68.01, 69.33, 70.71, 72.07, 73.36, 74.60, 75.79, 76.99, 78.17, 79.39, 80.49, 81.55, 82.56, 83.56, 84.48, 85.34, 86.17, 86.86, 87.47, 87.99, 88.49, 88.92, 89.35, 89.67, 89.90, 90.00],
        pr=[0.234, 0.376, 0.484, 0.569, 0.657, 0.736, 0.814, 0.886, 0.953, 1.023, 1.103, 1.171, 1.239, 1.303, 1.345, 1.363, 1.384, 1.427, 1.465, 1.482, 1.487, 1.492, 1.496, 1.501, 1.512, 1.528, 1.544, 1.553, 1.564, 1.589, 1.618, 1.629, 1.624, 1.608, 1.582, 1.553, 1.533, 1.528, 1.509, 1.470, 1.448, 1.444, 1.434, 1.424, 1.414, 1.382, 1.328, 1.262, 1.180, 1.081, 0.968, 0.877, 0.801, 0.738, 0.670, 0.595, 0.521, 0.453, 0.343, 0.197]),
    "25x65nm": dict(
        r=[0.0416, 0.0549, 0.0682, 0.0816, 0.0962, 0.1095, 0.1229, 0.1362, 0.1495, 0.1628, 0.1775, 0.1908, 0.2042, 0.2175, 0.2308, 0.2441, 0.2575, 0.2721, 0.2854, 0.2988, 0.3121, 0.3254, 0.3388, 0.3534, 0.3667, 0.3801, 0.3934, 0.4067, 0.4200, 0.4334, 0.4480, 0.4614, 0.4747, 0.4880, 0.5013, 0.5147, 0.5280, 0.5426, 0.5560, 0.5693, 0.5826, 0.5959, 0.6093, 0.6239, 0.6373, 0.6506, 0.6639, 0.6772, 0.6906, 0.7039, 0.7186, 0.7319, 0.7452, 0.7585, 0.7719, 0.7852, 0.7998, 0.8132, 0.8265, 0.8398],
        theta=[2.60, 5.77, 8.41, 10.82, 13.34, 15.56, 17.75, 19.93, 22.09, 24.23, 26.54, 28.59, 30.61, 32.61, 34.57, 36.51, 38.41, 40.47, 42.29, 44.07, 45.83, 47.59, 49.34, 51.24, 52.93, 54.58, 56.20, 57.80, 59.38, 60.93, 62.60, 64.10, 65.58, 67.03, 68.45, 69.83, 71.18, 72.64, 73.93, 75.18, 76.38, 77.54, 78.64, 79.81, 80.84, 81.83, 82.78, 83.68, 84.52, 85.30, 86.08, 86.75, 87.36, 87.91, 88.42, 88.87, 89.30, 89.62, 89.85, 90.00],
        lo=[2.06, 5.28, 7.92, 10.30, 12.80, 15.00, 17.13, 19.25, 21.33, 23.39, 25.61, 27.64, 29.63, 31.62, 33.56, 35.49, 37.39, 39.44, 41.28, 43.06, 44.80, 46.50, 48.23, 50.09, 51.80, 53.41, 55.00, 56.57, 58.11, 59.67, 61.32, 62.84, 64.33, 65.77, 67.15, 68.53, 69.91, 71.42, 72.74, 74.00, 75.22, 76.42, 77.59, 78.85, 79.93, 80.98, 81.97, 82.92, 83.86, 84.69, 85.52, 86.22, 86.91, 87.53, 88.10, 88.62, 89.11, 89.48, 89.75, 89.94],
        hi=[3.13, 6.26, 8.93, 11.37, 13.95, 16.23, 18.47, 20.68, 22.88, 25.07, 27.46, 29.58, 31.64, 33.71, 35.72, 37.70, 39.61, 41.69, 43.58, 45.38, 47.11, 48.86, 50.60, 52.49, 54.19, 55.89, 57.53, 59.15, 60.70, 62.29, 63.98, 65.48, 66.92, 68.36, 69.72, 71.07, 72.38, 73.78, 75.01, 76.18, 77.32, 78.47, 79.56, 80.71, 81.73, 82.68, 83.59, 84.44, 85.23, 85.98, 86.71, 87.30, 87.85, 88.33, 88.76, 89.17, 89.53, 89.79, 89.97, 90.00],
        pr=[0.233, 0.369, 0.479, 0.576, 0.679, 0.774, 0.872, 0.969, 1.060, 1.140, 1.213, 1.274, 1.340, 1.399, 1.448, 1.497, 1.539, 1.567, 1.585, 1.611, 1.653, 1.700, 1.729, 1.740, 1.745, 1.747, 1.750, 1.762, 1.766, 1.759, 1.754, 1.753, 1.748, 1.732, 1.706, 1.679, 1.658, 1.635, 1.602, 1.556, 1.501, 1.443, 1.391, 1.346, 1.308, 1.257, 1.200, 1.136, 1.056, 0.974, 0.898, 0.832, 0.762, 0.696, 0.631, 0.557, 0.460, 0.362, 0.253, 0.139]),
}

def load_hugh_curves():
    return {k: {kk: np.asarray(vv, float) for kk, vv in v.items()}
            for k, v in _HUGH_HARDCODED.items()}

hugh = load_hugh_curves()
print("empirical curves (hard-coded):", list(hugh.keys()))
""")

md(r"""## 6. The full pipeline

`simulate(cfg)` chains all four ingredients:
1. sample isotropic orientations;
2. assign each to a "recording" (group) with its own $\rho$ (from its LSPR draw —
   this is where **rod-to-rod $\rho$ variation** enters), its own peak signal
   $I_{\mathrm{max}}$, and its own background + interference phase;
3. build the rod channels (optics + transverse), scale to each rod's $I_{\mathrm{max}}$;
4. add background, coherently or not (interference);
5. compute $(x,y,r)$.

It returns the raw pooled $r$ (the physical $p(r)$) and the balanced sample used
for Hugh's $\theta(r)$.
""")

code(r"""def simulate(cfg, rng=None):
    rng = rng or np.random.default_rng(cfg.seed)
    A, B, C = effective_ABC(cfg)

    theta, phi = sample_isotropic(cfg.n_total, rng)
    gid = rng.integers(0, cfg.n_groups, size=cfg.n_total)

    # per-recording rho from an LSPR draw (aspect-ratio polydispersity)
    lamL = rng.uniform(*cfg.lambda_L_spread_nm, size=cfg.n_groups)
    rho_g = np.array([rho_of_lambdaL(cfg, L) for L in lamL])
    rho_pt = rho_g[gid]

    C0, C45, C90, C135 = channels_transverse(theta, phi, A, B, C, rho_pt)
    # Normalise so the theta=90 longitudinal peak (rho=0) equals I_max, then give
    # each rod its own I_max (rod-to-rod signal spread) in absolute counts.
    peak_ref = 0.5 * (A + B + C)
    Imax_rod = cfg.avg_signal_ct * rng.lognormal(0.0, cfg.sigma_signal_rod, size=cfg.n_groups)
    scale = Imax_rod[gid] / peak_ref
    rod = (C0*scale, C45*scale, C90*scale, C135*scale)

    bg_tab = two_layer_bg(cfg, rng)
    bg = (bg_tab[gid, 0], bg_tab[gid, 1], bg_tab[gid, 2], bg_tab[gid, 3])
    # per-rod background phase: centre + uniform spread (spread=360 => fully random)
    half = np.radians(cfg.phase_spread_deg) / 2.0
    delta = (np.radians(cfg.phase_center_deg)
             + rng.uniform(-half, half, size=cfg.n_groups))[gid]

    d0, d45, d90, d135 = detect(rod, bg, delta, cfg.eta)
    x, y, r = xy_from_channels(d0, d45, d90, d135)
    r_bal = balanced_r_sample(x, y, rng)
    return {"A": A, "B": B, "C": C, "rho_g": rho_g,
            "r": r, "x": x, "y": y, "r_bal": r_bal}

out = simulate(cfg)
print(f"r_max(rho range) = {r_max_of_rho(*[out[k] for k in 'ABC'], out['rho_g'].max()):.3f}"
      f" .. {r_max_of_rho(*[out[k] for k in 'ABC'], out['rho_g'].min()):.3f}")
print(f"median r = {np.median(out['r']):.3f}")
""")

md(r"""## 7. Results: $p(r)$ and $\theta(r)$

The left panel is the physical $p(r)$ (the spike-vs-bell story); the right panel is
Hugh's $\theta(r)$ compared with the empirical 40×65 nm curve. Change any knob in
`Config` above, re-run from Section 6, and compare. The dotted grey line is the
analytic Fourkas prediction (no background, no transverse) — the thing the data
depart from.
""")

code(r"""def show(cfg, out, empirical="40x65nm"):
    A, B, C = out["A"], out["B"], out["C"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- p(r) ---
    edges = np.linspace(0, 1, 121); ctr = 0.5*(edges[:-1] + edges[1:])
    # analytic no-bg reference
    th_iso, ph_iso = sample_isotropic(cfg.n_total, np.random.default_rng(cfg.seed + 7))
    C0r, C45r, C90r, C135r = channels_transverse(th_iso, ph_iso, A, B, C, 0.0)
    r_ref = xy_from_channels(C0r, C45r, C90r, C135r)[2]
    h_ref, _ = np.histogram(r_ref, bins=edges, density=True)
    axL.plot(ctr, h_ref, "0.55", ls=":", lw=1.8, label="analytic Fourkas (no bg, no transverse)")
    h, _ = np.histogram(out["r"], bins=edges, density=True)
    lab = (f"model: bg={cfg.avg_bg_ct:.0f}ct / signal={cfg.avg_signal_ct:.0f}ct "
           f"(~{100*cfg.avg_bg_ct/cfg.avg_signal_ct:.0f}%), amp={cfg.amp_ratio_T_over_L}, "
           f"eta={cfg.eta}")
    axL.plot(ctr, h, "#d62728", lw=2.4, label=lab)
    if empirical in hugh:
        axL.plot(hugh[empirical]["r"], hugh[empirical]["pr"], "--", color="#8c564b",
                 lw=2.0, label=f"empirical {empirical}")
    axL.set_xlabel("r"); axL.set_ylabel("p(r)"); axL.set_xlim(0, 1); axL.set_ylim(bottom=0)
    axL.grid(alpha=0.25); axL.legend(frameon=False, fontsize=9)
    axL.set_title("p(r): background turns the pile-up spike into a bell")

    # --- theta(r) ---
    tt = np.linspace(0, np.pi/2, 400)
    cc, th = hugh_theta_of_r(out["r_bal"])
    rmse = np.nan
    if empirical in hugh:
        e = hugh[empirical]
        m = (cc >= e["r"].min()) & (cc <= e["r"].max())
        rmse = float(np.sqrt(np.mean((th[m] - np.interp(cc[m], e["r"], e["theta"])) ** 2)))
    axR.plot(r_formula(tt, A, B, C), np.degrees(tt), "k:", lw=1.6,
             label="analytic Fourkas (no bg, no transverse)")
    axR.plot(cc, th, "#d62728", lw=2.4, label="model")
    if empirical in hugh:
        e = hugh[empirical]
        axR.fill_between(e["r"], e["lo"], e["hi"], color="#8c564b", alpha=0.18)
        axR.plot(e["r"], e["theta"], "--", color="#8c564b", lw=2.4,
                 label=f"empirical {empirical}")
    axR.set_xlabel("r"); axR.set_ylabel(r"$\theta$ (deg)")
    axR.set_xlim(0, 1); axR.set_ylim(0, 92); axR.grid(alpha=0.25)
    axR.legend(frameon=False, fontsize=9, loc="lower right")
    axR.set_title(rf"Hugh $\theta(r)$ vs empirical — RMSE {rmse:.2f}°")
    fig.tight_layout(); plt.show()

show(cfg, out)
""")

md(r"""## 8. Fit sweep — background only (`avg_bg_ct`, `sigma_bg_rod`, `sigma_bg_ch`)

Grid-search the three background knobs — `avg_bg_ct`, `sigma_bg_rod`,
`sigma_bg_ch` — for the combination whose Hugh $\theta(r)$ best matches the
empirical 40×65 nm curve (RMSE in $\theta$ over the overlapping $r$ range), with the
transverse amplitude and interference **held at their `Config` values**. A smaller
`n_total` is used for speed. Each grid **includes the current `Config` value**, so
the sweep never reports a worse "best" than your current settings. The left panel is
the RMSE map over (`avg_bg_ct`, `sigma_bg_rod`) at the best `sigma_bg_ch`; the right
panel overlays the best-fit $\theta(r)$ on the data.
""")

code(r"""def theta_rmse(cfg_, empirical="40x65nm"):
    o = simulate(cfg_)
    cc, th = hugh_theta_of_r(o["r_bal"])
    e = hugh[empirical]
    m = (cc >= e["r"].min()) & (cc <= e["r"].max())
    th_emp = np.interp(cc[m], e["r"], e["theta"])
    return float(np.sqrt(np.mean((th[m] - th_emp) ** 2)))

def default_reference(n_total, empirical="40x65nm"):
    # simulate the DEFAULT cfg (at reduced n_total) -> (r-sample, cc, theta, rmse)
    o = simulate(replace(cfg, n_total=n_total))
    cc, th = hugh_theta_of_r(o["r_bal"])
    e = hugh[empirical]; m = (cc >= e["r"].min()) & (cc <= e["r"].max())
    rmse = float(np.sqrt(np.mean((th[m] - np.interp(cc[m], e["r"], e["theta"])) ** 2)))
    return o, cc, th, rmse

def grid_with(base, val, nd=4):
    # base grid PLUS the current Config value (dedup + sorted), so every sweep always
    # evaluates the current config point and only reports a different best if better.
    vals = {round(float(x), nd) for x in base}
    vals.add(round(float(val), nd))
    return sorted(vals)

BG_FRAC_CUR = cfg.avg_bg_ct / cfg.avg_signal_ct
BG_FRACS  = grid_with([0.05, 0.08, 0.11, 0.15, 0.20, 0.25, 0.30, 0.40], BG_FRAC_CUR)
BG_LEVELS = [round(f * cfg.avg_signal_ct, 3) for f in BG_FRACS]           # -> counts
SIG_ROD   = grid_with([round(0.25 + 0.05*i, 2) for i in range(10)], cfg.sigma_bg_rod)
SIG_CH    = grid_with([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40], cfg.sigma_bg_ch)
NFAST     = 250_000                             # smaller sample for the grid

results = []
for bg in BG_LEVELS:
    for sr in SIG_ROD:
        for sc in SIG_CH:
            c = replace(cfg, avg_bg_ct=bg, sigma_bg_rod=sr, sigma_bg_ch=sc,
                        n_total=NFAST)
            results.append((theta_rmse(c), bg, sr, sc))
results.sort()
best = results[0]
print("best 5  (RMSE deg | avg_bg_ct | sigma_bg_rod | sigma_bg_ch):")
for rmse, bg, sr, sc in results[:5]:
    print(f"  {rmse:5.2f}  |  {bg:6.1f}  |  {sr:.2f}  |  {sc:.2f}")
odef, ccd, thd, rmse_def = default_reference(NFAST)
print(f"default config RMSE = {rmse_def:.2f} deg  "
      f"(bg={cfg.avg_bg_ct:.0f}, sig_rod={cfg.sigma_bg_rod}, sig_ch={cfg.sigma_bg_ch}, amp={cfg.amp_ratio_T_over_L})")

edges = np.linspace(0, 1, 121); ctr = 0.5*(edges[:-1] + edges[1:])
tt = np.linspace(0, np.pi/2, 400)

fig, (axH, axT, axP) = plt.subplots(1, 3, figsize=(16.5, 4.9))

# (1) RMSE heatmap over (avg_bg_ct, sigma_bg_rod) at the best sigma_bg_ch
sc_best = best[3]
Z = np.full((len(SIG_ROD), len(BG_LEVELS)), np.nan)
for rmse, bg, sr, sc in results:
    if sc == sc_best:
        Z[SIG_ROD.index(sr), BG_LEVELS.index(bg)] = rmse
im = axH.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r")
axH.set_xticks(range(len(BG_LEVELS))); axH.set_xticklabels([f"{f*100:.0f}%" for f in BG_FRACS], rotation=90, fontsize=7)
axH.set_yticks(range(len(SIG_ROD))); axH.set_yticklabels(SIG_ROD, fontsize=7)
axH.set_xlabel("avg_bg_ct (% of signal)"); axH.set_ylabel("sigma_bg_rod")
axH.set_title(f"$\\theta(r)$ RMSE (deg) at $\\sigma_{{ch}}$={sc_best}")
fig.colorbar(im, ax=axH, shrink=0.85)

# best-fit run + analytic (no-bg) reference sample
cbest = replace(cfg, avg_bg_ct=best[1], sigma_bg_rod=best[2], sigma_bg_ch=best[3])
ob = simulate(cbest)
A_, B_, C_ = ob["A"], ob["B"], ob["C"]
th_iso, ph_iso = sample_isotropic(cbest.n_total, np.random.default_rng(cbest.seed + 7))
r_ana = xy_from_channels(*channels_transverse(th_iso, ph_iso, A_, B_, C_, 0.0))[2]
e = hugh["40x65nm"]

# (2) theta(r): analytic + empirical + default + best fit
cc, th = hugh_theta_of_r(ob["r_bal"])
axT.plot(r_formula(tt, A_, B_, C_), np.degrees(tt), "k:", lw=1.6,
         label="analytic Fourkas (no bg, no transverse)")
axT.fill_between(e["r"], e["lo"], e["hi"], color="#8c564b", alpha=0.18)
axT.plot(e["r"], e["theta"], "--", color="#8c564b", lw=2.2, label="empirical 40 nm")
axT.plot(ccd, thd, "#1f77b4", lw=1.4, ls="-.", label=f"default (RMSE {rmse_def:.2f}°)")
axT.plot(cc, th, "#d62728", lw=2.2, label=f"best fit (RMSE {best[0]:.2f}°)")
axT.set_xlim(0, 1); axT.set_ylim(0, 92); axT.set_xlabel("r"); axT.set_ylabel(r"$\theta$ (deg)")
axT.grid(alpha=0.25); axT.legend(frameon=False, fontsize=8, loc="lower right")
axT.set_title(f"θ(r): bg={best[1]}ct, σ_rod={best[2]}, σ_ch={best[3]}")

# (3) p(r): analytic + empirical + default + best fit
h_ana, _ = np.histogram(r_ana, bins=edges, density=True)
h_def, _ = np.histogram(odef["r"], bins=edges, density=True)
h_best, _ = np.histogram(ob["r"], bins=edges, density=True)
axP.plot(ctr, h_ana, "0.55", ls=":", lw=1.8, label="analytic (no bg, no transverse)")
axP.plot(e["r"], e["pr"], "--", color="#8c564b", lw=2.0, label="empirical 40 nm")
axP.plot(ctr, h_def, "#1f77b4", ls="-.", lw=1.4, label="default")
axP.plot(ctr, h_best, "#d62728", lw=2.2, label="best fit")
axP.set_xlim(0, 1); axP.set_ylim(bottom=0); axP.set_xlabel("r"); axP.set_ylabel("p(r)")
axP.grid(alpha=0.25); axP.legend(frameon=False, fontsize=8)
axP.set_title("p(r): best fit vs default vs empirical vs analytic")

fig.tight_layout(); plt.show()
""")

md(r"""## 9. Fit sweep including transverse amplitude (`amp_ratio_T_over_L`)

Same grid search as Section 8 but with the **transverse amplitude**
`amp_ratio_T_over_L` added as a fourth axis. The left panel shows the *best
achievable* RMSE at each amp (minimised over the background knobs) — i.e. whether
allowing transverse modes helps the fit at all; the right panel overlays the
overall best fit. Interference is held at the `Config` value. Each grid **includes
the current `Config` value** (so the current point is always evaluated).
""")

code(r"""# 4-D fit sweep: background (as Section 8) + transverse amplitude.
BG_FRAC_CUR4 = cfg.avg_bg_ct / cfg.avg_signal_ct
BG_FRACS4  = grid_with([0.05, 0.10, 0.15, 0.20, 0.30, 0.40], BG_FRAC_CUR4)
BG_LEVELS4 = [round(f * cfg.avg_signal_ct, 3) for f in BG_FRACS4]         # -> counts
SIG_ROD4   = grid_with([0.30, 0.40, 0.50, 0.60, 0.70], cfg.sigma_bg_rod)
SIG_CH4    = grid_with([0.0, 0.10, 0.20, 0.30], cfg.sigma_bg_ch)
AMP4       = grid_with([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], cfg.amp_ratio_T_over_L)
NFAST4     = 250_000

res4 = []
for bg in BG_LEVELS4:
    for sr in SIG_ROD4:
        for sc in SIG_CH4:
            for amp in AMP4:
                c = replace(cfg, avg_bg_ct=bg, sigma_bg_rod=sr, sigma_bg_ch=sc,
                            amp_ratio_T_over_L=amp, n_total=NFAST4)
                res4.append((theta_rmse(c), bg, sr, sc, amp))
res4.sort()
best4 = res4[0]
print("best 8  (RMSE deg | avg_bg_ct | sigma_bg_rod | sigma_bg_ch | amp):")
for rmse, bg, sr, sc, amp in res4[:8]:
    print(f"  {rmse:5.2f}  |  {bg:6.1f}  |  {sr:.2f}  |  {sc:.2f}  |  {amp:.2f}")
odef, ccd, thd, rmse_def = default_reference(NFAST4)
print(f"default config RMSE = {rmse_def:.2f} deg  "
      f"(bg={cfg.avg_bg_ct:.0f}, sig_rod={cfg.sigma_bg_rod}, sig_ch={cfg.sigma_bg_ch}, amp={cfg.amp_ratio_T_over_L})")
print("NB: Section 8's grid is finer than this 4-D grid, so its best RMSE can be lower "
      "even though amp is included here.")

# best achievable RMSE at each amp (min over background knobs): does amp help?
amp_best = {a: min(r[0] for r in res4 if r[4] == a) for a in AMP4}

edges = np.linspace(0, 1, 121); ctr = 0.5*(edges[:-1] + edges[1:])
tt = np.linspace(0, np.pi/2, 400)

fig, (axA, axT, axP) = plt.subplots(1, 3, figsize=(16.5, 4.9))

# (1) does transverse help? best achievable RMSE at each amp
axA.plot(list(amp_best), list(amp_best.values()), "o-", color="#1f77b4")
axA.axhline(rmse_def, color="0.4", ls="-.", lw=1.2, label=f"default ({rmse_def:.2f}°)")
axA.axvline(cfg.amp_ratio_T_over_L, color="0.7", ls=":", lw=1.0)
axA.set_xlabel("amp_ratio_T_over_L"); axA.set_ylabel("best achievable RMSE (deg)")
axA.grid(alpha=0.25); axA.legend(frameon=False, fontsize=8)
axA.set_title("Does transverse help?\n(min RMSE over bg, σ_rod, σ_ch)")

cb = replace(cfg, avg_bg_ct=best4[1], sigma_bg_rod=best4[2], sigma_bg_ch=best4[3],
             amp_ratio_T_over_L=best4[4])
ob = simulate(cb)
A_, B_, C_ = ob["A"], ob["B"], ob["C"]
th_iso, ph_iso = sample_isotropic(cb.n_total, np.random.default_rng(cb.seed + 7))
r_ana = xy_from_channels(*channels_transverse(th_iso, ph_iso, A_, B_, C_, 0.0))[2]
e = hugh["40x65nm"]

# (2) theta(r): analytic + empirical + default + best fit
cc, th = hugh_theta_of_r(ob["r_bal"])
axT.plot(r_formula(tt, A_, B_, C_), np.degrees(tt), "k:", lw=1.6,
         label="analytic Fourkas (no bg, no transverse)")
axT.fill_between(e["r"], e["lo"], e["hi"], color="#8c564b", alpha=0.18)
axT.plot(e["r"], e["theta"], "--", color="#8c564b", lw=2.2, label="empirical 40 nm")
axT.plot(ccd, thd, "#1f77b4", lw=1.4, ls="-.", label=f"default (RMSE {rmse_def:.2f}°)")
axT.plot(cc, th, "#d62728", lw=2.2, label=f"best fit (RMSE {best4[0]:.2f}°)")
axT.set_xlim(0, 1); axT.set_ylim(0, 92); axT.set_xlabel("r"); axT.set_ylabel(r"$\theta$ (deg)")
axT.grid(alpha=0.25); axT.legend(frameon=False, fontsize=8, loc="lower right")
axT.set_title(f"θ(r): bg={best4[1]}, σ_rod={best4[2]}, σ_ch={best4[3]}, amp={best4[4]}")

# (3) p(r): analytic + empirical + default + best fit
h_ana, _ = np.histogram(r_ana, bins=edges, density=True)
h_def, _ = np.histogram(odef["r"], bins=edges, density=True)
h_best, _ = np.histogram(ob["r"], bins=edges, density=True)
axP.plot(ctr, h_ana, "0.55", ls=":", lw=1.8, label="analytic (no bg, no transverse)")
axP.plot(e["r"], e["pr"], "--", color="#8c564b", lw=2.0, label="empirical 40 nm")
axP.plot(ctr, h_def, "#1f77b4", ls="-.", lw=1.4, label="default")
axP.plot(ctr, h_best, "#d62728", lw=2.2, label="best fit")
axP.set_xlim(0, 1); axP.set_ylim(bottom=0); axP.set_xlabel("r"); axP.set_ylabel("p(r)")
axP.grid(alpha=0.25); axP.legend(frameon=False, fontsize=8)
axP.set_title("p(r): best fit vs default vs empirical vs analytic")

fig.tight_layout(); plt.show()
""")

md(r"""## 10. Mechanism figure + best-fit vs set-values comparison

A pedagogical 4-panel figure (à la the report's "fig 9") showing *why*
per-recording background turns the sharp $r_{\mathrm{max}}$ pile-up into a smooth
bell: (a) the ideal anisotropy cloud coloured by θ band; (b) after one background
the cloud shrinks toward an off-centre point; (c) where each θ band lands in
$p(r)$; (d) many recordings, each shifted differently, sum to a smooth bell.

We render it (and the Section 7 comparison) twice: once with the **set values**
in `Config`, and once with the **best-fit values from the Section 9 sweep**, so
the two can be compared side by side.
""")

code(r"""def figure_mechanism(cfg, title=""):
    from matplotlib.colors import BoundaryNorm
    A, B, C = effective_ABC(cfg)
    r_max = C / (A + B)
    rng = np.random.default_rng(cfg.seed + 3)
    theta, phi = sample_isotropic(300_000, rng)
    theta_deg = np.degrees(theta)

    # ideal channels (rho=0), scaled so the theta=90 peak = avg_signal_ct
    C0, C45, C90, C135 = channels_transverse(theta, phi, A, B, C, 0.0)
    scale = cfg.avg_signal_ct / (0.5 * (A + B + C))
    C0 *= scale; C45 *= scale; C90 *= scale; C135 *= scale
    x_id, y_id, r_id = xy_from_channels(C0, C45, C90, C135)

    # one representative, sum-invariant, asymmetric background (in counts)
    b = cfg.avg_bg_ct * np.array([1.15, 1.0, 0.7, 0.85])
    s0 = b[0] + b[2]; dd = b[1] - b[3]; b[1] = 0.5*(s0 + dd); b[3] = 0.5*(s0 - dd)
    x_bg, y_bg, r_bg = xy_from_channels(C0 + b[0], C45 + b[1], C90 + b[2], C135 + b[3])
    sink = ((b[0]-b[2])/(b[0]+b[2]), (b[1]-b[3])/(b[1]+b[3]))

    band_edges = np.arange(0, 91, 10); n_band = len(band_edges) - 1
    cmap = plt.get_cmap("turbo", n_band); norm = BoundaryNorm(band_edges, cmap.N)
    band_colors = [cmap(i) for i in range(n_band)]
    band_id = np.clip(np.digitize(theta_deg, band_edges) - 1, 0, n_band - 1)
    sub = rng.choice(x_id.size, size=9000, replace=False)

    fig = plt.figure(figsize=(13.8, 9.0))
    gs = fig.add_gridspec(2, 2, wspace=0.24, hspace=0.42,
                          left=0.07, right=0.94, top=0.90, bottom=0.08)

    # (a) ideal cloud
    axa = fig.add_subplot(gs[0, 0])
    axa.scatter(x_id[sub], y_id[sub], c=theta_deg[sub], cmap=cmap, norm=norm,
                s=6, alpha=0.6, linewidths=0)
    axa.add_patch(plt.Circle((0, 0), r_max, fill=False, ls="--", color="k", lw=1.4))
    axa.scatter([0], [0], s=80, marker="+", color="k", zorder=6)
    axa.text(0, r_max + 0.03, r"$r_{\max}$", ha="center", fontsize=11)
    axa.set_aspect("equal"); axa.set_xlim(-1, 1); axa.set_ylim(-1, 1)
    axa.set_xlabel("x"); axa.set_ylabel("y")
    axa.set_title("(a) Ideal anisotropy (no bg), coloured by θ band")

    # (b) after one background
    axb = fig.add_subplot(gs[0, 1])
    sc = axb.scatter(x_bg[sub], y_bg[sub], c=theta_deg[sub], cmap=cmap, norm=norm,
                     s=6, alpha=0.6, linewidths=0)
    axb.add_patch(plt.Circle((0, 0), r_max, fill=False, ls="--", color="k", lw=1.4))
    axb.scatter([0], [0], s=80, marker="+", color="k", zorder=6)
    axb.scatter([sink[0]], [sink[1]], s=170, marker="*", color="w",
                edgecolor="k", lw=1.3, zorder=7)
    axb.annotate("dim rods (small θ)\ncollapse here", xy=sink,
                 xytext=(sink[0] - 0.10, sink[1] + 0.5), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    axb.set_aspect("equal"); axb.set_xlim(-1, 1); axb.set_ylim(-1, 1)
    axb.set_xlabel("x"); axb.set_ylabel("y")
    axb.set_title("(b) After one background: shrunk toward an off-centre point")
    txt = (f"background (counts), sum-invariant\n  b0={b[0]:.1f}   b45={b[1]:.1f}\n"
           f"  b90={b[2]:.1f}   b135={b[3]:.1f}\n"
           f"peak signal I_max = {cfg.avg_signal_ct:.0f}\n"
           f"  bg/peak ~ {100*cfg.avg_bg_ct/cfg.avg_signal_ct:.0f}%")
    axb.text(-0.96, -0.96, txt, ha="left", va="bottom", fontsize=8.2, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="0.6"))
    fig.colorbar(sc, ax=axb, boundaries=band_edges, ticks=band_edges,
                 fraction=0.046, pad=0.04).set_label(r"$\theta$ (deg)")

    # (c) which theta band lands where in p(r)
    axc = fig.add_subplot(gs[1, 0])
    r_bins = np.linspace(0, 1, 120); ctr = 0.5*(r_bins[:-1] + r_bins[1:])
    width = r_bins[1] - r_bins[0]; Np = theta.size
    ideal_bands, bg_bands = [], []
    for i in range(n_band):
        m = band_id == i
        hi, _ = np.histogram(r_id[m], bins=r_bins); hb, _ = np.histogram(r_bg[m], bins=r_bins)
        ideal_bands.append(hi / (Np*width)); bg_bands.append(hb / (Np*width))
    si = 1.0 / max(np.sum(ideal_bands, 0).max(), 1e-12)
    sb = 1.0 / max(np.sum(bg_bands, 0).max(), 1e-12)
    axc.stackplot(ctr, *[bb*si for bb in ideal_bands], colors=band_colors, edgecolor="none")
    axc.stackplot(ctr, *[-bb*sb for bb in bg_bands], colors=band_colors, edgecolor="none")
    axc.axhline(0, color="k", lw=0.9); axc.axvline(r_max, ls="--", color="k", lw=1.0)
    axc.text(0.985, 0.86, "ideal p(r)", transform=axc.transAxes, ha="right", fontsize=11)
    axc.text(0.985, 0.14, "after one bg", transform=axc.transAxes, ha="right", fontsize=11)
    axc.set_xlim(0, 1); axc.set_ylim(-1.15, 1.15)
    axc.set_yticks([-1, -0.5, 0, 0.5, 1]); axc.set_yticklabels(["1", "0.5", "0", "0.5", "1"])
    axc.set_xlabel("r"); axc.set_ylabel("relative density (per θ band)")
    axc.set_title("(c) Which θ band lands where in p(r)")

    # (d) many recordings -> smooth bell
    axd = fig.add_subplot(gs[1, 1])
    bgs = two_layer_bg(cfg, np.random.default_rng(cfg.seed + 22))
    palette = plt.get_cmap("tab20"); pooled = []
    n_rec = min(14, cfg.n_groups)
    for i in range(n_rec):
        _, _, rr = xy_from_channels(C0 + bgs[i, 0], C45 + bgs[i, 1],
                                    C90 + bgs[i, 2], C135 + bgs[i, 3])
        pooled.append(rr)
        h, _ = np.histogram(rr, bins=r_bins, density=True)
        axd.plot(ctr, h, lw=1.0, color=palette(i % 20), alpha=0.55)
    hp, _ = np.histogram(np.concatenate(pooled), bins=r_bins, density=True)
    axd.plot(ctr, hp, lw=3.2, color="k", label="pooled (sum) = smooth bell")
    axd.axvline(r_max, ls="--", color="grey", lw=1.0)
    axd.set_xlim(0, 1); axd.set_xlabel("r"); axd.set_ylabel("density")
    axd.set_title("(d) Different recordings shift their bump; sum is a smooth bell")
    axd.legend(frameon=False, loc="upper left", fontsize=10)

    fig.suptitle(title or "Why per-recording background makes a bell", y=0.985, fontsize=12.5)
    plt.show()
""")

code(r"""# best-fit Config from the Section 9 sweep
cfg_best = replace(cfg, avg_bg_ct=best4[1], sigma_bg_rod=best4[2],
                   sigma_bg_ch=best4[3], amp_ratio_T_over_L=best4[4])

print(f"SET values:      bg={cfg.avg_bg_ct:.0f}ct  sig_rod={cfg.sigma_bg_rod}  "
      f"sig_ch={cfg.sigma_bg_ch}  amp={cfg.amp_ratio_T_over_L}")
show(cfg, simulate(cfg))
figure_mechanism(cfg, f"SET values  (bg={cfg.avg_bg_ct:.0f}ct, "
                      f"σ_rod={cfg.sigma_bg_rod}, σ_ch={cfg.sigma_bg_ch}, amp={cfg.amp_ratio_T_over_L})")

print(f"BEST FIT (sec 9): bg={cfg_best.avg_bg_ct:.0f}ct  sig_rod={cfg_best.sigma_bg_rod}  "
      f"sig_ch={cfg_best.sigma_bg_ch}  amp={cfg_best.amp_ratio_T_over_L}")
show(cfg_best, simulate(cfg_best))
figure_mechanism(cfg_best, f"BEST FIT sec 9  (bg={cfg_best.avg_bg_ct:.0f}ct, "
                           f"σ_rod={cfg_best.sigma_bg_rod}, σ_ch={cfg_best.sigma_bg_ch}, amp={cfg_best.amp_ratio_T_over_L})")
""")

md(r"""## 11. Interference sweep (Richard's model) — vary interference, fix bg & amp

Test the **interference** picture: per channel
$I_j = I_{\mathrm{rod},j} + b_j + 2\eta\sqrt{I_{\mathrm{rod},j}\,b_j}\cos\delta$ with a
shared per-rod phase $\delta$ (`phase_center_deg` at the chosen focus,
`phase_spread_deg` across rods/FOV). Here the **background and transverse amplitude
are held at the set `Config` values**; we sweep only the interference parameters
$\eta$ and `phase_spread_deg` and score the Hugh $\theta(r)$ RMSE against the
empirical 40 nm curve. (`phase_spread_deg=360` = fully random phase, `=0` = every
rod in phase; `phase_center_deg` is held at the set value.)
""")

code(r"""# Interference-parameter sweep at the SET bg & amp values.
ETA_GRID    = grid_with([0.0, 0.25, 0.5, 0.75, 1.0], cfg.eta)
SPREAD_GRID = sorted(set([0, 60, 120, 180, 240, 360] + [int(round(cfg.phase_spread_deg))]))
NFAST_I     = 300_000

resI = []
for et in ETA_GRID:
    for sp in SPREAD_GRID:
        c = replace(cfg, eta=et, phase_spread_deg=sp, n_total=NFAST_I)
        resI.append((theta_rmse(c), et, sp))
resI.sort()
bestI = resI[0]
print("best 5  (RMSE deg | eta | phase_spread_deg)   [bg & amp fixed at set values]:")
for rmse, et, sp in resI[:5]:
    print(f"  {rmse:5.2f}  |  {et:.2f}  |  {sp:3d}")
odef, ccd, thd, rmse_def = default_reference(NFAST_I)
print(f"default config RMSE = {rmse_def:.2f} deg  (eta={cfg.eta}, i.e. interference off)")

edges = np.linspace(0, 1, 121); ctr = 0.5*(edges[:-1] + edges[1:])
tt = np.linspace(0, np.pi/2, 400)
fig, (axH, axT, axP) = plt.subplots(1, 3, figsize=(16.5, 4.9))

# (1) RMSE heatmap over (eta, phase_spread)
Z = np.full((len(ETA_GRID), len(SPREAD_GRID)), np.nan)
for rmse, et, sp in resI:
    Z[ETA_GRID.index(et), SPREAD_GRID.index(sp)] = rmse
im = axH.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r")
axH.set_xticks(range(len(SPREAD_GRID))); axH.set_xticklabels(SPREAD_GRID)
axH.set_yticks(range(len(ETA_GRID))); axH.set_yticklabels(ETA_GRID)
axH.set_xlabel("phase_spread_deg"); axH.set_ylabel("eta")
axH.set_title("θ(r) RMSE (deg)  (bg & amp = set values)")
fig.colorbar(im, ax=axH, shrink=0.85)

# best-fit run + references
cb = replace(cfg, eta=bestI[1], phase_spread_deg=bestI[2])
ob = simulate(cb)
A_, B_, C_ = ob["A"], ob["B"], ob["C"]
th_iso, ph_iso = sample_isotropic(cb.n_total, np.random.default_rng(cb.seed + 7))
r_ana = xy_from_channels(*channels_transverse(th_iso, ph_iso, A_, B_, C_, 0.0))[2]
e = hugh["40x65nm"]

# (2) theta(r): analytic + empirical + default + best fit
cc, th = hugh_theta_of_r(ob["r_bal"])
axT.plot(r_formula(tt, A_, B_, C_), np.degrees(tt), "k:", lw=1.6,
         label="analytic Fourkas (no bg, no transverse)")
axT.fill_between(e["r"], e["lo"], e["hi"], color="#8c564b", alpha=0.18)
axT.plot(e["r"], e["theta"], "--", color="#8c564b", lw=2.2, label="empirical 40 nm")
axT.plot(ccd, thd, "#1f77b4", lw=1.4, ls="-.", label=f"default / η=0 (RMSE {rmse_def:.2f}°)")
axT.plot(cc, th, "#d62728", lw=2.2, label=f"best fit (RMSE {bestI[0]:.2f}°)")
axT.set_xlim(0, 1); axT.set_ylim(0, 92); axT.set_xlabel("r"); axT.set_ylabel(r"$\theta$ (deg)")
axT.grid(alpha=0.25); axT.legend(frameon=False, fontsize=8, loc="lower right")
axT.set_title(f"θ(r): eta={bestI[1]}, spread={bestI[2]}° (bg,amp set)")

# (3) p(r): analytic + empirical + default + best fit
h_ana, _ = np.histogram(r_ana, bins=edges, density=True)
h_def, _ = np.histogram(odef["r"], bins=edges, density=True)
h_best, _ = np.histogram(ob["r"], bins=edges, density=True)
axP.plot(ctr, h_ana, "0.55", ls=":", lw=1.8, label="analytic (no bg, no transverse)")
axP.plot(e["r"], e["pr"], "--", color="#8c564b", lw=2.0, label="empirical 40 nm")
axP.plot(ctr, h_def, "#1f77b4", ls="-.", lw=1.4, label="default / η=0")
axP.plot(ctr, h_best, "#d62728", lw=2.2, label="best fit")
axP.set_xlim(0, 1); axP.set_ylim(bottom=0); axP.set_xlabel("r"); axP.set_ylabel("p(r)")
axP.grid(alpha=0.25); axP.legend(frameon=False, fontsize=8)
axP.set_title("p(r): best fit vs default vs empirical vs analytic")
fig.tight_layout(); plt.show()
""")

md(r"""## 12. Ablation — which ingredient matters most? (turn each off from default)

Starting from the **default** `Config`, we set each parameter to its "off" value
one at a time (everything else unchanged) and re-measure the Hugh $\theta(r)$ RMSE
vs empirical. This isolates each ingredient's *marginal* contribution to the
current default model. Parameters already at their off-value (e.g. `eta`) are
flagged `(=default)` and produce no change. The bar chart ranks the RMSE; the
$\theta(r)$/$p(r)$ panels overlay every ablation against analytic and empirical.
""")

code(r"""ABLATIONS = [
    ("default",              {}),
    ("amp = 0",              dict(amp_ratio_T_over_L=0.0)),
    ("avg_bg_ct = 0",        dict(avg_bg_ct=0.0)),
    ("sigma_bg_rod = 0",     dict(sigma_bg_rod=0.0)),
    ("sigma_bg_ch = 0",      dict(sigma_bg_ch=0.0)),
    ("sigma_signal_rod = 0", dict(sigma_signal_rod=0.0)),
    ("sum_invariance off",   dict(enforce_sum_invariance=False)),
    ("eta = 0",              dict(eta=0.0)),
]
NFAST_AB = 400_000
edges = np.linspace(0, 1, 121); ctr = 0.5*(edges[:-1] + edges[1:])
tt = np.linspace(0, np.pi/2, 400)
e = hugh["40x65nm"]
Aa, Ba, Ca = effective_ABC(cfg)
th0, ph0 = sample_isotropic(NFAST_AB, np.random.default_rng(cfg.seed + 7))
r_ana = xy_from_channels(*channels_transverse(th0, ph0, Aa, Ba, Ca, 0.0))[2]

fig, (axR, axT, axP) = plt.subplots(1, 3, figsize=(17, 5.2))
axT.plot(r_formula(tt, Aa, Ba, Ca), np.degrees(tt), "k:", lw=1.6,
         label="analytic Fourkas (no bg, no transverse)")
axT.fill_between(e["r"], e["lo"], e["hi"], color="#8c564b", alpha=0.15)
axT.plot(e["r"], e["theta"], "--", color="#8c564b", lw=2.0, label="empirical 40 nm")
h_ana, _ = np.histogram(r_ana, bins=edges, density=True)
axP.plot(ctr, h_ana, "k:", lw=1.6, label="analytic (no bg, no transverse)")
axP.plot(e["r"], e["pr"], "--", color="#8c564b", lw=2.0, label="empirical 40 nm")

labels, rmses = [], []
palette = plt.get_cmap("tab10")
for i, (lab, kw) in enumerate(ABLATIONS):
    is_default_val = bool(kw) and all(getattr(cfg, k) == v for k, v in kw.items())
    c = replace(cfg, n_total=NFAST_AB, **kw)
    o = simulate(c)
    cc, th = hugh_theta_of_r(o["r_bal"])
    m = (cc >= e["r"].min()) & (cc <= e["r"].max())
    rm = float(np.sqrt(np.mean((th[m] - np.interp(cc[m], e["r"], e["theta"])) ** 2)))
    tag = lab + (" (=default)" if is_default_val else "")
    labels.append(tag); rmses.append(rm)
    col = "k" if lab == "default" else palette(i % 10)
    lw = 2.6 if lab == "default" else 1.5
    axT.plot(cc, th, color=col, lw=lw, label=f"{tag}  {rm:.2f}°")
    hh, _ = np.histogram(o["r"], bins=edges, density=True)
    axP.plot(ctr, hh, color=col, lw=lw)

yy = list(range(len(labels)))
axR.barh(yy, rmses, color=["k"] + [palette(i % 10) for i in range(1, len(labels))])
axR.axvline(rmses[0], color="0.4", ls="--", lw=1.2, label=f"default {rmses[0]:.2f}°")
axR.set_yticks(yy); axR.set_yticklabels(labels, fontsize=8)
axR.invert_yaxis(); axR.set_xlabel("θ(r) RMSE (deg)")
axR.grid(alpha=0.25, axis="x"); axR.legend(frameon=False, fontsize=8)
axR.set_title("Ablation: RMSE with each ingredient OFF")

axT.set_xlim(0, 1); axT.set_ylim(0, 92); axT.set_xlabel("r"); axT.set_ylabel(r"$\theta$ (deg)")
axT.grid(alpha=0.25); axT.legend(frameon=False, fontsize=7, loc="lower right")
axT.set_title("θ(r) for each ablation")
axP.set_xlim(0, 1); axP.set_ylim(bottom=0); axP.set_xlabel("r"); axP.set_ylabel("p(r)")
axP.grid(alpha=0.25); axP.legend(frameon=False, fontsize=7)
axP.set_title("p(r) for each ablation")
fig.tight_layout(); plt.show()

d0 = rmses[0]
print("effect of turning each ingredient OFF (ΔRMSE vs default; + = worse fit):")
for lab, rm in sorted(zip(labels[1:], rmses[1:]), key=lambda t: -abs(t[1] - d0)):
    print(f"  {lab:24s}  RMSE {rm:6.2f}   Δ {rm - d0:+6.2f}")
""")

md(r"""## 13. Summary for discussion

- **Background *level* is the dominant ingredient.** The ablation (Section 12) is
  unambiguous: removing the background (`avg_bg_ct=0`) sends the $\theta(r)$ RMSE
  from ~6° to ~21°, while turning off any *other* single ingredient changes it by
  $<0.5°$. Background is what turns the analytic pile-up spike into the observed
  bell.
- **The background *spreads* matter, and the point-measured values are too small.**
  At the default (measured) settings — 15% bg, $\sigma_{\mathrm{bg,rod}}\approx0.46$,
  $\sigma_{\mathrm{bg,ch}}\approx0.06$ — the fit is only RMSE $\approx5.9°$. Letting the
  sweep choose gives RMSE $\approx2.0°$ at the *same* 15% level but with larger
  spreads ($\sigma_{\mathrm{bg,rod}}\approx0.7$, $\sigma_{\mathrm{bg,ch}}\approx0.2$–$0.3$).
  The cross-channel term is the biggest gap: the time-averaged FOV profile gives
  $\sigma_{\mathrm{ch}}\approx0.06$, but matching $\theta(r)$ wants ~0.2–0.3 (likely
  per-frame noise the average hides).
- **Transverse modes** ($\rho$): adding amp barely helps (best $\approx1.84°$ at
  amp$=0.2$ vs $\approx2.0°$ at amp$=0$ — within noise). Not needed to explain the bell.
- **Interference** (Richard's model): with bg & amp held at the measured values,
  sweeping $\eta$ and phase spread improves the fit ($5.99°\to5.62°$) but cannot
  reach the ~2° that more background spread achieves. It helps, but at the measured
  (narrow) background it is not sufficient alone — and it remains the only mechanism
  that can push points past the optical ceiling.
- **Optics** set $A,B,C$ and the hard ceiling $r_{\mathrm{max}}\approx0.82$. The data
  reach $r\approx0.87$, so the high-$r$ tail is the one feature no additive term
  reproduces — revisit the NA / hole / index constants.

**Headline:** background — its *level* above all, then its rod-to-rod and
cross-channel *spread* — reproduces the 40 nm $\theta(r)$. Transverse and
interference are second-order at the measured background.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out_path = Path(__file__).resolve().parent.parent / "fourkas_theta_r_model.ipynb"
nbf.write(nb, str(out_path))
print("Wrote", out_path)
