"""
Transverse-mode model for the Fourkas r-theta analysis.

Motivation
----------
The rod is a spherocylinder with uniaxial symmetry, so its plasmon
polarisability is
        alpha = alpha_L * (u x u) + alpha_T * (I - u x u)
i.e. one LONGITUDINAL mode (dipole along the long axis u) and TWO degenerate
TRANSVERSE modes (dipoles in the plane perpendicular to u). The earlier model
kept only the longitudinal mode (I_tot ~ sin^2 theta), which forces the signal
to zero at theta = 0. Including the transverse response changes that.

Model (incoherent 3-dipole picture under circular excitation)
-------------------------------------------------------------
Circular excitation in the x-y plane drives each mode by the time-averaged
projection of E:
    longitudinal drive  = <|E.u|^2>       = sin^2(theta)/2
    transverse  drive   = <|E_perp|^2>    = 1 - sin^2(theta)/2
A unit dipole along u is detected in Fourkas channel j (0,45,90,135) as
    F_dip(u; j) = A + B s^2 + C s^2 * m_j ,   m_j in {cos2phi, sin2phi, -cos2phi, -sin2phi}
Using the identity  (longitudinal dipole) + (2 transverse dipoles) = isotropic,
the combined 2-transverse response is
    F_2T(u; j) = (3A + 2B) - F_dip(u; j)          [isotropic sum minus longitudinal]

With g_L = |alpha_L|^2, g_T = |alpha_T|^2 and rho = g_T/g_L, the detected channel
intensity is
    C_j = (s^2/2) * F_dip   +   rho * (1 - s^2/2) * F_2T
(overall scale g_L dropped; ratios x,y are scale-free).

Consequences (derived, then simulated)
--------------------------------------
* Signal is NO LONGER zero at theta = 0:  I_tot(0) ~ rho * (A+B) > 0 (bright centre).
* But r STILL -> 0 at theta = 0 (no azimuth when the rod points up).
* The anisotropy modulation is  M(theta) ~ C s^2 [ s^2/2 - rho(1 - s^2/2) ], which
  CHANGES SIGN at  s^2* = 2 rho/(1+rho).  So r(theta) is NON-MONOTONIC: a given r
  maps to TWO theta values (a transverse-dominated, phase-flipped branch at small
  theta and a longitudinal branch at large theta). Hugh's monotonic CDF inversion
  is therefore not strictly valid once rho > 0.
* r_max (at theta=90) shrinks:  r_max = C(1-rho) / [ (A+B) + rho(2A+B) ].

Outputs go to ../transverse_mode_output/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

import make_report_figures as M

OUTDIR = Path(__file__).resolve().parent.parent / "transverse_mode_output"
OUTDIR.mkdir(exist_ok=True)

# ── optical / measurement configuration (VERIFY these, then re-run) ────────────
LASER_NM    = 633.0       # excitation wavelength (nm)
BACKSCATTER = True        # elastic backscatter: excitation and detection share λ
NA_COLLECT  = 1.3         # objective NA
N_MEDIUM    = 1.33        # medium / immersion refractive index (water)

# Hole-mirror (annular back-focal-plane) collection — see Section 10 of
# Fourkas_processing_v4.ipynb. A central hole (NA < NA_IN) is blocked and only the
# annulus NA_IN..NA_OUT is collected, which changes the effective Fourkas A, B, C.
USE_HOLE_CORRECTION = True
NA_IN  = 0.38             # inner NA (hole edge)    [v4 Section 10 default]
NA_OUT = 1.3              # outer NA (annulus rim)  [v4 Section 10 default]

# Transverse/longitudinal strength ratio  rho(λ) = |alpha_T(λ)|^2 / |alpha_L(λ)|^2.
# For elastic backscatter, excitation and emission share the same λ, so this ratio
# is exactly the intensity weight of the transverse vs longitudinal channels.
# DEFAULTS are placeholders from the C12-40-600 spec sheet — REPLACE with the real
# single-particle spectrum (LSPR/TSPR peaks + linewidths) once confirmed.
LAMBDA_L_NM = 600.0       # longitudinal LSPR peak (nm)
LAMBDA_T_NM = 520.0       # transverse SPR peak (nm)
FWHM_L_NM   = 60.0        # longitudinal linewidth (nm, FWHM)
FWHM_T_NM   = 60.0        # transverse linewidth (nm, FWHM)   [spec "SPR linewidth 80% = 60"]
AMP_RATIO_T_OVER_L = 1.0  # peak-amplitude ratio |alpha_T,peak| / |alpha_L,peak|
                          # (1.0 = equal peak strength; set <1 if longitudinal dominates)

N_TOTAL = 1_500_000
RHO_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]
RHO_COLORS = {0.0: "#888888", 0.1: "#1f77b4", 0.25: "#2ca02c",
              0.5: "#ff7f0e", 1.0: "#d62728"}
SEED = 20260718

# Background measured from Hugh's UNCORRECTED background CSVs:
#   frac_med = bg / peak-signal (= bg% at the brightest orientation, the min bg%)
#   log_sigma from the rod-to-rod coefficient of variation (CV ~ 0.31 both sizes)
CSV_MEAS = {
    "25x65nm": dict(frac_med=0.21, log_sigma=0.31),
    "40x65nm": dict(frac_med=0.10, log_sigma=0.31),
}


def fourkas_ABC_annular(na_in, na_out, n):
    """Effective Fourkas A, B, C for annular (hole-mirror) collection NA_in..NA_out.
    Mirrors Section 10 of Fourkas_processing_v4.ipynb (J1/J2/J3 aperture integrals)."""
    a_in = np.arcsin(min(na_in / n, 1.0))
    a_out = np.arcsin(min(na_out / n, 1.0))
    J1 = quad(lambda b: (2 * np.cos(b) ** 2 + 0.75 * np.sin(b) ** 4) * np.sin(b), a_in, a_out)[0]
    J2 = quad(lambda b: 0.25 * np.sin(b) ** 4 * np.sin(b), a_in, a_out)[0]
    J3 = quad(lambda b: np.sin(b) ** 2 * np.cos(b) ** 2 * np.sin(b), a_in, a_out)[0]
    C = J1 - J2
    A = 2.0 * J3
    B = J1 + J2 - 2.0 * J3
    return A, B, C


def _lorentz_sq(lam, lam0, fwhm):
    """|Lorentzian|^2 line-shape in wavelength: peak = 1 at lam0, FWHM = fwhm."""
    hw = 0.5 * fwhm
    return hw * hw / ((lam - lam0) ** 2 + hw * hw)


def rho_backscatter(laser_nm=None):
    """Estimate rho = |alpha_T|^2/|alpha_L|^2 at the laser wavelength from the two
    plasmon resonances (each modelled as a Lorentzian). Backscatter => single λ."""
    lam = LASER_NM if laser_nm is None else laser_nm
    gL = _lorentz_sq(lam, LAMBDA_L_NM, FWHM_L_NM)
    gT = _lorentz_sq(lam, LAMBDA_T_NM, FWHM_T_NM)
    return float((AMP_RATIO_T_OVER_L ** 2) * gT / gL)


def effective_ABC():
    """A, B, C to use downstream, honouring the hole-mirror toggle."""
    if USE_HOLE_CORRECTION:
        return fourkas_ABC_annular(NA_IN, NA_OUT, N_MEDIUM)
    _, A, B, C = M.fourkas_ABC(NA_COLLECT, N_MEDIUM)
    return A, B, C


def channels_transverse(theta, phi, A, B, C, rho):
    """Detected 4-channel intensities including transverse modes.

    `rho` may be a scalar or an array broadcastable to theta (per-point ratio).
    """
    s2 = np.sin(theta) ** 2
    long_drive = s2 / 2.0
    trans_drive = 1.0 - s2 / 2.0
    iso = 3.0 * A + 2.0 * B

    def chan(m):
        Fdip = A + B * s2 + C * s2 * m
        F2T = iso - Fdip
        return long_drive * Fdip + rho * trans_drive * F2T

    c2, s2p = np.cos(2 * phi), np.sin(2 * phi)
    return chan(c2), chan(s2p), chan(-c2), chan(-s2p)


def r_theta_signed(theta, A, B, C, rho):
    """Signed analytic r(theta) at the modulation maximum (cos2phi=1)."""
    s2 = np.sin(theta) ** 2
    num = C * s2 * (s2 / 2.0 - rho * (1.0 - s2 / 2.0))
    den = (s2 / 2.0) * (A + B * s2) + rho * (1.0 - s2 / 2.0) * (2 * A + 2 * B - B * s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def itot_theta(theta, A, B, C, rho):
    s2 = np.sin(theta) ** 2
    return 4.0 * ((s2 / 2.0) * (A + B * s2)
                  + rho * (1.0 - s2 / 2.0) * (2 * A + B * (2.0 - s2)))


def r_max_of_rho(A, B, C, rho):
    return C * (1.0 - rho) / ((A + B) + rho * (2 * A + B))


def xy_from_channels(C0, C45, C90, C135):
    x = (C0 - C90) / (C0 + C90)
    y = (C45 - C135) / (C45 + C135)
    return x, y, np.sqrt(x * x + y * y)


def main():
    A, B, C = effective_ABC()
    rho_est = rho_backscatter()
    curves = M.load_hugh_curves()

    print("── configuration ─────────────────────────────────────────────")
    print(f"laser = {LASER_NM:.0f} nm   backscatter = {BACKSCATTER}   "
          f"NA = {NA_COLLECT}   n = {N_MEDIUM}")
    if USE_HOLE_CORRECTION:
        print(f"hole-mirror annulus: NA_in = {NA_IN}, NA_out = {NA_OUT}  (v4 Section 10)")
    else:
        print("hole correction OFF (full-cone Fourkas)")
    print(f"effective Fourkas:  A={A:.5g}  B={B:.5g}  C={C:.5g}   "
          f"r_max(ρ=0) = {r_max_of_rho(A, B, C, 0.0):.4f}")
    print(f"resonances: LSPR {LAMBDA_L_NM:.0f} nm (FWHM {FWHM_L_NM:.0f}), "
          f"TSPR {LAMBDA_T_NM:.0f} nm (FWHM {FWHM_T_NM:.0f}), amp ratio {AMP_RATIO_T_OVER_L}")
    print(f"=> estimated ρ @ {LASER_NM:.0f} nm = {rho_est:.3f}   "
          f"(r_max = {r_max_of_rho(A, B, C, rho_est):+.4f})")
    print("───────────────────────────────────────────────────────────────")

    print("rho     r_max      theta*(zero-crossing, deg)")
    for rho in RHO_LEVELS + [rho_est]:
        s2star = 2 * rho / (1 + rho) if rho > 0 else np.nan
        thstar = np.degrees(np.arcsin(np.sqrt(s2star))) if 0 <= s2star <= 1 else np.nan
        tag = "  <- estimate" if rho == rho_est else ""
        print(f"{rho:6.3f}  {r_max_of_rho(A, B, C, rho):+.4f}    {thstar:6.2f}{tag}")

    # (rho, colour, linewidth, label, is_estimate) for every curve drawn
    plot_specs = [(rho, RHO_COLORS[rho], 2.0, f"ρ={rho:.2f}", False) for rho in RHO_LEVELS]
    plot_specs.append((rho_est, "#000000", 3.2,
                       f"ρ≈{rho_est:.2f}  (est. @ {LASER_NM:.0f} nm)", True))

    # simulate pooled p(r) and Hugh theta(r) for each rho (isotropic, NO background)
    rng = np.random.default_rng(SEED)
    theta, phi = M.sample_isotropic(N_TOTAL, rng)
    sims = {}
    for rho, *_rest in plot_specs:
        C0, C45, C90, C135 = channels_transverse(theta, phi, A, B, C, rho)
        x, y, r = xy_from_channels(C0, C45, C90, C135)
        r_bal = M.balanced_r_sample(x, y, np.random.default_rng(SEED + int(rho * 1000) + 1))
        ctr, th_deg, _, _ = M.hugh_theta_of_r(r_bal)
        sims[rho] = {"r": r, "ctr": ctr, "theta": th_deg}

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13.8, 9.0))
    gs = fig.add_gridspec(2, 2, wspace=0.24, hspace=0.42,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    th_grid = np.linspace(0, np.pi / 2, 400)
    th_deg_grid = np.degrees(th_grid)

    # (a) signed analytic r(theta)
    ax = fig.add_subplot(gs[0, 0])
    for rho, col, lw, lab, _est in plot_specs:
        ax.plot(th_deg_grid, r_theta_signed(th_grid, A, B, C, rho),
                lw=lw, color=col, label=lab)
    ax.axhline(0, color="grey", lw=1.0, ls=":")
    ax.set_xlabel(r"$\theta$ (deg)"); ax.set_ylabel("signed $r(\\theta)$")
    ax.set_xlim(0, 90)
    ax.set_title("(a)  $r(\\theta)$ vs transverse ratio ρ = $|\\alpha_T|^2/|\\alpha_L|^2$\n"
                 "sign flip → non-monotonic; $r_{\\max}$ shrinks with ρ")
    ax.legend(frameon=False, fontsize=8.5, title="transverse ratio")

    # (b) I_tot(theta)
    ax = fig.add_subplot(gs[0, 1])
    for rho, col, lw, lab, _est in plot_specs:
        it = itot_theta(th_grid, A, B, C, rho)
        ax.plot(th_deg_grid, it / it.max(), lw=lw, color=col, label=lab)
    ax.set_xlabel(r"$\theta$ (deg)"); ax.set_ylabel(r"$I_{\rm tot}(\theta)$ (norm.)")
    ax.set_xlim(0, 90); ax.set_ylim(0, 1.05)
    ax.set_title("(b)  Total emitted signal vs θ\n"
                 "transverse modes give a bright, non-zero centre ($\\theta=0$)")
    ax.legend(frameon=False, fontsize=8.5)

    # (c) pooled p(r), isotropic, NO background
    ax = fig.add_subplot(gs[1, 0])
    r_bins = np.linspace(0, 1, 140)
    ctr_bins = 0.5 * (r_bins[:-1] + r_bins[1:])
    for rho, col, lw, lab, _est in plot_specs:
        h, _ = np.histogram(sims[rho]["r"], bins=r_bins, density=True)
        ax.plot(ctr_bins, h, lw=lw, color=col, label=lab)
    ax.set_xlabel("r"); ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.set_title("(c)  Pooled $p(r)$, isotropic tumbling, NO background\n"
                 "transverse modes alone move the pile-up inward & broaden it")
    ax.legend(frameon=False, fontsize=8.5)

    # (d) theta(r) via Hugh's method
    ax = fig.add_subplot(gs[1, 1])
    for rho, col, lw, lab, _est in plot_specs:
        ax.plot(sims[rho]["ctr"], sims[rho]["theta"], lw=lw, color=col, label=lab)
    if "25x65nm" in curves:
        c = curves["25x65nm"]
        ax.fill_between(c["r"], c["lo"], c["hi"], color="#8c564b", alpha=0.18)
        ax.plot(c["r"], c["theta"], lw=2.6, color="#8c564b", ls="--",
                label="Hugh empirical 25×65 nm")
    ax.set_xlabel("r"); ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 92)
    ax.set_title("(d)  Hugh CDF-inversion $\\theta(r)$ vs empirical\n"
                 "(no background — transverse modes only)")
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")

    _tag = "hole-corrected" if USE_HOLE_CORRECTION else "full-cone"
    fig.suptitle(f"Transverse plasmon modes @ {LASER_NM:.0f} nm backscatter "
                 f"({_tag} A,B,C, n={N_MEDIUM}); estimated ρ≈{rho_est:.2f} highlighted (black)",
                 y=0.985, fontsize=12.0)
    fig.savefig(OUTDIR / "transverse_mode_effect.png", dpi=200)
    plt.close(fig)
    print(f"\nWritten {OUTDIR / 'transverse_mode_effect.png'}")

    figure_polydispersity(A, B, C, theta, phi, curves, rho_med=rho_est)
    figure_fixed_rho_background(A, B, C, theta, phi, curves, rho_fixed=rho_est)
    figure_csv_measured(A, B, C, theta, phi, curves, rho_est)


def figure_polydispersity(A, B, C, theta, phi, curves, rho_med=0.20):
    """Does rod-to-rod ρ variation (aspect-ratio polydispersity) pool into a bell,
    with only a SMALL background, and match Hugh's empirical curve?"""
    print("\nSecond figure: per-recording ρ spread (polydispersity)")
    N = theta.size
    N_GROUPS = 115
    rng = np.random.default_rng(SEED + 999)
    gid = rng.integers(0, N_GROUPS, size=N)

    # per-recording rho: lognormal around a median (aspect-ratio / detuning spread).
    # Median defaults to the estimated backscatter rho at the laser wavelength.
    RHO_MED, RHO_SIG = float(rho_med), 0.55
    rho_g = RHO_MED * rng.lognormal(0.0, RHO_SIG, size=N_GROUPS)
    rho_pt = rho_g[gid]

    scenarios = {}

    # (A) rho spread only, no background
    C0, C45, C90, C135 = channels_transverse(theta, phi, A, B, C, rho_pt)
    base_scale = float(np.max(C0))
    xA, yA, rA = xy_from_channels(C0, C45, C90, C135)
    scenarios["ρ-spread only (no bg)"] = ("#1f77b4", rA, xA, yA)

    # (B) rho spread + small 5% per-recording background (sum-invariance enforced)
    bg = M.make_group_backgrounds(N_GROUPS, base_scale, 0.05, 0.6,
                                   np.random.default_rng(SEED + 111))
    C0b = C0 + bg[gid, 0]; C45b = C45 + bg[gid, 1]
    C90b = C90 + bg[gid, 2]; C135b = C135 + bg[gid, 3]
    xB, yB, rB = xy_from_channels(C0b, C45b, C90b, C135b)
    scenarios["ρ-spread + 5% bg"] = ("#d62728", rB, xB, yB)

    # (C) fixed single rho (median) + no bg, for contrast (stays spiky)
    C0c, C45c, C90c, C135c = channels_transverse(theta, phi, A, B, C, RHO_MED)
    xC, yC, rC = xy_from_channels(C0c, C45c, C90c, C135c)
    scenarios[f"fixed ρ={RHO_MED:.2f} (no bg)"] = ("#7f7f7f", rC, xC, yC)

    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(1, 2, wspace=0.24, left=0.07, right=0.985,
                          top=0.86, bottom=0.13)

    r_bins = np.linspace(0, 1, 140)
    ctr_bins = 0.5 * (r_bins[:-1] + r_bins[1:])

    ax = fig.add_subplot(gs[0])
    for label, (col, r, _, _) in scenarios.items():
        h, _ = np.histogram(r, bins=r_bins, density=True)
        ax.plot(ctr_bins, h, lw=2.2, color=col, label=label)
    ax.set_xlabel("r"); ax.set_ylabel("density"); ax.set_xlim(0, 1)
    ax.set_title("Pooled $p(r)$: rod-to-rod ρ spread washes the spike into a bell\n"
                 f"(ρ ~ lognormal median {RHO_MED:.2f}, σ={RHO_SIG})")
    ax.legend(frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[1])
    for label, (col, r, x, y) in scenarios.items():
        r_bal = M.balanced_r_sample(x, y, np.random.default_rng(hash(label) % 2**31))
        ctr, th_deg, _, _ = M.hugh_theta_of_r(r_bal)
        ax.plot(ctr, th_deg, lw=2.2, color=col, label=label)
    tt = np.linspace(0, np.pi / 2, 400)
    ax.plot(M.r_formula(tt, A, B, C), np.degrees(tt), lw=1.8, color="k", ls=":",
            label="analytic Fourkas (ρ=0)")
    if "25x65nm" in curves:
        c = curves["25x65nm"]
        ax.fill_between(c["r"], c["lo"], c["hi"], color="#8c564b", alpha=0.18)
        ax.plot(c["r"], c["theta"], lw=2.6, color="#8c564b", ls="--",
                label="Hugh empirical 25×65 nm")
    ax.set_xlabel("r"); ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 92)
    ax.set_title(r"Hugh $\theta(r)$: transverse polydispersity + small bg vs empirical")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle("Alternative to large background: rod-to-rod transverse-mode variation (aspect-ratio polydispersity) "
                 "+ a small background", y=0.98, fontsize=12.5)
    fig.savefig(OUTDIR / "transverse_polydispersity.png", dpi=200)
    plt.close(fig)
    print(f"Written {OUTDIR / 'transverse_polydispersity.png'}")


def figure_fixed_rho_background(A, B, C, theta, phi, curves, rho_fixed=0.146,
                                bg_levels=(0.05, 0.15, 0.30), log_sigma=0.6):
    """FIXED transverse ratio rho (same for every rod, set by laser/resonances) with
    per-chunk VARYING background, pooled over many chunks. Answers: with the physical
    rho baked in, does the usual per-recording background heterogeneity still produce
    the bell + bend theta(r) onto Hugh's empirical curve?"""
    print(f"\nThird figure: fixed ρ={rho_fixed:.3f} + per-chunk varying background")
    N = theta.size
    N_GROUPS = 115
    rng = np.random.default_rng(SEED + 4242)
    gid = rng.integers(0, N_GROUPS, size=N)

    # channels at the FIXED physical rho (no background yet)
    C0, C45, C90, C135 = channels_transverse(theta, phi, A, B, C, rho_fixed)
    base_scale = float(np.max(C0))

    colors = {0.05: "#d62728", 0.15: "#ff7f0e", 0.30: "#1f77b4"}

    # no-background reference at this rho (sharp pile-up at the reduced r_max)
    _, _, r_nobg = xy_from_channels(C0, C45, C90, C135)
    scenarios = {f"fixed ρ={rho_fixed:.2f}, no bg": ("#000000", r_nobg, None, None)}

    for frac in bg_levels:
        bg = M.make_group_backgrounds(N_GROUPS, base_scale, frac, log_sigma,
                                      np.random.default_rng(SEED + int(frac * 1000) + 5))
        C0b = C0 + bg[gid, 0]; C45b = C45 + bg[gid, 1]
        C90b = C90 + bg[gid, 2]; C135b = C135 + bg[gid, 3]
        x, y, r = xy_from_channels(C0b, C45b, C90b, C135b)
        scenarios[f"+ per-chunk bg, median {int(frac*100)}%"] = (
            colors.get(frac, "#2ca02c"), r, x, y)

    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(1, 2, wspace=0.24, left=0.07, right=0.985,
                          top=0.86, bottom=0.13)

    r_bins = np.linspace(0, 1, 140)
    ctr_bins = 0.5 * (r_bins[:-1] + r_bins[1:])

    ax = fig.add_subplot(gs[0])
    for label, (col, r, _x, _y) in scenarios.items():
        h, _ = np.histogram(r, bins=r_bins, density=True)
        lw = 3.0 if "no bg" in label else 2.0
        ax.plot(ctr_bins, h, lw=lw, color=col, label=label)
    ax.axvline(r_max_of_rho(A, B, C, rho_fixed), ls="--", color="grey", lw=1.0)
    ax.set_xlabel("r"); ax.set_ylabel("density"); ax.set_xlim(0, 1)
    ax.set_title(f"Pooled $p(r)$: fixed ρ={rho_fixed:.2f} + per-chunk background\n"
                 f"(dashed grey = $r_{{\\max}}$(ρ)={r_max_of_rho(A,B,C,rho_fixed):.2f}, "
                 f"log-σ={log_sigma})")
    ax.legend(frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[1])
    for label, (col, r, x, y) in scenarios.items():
        if x is None:
            continue
        r_bal = M.balanced_r_sample(x, y, np.random.default_rng(abs(hash(label)) % 2**31))
        ctr, th_deg, _, _ = M.hugh_theta_of_r(r_bal)
        ax.plot(ctr, th_deg, lw=2.2, color=col, label=label)
    tt = np.linspace(0, np.pi / 2, 400)
    ax.plot(M.r_formula(tt, A, B, C), np.degrees(tt), lw=1.8, color="k", ls=":",
            label="analytic Fourkas (ρ=0)")
    if "25x65nm" in curves:
        c = curves["25x65nm"]
        ax.fill_between(c["r"], c["lo"], c["hi"], color="#8c564b", alpha=0.18)
        ax.plot(c["r"], c["theta"], lw=2.6, color="#8c564b", ls="--",
                label="Hugh empirical 25×65 nm")
    ax.set_xlabel("r"); ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 92)
    ax.set_title(r"Hugh $\theta(r)$: fixed ρ + varying background vs empirical")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle(f"Fixed physical ρ={rho_fixed:.2f} (laser {LASER_NM:.0f} nm) + per-chunk background heterogeneity, "
                 "pooled over 115 chunks", y=0.98, fontsize=12.0)
    fig.savefig(OUTDIR / "transverse_fixed_rho_background.png", dpi=200)
    plt.close(fig)
    print(f"Written {OUTDIR / 'transverse_fixed_rho_background.png'}")


def figure_csv_measured(A, B, C, theta, phi, curves, rho_est):
    """Transverse modes + the ACTUAL background level & variance measured from Hugh's
    uncorrected background CSVs (per dataset): 25nm frac~0.21, 40nm frac~0.10, σ~0.31.
    Compares pooled p(r) and Hugh θ(r) against each dataset's empirical curve."""
    print("\nFourth figure: CSV-measured background + transverse modes")
    N = theta.size
    N_GROUPS = 115
    r_bins = np.linspace(0, 1, 140)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    for row, (size, meas) in enumerate(CSV_MEAS.items()):
        frac = meas["frac_med"]; sig = meas["log_sigma"]
        gid = np.random.default_rng(SEED + row).integers(0, N_GROUPS, size=N)

        def build(rho, add_bg):
            C0, C45, C90, C135 = channels_transverse(theta, phi, A, B, C, rho)
            base = float(np.max(C0))
            if add_bg:
                bg = M.make_group_backgrounds(N_GROUPS, base, frac, sig,
                                              np.random.default_rng(SEED + row + 7))
                C0 = C0 + bg[gid, 0]; C45 = C45 + bg[gid, 1]
                C90 = C90 + bg[gid, 2]; C135 = C135 + bg[gid, 3]
            return xy_from_channels(C0, C45, C90, C135)

        rho_cases = [(0.0, "#1f77b4"), (rho_est, "#d62728")]

        # p(r)
        axp = axes[row, 0]
        _, _, r_nobg = build(0.0, False)
        h, _ = np.histogram(r_nobg, bins=r_bins, density=True)
        axp.plot(ctr, h, color="k", lw=2.4, label="ρ=0, no bg (Fourkas)")
        for rho, col in rho_cases:
            _, _, r = build(rho, True)
            h, _ = np.histogram(r, bins=r_bins, density=True)
            axp.plot(ctr, h, color=col, lw=2.0, label=f"ρ={rho:.2f} + measured bg")
        axp.set_xlabel("r"); axp.set_ylabel("density"); axp.set_xlim(0, 1)
        axp.set_title(f"{size}: pooled $p(r)$  (measured bg frac={frac:.2f}, σ={sig:.2f})")
        axp.legend(frameon=False, fontsize=8.5)

        # theta(r)
        axt = axes[row, 1]
        tt = np.linspace(0, np.pi / 2, 400)
        axt.plot(M.r_formula(tt, A, B, C), np.degrees(tt), "k:", lw=1.6,
                 label="analytic Fourkas (ρ=0)")
        for rho, col in rho_cases:
            x, y, r = build(rho, True)
            r_bal = M.balanced_r_sample(x, y, np.random.default_rng(SEED + row + int(rho * 100) + 3))
            cc, th, _, _ = M.hugh_theta_of_r(r_bal)
            axt.plot(cc, th, color=col, lw=2.0, label=f"ρ={rho:.2f} + measured bg")
        if size in curves:
            c = curves[size]
            axt.fill_between(c["r"], c["lo"], c["hi"], color="#8c564b", alpha=0.18)
            axt.plot(c["r"], c["theta"], color="#8c564b", lw=2.4, ls="--",
                     label=f"Hugh empirical {size}")
        axt.set_xlabel("r"); axt.set_ylabel(r"$\theta$ (deg)")
        axt.set_xlim(0, 1); axt.set_ylim(0, 92)
        axt.set_title(f"{size}: Hugh $\\theta(r)$ vs empirical")
        axt.legend(frameon=False, fontsize=8.0, loc="lower right")

    fig.suptitle("Transverse modes + CSV-measured uncorrected background "
                 "(25nm frac≈0.21, 40nm frac≈0.10, σ≈0.31); ρ from 633 nm estimate",
                 y=0.99, fontsize=12.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTDIR / "transverse_csv_measured_background.png", dpi=200)
    plt.close(fig)
    print(f"Written {OUTDIR / 'transverse_csv_measured_background.png'}")


if __name__ == "__main__":
    main()
