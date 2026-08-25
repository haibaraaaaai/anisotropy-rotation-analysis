"""
Grid: one panel per transverse amplitude ratio (0 -> 1). Each panel shows Hugh
theta(r) for three background levels (15%, 25%, 40% of peak signal), with the
per-rod LSPR spread (575-625 nm) setting rho, and the two-layer background
variance (rod-to-rod sigma=0.49, cross-channel sigma=0.30). amp=0 => no transverse.

Empirical 40x65 nm and analytic Fourkas overlaid on every panel.

Output: ../transverse_mode_output/amp_bg_grid_40nm.png
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_report_figures as M
import transverse_mode_model as T

OUT = T.OUTDIR
N_TOTAL = 1_000_000
N_GROUPS = 115
SEED = 20260723
SIGNAL_PEAK_CT = 400.0
SIGMA_ROD = 0.49
SIGMA_CH = 0.30
LAMBDA_L_RANGE = (575.0, 625.0)

AMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BG_LEVELS = [(0.15, "#2ca02c"), (0.25, "#ff7f0e"), (0.40, "#d62728")]


def two_layer_bg(n_groups, median_ct, rng):
    level = median_ct * rng.lognormal(0.0, SIGMA_ROD, size=n_groups)
    b = level[:, None] * rng.lognormal(0.0, SIGMA_CH, size=(n_groups, 4))
    sum0 = b[:, 0] + b[:, 2]; diff = b[:, 1] - b[:, 3]
    b[:, 1] = 0.5 * (sum0 + diff); b[:, 3] = 0.5 * (sum0 - diff)
    neg = (b[:, 1] < 0) | (b[:, 3] < 0)
    b[neg, 1] = 0.5 * sum0[neg]; b[neg, 3] = 0.5 * sum0[neg]
    return b


def rho_groups(amp, rng):
    lamL = rng.uniform(*LAMBDA_L_RANGE, size=N_GROUPS)
    gL = T._lorentz_sq(T.LASER_NM, lamL, T.FWHM_L_NM)
    gT = T._lorentz_sq(T.LASER_NM, T.LAMBDA_T_NM, T.FWHM_T_NM)
    return (amp ** 2) * gT / gL


def main():
    A, B, C = T.effective_ABC()
    curves = M.load_hugh_curves()
    theta, phi = M.sample_isotropic(N_TOTAL, np.random.default_rng(SEED))
    tt = np.linspace(0, np.pi / 2, 400)
    r_fourkas = M.r_formula(tt, A, B, C)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), sharex=True, sharey=True)
    for ax, amp in zip(axes.ravel(), AMPS):
        rng = np.random.default_rng(SEED + int(amp * 100) + 1)
        gid = rng.integers(0, N_GROUPS, size=N_TOTAL)
        rho_g = rho_groups(amp, np.random.default_rng(SEED + int(amp * 100) + 2))
        rho_pt = rho_g[gid]
        rmax_lo = T.r_max_of_rho(A, B, C, rho_g.max())
        rmax_hi = T.r_max_of_rho(A, B, C, rho_g.min())

        C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, rho_pt)
        scale = SIGNAL_PEAK_CT / float(np.max(C0))
        C0 *= scale; C45 *= scale; C90 *= scale; C135 *= scale

        ax.plot(r_fourkas, np.degrees(tt), "k:", lw=1.4, label="Fourkas ρ=0")
        for frac, col in BG_LEVELS:
            bg = two_layer_bg(N_GROUPS, frac * SIGNAL_PEAK_CT,
                              np.random.default_rng(SEED + int(amp * 100) + int(frac * 1000)))
            d0 = C0 + bg[gid, 0]; d45 = C45 + bg[gid, 1]
            d90 = C90 + bg[gid, 2]; d135 = C135 + bg[gid, 3]
            x, y, _ = T.xy_from_channels(d0, d45, d90, d135)
            r_bal = M.balanced_r_sample(x, y, np.random.default_rng(SEED + int(frac * 1000) + int(amp * 50)))
            cc, th, _, _ = M.hugh_theta_of_r(r_bal)
            ax.plot(cc, th, color=col, lw=2.0, label=f"{int(frac*100)}% bg")
        if "40x65nm" in curves:
            c_ = curves["40x65nm"]
            ax.plot(c_["r"], c_["theta"], color="0.2", ls="--", lw=2.2, label="empirical 40 nm")

        tag = "no transverse" if amp == 0 else f"ρ={rho_g.min():.2f}–{rho_g.max():.2f}"
        ax.set_title(f"amp={amp:.1f}   ({tag}; $r_{{max}}$={rmax_lo:.2f}–{rmax_hi:.2f})", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 92); ax.grid(alpha=0.25)
        if amp == AMPS[0]:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("r")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta$ (deg)")

    fig.suptitle("40×65 nm: Hugh $\\theta(r)$ vs transverse amplitude ratio (panels) × background level (curves)\n"
                 "LSPR spread 575–625 nm, two-layer bg variance (σ_rod=0.49, σ_ch=0.30), peak signal 400 ct",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "amp_bg_grid_40nm.png", dpi=190)
    plt.close(fig)
    print(f"Written {OUT / 'amp_bg_grid_40nm.png'}")


if __name__ == "__main__":
    main()
