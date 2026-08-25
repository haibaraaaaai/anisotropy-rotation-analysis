"""
Sweep the transverse/longitudinal PEAK amplitude ratio (AMP_RATIO_T_OVER_L) with
the per-rod LSPR spread (575-625 nm), no background, no shot noise. Find where the
pooled r-distribution's high-r reach lands, and compare theta(r) to empirical 40 nm.

rho_rod = amp^2 * g_T(633)/g_L(633; lambda_L),  lambda_L ~ U(575,625) per rod.
r_max(rho=0) = C/(A+B) is the hard ceiling set by the optics.

Output: ../transverse_mode_output/amp_ratio_sweep.png
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_report_figures as M
import transverse_mode_model as T

OUT = T.OUTDIR
N_TOTAL = 1_200_000
N_GROUPS = 115
SEED = 20260721
LAMBDA_L_RANGE = (575.0, 625.0)
AMP_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4]


def rho_of(lamL, amp):
    gL = T._lorentz_sq(T.LASER_NM, lamL, T.FWHM_L_NM)
    gT = T._lorentz_sq(T.LASER_NM, T.LAMBDA_T_NM, T.FWHM_T_NM)
    return (amp ** 2) * gT / gL


def main():
    A, B, C = T.effective_ABC()
    ceiling = T.r_max_of_rho(A, B, C, 0.0)
    curves = M.load_hugh_curves()
    rng = np.random.default_rng(SEED)
    theta, phi = M.sample_isotropic(N_TOTAL, rng)
    gid = rng.integers(0, N_GROUPS, size=N_TOTAL)
    lamL_g = np.random.default_rng(SEED + 1).uniform(*LAMBDA_L_RANGE, size=N_GROUPS)

    r_bins = np.linspace(0, 1, 150)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])
    cmap = plt.get_cmap("viridis")
    print(f"r_max ceiling (rho=0) = {ceiling:.3f}")
    print(f"{'amp':>5} {'rho_med':>8} {'rho_min':>8} {'rmax(rho_min)':>13} {'pooled p99.5(r)':>15}")

    fig, (axP, axT, axR) = plt.subplots(1, 3, figsize=(16.5, 5.2))
    reach = []
    for i, amp in enumerate(AMP_RATIOS):
        rho_g = rho_of(lamL_g, amp)
        rho_pt = rho_g[gid]
        C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, rho_pt)
        x, y, r = T.xy_from_channels(C0, C45, C90, C135)
        p995 = float(np.percentile(r, 99.5))
        reach.append(p995)
        rmax_min = T.r_max_of_rho(A, B, C, rho_g.min())
        print(f"{amp:>5.2f} {np.median(rho_g):>8.3f} {rho_g.min():>8.3f} "
              f"{rmax_min:>13.3f} {p995:>15.3f}")
        col = cmap(i / (len(AMP_RATIOS) - 1))
        h, _ = np.histogram(r, bins=r_bins, density=True)
        axP.plot(ctr, h, color=col, lw=1.8, label=f"amp={amp:.2f}")
        r_bal = M.balanced_r_sample(x, y, np.random.default_rng(SEED + i + 3))
        cc, th, _, _ = M.hugh_theta_of_r(r_bal)
        axT.plot(cc, th, color=col, lw=1.8, label=f"amp={amp:.2f}")

    # p(r)
    axP.axvline(ceiling, color="k", ls="--", lw=1.0)
    axP.set_xlim(0, 1); axP.set_xlabel("r"); axP.set_ylabel("density")
    axP.set_title("Pooled $p(r)$ vs amp ratio\n(dashed = $r_{\\max}$ ceiling %.2f)" % ceiling)
    axP.legend(frameon=False, fontsize=8, ncol=2)

    # theta(r)
    tt = np.linspace(0, np.pi / 2, 400)
    axT.plot(M.r_formula(tt, A, B, C), np.degrees(tt), "k:", lw=1.6, label="Fourkas ρ=0")
    if "40x65nm" in curves:
        c_ = curves["40x65nm"]
        axT.plot(c_["r"], c_["theta"], color="crimson", ls="--", lw=2.4, label="empirical 40 nm")
    axT.set_xlim(0, 1); axT.set_ylim(0, 92)
    axT.set_xlabel("r"); axT.set_ylabel(r"$\theta$ (deg)")
    axT.set_title(r"Hugh $\theta(r)$ vs empirical")
    axT.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

    # reach vs amp ratio
    axR.plot(AMP_RATIOS, reach, "o-", color="#1f77b4", lw=2)
    axR.axhline(ceiling, color="k", ls="--", lw=1.2, label=f"optics ceiling {ceiling:.2f}")
    axR.axhline(0.85, color="crimson", ls=":", lw=1.4, label="target 0.85")
    if "40x65nm" in curves:
        axR.axhline(float(np.max(curves["40x65nm"]["r"])), color="crimson", ls="--",
                    lw=1.0, alpha=0.6, label="empirical max r")
    axR.set_xlabel("transverse/longitudinal peak amplitude ratio")
    axR.set_ylabel("pooled high-r reach  (p99.5 of r)")
    axR.set_title("High-r reach vs amp ratio")
    axR.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Amplitude-ratio sweep with LSPR 575–625 nm spread (no background, no shot noise) — "
                 "high-r reach is capped by the optics $r_{\\max}$", fontsize=12.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "amp_ratio_sweep.png", dpi=200)
    plt.close(fig)
    print(f"\nWritten {OUT / 'amp_ratio_sweep.png'}")


if __name__ == "__main__":
    main()
