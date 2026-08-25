"""
rho spread ONLY — no background, no shot noise.

Each rod's longitudinal LSPR sits somewhere in the spec range 575-625 nm (poly-
dispersity), so at the 633 nm laser each rod has a different transverse/longitudinal
ratio rho, hence a different r_max. We draw lambda_L ~ U(575, 625) per recording,
keep the transverse resonance at 520 nm, use the current hole-corrected optics
(NA=1.3, n=1.33), and pool many rods. Question: does rho polydispersity ALONE
reshape p(r) / bend theta(r) toward the data, with no background at all?

Output: ../transverse_mode_output/rho_spread_only.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_report_figures as M
import transverse_mode_model as T

OUT = T.OUTDIR
N_TOTAL = 1_500_000
N_GROUPS = 115
SEED = 20260721
LAMBDA_L_RANGE = (575.0, 625.0)     # spec Peak SPR range (longitudinal LSPR)


def rho_of_lambdaL(lamL):
    gL = T._lorentz_sq(T.LASER_NM, lamL, T.FWHM_L_NM)
    gT = T._lorentz_sq(T.LASER_NM, T.LAMBDA_T_NM, T.FWHM_T_NM)
    return (T.AMP_RATIO_T_OVER_L ** 2) * gT / gL


def main():
    A, B, C = T.effective_ABC()
    curves = M.load_hugh_curves()
    rng = np.random.default_rng(SEED)
    theta, phi = M.sample_isotropic(N_TOTAL, rng)
    gid = rng.integers(0, N_GROUPS, size=N_TOTAL)

    # per-rod longitudinal resonance -> per-rod rho -> per-rod r_max
    lamL_g = np.random.default_rng(SEED + 1).uniform(*LAMBDA_L_RANGE, size=N_GROUPS)
    rho_g = np.array([rho_of_lambdaL(l) for l in lamL_g])
    rmax_g = np.array([T.r_max_of_rho(A, B, C, r) for r in rho_g])
    rho_med = float(np.median(rho_g))

    print(f"hole-corrected A={A:.4g} B={B:.4g} C={C:.4g}")
    print(f"lambda_L in {LAMBDA_L_RANGE} nm  ->  rho in "
          f"[{rho_g.min():.3f}, {rho_g.max():.3f}] (median {rho_med:.3f})")
    print(f"                                r_max in "
          f"[{rmax_g.min():.3f}, {rmax_g.max():.3f}]")

    def pooled(rho_pt):
        C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, rho_pt)
        return T.xy_from_channels(C0, C45, C90, C135)

    x_s, y_s, r_s = pooled(rho_g[gid])                 # rho-spread
    x_f, y_f, r_f = pooled(np.full(N_TOTAL, rho_med))  # fixed median rho
    r_nobg0 = pooled(np.zeros(N_TOTAL))[2]             # analytic Fourkas (rho=0)

    r_bins = np.linspace(0, 1, 150)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # (A) rho and r_max distribution across rods
    axA.hist(rho_g, bins=20, color="#4c72b0", alpha=0.8)
    axA.set_xlabel("per-rod ρ"); axA.set_ylabel("rods")
    axA.set_title(f"ρ from λ_L∈{LAMBDA_L_RANGE[0]:.0f}–{LAMBDA_L_RANGE[1]:.0f} nm\n"
                  f"ρ∈[{rho_g.min():.2f}, {rho_g.max():.2f}], "
                  f"$r_{{\\max}}$∈[{rmax_g.min():.2f}, {rmax_g.max():.2f}]")
    axt = axA.twiny()
    axt.set_xlim(axA.get_xlim())
    axt.set_xticks([rho_g.min(), rho_med, rho_g.max()])
    axt.set_xticklabels([f"$r_{{max}}$={rmax_g.max():.2f}", f"{np.median(rmax_g):.2f}",
                         f"{rmax_g.min():.2f}"], fontsize=8)

    # (B) pooled p(r)
    for r, col, lab, lw in ((r_nobg0, "k", "analytic Fourkas (ρ=0)", 2.6),
                            (r_f, "#7f7f7f", f"fixed ρ={rho_med:.2f}", 2.0),
                            (r_s, "#d62728", "ρ spread (575–625 nm)", 2.6)):
        h, _ = np.histogram(r, bins=r_bins, density=True)
        axB.plot(ctr, h, color=col, lw=lw, label=lab)
    axB.set_xlim(0, 1); axB.set_xlabel("r"); axB.set_ylabel("density")
    axB.set_title("Pooled $p(r)$ — no background, no shot noise")
    axB.legend(frameon=False, fontsize=9)

    # (C) Hugh theta(r)
    tt = np.linspace(0, np.pi / 2, 400)
    axC.plot(M.r_formula(tt, A, B, C), np.degrees(tt), "k:", lw=2.0, label="analytic Fourkas")
    for (x, y), col, lab in (((x_f, y_f), "#7f7f7f", f"fixed ρ={rho_med:.2f}"),
                             ((x_s, y_s), "#d62728", "ρ spread (575–625 nm)")):
        r_bal = M.balanced_r_sample(x, y, np.random.default_rng(SEED + hash(lab) % 999))
        cc, th, _, _ = M.hugh_theta_of_r(r_bal)
        axC.plot(cc, th, color=col, lw=2.4, label=lab)
    if "40x65nm" in curves:
        c_ = curves["40x65nm"]
        axC.fill_between(c_["r"], c_["lo"], c_["hi"], color="0.4", alpha=0.2)
        axC.plot(c_["r"], c_["theta"], color="0.25", ls="--", lw=2.2, label="empirical 40×65 nm")
    axC.set_xlim(0, 1); axC.set_ylim(0, 92)
    axC.set_xlabel("r"); axC.set_ylabel(r"$\theta$ (deg)")
    axC.set_title(r"Hugh $\theta(r)$ vs empirical")
    axC.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle("ρ polydispersity alone (LSPR 575–625 nm @ 633 nm laser, hole-corrected optics) — "
                 "no background, no shot noise", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "rho_spread_only.png", dpi=200)
    plt.close(fig)
    print(f"\nWritten {OUT / 'rho_spread_only.png'}")


if __name__ == "__main__":
    main()
