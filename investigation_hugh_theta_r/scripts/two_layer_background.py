"""
Two-layer background model, in COUNTS (pure background, no transverse).

Everything is entered in 12-bit camera counts so it is intuitive:
    SIGNAL_PEAK_CT   : peak (brightest-orientation) per-channel signal
    BG_MEDIAN_CT     : fixed/median background level
    SIGMA_ROD        : rod-to-rod (per-recording) lognormal spread of the level
    SIGMA_CH         : cross-channel lognormal spread within a recording

Two layers:
    level_g = BG_MEDIAN_CT * lognormal(0, SIGMA_ROD)          # per recording
    b_ch    = level_g       * lognormal(0, SIGMA_CH)          # per channel
    (then b45,b135 adjusted so b0+b90 = b45+b135)

Signal channels (Fourkas, rho=0) are scaled so their peak = SIGNAL_PEAK_CT, then
background counts are added directly. Pool over N_GROUPS recordings, phi-balanced
CDF inversion (original pipeline). Compare to empirical 40x65 nm.

Output: ../transverse_mode_output/two_layer_background_40nm.png
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_report_figures as M
import transverse_mode_model as T

OUT = T.OUTDIR
N_TOTAL = 1_500_000
N_GROUPS = 115
SEED = 20260722

# --- everything in 12-bit counts (measured from 40 nm data) ---
SIGNAL_PEAK_CT = 400.0     # brightest-orientation per-channel signal
BG_MEDIAN_CT = 58.0        # fixed/median background
SIGMA_ROD = 0.49           # rod-to-rod spread of the background level
SIGMA_CH = 0.30            # cross-channel imbalance within a recording


def two_layer_bg(n_groups, bg_median_ct, sigma_rod, sigma_ch, rng, enforce=True):
    """Per-recording background in counts, columns [b0, b45, b90, b135]."""
    level = bg_median_ct * rng.lognormal(0.0, sigma_rod, size=n_groups)      # per rod
    b = level[:, None] * rng.lognormal(0.0, sigma_ch, size=(n_groups, 4))    # per channel
    if enforce:
        sum0 = b[:, 0] + b[:, 2]
        diff = b[:, 1] - b[:, 3]
        b[:, 1] = 0.5 * (sum0 + diff)
        b[:, 3] = 0.5 * (sum0 - diff)
        neg = (b[:, 1] < 0) | (b[:, 3] < 0)
        b[neg, 1] = 0.5 * sum0[neg]; b[neg, 3] = 0.5 * sum0[neg]
    return b


def simulate(theta, phi, A, B, C, bg_median_ct, sigma_rod, sigma_ch, seed):
    rng = np.random.default_rng(seed)
    gid = rng.integers(0, N_GROUPS, size=theta.size)
    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, 0.0)   # rho=0
    scale = SIGNAL_PEAK_CT / float(np.max(C0))                             # -> counts
    C0 *= scale; C45 *= scale; C90 *= scale; C135 *= scale
    bg = two_layer_bg(N_GROUPS, bg_median_ct, sigma_rod, sigma_ch, np.random.default_rng(seed + 1))
    d0 = C0 + bg[gid, 0]; d45 = C45 + bg[gid, 1]
    d90 = C90 + bg[gid, 2]; d135 = C135 + bg[gid, 3]
    return T.xy_from_channels(d0, d45, d90, d135)


def main():
    A, B, C = T.effective_ABC()
    r_max = T.r_max_of_rho(A, B, C, 0.0)
    curves = M.load_hugh_curves()
    theta, phi = M.sample_isotropic(N_TOTAL, np.random.default_rng(SEED))

    print(f"signal peak = {SIGNAL_PEAK_CT:.0f} ct, r_max = {r_max:.3f}")
    print(f"fixed bg = {BG_MEDIAN_CT:.0f} ct = {BG_MEDIAN_CT/SIGNAL_PEAK_CT:.0%} of peak; "
          f"rod-to-rod sigma={SIGMA_ROD}, cross-channel sigma={SIGMA_CH}")

    r_bins = np.linspace(0, 1, 150)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])
    fig, (axP, axT) = plt.subplots(1, 2, figsize=(14.0, 5.8))

    # no-bg reference
    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, 0.0)
    r_nobg = T.xy_from_channels(C0, C45, C90, C135)[2]
    h, _ = np.histogram(r_nobg, bins=r_bins, density=True)
    axP.plot(ctr, h, "k", lw=2.4, label="no background (Fourkas)")
    tt = np.linspace(0, np.pi / 2, 400)
    axT.plot(M.r_formula(tt, A, B, C), np.degrees(tt), "k:", lw=1.8, label="analytic Fourkas")

    # measured level + two higher levels to see how much bg is needed
    cases = [(BG_MEDIAN_CT, "#2ca02c", "measured 58 ct (15%)"),
             (100.0, "#ff7f0e", "100 ct (25%)"),
             (160.0, "#d62728", "160 ct (40%)")]
    for med, col, lab in cases:
        x, y, r = simulate(theta, phi, A, B, C, med, SIGMA_ROD, SIGMA_CH, SEED + int(med))
        h, _ = np.histogram(r, bins=r_bins, density=True)
        axP.plot(ctr, h, color=col, lw=2.2, label=lab)
        r_bal = M.balanced_r_sample(x, y, np.random.default_rng(SEED + int(med) + 5))
        cc, th, _, _ = M.hugh_theta_of_r(r_bal)
        axT.plot(cc, th, color=col, lw=2.2, label=lab)

    if "40x65nm" in curves:
        c_ = curves["40x65nm"]
        axT.fill_between(c_["r"], c_["lo"], c_["hi"], color="0.4", alpha=0.2)
        axT.plot(c_["r"], c_["theta"], color="0.2", ls="--", lw=2.4, label="empirical 40×65 nm")

    axP.axvline(r_max, color="grey", ls="--", lw=1.0)
    axP.set_xlim(0, 1); axP.set_xlabel("r"); axP.set_ylabel("density")
    axP.set_title(f"Pooled $p(r)$ — two-layer bg (σ_rod={SIGMA_ROD}, σ_ch={SIGMA_CH})\n"
                  f"peak signal {SIGNAL_PEAK_CT:.0f} ct, no transverse")
    axP.legend(frameon=False, fontsize=9)

    axT.set_xlim(0, 1); axT.set_ylim(0, 92)
    axT.set_xlabel("r"); axT.set_ylabel(r"$\theta$ (deg)")
    axT.set_title(r"Hugh $\theta(r)$ vs empirical 40×65 nm")
    axT.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle("Pure background, two-layer variance (rod-to-rod + cross-channel), counts — 40×65 nm",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "two_layer_background_40nm.png", dpi=200)
    plt.close(fig)
    print(f"\nWritten {OUT / 'two_layer_background_40nm.png'}")


if __name__ == "__main__":
    main()
