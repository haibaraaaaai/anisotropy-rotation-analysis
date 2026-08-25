"""
p(r) for amp = 0, 0.1, 0.2 at the measured 15% background (two-layer variance).

Shows how the r probability density (physical isotropic sample, NOT balanced)
turns from the analytic Jacobian pile-up spike near r_max into the observed bell.

Output: ../transverse_mode_output/p_of_r_amp_40nm.png
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
SEED = 20260724
SIGNAL_PEAK_CT = 400.0
SIGMA_ROD = 0.49
SIGMA_CH = 0.30
LAMBDA_L_RANGE = (575.0, 625.0)
BG_FRAC = 0.15

AMPS = [(0.0, "#1f77b4"), (0.1, "#2ca02c"), (0.2, "#d62728")]


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


def r_of_sample(amp, add_bg, rng):
    theta, phi = M.sample_isotropic(N_TOTAL, rng)
    gid = rng.integers(0, N_GROUPS, size=N_TOTAL)
    rho_pt = rho_groups(amp, rng)[gid]
    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, rho_pt)
    scale = SIGNAL_PEAK_CT / float(np.max(C0))
    C0 *= scale; C45 *= scale; C90 *= scale; C135 *= scale
    if add_bg:
        bg = two_layer_bg(N_GROUPS, BG_FRAC * SIGNAL_PEAK_CT, rng)
        C0 = C0 + bg[gid, 0]; C45 = C45 + bg[gid, 1]
        C90 = C90 + bg[gid, 2]; C135 = C135 + bg[gid, 3]
    _, _, r = T.xy_from_channels(C0, C45, C90, C135)
    return r


A, B, C = T.effective_ABC()


def main():
    edges = np.linspace(0, 1, 121)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    # analytic clean reference (no bg, no transverse) -> Jacobian pile-up spike
    r_clean = r_of_sample(0.0, add_bg=False, rng=np.random.default_rng(SEED))
    h, _ = np.histogram(r_clean, bins=edges, density=True)
    ax.plot(ctr, h, color="0.55", lw=1.6, ls=":",
            label="no bg, no transverse (analytic pile-up)")

    for amp, col in AMPS:
        r = r_of_sample(amp, add_bg=True, rng=np.random.default_rng(SEED + int(amp * 100) + 1))
        h, _ = np.histogram(r, bins=edges, density=True)
        ax.plot(ctr, h, color=col, lw=2.2, label=f"amp={amp:.1f}  +15% bg")

    ax.set_xlabel("r")
    ax.set_ylabel("p(r)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("40×65 nm: p(r) at 15% background (two-layer variance)\n"
                 "amp = 0, 0.1, 0.2 — background turns the pile-up spike into a bell",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "p_of_r_amp_40nm.png", dpi=190)
    plt.close(fig)
    print(f"Written {OUT / 'p_of_r_amp_40nm.png'}")


if __name__ == "__main__":
    main()
