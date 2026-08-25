"""
Interference term on top of 15% two-layer background (sigma_ch=0.30, sigma_rod=0.49).

For each amp in {0, 0.1, 0.2} we build the pooled r sample three ways and plot p(r):
  - no bg (analytic pile-up reference)
  - incoherent + 15% bg
  - coherent + 15% bg  (interference: I_j = |sqrt(I_rod,j) + sqrt(I_bg,j) e^{i delta}|^2,
    only the polarised parts interfere, per-rod random phase delta)

One separate figure per amp.

Output: ../transverse_mode_output/interference_p_of_r_amp{X}.png
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_report_figures as M
import transverse_mode_model as T
import coherent_background_model as CB

OUT = T.OUTDIR
N_TOTAL = 1_000_000
N_GROUPS = 115
SEED = 20260725
SIGNAL_PEAK_CT = 400.0
SIGMA_ROD = 0.49
SIGMA_CH = 0.30
LAMBDA_L_RANGE = (575.0, 625.0)
BG_FRAC = 0.15

AMPS = [0.0, 0.1, 0.2]

A, B, C = T.effective_ABC()


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


def build_r(amp, method, rng):
    """Pooled r for given amp and detection method ('incoherent'/'coherent')."""
    theta, phi = M.sample_isotropic(N_TOTAL, rng)
    gid = rng.integers(0, N_GROUPS, size=N_TOTAL)
    rho_pt = rho_groups(amp, rng)[gid]
    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, rho_pt)
    scale = SIGNAL_PEAK_CT / float(np.max(C0))
    C0 *= scale; C45 *= scale; C90 *= scale; C135 *= scale

    bg = two_layer_bg(N_GROUPS, BG_FRAC * SIGNAL_PEAK_CT, rng)
    bg_pt = (bg[gid, 0], bg[gid, 1], bg[gid, 2], bg[gid, 3])
    delta = rng.uniform(0.0, 2.0 * np.pi, size=N_GROUPS)[gid]
    d0, d45, d90, d135 = CB.detect((C0, C45, C90, C135), bg_pt, delta, method)
    return T.xy_from_channels(d0, d45, d90, d135)[2]


def r_nobg(amp, rng):
    theta, phi = M.sample_isotropic(N_TOTAL, rng)
    gid = rng.integers(0, N_GROUPS, size=N_TOTAL)
    rho_pt = rho_groups(amp, rng)[gid]
    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, rho_pt)
    return T.xy_from_channels(C0, C45, C90, C135)[2]


def main():
    edges = np.linspace(0, 1, 121)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    for amp in AMPS:
        r0 = r_nobg(amp, np.random.default_rng(SEED + int(amp * 100)))
        ri = build_r(amp, "incoherent", np.random.default_rng(SEED + int(amp * 100) + 1))
        rc = build_r(amp, "coherent", np.random.default_rng(SEED + int(amp * 100) + 2))

        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        for r, col, lab, ls in (
            (r0, "0.55", "no bg (analytic pile-up)", ":"),
            (ri, "#1f77b4", "incoherent + 15% bg", "-"),
            (rc, "#d62728", "coherent + 15% bg (interference)", "-"),
        ):
            h, _ = np.histogram(r, bins=edges, density=True)
            ax.plot(ctr, h, color=col, lw=2.2, ls=ls, label=lab)
        ax.set_xlabel("r"); ax.set_ylabel("p(r)")
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0); ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        tag = "no transverse" if amp == 0 else f"amp={amp:.1f}"
        ax.set_title(f"40×65 nm: p(r) — interference vs incoherent at 15% bg  ({tag})\n"
                     "two-layer bg (σ_rod=0.49, σ_ch=0.30), per-rod random phase δ",
                     fontsize=11.0)
        fig.tight_layout()
        out = OUT / f"interference_p_of_r_amp{amp:.1f}.png"
        fig.savefig(out, dpi=190)
        plt.close(fig)
        print(f"Written {out}  "
              f"[median r: nobg={np.median(r0):.3f} incoh={np.median(ri):.3f} coh={np.median(rc):.3f}]")


if __name__ == "__main__":
    main()
