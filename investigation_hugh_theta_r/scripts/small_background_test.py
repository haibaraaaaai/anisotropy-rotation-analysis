"""
Small-background test.

Colleague feedback: real backgrounds are expected to be SMALL (few %), not the
~30% used in the main grouped-background fit. But because the rod signal goes to
ZERO at theta=0 (I_tot = sin^2(theta): pure long-dipole drive, no transverse
component), the anisotropy of a dim point is set ENTIRELY by the background
imbalance across channels:

    x_dim = (b0 - b90) / (b0 + b90),   y_dim = (b45 - b135) / (b45 + b135)

This ratio is scale-free: it depends on the *relative* channel imbalance, not
on the absolute background level. So a small (5%) background that still varies
across channels shifts dim points just as far as a large one. What scales with
the level is only how far the BRIGHT points (near r_max) get pulled inward.

Question tested here: with a small (5%) but cross-channel-varying background,
    (1) does the pooled p(r) still become a smooth bell, or does the r_max
        Jacobian pile-up survive?
    (2) does Hugh's CDF-inversion still bend theta(r) toward the empirical curve?

We reuse the exact machinery (Fourkas channels, per-recording lognormal bg with
b0+b90 = b45+b135 enforced, Hugh phi-balancing + CDF inversion) from
make_report_figures.py and simply sweep the median background fraction.

Output: ../small_background_test/small_background_test.png (+ per-level curves).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# reuse the already-validated helpers
import make_report_figures as M

OUTDIR = Path(__file__).resolve().parent.parent / "small_background_test"
OUTDIR.mkdir(exist_ok=True)

# --------- config ---------
N_TOTAL = 2_000_000
N_GROUPS = 115
LOG_SIGMA = 0.6            # cross-channel / cross-recording imbalance (kept strong)
FRAC_LEVELS = [0.05, 0.15, 0.30]
LEVEL_COLORS = {0.05: "#d62728", 0.15: "#ff7f0e", 0.30: "#1f77b4"}
SEED = 20260718


def main():
    curves = M.load_hugh_curves()
    NA, N_MED = 1.3, 1.5
    _, A, B, C = M.fourkas_ABC(NA, N_MED)
    r_max = M.r_formula(np.pi / 2, A, B, C)
    print(f"A={A:.5g} B={B:.5g} C={C:.5g}  r_max={r_max:.4f}")

    # no-background reference pool (Jacobian spike at r_max)
    rng0 = np.random.default_rng(SEED)
    sim_nobg = M.simulate_pool(N_TOTAL, 1, 0.0, 0.0, rng0)

    results = {}
    for frac in FRAC_LEVELS:
        rng = np.random.default_rng(SEED + int(frac * 1000))
        sim = M.simulate_pool(N_TOTAL, N_GROUPS, frac, LOG_SIGMA, rng)
        r_bal = M.balanced_r_sample(sim["x"], sim["y"],
                                    np.random.default_rng(SEED + 7 + int(frac * 1000)))
        ctr, theta_deg, _, _ = M.hugh_theta_of_r(r_bal)
        results[frac] = {"sim": sim, "ctr": ctr, "theta": theta_deg,
                         "r": sim["r"]}
        # diagnostics: does the r_max pile-up survive?
        r = sim["r"]
        frac_near_rmax = float(np.mean(r > 0.9 * r_max))
        print(f"frac_med={frac:.2f}: pooled r  mean={r.mean():.3f}  "
              f"median={np.median(r):.3f}  p95={np.percentile(r,95):.3f}  "
              f"fraction with r>0.9*r_max = {frac_near_rmax:.3f}")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13.8, 9.0))
    gs = fig.add_gridspec(2, 2, wspace=0.24, hspace=0.42,
                          left=0.07, right=0.94, top=0.90, bottom=0.08)

    r_bins = np.linspace(0, 1, 140)
    ctr_bins = 0.5 * (r_bins[:-1] + r_bins[1:])

    # (a) pooled p(r) for each level vs no-bg
    ax = fig.add_subplot(gs[0, 0])
    h0, _ = np.histogram(sim_nobg["r"], bins=r_bins, density=True)
    ax.fill_between(ctr_bins, h0, color="0.75", alpha=0.7,
                    label="no background (Fourkas)")
    for frac in FRAC_LEVELS:
        h, _ = np.histogram(results[frac]["r"], bins=r_bins, density=True)
        ax.plot(ctr_bins, h, lw=2.2, color=LEVEL_COLORS[frac],
                label=f"{int(frac*100)}% median bg")
    ax.axvline(r_max, ls="--", color="k", lw=1.0)
    ax.text(r_max, ax.get_ylim()[1] * 0.92, r"  $r_{\max}$", fontsize=10)
    ax.set_xlabel("r"); ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.set_title("(a)  Pooled $p(r)$: does the bell form at small bg?\n"
                 "(strong cross-channel imbalance, log-σ=%.1f)" % LOG_SIGMA)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    # (b) theta(r) via Hugh's method for each level
    ax = fig.add_subplot(gs[0, 1])
    tt = np.linspace(0, np.pi / 2, 500)
    ax.plot(M.r_formula(tt, A, B, C), np.degrees(tt), lw=2.4, color="k",
            label="analytic Fourkas (no bg)")
    for frac in FRAC_LEVELS:
        ax.plot(results[frac]["ctr"], results[frac]["theta"], lw=2.2,
                color=LEVEL_COLORS[frac], label=f"{int(frac*100)}% bg → Hugh method")
    if "25x65nm" in curves:
        c = curves["25x65nm"]
        ax.fill_between(c["r"], c["lo"], c["hi"], color="#2ca02c", alpha=0.18)
        ax.plot(c["r"], c["theta"], lw=2.6, color="#2ca02c", ls="--",
                label="Hugh empirical 25×65 nm")
    ax.set_xlabel("r"); ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 92)
    ax.set_title(r"(b)  $\theta(r)$ from Hugh's CDF-inversion at each bg level")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    # (c) xy scatter at 5% coloured by theta band (dim shift vs bright stay)
    band_edges = np.arange(0, 91, 10)
    n_band = len(band_edges) - 1
    cmap = plt.get_cmap("turbo", n_band)
    norm = BoundaryNorm(band_edges, cmap.N)
    for panel, frac in ((gs[1, 0], 0.05), (gs[1, 1], 0.30)):
        ax = fig.add_subplot(panel)
        sim = results[frac]["sim"]
        theta_deg = np.degrees(sim["theta"])
        sub = np.random.default_rng(1).choice(sim["x"].size, size=9000, replace=False)
        sc = ax.scatter(sim["x"][sub], sim["y"][sub], c=theta_deg[sub],
                        cmap=cmap, norm=norm, s=5, alpha=0.55, linewidths=0)
        ax.add_patch(plt.Circle((0, 0), r_max, fill=False, ls="--", color="k", lw=1.3))
        ax.scatter([0], [0], s=70, marker="+", color="k", zorder=6)
        ax.set_aspect("equal")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"({'c' if frac == 0.05 else 'd'})  Anisotropy cloud, "
                     f"{int(frac*100)}% bg (colour = θ band)")
        if frac == 0.30:
            cb = fig.colorbar(sc, ax=ax, boundaries=band_edges, ticks=band_edges,
                              fraction=0.046, pad=0.04)
            cb.set_label(r"$\theta$ (deg)")

    fig.suptitle("Small (5%) but cross-channel-varying background: dim points still shift, "
                 "but the bright $r_{\\max}$ edge is far less smeared than at 30%",
                 y=0.985, fontsize=12.5)
    fig.savefig(OUTDIR / "small_background_test.png", dpi=200)
    plt.close(fig)
    print(f"\nWritten {OUTDIR / 'small_background_test.png'}")


if __name__ == "__main__":
    main()
