"""
Test hypothesis: per-recording background differences reproduce Hugh's r-density.

Method
------
1. Simulate a large pool of isotropic tumbling points using the exact Fourkas
   channel formulas (as in fourkas_background_comparison_output/
   isotropic_fourkas_background_compare.py). Use Itot = sin^2(theta) (rod
   emission scaling).
2. Split those points into N_GROUPS "recordings" (equal-size chunks).
3. For each recording, draw four channel backgrounds b0,b45,b90,b135 from
   independent lognormals with per-channel medians = FRAC_MED * <C0_ideal_max>
   and spread SIGMA. Then optionally enforce b0+b90 == b45+b135 by projecting
   onto that plane (rebalance b45,b135 to preserve their difference but match
   the sum).
4. Add each recording's fixed background to its own points, compute x,y,r.
5. Pool everything and run the same phi-balancing + CDF-inversion that Hugh
   uses. Compare to Hugh's real 25 nm curve.

Also compare the pooled p(r) shape to Hugh's data indirectly by plotting the
simulated pooled histogram and per-recording histogram spread.

Outputs go to fourkas_group_background_simulation/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
    PchipInterpolator = None

# --------- config ---------
N_TOTAL = 3_000_000          # total pooled points (all recordings combined)
N_GROUPS = 115               # number of "recordings"
NA = 1.3
N_MEDIUM = 1.5
ITOT_MODE = "sin2theta"      # "constant" or "sin2theta"

BG_FRAC_MED = 0.15           # median background per channel = 15% of max C0_ideal
BG_LOG_SIGMA = 0.6           # multiplicative spread across recordings (log-normal sigma)
ENFORCE_SUM_INV = True       # enforce b0+b90 == b45+b135
RANDOM_SEED = 20260714

OUTDIR = Path(__file__).resolve().parent.parent / "fourkas_group_background_simulation"

# Hugh's method parameters (matches bootstrap_balanced_theta_vs_r_recording_level.py)
PHI_BIN_DEG = 5.0
HIST_BINS = 160
HIST_SIGMA_BINS = 3.0
HIST_LO_PCT = 0.5
HIST_HI_PCT = 99.5
BALANCE_HALF_LEAST = True


# --------- Fourkas ---------
def fourkas_ABC(NA, n):
    alpha = np.arcsin(NA / n)
    ca = np.cos(alpha)
    A = 1/6 - 1/4 * ca + 1/12 * ca**3
    B = 1/8 * ca - 1/8 * ca**3
    C = 7/48 - 1/16 * ca - 1/16 * ca**2 - 1/48 * ca**3
    return alpha, A, B, C


def sample_isotropic(N, rng):
    cos_theta = rng.uniform(-1.0, 1.0, N)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2 * np.pi, N)
    theta_folded = np.minimum(theta, np.pi - theta)
    return theta_folded, phi


def channels(theta, phi, Itot, A, B, C):
    s2 = np.sin(theta) ** 2
    common = A + B * s2
    C0 = Itot * (common + C * s2 * np.cos(2 * phi))
    C45 = Itot * (common + C * s2 * np.sin(2 * phi))
    C90 = Itot * (common - C * s2 * np.cos(2 * phi))
    C135 = Itot * (common - C * s2 * np.sin(2 * phi))
    return C0, C45, C90, C135


def r_formula(theta, A, B, C):
    s2 = np.sin(theta) ** 2
    return C * s2 / (A + B * s2)


# --------- Hugh's method reimplementation ---------
def phi_deg_from_xy(x, y):
    return np.degrees(np.mod(0.5 * np.arctan2(y, x), np.pi))


def gaussian_smooth_hist(counts, sigma_bins):
    sigma = float(max(0.0, sigma_bins))
    if sigma <= 0:
        return counts.copy()
    radius = max(1, int(round(4.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(counts, kernel, mode="same")


def fit_uniform_costheta_curve(r_vals):
    r = np.asarray(r_vals, dtype=np.float64)
    r = r[np.isfinite(r)]
    lo = float(np.percentile(r, HIST_LO_PCT))
    hi = float(np.percentile(r, HIST_HI_PCT))
    counts, edges = np.histogram(r, bins=HIST_BINS, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    smooth = np.maximum(gaussian_smooth_hist(counts, HIST_SIGMA_BINS), 0.0)
    area = np.sum(smooth * widths)
    if area > 0:
        smooth /= area
    cdf = np.clip(np.cumsum(smooth * widths), 0.0, 1.0)
    theta_deg = np.degrees(np.arccos(np.clip(1.0 - cdf, 0.0, 1.0)))
    theta_deg = np.maximum.accumulate(theta_deg)
    return centers, theta_deg


def interp_monotonic(r_centers, theta_deg, r_grid):
    if PchipInterpolator is not None:
        out = PchipInterpolator(r_centers, theta_deg, extrapolate=True)(r_grid)
    else:
        out = np.interp(r_grid, r_centers, theta_deg,
                        left=theta_deg[0], right=theta_deg[-1])
    out = np.maximum.accumulate(np.asarray(out, dtype=np.float64))
    return np.clip(out, 0.0, 90.0)


def balanced_r_sample(x, y, rng):
    r = np.sqrt(x * x + y * y)
    phi = phi_deg_from_xy(x, y)
    edges = np.arange(0.0, 180.0 + PHI_BIN_DEG, PHI_BIN_DEG)
    bin_id = np.digitize(phi, edges) - 1
    bin_id = np.clip(bin_id, 0, len(edges) - 2)
    idx_by_bin = [np.flatnonzero(bin_id == i) for i in range(len(edges) - 1)]
    counts = [int(idx.size) for idx in idx_by_bin]
    if min(counts) == 0:
        # skip empty phi bins
        idx_by_bin = [idx for idx, c in zip(idx_by_bin, counts) if c > 0]
        counts = [c for c in counts if c > 0]
    min_count = min(counts)
    n_per_bin = max(1, min_count // 2) if BALANCE_HALF_LEAST else min_count
    picks = [np.asarray(rng.choice(idx, size=n_per_bin, replace=False), dtype=np.int64)
             for idx in idx_by_bin]
    sel = np.concatenate(picks)
    return r[sel], sel


# --------- experiment ---------
def make_group_backgrounds(n_groups, base_scale, rng, enforce_sum_inv):
    """Return array shape (n_groups, 4) with columns [b0, b45, b90, b135]."""
    med = BG_FRAC_MED * base_scale
    b = med * rng.lognormal(mean=0.0, sigma=BG_LOG_SIGMA, size=(n_groups, 4))
    if enforce_sum_inv:
        # Adjust b45, b135 so that b45+b135 == b0+b90 while preserving b45-b135.
        sum0 = b[:, 0] + b[:, 2]
        diff_ac = b[:, 1] - b[:, 3]
        new_b45 = 0.5 * (sum0 + diff_ac)
        new_b135 = 0.5 * (sum0 - diff_ac)
        # if either would go negative, floor at zero
        floor = 0.0
        neg = (new_b45 < floor) | (new_b135 < floor)
        if np.any(neg):
            # simple fallback: split sum0 equally
            new_b45[neg] = 0.5 * sum0[neg]
            new_b135[neg] = 0.5 * sum0[neg]
        b[:, 1] = new_b45
        b[:, 3] = new_b135
    return b


def main():
    OUTDIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    alpha, A, B, C = fourkas_ABC(NA, N_MEDIUM)
    r_max_formula = float(r_formula(np.pi / 2, A, B, C))
    print(f"alpha={np.rad2deg(alpha):.3f} deg  A={A:.6g} B={B:.6g} C={C:.6g}  r_max={r_max_formula:.4f}")

    # 1) simulate a big pool of orientations
    theta, phi = sample_isotropic(N_TOTAL, rng)
    Itot = np.sin(theta) ** 2 if ITOT_MODE == "sin2theta" else np.ones_like(theta)
    C0_id, C45_id, C90_id, C135_id = channels(theta, phi, Itot, A, B, C)
    base_scale = float(np.max(C0_id))
    print(f"max C0_ideal = {base_scale:.4f}")

    # 2) split into groups
    idx_perm = rng.permutation(N_TOTAL)
    group_size = N_TOTAL // N_GROUPS
    group_id = np.repeat(np.arange(N_GROUPS), group_size)
    if group_id.size < N_TOTAL:
        group_id = np.concatenate([group_id,
                                   np.full(N_TOTAL - group_id.size, N_GROUPS - 1)])
    group_id_shuffled = np.empty(N_TOTAL, dtype=np.int64)
    group_id_shuffled[idx_perm] = group_id  # ensure random assignment

    # 3) draw backgrounds
    bg = make_group_backgrounds(N_GROUPS, base_scale, rng, ENFORCE_SUM_INV)
    print(f"background medians / max_C0 : "
          f"{[float(np.median(bg[:, k]) / base_scale) for k in range(4)]}")
    if ENFORCE_SUM_INV:
        chk = np.abs((bg[:, 0] + bg[:, 2]) - (bg[:, 1] + bg[:, 3])).max()
        print(f"max |b0+b90 - b45-b135| across groups after enforcement = {chk:.3e}")

    # 4) add group backgrounds
    b0 = bg[group_id_shuffled, 0]
    b45 = bg[group_id_shuffled, 1]
    b90 = bg[group_id_shuffled, 2]
    b135 = bg[group_id_shuffled, 3]
    C0 = C0_id + b0
    C45 = C45_id + b45
    C90 = C90_id + b90
    C135 = C135_id + b135

    x = (C0 - C90) / (C0 + C90)
    y = (C45 - C135) / (C45 + C135)
    r = np.sqrt(x * x + y * y)

    # 5) also do the "no per-group bg" reference (all groups share the same 15%)
    b_ref = BG_FRAC_MED * base_scale
    C0r, C45r, C90r, C135r = (C0_id + b_ref, C45_id + b_ref,
                              C90_id + b_ref, C135_id + b_ref)
    xr = (C0r - C90r) / (C0r + C90r)
    yr = (C45r - C135r) / (C45r + C135r)
    rr = np.sqrt(xr * xr + yr * yr)

    # 6) Hugh method on both
    sub_rng = np.random.default_rng(12345)
    r_bal, sel = balanced_r_sample(x, y, sub_rng)
    r_bal_ref, sel_ref = balanced_r_sample(xr, yr, np.random.default_rng(12345))

    r_centers_group, theta_group = fit_uniform_costheta_curve(r_bal)
    r_centers_ref, theta_ref = fit_uniform_costheta_curve(r_bal_ref)

    r_grid = np.linspace(0.0, min(1.0, r_max_formula), 600)
    theta_group_i = interp_monotonic(r_centers_group, theta_group, r_grid)
    theta_ref_i = interp_monotonic(r_centers_ref, theta_ref, r_grid)

    # analytic Fourkas
    theta_line = np.linspace(0, np.pi / 2, 800)
    r_line = r_formula(theta_line, A, B, C)

    # Hugh's real 25 nm curve, if available
    hugh_csv = OUTDIR.parent / "theta_r_curve_25x65nm_recording_bootstrap.csv"
    hugh_curve = None
    if hugh_csv.exists():
        import csv as _csv
        rows = list(_csv.DictReader(hugh_csv.open("r", encoding="utf-8")))
        hugh_curve = (
            np.asarray([float(row["r"]) for row in rows]),
            np.asarray([float(row["theta_deg_center"]) for row in rows]),
        )

    # ----- Plot 1: theta(r) comparison -----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(r_line, np.rad2deg(theta_line), lw=2.2, color="k",
            label="analytic Fourkas (no bg)")
    ax.plot(r_grid, theta_ref_i, lw=2.0, color="tab:blue",
            label=f"sim, uniform {int(BG_FRAC_MED*100)}% bg (Hugh method)")
    ax.plot(r_grid, theta_group_i, lw=2.0, color="tab:red",
            label=f"sim, {N_GROUPS} groups w/ per-group bg (Hugh method)")
    if hugh_curve is not None:
        ax.plot(hugh_curve[0], hugh_curve[1], lw=2.2, color="tab:green", ls="--",
                label="Hugh empirical 25x65nm")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title(
        f"theta(r): per-recording background heterogeneity vs Hugh empirical\n"
        f"Itot={ITOT_MODE}, median bg = {int(BG_FRAC_MED*100)}% of max C0, "
        f"lognormal sigma={BG_LOG_SIGMA}, sum-inv enforced={ENFORCE_SUM_INV}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "theta_r_group_background_vs_Hugh.png", dpi=190)
    plt.close(fig)

    # ----- Plot 2: pooled r-distribution -----
    r_bins = np.linspace(0, 1.0, 120)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rr, bins=r_bins, density=True, histtype="step", lw=1.7,
            color="tab:blue", label="uniform 15% bg")
    ax.hist(r, bins=r_bins, density=True, histtype="step", lw=1.7,
            color="tab:red", label=f"per-group bg (N={N_GROUPS})")
    ax.axvline(r_max_formula, ls="--", c="k", lw=0.8,
               label=f"Fourkas r_max={r_max_formula:.3f}")
    ax.set_xlabel("r")
    ax.set_ylabel("density")
    ax.set_title("Pooled p(r) simulated")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "pooled_r_density.png", dpi=190)
    plt.close(fig)

    # ----- Plot 3: per-recording r-density spread -----
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in range(N_GROUPS):
        mask = group_id_shuffled == g
        if mask.sum() < 200:
            continue
        h, _ = np.histogram(r[mask], bins=r_bins, density=True)
        ax.plot(0.5 * (r_bins[:-1] + r_bins[1:]), h, color="grey",
                lw=0.4, alpha=0.4)
    h_pool, _ = np.histogram(r, bins=r_bins, density=True)
    ax.plot(0.5 * (r_bins[:-1] + r_bins[1:]), h_pool, color="tab:red",
            lw=2.2, label="pooled")
    ax.axvline(r_max_formula, ls="--", c="k", lw=0.8)
    ax.set_xlabel("r")
    ax.set_ylabel("density")
    ax.set_title(f"Per-'recording' r densities (grey), pooled (red)  "
                 f"[per-group bg, sum-inv={ENFORCE_SUM_INV}]")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "per_group_r_density_overlay.png", dpi=190)
    plt.close(fig)

    print(f"\nDone. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
