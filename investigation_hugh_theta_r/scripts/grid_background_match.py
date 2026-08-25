"""
Grid search over per-recording background parameters to best match Hugh's
empirical theta(r) curve (25x65nm).

We vary
    BG_FRAC_MED  in a small list
    BG_LOG_SIGMA in a small list
For each combination, simulate isotropic tumbling with per-group backgrounds
(sum-invariance enforced, as in simulate_grouped_background.py), pool the
points, run Hugh's method, and compute RMS distance to his empirical curve
over the overlapping r-range.

Outputs
    grid_summary.csv                          (all combinations sorted by RMSE)
    theta_r_all_combinations.png              (all curves overlaid vs Hugh)
    theta_r_best_matches.png                  (top 5 combinations, close-up)
"""

from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None

# ---------- config ----------
N_TOTAL = 2_000_000     # smaller than the single-run script for speed
N_GROUPS = 115
NA = 1.3
N_MEDIUM = 1.5
ITOT_MODE = "sin2theta"
ENFORCE_SUM_INV = True
RANDOM_SEED = 20260714

BG_FRAC_MED_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
BG_LOG_SIGMA_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]

PHI_BIN_DEG = 5.0
HIST_BINS = 160
HIST_SIGMA_BINS = 3.0
HIST_LO_PCT = 0.5
HIST_HI_PCT = 99.5

HUGH_CSV = Path(__file__).resolve().parent.parent / "theta_r_curve_25x65nm_recording_bootstrap.csv"
OUTDIR = Path(__file__).resolve().parent.parent / "fourkas_group_background_grid"


# ---------- Fourkas ----------
def fourkas_ABC(NA, n):
    alpha = np.arcsin(NA / n)
    ca = np.cos(alpha)
    A = 1 / 6 - 1 / 4 * ca + 1 / 12 * ca**3
    B = 1 / 8 * ca - 1 / 8 * ca**3
    C = 7 / 48 - 1 / 16 * ca - 1 / 16 * ca**2 - 1 / 48 * ca**3
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


# ---------- Hugh's method ----------
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
    idx_by_bin = [idx for idx, c in zip(idx_by_bin, counts) if c > 0]
    counts = [c for c in counts if c > 0]
    min_count = min(counts)
    n_per_bin = max(1, min_count // 2)
    picks = [np.asarray(rng.choice(idx, size=n_per_bin, replace=False), dtype=np.int64)
             for idx in idx_by_bin]
    return r[np.concatenate(picks)]


def make_group_backgrounds(n_groups, base_scale, frac_med, log_sigma, rng, enforce_sum_inv):
    med = frac_med * base_scale
    b = med * rng.lognormal(mean=0.0, sigma=log_sigma, size=(n_groups, 4))
    if enforce_sum_inv:
        sum0 = b[:, 0] + b[:, 2]
        diff_ac = b[:, 1] - b[:, 3]
        new_b45 = 0.5 * (sum0 + diff_ac)
        new_b135 = 0.5 * (sum0 - diff_ac)
        neg = (new_b45 < 0.0) | (new_b135 < 0.0)
        if np.any(neg):
            new_b45[neg] = 0.5 * sum0[neg]
            new_b135[neg] = 0.5 * sum0[neg]
        b[:, 1] = new_b45
        b[:, 3] = new_b135
    return b


# ---------- experiment ----------
def load_hugh_curve(csv_path):
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    r = np.asarray([float(row["r"]) for row in rows], dtype=np.float64)
    th = np.asarray([float(row["theta_deg_center"]) for row in rows], dtype=np.float64)
    return r, th


def simulate_one(theta, phi, C0_id, C45_id, C90_id, C135_id, base_scale, frac_med, log_sigma, rng):
    bg = make_group_backgrounds(N_GROUPS, base_scale, frac_med, log_sigma, rng, ENFORCE_SUM_INV)
    n = theta.size
    # group assignment: reshuffle indices so each group is a random subset
    perm = rng.permutation(n)
    group_id = np.empty(n, dtype=np.int64)
    per_group = n // N_GROUPS
    for g in range(N_GROUPS):
        start = g * per_group
        end = (g + 1) * per_group if g < N_GROUPS - 1 else n
        group_id[perm[start:end]] = g
    b0 = bg[group_id, 0]
    b45 = bg[group_id, 1]
    b90 = bg[group_id, 2]
    b135 = bg[group_id, 3]
    C0 = C0_id + b0
    C45 = C45_id + b45
    C90 = C90_id + b90
    C135 = C135_id + b135
    x = (C0 - C90) / (C0 + C90)
    y = (C45 - C135) / (C45 + C135)
    r_bal = balanced_r_sample(x, y, rng)
    centers, theta_deg = fit_uniform_costheta_curve(r_bal)
    return centers, theta_deg, np.sqrt(x * x + y * y)


def compute_rmse(sim_r, sim_theta, hugh_r, hugh_theta):
    lo = max(sim_r.min(), hugh_r.min())
    hi = min(sim_r.max(), hugh_r.max())
    if hi <= lo:
        return np.nan, None
    r_grid = np.linspace(lo, hi, 400)
    sim_i = interp_monotonic(sim_r, sim_theta, r_grid)
    hugh_i = np.interp(r_grid, hugh_r, hugh_theta)
    rmse = float(np.sqrt(np.mean((sim_i - hugh_i) ** 2)))
    return rmse, r_grid


def main():
    OUTDIR.mkdir(exist_ok=True)
    hugh = load_hugh_curve(HUGH_CSV)
    if hugh is None:
        raise SystemExit(f"Missing Hugh CSV: {HUGH_CSV}")
    hugh_r, hugh_theta = hugh

    rng = np.random.default_rng(RANDOM_SEED)
    alpha, A, B, C = fourkas_ABC(NA, N_MEDIUM)

    theta, phi = sample_isotropic(N_TOTAL, rng)
    Itot = np.sin(theta) ** 2 if ITOT_MODE == "sin2theta" else np.ones_like(theta)
    C0_id, C45_id, C90_id, C135_id = channels(theta, phi, Itot, A, B, C)
    base_scale = float(np.max(C0_id))
    print(f"alpha={np.rad2deg(alpha):.2f} deg  A={A:.6g} B={B:.6g} C={C:.6g}  max_C0_ideal={base_scale:.4f}")
    print(f"Hugh empirical r range = [{hugh_r.min():.3f}, {hugh_r.max():.3f}]  "
          f"theta range = [{hugh_theta.min():.2f}, {hugh_theta.max():.2f}] deg")
    print(f"Grid: {len(BG_FRAC_MED_GRID)} * {len(BG_LOG_SIGMA_GRID)} = "
          f"{len(BG_FRAC_MED_GRID)*len(BG_LOG_SIGMA_GRID)} combinations, N_TOTAL={N_TOTAL:,}, N_GROUPS={N_GROUPS}\n")

    results = []
    curves = {}
    for frac_med, log_sigma in product(BG_FRAC_MED_GRID, BG_LOG_SIGMA_GRID):
        local_rng = np.random.default_rng(RANDOM_SEED + int(frac_med * 1000) * 100 + int(log_sigma * 100))
        centers, theta_deg, r_all = simulate_one(
            theta, phi, C0_id, C45_id, C90_id, C135_id,
            base_scale, frac_med, log_sigma, local_rng,
        )
        rmse, r_grid = compute_rmse(centers, theta_deg, hugh_r, hugh_theta)
        sim_r_max = float(centers.max())
        sim_r_mean = float(np.mean(r_all))
        results.append({
            "bg_frac_med": frac_med,
            "bg_log_sigma": log_sigma,
            "rmse_deg": rmse,
            "sim_r_max_p99p5": sim_r_max,
            "sim_r_mean": sim_r_mean,
        })
        curves[(frac_med, log_sigma)] = (centers, theta_deg)
        print(f"  bg_med={frac_med:.2f}  log_sigma={log_sigma:.2f}  "
              f"rmse={rmse:.2f} deg  sim_r_max={sim_r_max:.3f}  sim_r_mean={sim_r_mean:.3f}")

    # sort and save
    results.sort(key=lambda d: d["rmse_deg"] if np.isfinite(d["rmse_deg"]) else np.inf)
    with (OUTDIR / "grid_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    # ----- overlay all -----
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    n_combos = len(results)
    for i, res in enumerate(reversed(results)):
        c, th = curves[(res["bg_frac_med"], res["bg_log_sigma"])]
        col = cmap(i / max(1, n_combos - 1))
        ax.plot(c, th, lw=0.8, color=col, alpha=0.55)
    ax.plot(hugh_r, hugh_theta, lw=2.6, color="k", label="Hugh 25x65nm empirical")
    # analytic
    theta_line = np.linspace(0, np.pi / 2, 500)
    ax.plot(r_formula(theta_line, A, B, C), np.rad2deg(theta_line),
            lw=2.0, ls="--", color="tab:red", label="analytic Fourkas (no bg)")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title(f"theta(r) across all grid points (dark=high RMSE, light=low RMSE)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "theta_r_all_combinations.png", dpi=190)
    plt.close(fig)

    # ----- top 5 -----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(hugh_r, hugh_theta, lw=3.0, color="k", label="Hugh 25x65nm empirical")
    top = results[:5]
    for res in top:
        c, th = curves[(res["bg_frac_med"], res["bg_log_sigma"])]
        ax.plot(c, th, lw=1.8,
                label=f"bg_med={res['bg_frac_med']:.2f} "
                      f"log_sig={res['bg_log_sigma']:.2f}  "
                      f"RMSE={res['rmse_deg']:.2f}°")
    ax.plot(r_formula(theta_line, A, B, C), np.rad2deg(theta_line),
            lw=1.6, ls="--", color="grey", label="analytic Fourkas")
    ax.set_xlabel("r")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Top 5 grid points closest to Hugh's empirical theta(r)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "theta_r_best_matches.png", dpi=190)
    plt.close(fig)

    # ----- heatmap of RMSE -----
    frac_grid = np.asarray(BG_FRAC_MED_GRID)
    sig_grid = np.asarray(BG_LOG_SIGMA_GRID)
    rmse_mat = np.full((len(frac_grid), len(sig_grid)), np.nan)
    for res in results:
        i = list(frac_grid).index(res["bg_frac_med"])
        j = list(sig_grid).index(res["bg_log_sigma"])
        rmse_mat[i, j] = res["rmse_deg"]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rmse_mat, origin="lower", aspect="auto",
                   extent=(sig_grid.min() - 0.05, sig_grid.max() + 0.05,
                           frac_grid.min() - 0.02, frac_grid.max() + 0.02),
                   cmap="magma_r")
    for i, f in enumerate(frac_grid):
        for j, s in enumerate(sig_grid):
            v = rmse_mat[i, j]
            if np.isfinite(v):
                ax.text(s, f, f"{v:.1f}", ha="center", va="center",
                        color="white" if v > np.nanmedian(rmse_mat) else "black",
                        fontsize=8)
    ax.set_xlabel("BG_LOG_SIGMA")
    ax.set_ylabel("BG_FRAC_MED")
    ax.set_title("RMSE to Hugh empirical theta(r), deg")
    fig.colorbar(im, ax=ax, label="RMSE (deg)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "rmse_heatmap.png", dpi=190)
    plt.close(fig)

    print("\nTop 5 by RMSE:")
    for res in top:
        print(f"  bg_med={res['bg_frac_med']:.2f}  log_sigma={res['bg_log_sigma']:.2f}  "
              f"RMSE={res['rmse_deg']:.2f} deg  sim_r_max={res['sim_r_max_p99p5']:.3f}  "
              f"sim_r_mean={res['sim_r_mean']:.3f}")
    print(f"\nOutputs in {OUTDIR}")


if __name__ == "__main__":
    main()
