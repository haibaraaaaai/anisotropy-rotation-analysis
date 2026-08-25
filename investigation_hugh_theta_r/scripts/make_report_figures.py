"""
Presentation-quality figures for the Fourkas / Hugh investigation report.

Writes all figures into ../report/figs/. Uses the artifacts produced by:
    scripts/xy_step_vs_range.py
    scripts/per_recording_diagnostics.py          (used for per-recording CSV)
    scripts/test_sum_invariance.py                (used for per-rod CSV)
    scripts/simulate_grouped_background.py        (regenerates grouped sim)
    scripts/grid_background_match.py              (regenerates grid summary)

Regenerates the underlying quick data only if the cached CSVs are missing.
"""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None

# ---------- global style ----------
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.constrained_layout.use": False,
})

HUGH_ROOT = Path(__file__).resolve().parent.parent
REPORT = HUGH_ROOT / "report"
FIGDIR = REPORT / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

C_FOURKAS = "#1b1b1b"
C_UNIFORM_BG = "#1f77b4"
C_GROUP_BG = "#d62728"
C_HUGH = "#2ca02c"
C_ACCENT = "#ff7f0e"


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


def gaussian_smooth(counts, sigma_bins=3.0):
    sigma = float(max(0.0, sigma_bins))
    if sigma <= 0:
        return counts.copy()
    radius = max(1, int(round(4.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(counts, kernel, mode="same")


def hugh_theta_of_r(r_vals, bins=160, lo_pct=0.5, hi_pct=99.5, sigma_bins=3.0):
    r = np.asarray(r_vals, dtype=np.float64)
    r = r[np.isfinite(r)]
    lo = float(np.percentile(r, lo_pct))
    hi = float(np.percentile(r, hi_pct))
    counts, edges = np.histogram(r, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    smooth = np.maximum(gaussian_smooth(counts, sigma_bins), 0.0)
    area = np.sum(smooth * widths)
    if area > 0:
        smooth /= area
    cdf = np.clip(np.cumsum(smooth * widths), 0.0, 1.0)
    theta_deg = np.degrees(np.arccos(np.clip(1.0 - cdf, 0.0, 1.0)))
    theta_deg = np.maximum.accumulate(theta_deg)
    return centers, theta_deg, smooth, edges


def balanced_r_sample(x, y, rng, phi_bin_deg=5.0):
    r = np.sqrt(x * x + y * y)
    phi = phi_deg_from_xy(x, y)
    edges = np.arange(0.0, 180.0 + phi_bin_deg, phi_bin_deg)
    bin_id = np.digitize(phi, edges) - 1
    bin_id = np.clip(bin_id, 0, len(edges) - 2)
    idx_by_bin = [np.flatnonzero(bin_id == i) for i in range(len(edges) - 1)]
    idx_by_bin = [ib for ib in idx_by_bin if ib.size > 0]
    counts = [ib.size for ib in idx_by_bin]
    n_per_bin = max(1, min(counts) // 2)
    picks = [rng.choice(ib, size=n_per_bin, replace=False) for ib in idx_by_bin]
    return r[np.concatenate(picks)]


def make_group_backgrounds(n_groups, base_scale, frac_med, log_sigma, rng, enforce=True):
    med = frac_med * base_scale
    b = med * rng.lognormal(mean=0.0, sigma=log_sigma, size=(n_groups, 4))
    if enforce:
        sum0 = b[:, 0] + b[:, 2]
        diff_ac = b[:, 1] - b[:, 3]
        new_b45 = 0.5 * (sum0 + diff_ac)
        new_b135 = 0.5 * (sum0 - diff_ac)
        neg = (new_b45 < 0) | (new_b135 < 0)
        if np.any(neg):
            new_b45[neg] = 0.5 * sum0[neg]
            new_b135[neg] = 0.5 * sum0[neg]
        b[:, 1] = new_b45
        b[:, 3] = new_b135
    return b


def simulate_pool(N_TOTAL, N_GROUPS, frac_med, log_sigma, rng,
                  NA=1.3, N_MED=1.5, itot_mode="sin2theta"):
    _, A, B, C = fourkas_ABC(NA, N_MED)
    theta, phi = sample_isotropic(N_TOTAL, rng)
    Itot = np.sin(theta) ** 2 if itot_mode == "sin2theta" else np.ones_like(theta)
    C0_id, C45_id, C90_id, C135_id = channels(theta, phi, Itot, A, B, C)
    base_scale = float(np.max(C0_id))
    if N_GROUPS > 1:
        bg = make_group_backgrounds(N_GROUPS, base_scale, frac_med, log_sigma, rng)
        n = theta.size
        perm = rng.permutation(n)
        group_id = np.empty(n, dtype=np.int64)
        per = n // N_GROUPS
        for g in range(N_GROUPS):
            s = g * per
            e = (g + 1) * per if g < N_GROUPS - 1 else n
            group_id[perm[s:e]] = g
        b0 = bg[group_id, 0]; b45 = bg[group_id, 1]
        b90 = bg[group_id, 2]; b135 = bg[group_id, 3]
    else:
        b = frac_med * base_scale
        b0 = b45 = b90 = b135 = b
        group_id = np.zeros(theta.size, dtype=np.int64)
    C0 = C0_id + b0; C45 = C45_id + b45
    C90 = C90_id + b90; C135 = C135_id + b135
    x = (C0 - C90) / (C0 + C90)
    y = (C45 - C135) / (C45 + C135)
    return {
        "A": A, "B": B, "C": C, "theta": theta, "phi": phi,
        "x": x, "y": y, "r": np.sqrt(x * x + y * y),
        "group_id": group_id, "base_scale": base_scale,
    }


# ---------- data loaders ----------
def load_hugh_curves():
    curves = {}
    for tag, fname in (("25x65nm", "theta_r_curve_25x65nm_recording_bootstrap.csv"),
                       ("40x65nm", "theta_r_curve_40x65nm_recording_bootstrap.csv")):
        p = HUGH_ROOT / fname
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open("r", encoding="utf-8")))
        curves[tag] = {
            "r": np.asarray([float(r["r"]) for r in rows]),
            "theta": np.asarray([float(r["theta_deg_center"]) for r in rows]),
            "lo": np.asarray([float(r["theta_deg_lo_1sigma"]) for r in rows]),
            "hi": np.asarray([float(r["theta_deg_hi_1sigma"]) for r in rows]),
        }
    return curves


def load_recordings(root: Path, min_frames=128, filter_range=True):
    out = []
    for source in ("pending", "good", "bad"):
        d = root / source
        if not d.exists():
            continue
        for rod in sorted([p for p in d.iterdir() if p.is_dir()]):
            meta = rod / "capture_maxfps_15x15_meta.json"
            if not meta.exists():
                continue
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            xy = m.get("xy_series")
            if not isinstance(xy, list) or len(xy) < min_frames:
                continue
            xy = np.asarray(xy, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            good = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            xy = xy[good]
            if xy.shape[0] < min_frames:
                continue
            metrics = m.get("xy_metrics") or {}
            rx = float(metrics.get("range_x", 0.0))
            ry = float(metrics.get("range_y", 0.0))
            if filter_range and not (rx > 1.0 and ry > 1.0):
                continue
            out.append({"source": source, "rod": rod.name,
                        "x": xy[:, 0], "y": xy[:, 1],
                        "range_x": rx, "range_y": ry})
    return out


def autocorr_e_lag(sig, max_lag=400):
    s = sig - sig.mean()
    var = np.dot(s, s) / s.size
    if var <= 0:
        return np.nan
    prev = 1.0
    for k in range(1, min(max_lag, s.size - 1) + 1):
        c = np.dot(s[:-k], s[k:]) / (s.size - k) / var
        if c < np.exp(-1.0):
            if c == prev:
                return float(k)
            frac = (np.exp(-1.0) - prev) / (c - prev)
            return float((k - 1) + frac)
        prev = c
    return np.nan


# ==================================================================
# Figure 1: The problem
# ==================================================================
def figure_problem(curves):
    print("Figure 1: problem statement")
    NA, N_MED = 1.3, 1.5
    _, A, B, C = fourkas_ABC(NA, N_MED)

    fig = plt.figure(figsize=(11.5, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28, left=0.07, right=0.98, top=0.9, bottom=0.14)

    ax1 = fig.add_subplot(gs[0])
    theta_line = np.linspace(0, np.pi / 2, 500)
    r_line = r_formula(theta_line, A, B, C)
    ax1.plot(r_line, np.rad2deg(theta_line), lw=2.8, color=C_FOURKAS,
             label="Analytic Fourkas (isotropic, no bg)")
    if "25x65nm" in curves:
        c = curves["25x65nm"]
        ax1.fill_between(c["r"], c["lo"], c["hi"], color=C_HUGH, alpha=0.18)
        ax1.plot(c["r"], c["theta"], lw=2.3, color=C_HUGH,
                 label="Empirical (25×65 nm rods, tumbling)")
    if "40x65nm" in curves:
        c = curves["40x65nm"]
        ax1.plot(c["r"], c["theta"], lw=2.0, color="#8B0000", ls="--",
                 label="Empirical (40×65 nm rods, tumbling)")
    ax1.set_xlabel("Anisotropy magnitude  $r = \\sqrt{x^2 + y^2}$")
    ax1.set_ylabel(r"Folded polar angle  $\theta$  (deg)")
    ax1.set_title("Theory vs. experiment: $\\theta(r)$ for isotropically tumbling rods")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 92)
    ax1.legend(loc="upper left", frameon=False)

    # Show sample p(r): analytic (Jacobian-piled) vs empirical smooth bell
    ax2 = fig.add_subplot(gs[1])
    rng = np.random.default_rng(1)
    sim = simulate_pool(500_000, 1, 0.0, 0.0, rng)  # no bg, no groups
    r_bins = np.linspace(0, 1, 100)
    ax2.hist(sim["r"], bins=r_bins, density=True, histtype="stepfilled",
             color=C_FOURKAS, alpha=0.75, label="Fourkas simulation, no bg")
    ax2.axvline(r_formula(np.pi / 2, A, B, C), ls="--", color=C_FOURKAS,
                lw=1.0, alpha=0.9)
    # For "empirical" p(r) representative, pool tumbling 25nm points
    root25 = HUGH_ROOT / "tumbling 25nm glycerol"
    recs = load_recordings(root25, filter_range=True)
    if recs:
        r_all = np.concatenate([np.sqrt(rec["x"] ** 2 + rec["y"] ** 2) for rec in recs])
        ax2.hist(r_all, bins=r_bins, density=True, histtype="step",
                 lw=2.5, color=C_HUGH, label="Empirical 25×65 nm pooled")
    ax2.set_xlabel("r")
    ax2.set_ylabel("density")
    ax2.set_title("Where the disagreement lives: pooled $p(r)$")
    ax2.set_xlim(0, 1)
    ax2.legend(loc="upper left", frameon=False)

    fig.savefig(FIGDIR / "fig1_problem.png")
    plt.close(fig)


# ==================================================================
# Figure 2: What Hugh's method is (CDF-inversion recap)
# ==================================================================
def figure_method(curves):
    print("Figure 2: Hugh's method schematic")
    NA, N_MED = 1.3, 1.5
    _, A, B, C = fourkas_ABC(NA, N_MED)

    root25 = HUGH_ROOT / "tumbling 25nm glycerol"
    recs = load_recordings(root25, filter_range=True)
    r_all = np.concatenate([np.sqrt(rec["x"] ** 2 + rec["y"] ** 2) for rec in recs])

    fig = plt.figure(figsize=(12.5, 4.4))
    gs = GridSpec(1, 3, wspace=0.32, left=0.06, right=0.985, top=0.9, bottom=0.16)

    r_bins = np.linspace(np.percentile(r_all, 0.5), np.percentile(r_all, 99.5), 160)
    counts, edges = np.histogram(r_all, bins=r_bins, density=True)
    smooth = gaussian_smooth(counts, 3.0)
    smooth /= np.sum(smooth * np.diff(edges))
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.clip(np.cumsum(smooth * np.diff(edges)), 0.0, 1.0)
    theta_deg = np.degrees(np.arccos(np.clip(1.0 - cdf, 0.0, 1.0)))
    theta_deg = np.maximum.accumulate(theta_deg)

    ax = fig.add_subplot(gs[0])
    ax.plot(centers, smooth, lw=2.2, color=C_HUGH)
    ax.set_xlabel("r"); ax.set_ylabel("density")
    ax.set_title("(1)   Smoothed pooled $p(r)$")
    ax.set_xlim(centers.min(), centers.max())

    ax = fig.add_subplot(gs[1])
    ax.plot(centers, cdf, lw=2.2, color=C_HUGH)
    ax.set_xlabel("r"); ax.set_ylabel(r"CDF$_r(r)$")
    ax.set_title(r"(2)   CDF (assumed = $1 - \cos\theta$)")

    ax = fig.add_subplot(gs[2])
    ax.plot(centers, theta_deg, lw=2.4, color=C_HUGH, label="Hugh inversion")
    theta_line = np.linspace(0, np.pi / 2, 500)
    ax.plot(r_formula(theta_line, A, B, C), np.rad2deg(theta_line),
            lw=2.0, color=C_FOURKAS, ls="--", label="Analytic Fourkas")
    ax.set_xlabel("r"); ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title(r"(3)   $\theta(r) = \arccos(1-\mathrm{CDF}_r)$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 92)
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle("Hugh's method is a CDF-match: the output shape is set entirely by the input $p(r)$",
                 y=0.995, fontsize=12.5)
    fig.savefig(FIGDIR / "fig2_method.png")
    plt.close(fig)


# ==================================================================
# Figure 3: Sampling & pooling diagnostics
# ==================================================================
def figure_pooling(curves):
    print("Figure 3: pooling diagnostics")
    root25 = HUGH_ROOT / "tumbling 25nm glycerol"
    root40 = HUGH_ROOT / "glycerol suspended rods 17062026"

    fig = plt.figure(figsize=(13.5, 8.4))
    gs = GridSpec(2, 2, wspace=0.28, hspace=0.55, left=0.07, right=0.98, top=0.92, bottom=0.08)

    for row, (name, root) in enumerate((("25×65 nm", root25), ("40×65 nm", root40))):
        recs = load_recordings(root, filter_range=True)
        if not recs:
            continue
        # per-recording densities
        r_bins = np.linspace(0, 1, 60)
        centers_r = 0.5 * (r_bins[:-1] + r_bins[1:])
        ax = fig.add_subplot(gs[row, 0])
        # compute ac lag and stratify
        ac_lags = []
        r_lists = []
        for rec in recs:
            r = np.sqrt(rec["x"] ** 2 + rec["y"] ** 2)
            r_lists.append(r)
            ac_lags.append(autocorr_e_lag(rec["x"]))
        ac_arr = np.array(ac_lags, dtype=float)
        for r in r_lists:
            h, _ = np.histogram(r, bins=r_bins, density=True)
            ax.plot(centers_r, h, color="grey", lw=0.4, alpha=0.35)
        pool = np.concatenate(r_lists)
        h_pool, _ = np.histogram(pool, bins=r_bins, density=True)
        ax.plot(centers_r, h_pool, color="k", lw=2.4, label="Pooled")
        ax.set_xlabel("r"); ax.set_ylabel("density")
        ax.set_title(f"{name}:  per-recording $p(r)$ (grey, N={len(recs)}) vs. pooled")
        ax.legend(frameon=False, loc="upper left")
        ax.set_xlim(0, 1)

        ax = fig.add_subplot(gs[row, 1])
        finite = np.isfinite(ac_arr)
        q1, q2 = np.percentile(ac_arr[finite], [33.3, 66.7])
        colors = ["#1f77b4", "#ff7f0e", "#d62728"]
        labels = [f"fast rotators  (lag $\\leq$ {q1:.0f})",
                  f"mid rotators  ({q1:.0f} $<$ lag $\\leq$ {q2:.0f})",
                  f"slow rotators  (lag $>$ {q2:.0f})"]
        splits = [ac_arr <= q1, (ac_arr > q1) & (ac_arr <= q2), ac_arr > q2]
        for mask, lab, c in zip(splits, labels, colors):
            r_t = np.concatenate([r_lists[i] for i in np.where(mask & finite)[0]])
            h, _ = np.histogram(r_t, bins=r_bins, density=True)
            ax.plot(centers_r, h, color=c, lw=2.0,
                    label=f"{lab}  (n={int(mask.sum())})")
        ax.plot(centers_r, h_pool, color="k", lw=1.6, ls="--", label="pooled")
        ax.set_xlabel("r"); ax.set_ylabel("density")
        ax.set_title(f"{name}:  $p(r)$ stratified by autocorrelation-1/e lag")
        ax.legend(frameon=False, loc="upper right", fontsize=8.5)
        ax.set_xlim(0, 1)

    fig.suptitle("Per-recording heterogeneity: each rod's $p(r)$ looks different, and depends on rotational speed",
                 y=0.98, fontsize=13)
    fig.savefig(FIGDIR / "fig3_pooling.png")
    plt.close(fig)


# ==================================================================
# Figure 4: Sum invariance check
# ==================================================================
def figure_sum_invariance():
    print("Figure 4: sum invariance")
    csv_path = HUGH_ROOT / "sum_invariance_output" / "sum_invariance_per_rod.csv"
    if not csv_path.exists():
        print("  skip: sum_invariance_per_rod.csv missing (run test_sum_invariance.py first)")
        return
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    dsets = {}
    for r in rows:
        dsets.setdefault(r["dataset"], []).append(r)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(rows))
    ratios = np.array([float(r["whole_ratio_med"]) for r in rows])
    colors = ["#1f77b4" if r["dataset"] == "25nm" else "#d62728" for r in rows]
    bars = ax.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_ylim(0.93, 1.02)
    ax.set_ylabel(r"$\mathrm{median}\ (C_0+C_{90})\,/\,(C_{45}+C_{135})$")
    ax.set_xlabel("rod index")
    ax.set_title("Sum invariance $C_0+C_{90}=C_{45}+C_{135}$ holds to $\\sim$1–3%\n"
                 "(bias is systematic per rod → channel gain offset, not stochastic)")
    ax.set_xticks(x)
    labels = [f"{r['dataset']}-{i}" for i, r in enumerate(rows)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="25×65 nm"),
                       Patch(color="#d62728", label="40×65 nm")],
              frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_sum_invariance.png")
    plt.close(fig)


# ==================================================================
# Figure 5: The per-group background simulation reproduces Hugh
# ==================================================================
def figure_grouped_sim(curves):
    print("Figure 5: grouped-background simulation")
    NA, N_MED = 1.3, 1.5
    _, A, B, C = fourkas_ABC(NA, N_MED)

    # Uniform background reference (bg = 15% single value)
    rng = np.random.default_rng(2)
    sim_uniform = simulate_pool(2_000_000, 1, 0.15, 0.0, rng)
    r_bal_u = balanced_r_sample(sim_uniform["x"], sim_uniform["y"],
                                 np.random.default_rng(11))
    ctr_u, th_u, _, _ = hugh_theta_of_r(r_bal_u)

    # Per-group background (using best-fit-ish parameters)
    rng2 = np.random.default_rng(3)
    sim_group = simulate_pool(2_000_000, 115, 0.30, 0.40, rng2)
    r_bal_g = balanced_r_sample(sim_group["x"], sim_group["y"],
                                 np.random.default_rng(13))
    ctr_g, th_g, _, _ = hugh_theta_of_r(r_bal_g)

    theta_line = np.linspace(0, np.pi / 2, 500)
    r_line = r_formula(theta_line, A, B, C)

    fig = plt.figure(figsize=(12.5, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.2], wspace=0.28,
                  left=0.06, right=0.98, top=0.9, bottom=0.14)

    ax = fig.add_subplot(gs[0])
    r_bins = np.linspace(0, 1, 100)
    ax.hist(sim_uniform["r"], bins=r_bins, density=True, histtype="step",
            lw=2.2, color=C_UNIFORM_BG, label="Fourkas + uniform 15% bg")
    ax.hist(sim_group["r"], bins=r_bins, density=True, histtype="step",
            lw=2.2, color=C_GROUP_BG,
            label="Fourkas + per-recording bg\n(median 30%, log-σ=0.40)")
    ax.axvline(r_formula(np.pi / 2, A, B, C), color="k", ls="--", lw=1.0)
    ax.set_xlabel("r"); ax.set_ylabel("density")
    ax.set_title("Pooled simulated $p(r)$")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1])
    ax.plot(r_line, np.rad2deg(theta_line), lw=2.4, color=C_FOURKAS,
            label="Analytic Fourkas")
    ax.plot(ctr_u, th_u, lw=2.2, color=C_UNIFORM_BG,
            label="Sim (uniform 15% bg) → Hugh method")
    ax.plot(ctr_g, th_g, lw=2.2, color=C_GROUP_BG,
            label="Sim (per-recording bg) → Hugh method")
    if "25x65nm" in curves:
        c = curves["25x65nm"]
        ax.fill_between(c["r"], c["lo"], c["hi"], color=C_HUGH, alpha=0.2)
        ax.plot(c["r"], c["theta"], lw=2.6, color=C_HUGH,
                label="Empirical 25×65 nm")
    ax.set_xlabel("r"); ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title(r"$\theta(r)$: adding per-recording bg heterogeneity reproduces the data")
    ax.set_xlim(0, 1); ax.set_ylim(0, 92)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    fig.savefig(FIGDIR / "fig5_grouped_sim.png")
    plt.close(fig)


# ==================================================================
# Figure 6: Grid search
# ==================================================================
def figure_grid(curves):
    print("Figure 6: grid search")
    csv_path = HUGH_ROOT / "fourkas_group_background_grid" / "grid_summary.csv"
    if not csv_path.exists():
        print("  skip: grid_summary.csv missing (run grid_background_match.py first)")
        return
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    frac_vals = sorted({float(r["bg_frac_med"]) for r in rows})
    sig_vals = sorted({float(r["bg_log_sigma"]) for r in rows})
    rmse_mat = np.full((len(frac_vals), len(sig_vals)), np.nan)
    for r in rows:
        i = frac_vals.index(float(r["bg_frac_med"]))
        j = sig_vals.index(float(r["bg_log_sigma"]))
        rmse_mat[i, j] = float(r["rmse_deg"])

    # top 5 combos: regenerate curves quickly with a modest N to plot
    top = sorted(rows, key=lambda r: float(r["rmse_deg"]))[:5]

    NA, N_MED = 1.3, 1.5
    _, A, B, C = fourkas_ABC(NA, N_MED)

    fig = plt.figure(figsize=(13.5, 5.0))
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28,
                  left=0.06, right=0.98, top=0.9, bottom=0.14)

    ax = fig.add_subplot(gs[0])
    if "25x65nm" in curves:
        c = curves["25x65nm"]
        ax.fill_between(c["r"], c["lo"], c["hi"], color=C_HUGH, alpha=0.18)
        ax.plot(c["r"], c["theta"], lw=3.0, color=C_HUGH,
                label="Empirical 25×65 nm")
    theta_line = np.linspace(0, np.pi / 2, 500)
    ax.plot(r_formula(theta_line, A, B, C), np.rad2deg(theta_line),
            lw=1.8, color=C_FOURKAS, ls="--", label="Analytic Fourkas")
    palette = plt.get_cmap("tab10")
    for k, res in enumerate(top):
        rng = np.random.default_rng(1000 + k)
        sim = simulate_pool(1_000_000, 115, float(res["bg_frac_med"]),
                            float(res["bg_log_sigma"]), rng)
        r_bal = balanced_r_sample(sim["x"], sim["y"],
                                   np.random.default_rng(2000 + k))
        ctr, th, _, _ = hugh_theta_of_r(r_bal)
        ax.plot(ctr, th, lw=1.9, color=palette(k),
                label=(f"bg={float(res['bg_frac_med']):.2f}  "
                       f"log-σ={float(res['bg_log_sigma']):.2f}  "
                       f"RMSE={float(res['rmse_deg']):.2f}°"))
    ax.set_xlabel("r"); ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title("Top-5 grid points match empirical curve to $\\sim$2° RMSE")
    ax.set_xlim(0, 1); ax.set_ylim(0, 92)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    ax = fig.add_subplot(gs[1])
    im = ax.imshow(rmse_mat, origin="lower", cmap="magma_r",
                   aspect="auto",
                   extent=(min(sig_vals) - 0.05, max(sig_vals) + 0.05,
                           min(frac_vals) - 0.02, max(frac_vals) + 0.02))
    for i, f in enumerate(frac_vals):
        for j, s in enumerate(sig_vals):
            v = rmse_mat[i, j]
            if np.isfinite(v):
                ax.text(s, f, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > np.nanmedian(rmse_mat) else "black")
    ax.set_xlabel("Log-normal spread  σ (per-recording bg)")
    ax.set_ylabel("Median bg fraction (of max $C_0^{\\mathrm{ideal}}$)")
    ax.set_title("RMSE vs empirical $\\theta(r)$   (degrees)")
    fig.colorbar(im, ax=ax, label="RMSE (deg)")

    fig.savefig(FIGDIR / "fig6_grid.png")
    plt.close(fig)


# ==================================================================
# Figure 7: Summary schematic (all curves overlaid)
# ==================================================================
def figure_summary(curves):
    print("Figure 7: summary")
    NA, N_MED = 1.3, 1.5
    _, A, B, C = fourkas_ABC(NA, N_MED)

    rng = np.random.default_rng(4)
    sim_uniform = simulate_pool(1_000_000, 1, 0.15, 0.0, rng)
    r_bal_u = balanced_r_sample(sim_uniform["x"], sim_uniform["y"],
                                 np.random.default_rng(41))
    ctr_u, th_u, _, _ = hugh_theta_of_r(r_bal_u)

    rng2 = np.random.default_rng(5)
    sim_group = simulate_pool(1_500_000, 115, 0.30, 0.40, rng2)
    r_bal_g = balanced_r_sample(sim_group["x"], sim_group["y"],
                                 np.random.default_rng(51))
    ctr_g, th_g, _, _ = hugh_theta_of_r(r_bal_g)

    theta_line = np.linspace(0, np.pi / 2, 500)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.plot(r_formula(theta_line, A, B, C), np.rad2deg(theta_line),
            lw=3.0, color=C_FOURKAS, label="Analytic Fourkas (isotropic, no bg)")
    ax.plot(ctr_u, th_u, lw=2.3, color=C_UNIFORM_BG,
            label="Simulation + uniform 15% bg\n→ Hugh's inversion")
    ax.plot(ctr_g, th_g, lw=2.6, color=C_GROUP_BG,
            label="Simulation + per-recording bg (30% med, log-σ=0.40)\n→ Hugh's inversion")
    if "25x65nm" in curves:
        c = curves["25x65nm"]
        ax.fill_between(c["r"], c["lo"], c["hi"], color=C_HUGH, alpha=0.20)
        ax.plot(c["r"], c["theta"], lw=3.0, color=C_HUGH,
                label="Empirical 25×65 nm")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title("Bottom line: adding per-recording background heterogeneity\n"
                 "closes the gap between Fourkas theory and Hugh's empirical curve")
    ax.set_xlim(0, 1); ax.set_ylim(0, 92)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_summary.png")
    plt.close(fig)


def apply_background(theta, phi, bg4, A, B, C, itot_mode="sin2theta"):
    """Apply a fixed 4-channel background to ideal Fourkas channels and return (x, y)."""
    s2 = np.sin(theta) ** 2
    Itot = s2 if itot_mode == "sin2theta" else np.ones_like(theta)
    common = A + B * s2
    C0 = Itot * (common + C * s2 * np.cos(2 * phi))
    C45 = Itot * (common + C * s2 * np.sin(2 * phi))
    C90 = Itot * (common - C * s2 * np.cos(2 * phi))
    C135 = Itot * (common - C * s2 * np.sin(2 * phi))
    b0, b45, b90, b135 = bg4
    x = (C0 + b0 - C90 - b90) / (C0 + b0 + C90 + b90)
    y = (C45 + b45 - C135 - b135) / (C45 + b45 + C135 + b135)
    return x, y


# ==================================================================
# Figure 8: How the data are generated and pooled
# ==================================================================
def figure_generated_data(curves):
    print("Figure 8: generated data & pooling")
    NA, N_MED = 1.3, 1.5
    _, A, B, C = fourkas_ABC(NA, N_MED)
    r_max = r_formula(np.pi / 2, A, B, C)

    rng = np.random.default_rng(7)
    # ideal (no-bg) reference cloud
    theta, phi = sample_isotropic(200_000, rng)
    Itot = np.sin(theta) ** 2
    C0, C45, C90, C135 = channels(theta, phi, Itot, A, B, C)
    x_id = (C0 - C90) / (C0 + C90)
    y_id = (C45 - C135) / (C45 + C135)

    # per-group simulation (few groups so clouds are visible)
    N_GROUPS_SHOW = 6
    sim = simulate_pool(120_000, N_GROUPS_SHOW, 0.30, 0.45,
                        np.random.default_rng(8))

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = GridSpec(2, 2, wspace=0.26, hspace=0.42,
                  left=0.07, right=0.95, top=0.92, bottom=0.08)

    # (a) isotropic orientation sampling
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(theta, bins=90, range=(0, np.pi / 2), density=True, color="#4c72b0",
            alpha=0.85, edgecolor="none", label="simulated draws")
    tt = np.linspace(0, np.pi / 2, 200)
    ax.plot(tt, np.sin(tt), lw=2.2, color="k",
            label=r"$p(\theta)=\sin\theta$ (folded isotropic)")
    ax.set_xlabel(r"folded polar angle $\theta$ (deg)")
    ax.set_ylabel("density (per radian)")
    ax.set_title("(a)  Step 1: draw orientations isotropically on the sphere")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, np.pi / 2)
    ax.set_xticks(np.radians([0, 30, 60, 90]))
    ax.set_xticklabels([0, 30, 60, 90])

    # (b) ideal anisotropy disk, colored by theta
    ax = fig.add_subplot(gs[0, 1])
    sub = rng.choice(x_id.size, size=8000, replace=False)
    sc = ax.scatter(x_id[sub], y_id[sub], c=np.degrees(theta[sub]), s=4,
                    cmap="viridis", alpha=0.6, linewidths=0)
    circ = plt.Circle((0, 0), r_max, fill=False, ls="--", color="k", lw=1.4)
    ax.add_patch(circ)
    ax.text(0, r_max + 0.03, r"$r_{\max}$", ha="center", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(b)  Step 2: ideal anisotropy (no bg)\nradius encodes θ, angle = 2φ")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\theta$ (deg)")

    # (c) a few groups after their own background -> off-center shrunken clouds
    ax = fig.add_subplot(gs[1, 0])
    palette = plt.get_cmap("tab10")
    for g in range(N_GROUPS_SHOW):
        m = sim["group_id"] == g
        xg, yg = sim["x"][m], sim["y"][m]
        take = np.random.default_rng(100 + g).choice(
            xg.size, size=min(1500, xg.size), replace=False)
        ax.scatter(xg[take], yg[take], s=5, color=palette(g), alpha=0.45,
                   linewidths=0, label=f"recording {g + 1}")
        ax.scatter([xg.mean()], [yg.mean()], s=90, color=palette(g),
                   edgecolor="k", marker="X", zorder=5)
    circ = plt.Circle((0, 0), r_max, fill=False, ls="--", color="k", lw=1.4)
    ax.add_patch(circ)
    ax.scatter([0], [0], s=60, color="k", marker="+", zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(c)  Step 3: each recording gets its own background\n"
                 "→ its cloud is shrunk and shifted off-centre (X = centroid)")
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)

    # (d) pooled cloud + pooled p(r)
    ax = fig.add_subplot(gs[1, 1])
    take = np.random.default_rng(9).choice(sim["x"].size,
                                            size=8000, replace=False)
    ax.scatter(sim["x"][take], sim["y"][take], s=4, color="#555555",
               alpha=0.35, linewidths=0)
    circ = plt.Circle((0, 0), r_max, fill=False, ls="--", color="k", lw=1.4)
    ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(d)  Step 4: pool all recordings\n"
                 "the sharp $r_{\\max}$ edge is washed into a filled blob")

    fig.suptitle("Simulation pipeline: isotropic tumbling → Fourkas anisotropy → per-recording background → pooling",
                 y=0.985, fontsize=13)
    fig.savefig(FIGDIR / "fig8_generated_data.png")
    plt.close(fig)


# ==================================================================
# Figure 9: Why per-recording background produces the bell curve
# ==================================================================
def _bell_prepare():
    """Compute all arrays / constants shared by the combined and per-panel figures."""
    from matplotlib.colors import BoundaryNorm

    NA, N_MED = 1.3, 1.5
    _, A, B, C = fourkas_ABC(NA, N_MED)
    r_max = r_formula(np.pi / 2, A, B, C)

    rng = np.random.default_rng(21)
    theta, phi = sample_isotropic(300_000, rng)
    theta_deg = np.degrees(theta)
    Itot = np.sin(theta) ** 2
    C0, C45, C90, C135 = channels(theta, phi, Itot, A, B, C)
    x_id = (C0 - C90) / (C0 + C90)
    y_id = (C45 - C135) / (C45 + C135)
    r_id = np.sqrt(x_id ** 2 + y_id ** 2)
    base_scale = float(np.max(C0))
    chan_max = {"C0": float(C0.max()), "C45": float(C45.max()),
                "C90": float(C90.max()), "C135": float(C135.max())}

    # one representative, sum-invariance-respecting, unbalanced background
    # b0+b90 = b45+b135 ; but b0!=b90 so there is a real translation
    bg = np.array([0.35, 0.30, 0.20, 0.25]) * base_scale  # [b0,b45,b90,b135]
    b0, b45, b90, b135 = bg
    Bsum = b0 + b90
    dbx = b0 - b90
    dby = b45 - b135
    sink = (dbx / Bsum, dby / Bsum)

    x_bg, y_bg = apply_background(theta, phi, bg, A, B, C)
    r_bg = np.sqrt(x_bg ** 2 + y_bg ** 2)

    # discrete colour map: one colour per 10-degree theta band
    band_edges = np.arange(0, 91, 10)            # 0,10,...,90  -> 9 bands
    n_band = len(band_edges) - 1
    cmap = plt.get_cmap("turbo", n_band)
    norm = BoundaryNorm(band_edges, cmap.N)
    band_colors = [cmap(i) for i in range(n_band)]
    band_id = np.clip(np.digitize(theta_deg, band_edges) - 1, 0, n_band - 1)

    sub = rng.choice(x_id.size, size=9000, replace=False)

    return dict(A=A, B=B, C=C, r_max=r_max, theta=theta, phi=phi,
                theta_deg=theta_deg, x_id=x_id, y_id=y_id, r_id=r_id,
                x_bg=x_bg, y_bg=y_bg, r_bg=r_bg, base_scale=base_scale,
                chan_max=chan_max, bg=bg, sink=sink,
                band_edges=band_edges, n_band=n_band, cmap=cmap, norm=norm,
                band_colors=band_colors, band_id=band_id, sub=sub)


def _bell_panel_a(ax, d):
    ax.scatter(d["x_id"][d["sub"]], d["y_id"][d["sub"]], c=d["theta_deg"][d["sub"]],
               cmap=d["cmap"], norm=d["norm"], s=6, alpha=0.6, linewidths=0)
    ax.add_patch(plt.Circle((0, 0), d["r_max"], fill=False, ls="--", color="k", lw=1.4))
    ax.scatter([0], [0], s=80, marker="+", color="k", zorder=6)
    ax.text(0, d["r_max"] + 0.03, r"$r_{\max}$", ha="center", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(a)  Ideal anisotropy (no background), coloured by θ band")


def _bell_panel_b(ax, d, add_colorbar=True, fig=None):
    sc = ax.scatter(d["x_bg"][d["sub"]], d["y_bg"][d["sub"]], c=d["theta_deg"][d["sub"]],
                    cmap=d["cmap"], norm=d["norm"], s=6, alpha=0.6, linewidths=0)
    ax.add_patch(plt.Circle((0, 0), d["r_max"], fill=False, ls="--", color="k", lw=1.4))
    ax.scatter([0], [0], s=80, marker="+", color="k", zorder=6)
    ax.scatter([d["sink"][0]], [d["sink"][1]], s=170, marker="*", color="w",
               edgecolor="k", lw=1.3, zorder=7)
    ax.annotate("dim rods (small θ)\ncollapse here", xy=d["sink"],
                xytext=(d["sink"][0] - 0.10, d["sink"][1] + 0.5), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(b)  After one background: shrunk toward an off-centre point\n"
                 "(same colours — a point's θ does not change)")

    # raw-value table
    b0, b45, b90, b135 = d["bg"]
    cm = d["chan_max"]
    txt = (
        "background used (raw, arb. units)\n"
        f"   $b_{{0}}$   = {b0:.4f}\n"
        f"   $b_{{45}}$  = {b45:.4f}\n"
        f"   $b_{{90}}$  = {b90:.4f}\n"
        f"   $b_{{135}}$ = {b135:.4f}\n"
        f"   ($b_0{{+}}b_{{90}}={b0 + b90:.4f}$ = $b_{{45}}{{+}}b_{{135}}$)\n"
        "peak signal per channel\n"
        f"   max $C_{{0}}$   = {cm['C0']:.4f}\n"
        f"   max $C_{{45}}$  = {cm['C45']:.4f}\n"
        f"   max $C_{{90}}$  = {cm['C90']:.4f}\n"
        f"   max $C_{{135}}$ = {cm['C135']:.4f}"
    )
    ax.text(-0.96, -0.96, txt, ha="left", va="bottom", fontsize=8.2,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="0.6"))
    if add_colorbar and fig is not None:
        cb = fig.colorbar(sc, ax=ax, boundaries=d["band_edges"], ticks=d["band_edges"],
                          fraction=0.046, pad=0.04)
        cb.set_label(r"$\theta$ (deg)")
    return sc


def _bell_panel_c(ax, d):
    r_bins = np.linspace(0, 1, 120)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])
    width = r_bins[1] - r_bins[0]
    N = d["theta"].size
    ideal_bands, bg_bands = [], []
    for i in range(d["n_band"]):
        m = d["band_id"] == i
        hi, _ = np.histogram(d["r_id"][m], bins=r_bins)
        hb, _ = np.histogram(d["r_bg"][m], bins=r_bins)
        ideal_bands.append(hi / (N * width))
        bg_bands.append(hb / (N * width))
    tot_i = np.sum(ideal_bands, axis=0)
    tot_b = np.sum(bg_bands, axis=0)
    si = 1.0 / max(tot_i.max(), 1e-12)
    sb = 1.0 / max(tot_b.max(), 1e-12)
    ax.stackplot(ctr, *[b * si for b in ideal_bands], colors=d["band_colors"],
                 edgecolor="none")
    ax.stackplot(ctr, *[-b * sb for b in bg_bands], colors=d["band_colors"],
                 edgecolor="none")
    ax.axhline(0, color="k", lw=0.9)
    ax.axvline(d["r_max"], ls="--", color="k", lw=1.0)
    ax.text(0.985, 0.86, "ideal $p(r)$", transform=ax.transAxes, ha="right",
            va="center", fontsize=11)
    ax.text(0.985, 0.14, "after one bg", transform=ax.transAxes, ha="right",
            va="center", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["1", "0.5", "0", "0.5", "1"])
    ax.set_xlabel("r"); ax.set_ylabel("relative density (per θ band)")
    ax.set_title("(c)  Which θ band lands where in $p(r)$\n"
                 "high-θ (was near $r_{\\max}$) → shifted inward; low-θ → collapses to offset")


def _bell_panel_d(ax, d):
    r_bins = np.linspace(0, 1, 120)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])
    N_REC = 12
    bgs = make_group_backgrounds(N_REC, d["base_scale"], 0.30, 0.45,
                                 np.random.default_rng(22))
    pooled_r = []
    palette = plt.get_cmap("tab20")
    for i in range(N_REC):
        xr, yr = apply_background(d["theta"], d["phi"], bgs[i], d["A"], d["B"], d["C"])
        rr = np.sqrt(xr ** 2 + yr ** 2)
        pooled_r.append(rr)
        h, _ = np.histogram(rr, bins=r_bins, density=True)
        ax.plot(ctr, h, lw=1.0, color=palette(i % 20), alpha=0.55)
    pooled_r = np.concatenate(pooled_r)
    h_pool, _ = np.histogram(pooled_r, bins=r_bins, density=True)
    ax.plot(ctr, h_pool, lw=3.2, color="k", label="pooled (sum) = smooth bell")
    ax.axvline(d["r_max"], ls="--", color="grey", lw=1.0)
    ax.set_xlabel("r"); ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.set_title("(d)  Different recordings shift their bump to different places;\n"
                 "their sum is a smooth bell — no trace of $r_{\\max}$")
    ax.legend(frameon=False, loc="upper left", fontsize=10)


def figure_bell_mechanism(curves):
    print("Figure 9: bell-curve mechanism (combined)")
    d = _bell_prepare()
    fig = plt.figure(figsize=(13.8, 9.0))
    gs = GridSpec(2, 2, wspace=0.24, hspace=0.42,
                  left=0.07, right=0.94, top=0.90, bottom=0.08)
    _bell_panel_a(fig.add_subplot(gs[0, 0]), d)
    _bell_panel_b(fig.add_subplot(gs[0, 1]), d, add_colorbar=True, fig=fig)
    _bell_panel_c(fig.add_subplot(gs[1, 0]), d)
    _bell_panel_d(fig.add_subplot(gs[1, 1]), d)
    fig.suptitle("Why per-recording background makes a bell: background shrinks the cloud toward an off-centre point; pooling different shifts smears the edge",
                 y=0.985, fontsize=12.5)
    fig.savefig(FIGDIR / "fig9_bell_mechanism.png")
    plt.close(fig)


def figure_bell_panels(curves):
    print("Figure 9a-d: bell-curve mechanism (separate larger panels)")
    d = _bell_prepare()

    # (a)
    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    _bell_panel_a(ax, d)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9a_ideal_xy.png")
    plt.close(fig)

    # (b) with colorbar + raw-value table
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    _bell_panel_b(ax, d, add_colorbar=True, fig=fig)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9b_after_bg_xy.png")
    plt.close(fig)

    # (c)
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    _bell_panel_c(ax, d)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9c_pr_bands.png")
    plt.close(fig)

    # (d)
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    _bell_panel_d(ax, d)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9d_pooled_bell.png")
    plt.close(fig)


def main():
    curves = load_hugh_curves()
    figure_problem(curves)
    figure_method(curves)
    figure_pooling(curves)
    figure_sum_invariance()
    figure_grouped_sim(curves)
    figure_grid(curves)
    figure_summary(curves)
    figure_generated_data(curves)
    figure_bell_mechanism(curves)
    figure_bell_panels(curves)
    print(f"\nAll figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
