"""
Per-recording diagnostics for the pooling assumption.

For every recording we compute:
    r_min, r_mean, r_std, r_max
    autocorr 1/e lag of r (frames)
    within-recording phi coverage: fraction of 18 phi-bins (10 deg wide, 0..180)
        that have at least one point
    range_x, range_y (from meta, but recomputed for safety)

Then we make diagnostic plots per dataset:
    (a) r_max vs autocorr lag (scatter)
    (b) r_mean vs autocorr lag (scatter)
    (c) r_std vs autocorr lag (scatter)
    (d) per-recording r-histogram heatmap: rows sorted by autocorr lag
    (e) pooled r-histogram (all filter-pass), plus overlays of individual
        recordings, thinned for visibility
    (f) pooled r-histogram stratified by autocorr-lag tercile (fast / mid /
        slow rotators)

If the isotropic-ergodic-pooling assumption holds, per-recording r_max should
saturate near the Fourkas ceiling for all rods, and the three stratified pooled
histograms in (f) should overlap. If per-recording bias is real, r_max will be
strongly correlated with autocorr lag and the three curves in (f) will differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RANGE_THRESHOLD = 1.0
PHI_BIN_DEG = 10.0
R_MAX_FOURKAS = 0.75  # approx, for NA=1.3 n=1.5


def phi_from_xy(x, y):
    return np.mod(0.5 * np.arctan2(y, x), np.pi)


def autocorr_e_lag(sig, max_lag=400):
    s = sig - sig.mean()
    var = np.dot(s, s) / s.size
    if var <= 0:
        return np.nan
    n = min(max_lag, s.size - 1)
    prev_c = 1.0
    for k in range(1, n + 1):
        c = np.dot(s[:-k], s[k:]) / (s.size - k) / var
        if c < np.exp(-1.0):
            if c == prev_c:
                return float(k)
            frac = (np.exp(-1.0) - prev_c) / (c - prev_c)
            return float((k - 1) + frac)
        prev_c = c
    return np.nan


def load_recordings(root: Path):
    out = []
    for source in ("pending", "good", "bad"):
        src = root / source
        if not src.exists():
            continue
        for rod in sorted([p for p in src.iterdir() if p.is_dir()]):
            meta_path = rod / "capture_maxfps_15x15_meta.json"
            if not meta_path.exists():
                continue
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            xy = m.get("xy_series")
            if not isinstance(xy, list) or len(xy) < 128:
                continue
            xy = np.asarray(xy, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            good = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            xy = xy[good]
            if xy.shape[0] < 128:
                continue
            metrics = m.get("xy_metrics") or {}
            rx = float(metrics.get("range_x", 0.0))
            ry = float(metrics.get("range_y", 0.0))
            passes = rx > RANGE_THRESHOLD and ry > RANGE_THRESHOLD
            out.append({
                "source": source,
                "rod": rod.name,
                "x": xy[:, 0],
                "y": xy[:, 1],
                "range_x": rx,
                "range_y": ry,
                "passes": passes,
            })
    return out


def recording_stats(rec):
    x, y = rec["x"], rec["y"]
    r = np.sqrt(x * x + y * y)
    phi = phi_from_xy(x, y)
    edges = np.arange(0.0, 180.0 + PHI_BIN_DEG, PHI_BIN_DEG)
    counts, _ = np.histogram(np.rad2deg(phi), bins=edges)
    phi_cov = float(np.mean(counts > 0))
    return {
        "n": int(x.size),
        "r_min": float(r.min()),
        "r_mean": float(r.mean()),
        "r_std": float(r.std()),
        "r_max": float(r.max()),
        "ac_lag_r": autocorr_e_lag(r),
        "phi_coverage_frac": phi_cov,
        "r": r,
    }


def per_recording_hist(r_all, r_bins):
    counts, _ = np.histogram(r_all, bins=r_bins)
    tot = counts.sum()
    if tot > 0:
        return counts / tot
    return counts.astype(float)


def analyse_dataset(name: str, root: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    recs = load_recordings(root)
    if not recs:
        print(f"[{name}] no recordings")
        return
    stats = []
    for rec in recs:
        s = recording_stats(rec)
        s["source"] = rec["source"]
        s["rod"] = rec["rod"]
        s["passes"] = rec["passes"]
        stats.append(s)

    kept = [s for s in stats if s["passes"]]
    if not kept:
        print(f"[{name}] no filter-pass recordings")
        return

    ac = np.asarray([s["ac_lag_r"] for s in kept], dtype=np.float64)
    r_max_arr = np.asarray([s["r_max"] for s in kept], dtype=np.float64)
    r_mean_arr = np.asarray([s["r_mean"] for s in kept], dtype=np.float64)
    r_std_arr = np.asarray([s["r_std"] for s in kept], dtype=np.float64)
    phi_cov_arr = np.asarray([s["phi_coverage_frac"] for s in kept], dtype=np.float64)

    print(f"\n=== {name} ===")
    print(f"  filter-pass recordings: {len(kept)}")
    print(f"  autocorr 1/e lag (frames):     median={np.nanmedian(ac):.1f}  "
          f"q25={np.nanpercentile(ac,25):.1f}  q75={np.nanpercentile(ac,75):.1f}")
    print(f"  per-recording r_max:           median={np.median(r_max_arr):.3f}  "
          f"min={r_max_arr.min():.3f}  max={r_max_arr.max():.3f}")
    print(f"  per-recording r_mean:          median={np.median(r_mean_arr):.3f}")
    print(f"  per-recording r_std:           median={np.median(r_std_arr):.3f}")
    print(f"  per-recording phi coverage:    median={np.median(phi_cov_arr):.2f}  "
          f"(fraction of 18 phi-bins occupied)")
    # corr
    finite = np.isfinite(ac)
    if finite.sum() > 2:
        corr = np.corrcoef(ac[finite], r_max_arr[finite])[0, 1]
        print(f"  corr(autocorr_lag, r_max)      = {corr:+.3f}")
        corr = np.corrcoef(ac[finite], r_mean_arr[finite])[0, 1]
        print(f"  corr(autocorr_lag, r_mean)     = {corr:+.3f}")
        corr = np.corrcoef(ac[finite], r_std_arr[finite])[0, 1]
        print(f"  corr(autocorr_lag, r_std)      = {corr:+.3f}")

    # ----- Figure 1: scatter panels -----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, y, ylab in zip(
        axes,
        (r_max_arr, r_mean_arr, r_std_arr),
        ("per-recording r_max", "per-recording r_mean", "per-recording r_std"),
    ):
        ax.scatter(ac, y, s=14, alpha=0.7)
        ax.axhline(R_MAX_FOURKAS, ls="--", c="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("autocorr 1/e lag of r (frames, 1640 Hz)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{name}: per-recording r statistics vs rotational timescale")
    fig.tight_layout()
    fig.savefig(outdir / "per_recording_scatter.png", dpi=180)
    plt.close(fig)

    # ----- Figure 2: r-histogram heatmap, rows sorted by autocorr lag -----
    r_bins = np.linspace(0.0, 1.0, 61)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    hist_rows = np.stack([per_recording_hist(s["r"], r_bins) for s in kept], axis=0)
    order = np.argsort(np.where(np.isfinite(ac), ac, np.inf))
    hist_sorted = hist_rows[order]
    ac_sorted = ac[order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(
        hist_sorted,
        aspect="auto",
        origin="lower",
        extent=(r_bins[0], r_bins[-1], 0, hist_sorted.shape[0]),
        cmap="viridis",
    )
    ax.set_xlabel("r")
    ax.set_ylabel("recording index (sorted by autocorr lag; bottom=fast, top=slow)")
    ax.set_title(f"{name}: per-recording r-histograms (row-normalized)")
    ax.axvline(R_MAX_FOURKAS, ls="--", c="w", lw=1.0, alpha=0.7)
    fig.colorbar(im, ax=ax, label="p(r) per recording")
    fig.tight_layout()
    fig.savefig(outdir / "per_recording_hist_heatmap.png", dpi=180)
    plt.close(fig)

    # ----- Figure 3: pooled histogram + terciles by autocorr lag -----
    # terciles
    finite = np.isfinite(ac)
    ac_f = ac[finite]
    idx_finite = np.where(finite)[0]
    q1 = np.percentile(ac_f, 33.333)
    q2 = np.percentile(ac_f, 66.666)
    tercile_id = np.full(len(kept), -1)
    tercile_id[idx_finite[ac_f <= q1]] = 0
    tercile_id[idx_finite[(ac_f > q1) & (ac_f <= q2)]] = 1
    tercile_id[idx_finite[ac_f > q2]] = 2

    pooled_r_all = np.concatenate([s["r"] for s in kept])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pooled_r_all, bins=r_bins, density=True, histtype="step",
            lw=2.2, color="k", label=f"all filter-pass (n={len(kept)})")
    labels = [
        f"fast rotators (lag<={q1:.0f} fr)",
        f"mid rotators ({q1:.0f}<lag<={q2:.0f})",
        f"slow rotators (lag>{q2:.0f})",
    ]
    colors = ["tab:blue", "tab:orange", "tab:red"]
    for t, lab, c in zip((0, 1, 2), labels, colors):
        pick = [i for i, tid in enumerate(tercile_id) if tid == t]
        if not pick:
            continue
        r_t = np.concatenate([kept[i]["r"] for i in pick])
        ax.hist(r_t, bins=r_bins, density=True, histtype="step",
                lw=1.7, color=c, label=f"{lab}  ({len(pick)} rods)")
    ax.axvline(R_MAX_FOURKAS, ls="--", c="k", lw=0.8, alpha=0.5,
               label=f"Fourkas r_max≈{R_MAX_FOURKAS}")
    ax.set_xlabel("r")
    ax.set_ylabel("density")
    ax.set_title(f"{name}: pooled r-density stratified by rotational timescale")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "pooled_r_hist_terciles.png", dpi=180)
    plt.close(fig)

    # ----- Figure 4: individual per-recording r-density overlay (thinned) -----
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(kept):
        counts, _ = np.histogram(s["r"], bins=r_bins, density=True)
        ax.plot(r_centers, counts, lw=0.5, alpha=0.35, color="grey")
    # mean
    all_hist = np.stack(
        [np.histogram(s["r"], bins=r_bins, density=True)[0] for s in kept], axis=0
    )
    ax.plot(r_centers, all_hist.mean(axis=0), lw=2.2, color="k",
            label="mean of per-recording densities")
    counts_pool, _ = np.histogram(pooled_r_all, bins=r_bins, density=True)
    ax.plot(r_centers, counts_pool, lw=2.2, color="tab:red", ls="--",
            label="pooled density")
    ax.axvline(R_MAX_FOURKAS, ls="--", c="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("r")
    ax.set_ylabel("density")
    ax.set_title(f"{name}: individual per-recording r-densities (grey) vs pooled")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "per_recording_r_density_overlay.png", dpi=180)
    plt.close(fig)

    # Save stats CSV
    import csv
    with (outdir / "per_recording_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "rod", "passes", "n", "r_min", "r_mean", "r_std",
                    "r_max", "ac_lag_r", "phi_coverage_frac"])
        for s in stats:
            w.writerow([s["source"], s["rod"], s["passes"], s["n"],
                        f"{s['r_min']:.4f}", f"{s['r_mean']:.4f}",
                        f"{s['r_std']:.4f}", f"{s['r_max']:.4f}",
                        f"{s['ac_lag_r']:.2f}" if np.isfinite(s['ac_lag_r']) else "",
                        f"{s['phi_coverage_frac']:.3f}"])
    print(f"  wrote plots + CSV to {outdir}")


def main():
    hugh = Path(__file__).resolve().parent.parent
    out_root = hugh / "pooling_diagnostics_output"
    analyse_dataset(
        "25x65nm (tumbling 25nm glycerol)",
        hugh / "tumbling 25nm glycerol",
        out_root / "25nm",
    )
    analyse_dataset(
        "40x65nm (glycerol suspended rods 17062026)",
        hugh / "glycerol suspended rods 17062026",
        out_root / "40nm",
    )


if __name__ == "__main__":
    main()
