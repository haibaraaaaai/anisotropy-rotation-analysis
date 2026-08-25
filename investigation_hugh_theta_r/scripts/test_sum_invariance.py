"""
Test whether C0+C90 == C45+C135 on real polarcam data, per-frame.

The polarcam raw is a 2x2 micropolarizer array. For Sony IMX250MZR:
    (0,0)=90 deg    (0,1)=45 deg
    (1,0)=135 deg   (1,1)=0 deg
So the two "diagonal" sums are
    diag1 = I(0,0)+I(1,1) = C90 + C0
    diag2 = I(0,1)+I(1,0) = C45 + C135
which should be equal if the four polarization projections of a linearly
polarized dipole in a symmetric optical system add up.

We compute the per-frame sums two ways:
    (A) whole 14x256 raw ROI
    (B) rod-centered 8x8 super-pixel patch (16x16 raw) around the peak
We report ratio (diag1/diag2) and (diag1-diag2)/(diag1+diag2) distributions
over frames, and across a handful of rods per dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_raw(rod_dir: Path):
    npy = rod_dir / "capture_maxfps_15x15.npy"
    if not npy.exists():
        return None
    return np.load(npy)


def sum_parities_whole(raw):
    # raw: (T, H, W)
    d1 = raw[:, 0::2, 0::2].sum(axis=(1, 2)).astype(np.float64) + \
         raw[:, 1::2, 1::2].sum(axis=(1, 2)).astype(np.float64)
    d2 = raw[:, 0::2, 1::2].sum(axis=(1, 2)).astype(np.float64) + \
         raw[:, 1::2, 0::2].sum(axis=(1, 2)).astype(np.float64)
    return d1, d2


def sum_parities_patch(raw, half_super=4):
    """Sum each parity inside a (2*half_super)-super-pixel window centered on
    the time-averaged brightest super-pixel of the raw movie."""
    T, H, W = raw.shape
    mean_frame = raw.mean(axis=0)
    # collapse to super-pixel resolution by max over 2x2
    sp = mean_frame.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3))
    iy, ix = np.unravel_index(np.argmax(sp), sp.shape)
    y0 = max(0, iy - half_super)
    y1 = min(sp.shape[0], iy + half_super)
    x0 = max(0, ix - half_super)
    x1 = min(sp.shape[1], ix + half_super)
    yr0, yr1 = 2 * y0, 2 * y1
    xr0, xr1 = 2 * x0, 2 * x1
    sub = raw[:, yr0:yr1, xr0:xr1].astype(np.float64)
    d1 = sub[:, 0::2, 0::2].sum(axis=(1, 2)) + sub[:, 1::2, 1::2].sum(axis=(1, 2))
    d2 = sub[:, 0::2, 1::2].sum(axis=(1, 2)) + sub[:, 1::2, 0::2].sum(axis=(1, 2))
    return d1, d2, (yr0, yr1, xr0, xr1)


def diag_stats(d1, d2):
    ratio = d1 / d2
    imbal = (d1 - d2) / (d1 + d2)
    return {
        "ratio_median": float(np.median(ratio)),
        "ratio_iqr": (float(np.percentile(ratio, 25)),
                      float(np.percentile(ratio, 75))),
        "imbal_median": float(np.median(imbal)),
        "imbal_mad": float(np.median(np.abs(imbal - np.median(imbal)))),
        "imbal_p05": float(np.percentile(imbal, 5)),
        "imbal_p95": float(np.percentile(imbal, 95)),
    }


def scan_dataset(root: Path, n_rods: int = 8, sources=("good", "pending", "bad")):
    rods = []
    for src in sources:
        d = root / src
        if not d.exists():
            continue
        for rod in sorted([p for p in d.iterdir() if p.is_dir()]):
            npy = rod / "capture_maxfps_15x15.npy"
            meta = rod / "capture_maxfps_15x15_meta.json"
            if not (npy.exists() and meta.exists()):
                continue
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                xy = m.get("xy_metrics") or {}
                if float(xy.get("range_x", 0)) > 1.0 and float(xy.get("range_y", 0)) > 1.0:
                    rods.append((src, rod))
            except Exception:
                continue
            if len(rods) >= n_rods:
                break
        if len(rods) >= n_rods:
            break
    return rods


def main():
    hugh = Path(__file__).resolve().parent.parent
    outdir = hugh / "sum_invariance_output"
    outdir.mkdir(exist_ok=True)

    datasets = {
        "25nm": hugh / "tumbling 25nm glycerol",
        "40nm": hugh / "glycerol suspended rods 17062026",
    }

    all_summary = []
    n_rods_per_ds = 8

    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4.6 * len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, 2)

    for row_i, (name, root) in enumerate(datasets.items()):
        rods = scan_dataset(root, n_rods=n_rods_per_ds)
        if not rods:
            print(f"[{name}] no rods available")
            continue
        print(f"\n=== {name}: {len(rods)} rods ===")

        ax_whole = axes[row_i, 0]
        ax_patch = axes[row_i, 1]

        for i, (src, rod) in enumerate(rods):
            raw = load_raw(rod)
            if raw is None:
                continue
            d1w, d2w = sum_parities_whole(raw)
            sw = diag_stats(d1w, d2w)

            d1p, d2p, box = sum_parities_patch(raw, half_super=4)
            sp = diag_stats(d1p, d2p)

            summary = {
                "dataset": name,
                "rod": rod.name,
                "source": src,
                "n_frames": int(raw.shape[0]),
                "whole_ratio_med": sw["ratio_median"],
                "whole_imbal_med": sw["imbal_median"],
                "whole_imbal_p05": sw["imbal_p05"],
                "whole_imbal_p95": sw["imbal_p95"],
                "patch_ratio_med": sp["ratio_median"],
                "patch_imbal_med": sp["imbal_median"],
                "patch_imbal_p05": sp["imbal_p05"],
                "patch_imbal_p95": sp["imbal_p95"],
                "patch_box_yxyx": box,
            }
            all_summary.append(summary)
            print(f"  [{name} rod {i:02d}] {rod.name[:38]}")
            print(f"    WHOLE  diag1/diag2 median = {sw['ratio_median']:.4f}   "
                  f"imbalance median = {sw['imbal_median']:+.4f}  "
                  f"[p5,p95]=[{sw['imbal_p05']:+.4f},{sw['imbal_p95']:+.4f}]")
            print(f"    PATCH  diag1/diag2 median = {sp['ratio_median']:.4f}   "
                  f"imbalance median = {sp['imbal_median']:+.4f}  "
                  f"[p5,p95]=[{sp['imbal_p05']:+.4f},{sp['imbal_p95']:+.4f}]")

            imbal_w = (d1w - d2w) / (d1w + d2w)
            imbal_p = (d1p - d2p) / (d1p + d2p)
            ax_whole.hist(imbal_w, bins=80, histtype="step", lw=1.0, alpha=0.75,
                          label=rod.name[:24])
            ax_patch.hist(imbal_p, bins=80, histtype="step", lw=1.0, alpha=0.75)

        ax_whole.axvline(0.0, ls="--", c="k", lw=0.8)
        ax_whole.set_title(f"{name}  WHOLE 14x256  (C0+C90 vs C45+C135)")
        ax_whole.set_xlabel("(diag1-diag2)/(diag1+diag2) per frame")
        ax_whole.set_ylabel("count")
        ax_whole.grid(True, alpha=0.25)
        ax_whole.legend(fontsize=6, frameon=False)

        ax_patch.axvline(0.0, ls="--", c="k", lw=0.8)
        ax_patch.set_title(f"{name}  ROD PATCH 16x16 raw")
        ax_patch.set_xlabel("(diag1-diag2)/(diag1+diag2) per frame")
        ax_patch.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "diag_imbalance_hist.png", dpi=180)
    plt.close(fig)

    import csv
    keys = ["dataset", "rod", "source", "n_frames",
            "whole_ratio_med", "whole_imbal_med", "whole_imbal_p05", "whole_imbal_p95",
            "patch_ratio_med", "patch_imbal_med", "patch_imbal_p05", "patch_imbal_p95"]
    with (outdir / "sum_invariance_per_rod.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for s in all_summary:
            w.writerow({k: s[k] for k in keys})

    print(f"\nWrote per-rod CSV and imbalance histogram to {outdir}")


if __name__ == "__main__":
    main()
