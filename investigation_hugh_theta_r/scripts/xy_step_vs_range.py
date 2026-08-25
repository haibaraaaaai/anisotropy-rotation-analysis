"""
Quick diagnostic: for each Hugh recording, compare frame-to-frame changes in
(x, y) anisotropy against the total range visited in that recording.

For each recording we compute:
    dx, dy, dr, dphi         : consecutive-frame differences
    step_xy = sqrt(dx^2+dy^2): displacement per frame in the anisotropy plane
    range_x, range_y         : xmax-xmin, ymax-ymin
    step_xy / cloud_diameter : fraction of the cloud crossed per frame
    autocorr time of x, y    : lag (in frames) where autocorr drops below 1/e

The point: if step_xy per frame is already a large fraction of the cloud
diameter, the rod is largely decorrelated between successive samples and the
observed (x, y) distribution reflects time-averaged / undersampled sampling
rather than the instantaneous Fourkas map.

Run from /Users/dapingxu/Documents/Workspace/pyqtrod/Hugh/ .
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RANGE_THRESHOLD = 1.0


def phi_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mod(0.5 * np.arctan2(y, x), np.pi)


def autocorr_e_lag(sig: np.ndarray, max_lag: int = 200) -> float:
    """Return lag (in samples) at which autocorrelation of `sig` drops below 1/e.

    Returns np.nan if it never does within max_lag.
    """
    s = sig - sig.mean()
    var = np.dot(s, s) / s.size
    if var <= 0:
        return np.nan
    n = min(max_lag, s.size - 1)
    for k in range(1, n + 1):
        c = np.dot(s[:-k], s[k:]) / (s.size - k) / var
        if c < np.exp(-1.0):
            # linear interp between k-1 and k
            if k == 1:
                return 1.0
            c_prev = np.dot(s[: -(k - 1)], s[(k - 1) :]) / (s.size - (k - 1)) / var
            if c_prev == c:
                return float(k)
            frac = (np.exp(-1.0) - c_prev) / (c - c_prev)
            return float((k - 1) + frac)
    return np.nan


def scan(root: Path) -> list[dict]:
    rows: list[dict] = []
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
            if not isinstance(xy, list) or len(xy) < 32:
                continue
            xy = np.asarray(xy, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] < 2:
                continue
            good = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            xy = xy[good]
            if xy.shape[0] < 32:
                continue

            metrics = m.get("xy_metrics") or {}
            rx = float(metrics.get("range_x", 0.0))
            ry = float(metrics.get("range_y", 0.0))

            fps = float((m.get("actual") or {}).get("fps", np.nan))

            x = xy[:, 0]
            y = xy[:, 1]
            r = np.sqrt(x * x + y * y)
            phi = phi_from_xy(x, y)

            dx = np.diff(x)
            dy = np.diff(y)
            dr = np.diff(r)
            dphi = np.diff(phi)
            # wrap dphi into [-pi/2, pi/2] since phi is mod pi
            dphi = (dphi + np.pi / 2.0) % np.pi - np.pi / 2.0
            step_xy = np.sqrt(dx * dx + dy * dy)

            cloud_diam = float(max(rx, ry))
            r_range = float(np.ptp(r))

            row = {
                "source": source,
                "rod": rod.name,
                "n_frames": int(xy.shape[0]),
                "fps": fps,
                "range_x": rx,
                "range_y": ry,
                "cloud_diameter": cloud_diam,
                "r_mean": float(r.mean()),
                "r_std": float(r.std()),
                "r_range": r_range,
                "step_xy_mean": float(step_xy.mean()),
                "step_xy_median": float(np.median(step_xy)),
                "step_xy_p95": float(np.percentile(step_xy, 95)),
                "step_over_diameter_mean": float(step_xy.mean() / cloud_diam) if cloud_diam > 0 else np.nan,
                "step_over_diameter_p95": float(np.percentile(step_xy, 95) / cloud_diam) if cloud_diam > 0 else np.nan,
                "step_dphi_rms_rad": float(np.sqrt(np.mean(dphi * dphi))),
                "step_dr_rms": float(np.sqrt(np.mean(dr * dr))),
                "ac_lag_x_1over_e": autocorr_e_lag(x),
                "ac_lag_y_1over_e": autocorr_e_lag(y),
                "ac_lag_r_1over_e": autocorr_e_lag(r),
                "passes_range_filter": rx > RANGE_THRESHOLD and ry > RANGE_THRESHOLD,
            }
            rows.append(row)
    return rows


def summarise(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"[{name}] no recordings.")
        return
    kept = [r for r in rows if r["passes_range_filter"]]
    print(f"\n=== {name} ===")
    print(f"  total recordings          : {len(rows)}")
    print(f"  passing range_x>1 range_y>1: {len(kept)}")
    for label, subset in (("ALL", rows), ("FILTER-PASS", kept)):
        if not subset:
            continue
        arr = {k: np.asarray([r[k] for r in subset], dtype=np.float64) for k in (
            "step_over_diameter_mean",
            "step_over_diameter_p95",
            "step_dphi_rms_rad",
            "step_dr_rms",
            "ac_lag_x_1over_e",
            "ac_lag_y_1over_e",
            "ac_lag_r_1over_e",
            "cloud_diameter",
            "step_xy_mean",
            "step_xy_p95",
            "r_range",
            "fps",
        )}
        print(f"  --- {label} ({len(subset)}) ---")
        print(f"    fps                          median = {np.nanmedian(arr['fps']):.1f} Hz")
        print(f"    cloud_diameter               median = {np.nanmedian(arr['cloud_diameter']):.3f}")
        print(f"    step_xy_mean (per frame)     median = {np.nanmedian(arr['step_xy_mean']):.4f}")
        print(f"    step_xy_p95  (per frame)     median = {np.nanmedian(arr['step_xy_p95']):.4f}")
        print(f"    step / cloud_diameter mean   median = {np.nanmedian(arr['step_over_diameter_mean']):.3f}")
        print(f"    step / cloud_diameter p95    median = {np.nanmedian(arr['step_over_diameter_p95']):.3f}")
        print(f"    |dphi| rms per frame (rad)   median = {np.nanmedian(arr['step_dphi_rms_rad']):.3f}"
              f"  (~{np.rad2deg(np.nanmedian(arr['step_dphi_rms_rad'])):.1f} deg)")
        print(f"    autocorr 1/e lag  x (frames) median = {np.nanmedian(arr['ac_lag_x_1over_e']):.2f}")
        print(f"    autocorr 1/e lag  y (frames) median = {np.nanmedian(arr['ac_lag_y_1over_e']):.2f}")
        print(f"    autocorr 1/e lag  r (frames) median = {np.nanmedian(arr['ac_lag_r_1over_e']):.2f}")


def main() -> None:
    hugh = Path(__file__).resolve().parent.parent
    datasets = {
        "25x65nm (tumbling 25nm glycerol)": hugh / "tumbling 25nm glycerol",
        "40x65nm (glycerol suspended rods 17062026)": hugh / "glycerol suspended rods 17062026",
    }
    all_rows: dict[str, list[dict]] = {}
    for name, root in datasets.items():
        if not root.exists():
            print(f"[skip] {root} does not exist")
            continue
        rows = scan(root)
        all_rows[name] = rows
        summarise(name, rows)

    # write a compact CSV so we can look at per-recording detail
    import csv

    out_csv = hugh / "xy_step_vs_range_per_recording.csv"
    fields = [
        "dataset",
        "source",
        "rod",
        "passes_range_filter",
        "n_frames",
        "fps",
        "cloud_diameter",
        "r_mean",
        "r_std",
        "r_range",
        "step_xy_mean",
        "step_xy_median",
        "step_xy_p95",
        "step_over_diameter_mean",
        "step_over_diameter_p95",
        "step_dphi_rms_rad",
        "step_dr_rms",
        "ac_lag_x_1over_e",
        "ac_lag_y_1over_e",
        "ac_lag_r_1over_e",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, rows in all_rows.items():
            for r in rows:
                w.writerow({"dataset": name, **{k: r.get(k) for k in fields if k != "dataset"}})
    print(f"\nPer-recording CSV: {out_csv}")


if __name__ == "__main__":
    main()
