"""
Background-scale analysis from the RAW polarcam data.

Goal (per user): the anisotropy is a MEAN over a ~15x15 window (=~50 px per
polarisation channel), so rare bright background spots should be diluted. Measure,
across all accepted rods in both datasets:
  * the per-channel windowed-mean SIGNAL scale (counts),
  * the BACKGROUND level (estimated from the rod-free part of the 14x256 strip),
  * background as a % of signal (typical, and at the DIMMEST orientation),
  * channel-to-channel background imbalance (what actually shifts the anisotropy),
  * rod-to-rod ("spot to spot") spread,
  * the rare-bright-spot behaviour (median vs p99 vs max of background pixels),
  * the "background-only" anisotropy point r_bg = where a signal-free (θ→0) frame
    would land — this is the per-recording shift that (if spread out) smears the
    Fourkas r_max pile-up into a bell.

Channels via Sony IMX250MZR 2x2 parity:  (0,0)=90 (0,1)=45 (1,0)=135 (1,1)=0.
Window: raw 15 cols (±7) x full 14 rows, centred on center_px (matches maxfps_15x15).
Background: strip columns well away from the rod, same parity planes.

Output: ../background_scale_output/background_scale.png  + printed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
OUTDIR = HERE / "background_scale_output"
OUTDIR.mkdir(exist_ok=True)

DATASETS = {
    "25x65nm": HERE / "tumbling 25nm glycerol",
    "40x65nm": HERE / "glycerol suspended rods 17062026",
}
N_RODS_PER_DS = 60          # cap per dataset for runtime
WIN_HALF_COL = 7            # ±7 raw cols -> 15-col window
BG_GAP = 25                 # raw cols to skip on each side of the rod before bg starts
BG_SPAN = 70                # raw cols of background sampled on each side


def parity_planes(a):
    """a:(T,H,W) -> dict of parity planes, each (T, H/2, W/2)."""
    return {"90": a[:, 0::2, 0::2], "45": a[:, 0::2, 1::2],
            "135": a[:, 1::2, 0::2], "0": a[:, 1::2, 1::2]}


def analyse_rod(npy_path, meta):
    a = np.load(npy_path)                       # uint16 (T,H,W)
    T, H, W = a.shape
    roi = meta["actual"]["roi"]
    c_col = float(roi["cx"]) - float(roi["x"])  # in-roi centre column (raw)
    c_col = int(round(np.clip(c_col, WIN_HALF_COL, W - WIN_HALF_COL - 1)))

    P = parity_planes(a.astype(np.float64))
    Wp = W // 2
    pc = int(round(c_col / 2.0))                # centre in parity-column units
    half_pc = max(2, WIN_HALF_COL // 2)         # window half-width in parity cols
    win = slice(max(0, pc - half_pc), min(Wp, pc + half_pc + 1))

    # rod-free background parity-column mask
    lo0, lo1 = max(0, pc - (BG_GAP + BG_SPAN) // 2), max(0, pc - BG_GAP // 2)
    hi0, hi1 = min(Wp, pc + BG_GAP // 2), min(Wp, pc + (BG_GAP + BG_SPAN) // 2)
    bg_cols = np.r_[np.arange(lo0, lo1), np.arange(hi0, hi1)]
    if bg_cols.size < 8:
        bg_cols = np.r_[np.arange(0, max(4, pc - half_pc - 2)),
                        np.arange(min(Wp, pc + half_pc + 2), Wp)]

    chans = ("0", "90", "45", "135")
    # per-frame window mean (signal+bg) for each channel
    win_mean = {ch: P[ch][:, :, win].mean(axis=(1, 2)) for ch in chans}      # (T,)
    # background level per channel: robust (median over rod-free region & time)
    bg_med = {ch: float(np.median(P[ch][:, :, bg_cols])) for ch in chans}
    bg_mean = {ch: float(np.mean(P[ch][:, :, bg_cols])) for ch in chans}
    # rare-spot descriptors of the background pixel distribution
    bg_pix = np.concatenate([P[ch][:, :, bg_cols].ravel() for ch in chans])
    bg_p50, bg_p99, bg_max = (float(np.percentile(bg_pix, 50)),
                              float(np.percentile(bg_pix, 99)), float(bg_pix.max()))

    # signal = window mean - background (per channel, per frame)
    sig = {ch: win_mean[ch] - bg_med[ch] for ch in chans}
    tot_win = sum(win_mean[ch] for ch in chans)          # window brightness proxy
    tot_sig = sum(sig[ch] for ch in chans)
    sig_med = float(np.median(tot_sig))
    sig_dim = float(np.percentile(tot_sig, 2))           # dim (near θ→0) proxy

    b = np.array([bg_med["0"], bg_med["90"], bg_med["45"], bg_med["135"]])
    b_mean = float(b.mean())
    # background as a fraction of per-channel signal
    sig_ch_med = np.array([np.median(sig[ch]) for ch in chans])
    bg_frac_typ = b_mean / max(np.median(sig_ch_med), 1e-9)
    bg_frac_dim = b_mean / max(sig_dim / 4.0, 1e-9)
    # cross-channel background imbalance
    bg_imbalance = (b.max() - b.min()) / max(b_mean, 1e-9)
    # background-only anisotropy point (where a signal-free frame lands)
    b0, b90, b45, b135 = bg_med["0"], bg_med["90"], bg_med["45"], bg_med["135"]
    xbg = (b0 - b90) / max(b0 + b90, 1e-9)
    ybg = (b45 - b135) / max(b45 + b135, 1e-9)
    r_bg = float(np.hypot(xbg, ybg))

    return dict(sig_ch_med=float(np.median(sig_ch_med)), sig_med=sig_med, sig_dim=sig_dim,
                bg_mean=b_mean, bg_frac_typ=bg_frac_typ, bg_frac_dim=bg_frac_dim,
                bg_imbalance=bg_imbalance, r_bg=r_bg,
                bg_p50=bg_p50, bg_p99=bg_p99, bg_max=bg_max,
                spot_ratio=bg_p99 / max(bg_p50, 1e-9))


def scan(root):
    rows = []
    for src in ("good", "pending", "bad"):
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
                xm = m.get("xy_metrics") or {}
                if not (float(xm.get("range_x", 0)) > 1.0 and float(xm.get("range_y", 0)) > 1.0):
                    continue
                rows.append(analyse_rod(npy, m))
            except Exception as e:
                print(f"  skip {rod.name}: {e}")
                continue
            if len(rows) >= N_RODS_PER_DS:
                break
        if len(rows) >= N_RODS_PER_DS:
            break
    return rows


def summarise(name, rows):
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    print(f"\n=== {name}  (N={len(rows)} rods) ===")
    for k, lbl, fmt in [
        ("sig_ch_med", "per-channel signal (counts, median over frames)", "{:.1f}"),
        ("bg_mean", "background level (counts, per channel mean)", "{:.1f}"),
        ("bg_frac_typ", "bg / typical signal", "{:.3f}"),
        ("bg_frac_dim", "bg / dim-orientation signal (p2)", "{:.3f}"),
        ("bg_imbalance", "cross-channel bg imbalance (max-min)/mean", "{:.3f}"),
        ("r_bg", "background-only anisotropy r_bg (dim-limit shift)", "{:.3f}"),
        ("spot_ratio", "bg p99/median (rare bright spots)", "{:.2f}"),
    ]:
        v = col(k)
        print(f"  {lbl:52s} median={fmt.format(np.median(v))}  "
              f"[p10 {fmt.format(np.percentile(v,10))}, p90 {fmt.format(np.percentile(v,90))}]")
    return {k: col(k) for k in rows[0]}


def main():
    results = {}
    for name, root in DATASETS.items():
        print(f"\nscanning {name} ...")
        rows = scan(root)
        if rows:
            results[name] = summarise(name, rows)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    colors = {"25x65nm": "#2ca02c", "40x65nm": "#d62728"}
    panels = [
        ("bg_frac_typ", "bg / typical signal", axes[0, 0]),
        ("bg_frac_dim", "bg / dim-orientation signal", axes[0, 1]),
        ("bg_imbalance", "cross-channel bg imbalance", axes[0, 2]),
        ("r_bg", "background-only anisotropy $r_{bg}$", axes[1, 0]),
        ("spot_ratio", "bg p99/median (rare spots)", axes[1, 1]),
        ("sig_ch_med", "per-channel signal (counts)", axes[1, 2]),
    ]
    for key, title, ax in panels:
        for name, res in results.items():
            v = res[key]
            v = v[np.isfinite(v)]
            ax.hist(v, bins=20, alpha=0.55, color=colors.get(name, "grey"),
                    label=f"{name} (med {np.median(v):.3g})")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("rods"); ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.2)
    axes[1, 0].axvline(0.0, color="k", lw=0.8, ls=":")
    fig.suptitle("Background scale from raw 15×15-window means: how big is bg vs signal, "
                 "how unequal across channels, how spread rod-to-rod", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / "background_scale.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUTDIR / 'background_scale.png'}")


if __name__ == "__main__":
    main()
