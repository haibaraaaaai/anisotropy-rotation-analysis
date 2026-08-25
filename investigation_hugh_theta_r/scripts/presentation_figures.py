"""
Presentation figures — 40x65 nm (68x40) rods only.

Figure 1: single-frame 15x15 windows (each frame = one anisotropy data point) in
          raw camera counts, annotated with the per-channel 15x15 spatial average
          (c0,c45,c90,c135) and the (x,y) that frame contributes. No time-averaging.

Figure 2: pooled p(r) and Hugh theta(r) following the ORIGINAL pipeline
          (per-recording background drawn with measured variance, split into
          N_GROUPS recordings, phi-balanced sampling, CDF inversion), for four
          scenarios of background level x transverse modes, vs the empirical
          40x65 nm curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_report_figures as M
import transverse_mode_model as T

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "presentation_figures"
OUT.mkdir(exist_ok=True)

ROOT_40 = HERE / "glycerol suspended rods 17062026"
SIGMA = 0.31            # measured rod-to-rod / cross-channel background CV (40 nm)
N_GROUPS = 115          # number of recordings (matches the dataset)
SEED = 20260720

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 210, "font.size": 11,
    "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.25,
})


def pick_rods(root, n):
    out = []
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
                if float(xm.get("range_x", 0)) > 1 and float(xm.get("range_y", 0)) > 1:
                    out.append((rod, m))
            except Exception:
                continue
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return out


def ch_means(win):
    """Per-channel spatial average over the 15x15 window (Sony 2x2 parity)."""
    c90 = win[0::2, 0::2].mean(); c45 = win[0::2, 1::2].mean()
    c135 = win[1::2, 0::2].mean(); c0 = win[1::2, 1::2].mean()
    return c0, c45, c90, c135


def figure_raw_frames():
    print("Figure 1: single-frame 15x15 windows (40 nm)")
    rods = pick_rods(ROOT_40, 2)
    panels = []          # (win, (c0,c45,c90,c135), x, y, label)
    for rod, m in rods:
        a = np.load(rod / "capture_maxfps_15x15.npy").astype(float)
        roi = m["actual"]["roi"]
        cc = int(np.clip(round(roi["cx"] - roi["x"]), 7, a.shape[2] - 8))
        wsum = a[:, :, cc - 7:cc + 8].sum(axis=(1, 2))
        for pctl in (25, 60, 92):                 # dim / typical / bright frames
            f = int(np.argmin(np.abs(wsum - np.percentile(wsum, pctl))))
            win = a[f, :, cc - 7:cc + 8]
            c0, c45, c90, c135 = ch_means(win)
            x = (c0 - c90) / (c0 + c90); y = (c45 - c135) / (c45 + c135)
            panels.append((win, (c0, c45, c90, c135), x, y,
                           f"frame {f}  ({pctl}th pct brightness)"))

    vmax = float(np.percentile(np.concatenate([p[0].ravel() for p in panels]), 99))
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6))
    for ax, (win, (c0, c45, c90, c135), x, y, lab) in zip(axes.ravel(), panels):
        im = ax.imshow(win, cmap="inferno", aspect="equal", vmin=0, vmax=vmax)
        ax.set_title(
            f"{lab}\n"
            f"15×15 avg:  $c_0$={c0:.0f}  $c_{{90}}$={c90:.0f}  "
            f"$c_{{45}}$={c45:.0f}  $c_{{135}}$={c135:.0f}\n"
            f"(x, y) = ({x:+.2f}, {y:+.2f})", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.09, pad=0.03)
        cb.set_label("counts", fontsize=8)
    fig.suptitle("40×65 nm — single-frame 15×15 windows (each frame = one (x,y) data point; no time-averaging)\n"
                 "12-bit sensor: saturation = 4095 ct (live camera compresses to 8-bit / 255). Signal is weak — "
                 f"per-channel 15×15 averages ~10²–few·10² ct; colour capped at {vmax:.0f} ct",
                 fontsize=11.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig1_raw_frames_40nm.png")
    plt.close(fig)
    print(f"  written {OUT / 'fig1_raw_frames_40nm.png'}")


def simulate(theta, phi, A, B, C, rho, frac, clip, seed):
    """Original pipeline: N_GROUPS recordings, per-recording lognormal bg
    (median=frac, sigma=SIGMA), pooled, then phi-balanced CDF inversion."""
    rng = np.random.default_rng(seed)
    gid = rng.integers(0, N_GROUPS, size=theta.size)
    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, rho)
    base = float(np.max(C0))
    bg = M.make_group_backgrounds(N_GROUPS, base, frac, SIGMA, np.random.default_rng(seed + 1))
    d0 = C0 + bg[gid, 0]; d45 = C45 + bg[gid, 1]
    d90 = C90 + bg[gid, 2]; d135 = C135 + bg[gid, 3]
    if clip:
        d0 = np.clip(d0, 0, None); d45 = np.clip(d45, 0, None)
        d90 = np.clip(d90, 0, None); d135 = np.clip(d135, 0, None)
    return T.xy_from_channels(d0, d45, d90, d135)


def figure_pooled():
    print("Figure 2: pooled p(r) / theta(r), 40 nm, per-recording bg variance")
    A, B, C = T.effective_ABC()          # NA=1.3, n=1.33, hole correction
    rho = T.rho_backscatter()
    r_max = T.r_max_of_rho(A, B, C, 0.0)
    curves = M.load_hugh_curves()
    theta, phi = M.sample_isotropic(2_000_000, np.random.default_rng(SEED))

    scenarios = [
        ("10% bg, no transverse", dict(rho=0.0, frac=0.10, clip=False), "#1f77b4"),
        (f"10% bg + transverse (ρ={rho:.2f})", dict(rho=rho, frac=0.10, clip=False), "#d62728"),
        ("2% bg (removed), no transverse", dict(rho=0.0, frac=0.02, clip=True), "#2ca02c"),
        ("2% bg (removed) + transverse", dict(rho=rho, frac=0.02, clip=True), "#ff7f0e"),
    ]

    fig, (axp, axt) = plt.subplots(1, 2, figsize=(14.0, 5.8))
    r_bins = np.linspace(0, 1, 150)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])

    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, 0.0)
    r_nobg = T.xy_from_channels(C0, C45, C90, C135)[2]
    h, _ = np.histogram(r_nobg, bins=r_bins, density=True)
    axp.plot(ctr, h, color="k", lw=2.6, label="no background (Fourkas)")
    tt = np.linspace(0, np.pi / 2, 400)
    axt.plot(M.r_formula(tt, A, B, C), np.degrees(tt), color="k", ls=":", lw=2.0,
             label="analytic Fourkas")

    for lab, kw, col in scenarios:
        x, y, r = simulate(theta, phi, A, B, C, seed=SEED + hash(lab) % 1000, **kw)
        h, _ = np.histogram(r, bins=r_bins, density=True)
        axp.plot(ctr, h, color=col, lw=2.2, label=lab)
        r_bal = M.balanced_r_sample(x, y, np.random.default_rng(SEED + hash(lab) % 777))
        cc, th, _, _ = M.hugh_theta_of_r(r_bal)
        axt.plot(cc, th, color=col, lw=2.2, label=lab)

    if "40x65nm" in curves:
        c_ = curves["40x65nm"]
        axt.fill_between(c_["r"], c_["lo"], c_["hi"], color="0.4", alpha=0.2)
        axt.plot(c_["r"], c_["theta"], color="0.25", ls="--", lw=2.2,
                 label="empirical 40×65 nm")

    axp.axvline(r_max, color="grey", ls="--", lw=1.0)
    axp.set_xlim(0, 1); axp.set_xlabel("r"); axp.set_ylabel("density")
    axp.set_title("Pooled $p(r)$  (NA=1.3, n=1.33, hole-corrected; per-recording bg, σ=%.2f)" % SIGMA)
    axp.legend(frameon=False, fontsize=8.5, loc="upper left")

    axt.set_xlim(0, 1); axt.set_ylim(0, 92)
    axt.set_xlabel("r"); axt.set_ylabel(r"$\theta$ (deg)")
    axt.set_title(r"Hugh $\theta(r)$ inversion vs empirical 40×65 nm")
    axt.legend(frameon=False, fontsize=8.0, loc="lower right")

    fig.suptitle("40×65 nm: background level × transverse modes, per-recording variance pooled over "
                 f"{N_GROUPS} recordings", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig2_pooled_40nm.png")
    plt.close(fig)
    print(f"  written {OUT / 'fig2_pooled_40nm.png'}")


def figure_pergroup_demo():
    """Show explicitly that the pooled p(r) is the SUM over per-recording groups,
    each with its own baseline background + variance, and how the bell develops
    as the background level rises. 40 nm, rho=0 (no transverse)."""
    print("Figure 3: per-group decomposition + level sweep (40 nm)")
    A, B, C = T.effective_ABC()
    r_max = T.r_max_of_rho(A, B, C, 0.0)
    theta, phi = M.sample_isotropic(1_500_000, np.random.default_rng(SEED))
    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, 0.0)
    base = float(np.max(C0))
    r_bins = np.linspace(0, 1, 150)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # ---- Panel A: individual groups (10% each) summing to the pooled curve ----
    rng = np.random.default_rng(SEED + 3)
    gid = rng.integers(0, N_GROUPS, size=theta.size)
    bg = M.make_group_backgrounds(N_GROUPS, base, 0.10, SIGMA, np.random.default_rng(SEED + 4))
    d0 = C0 + bg[gid, 0]; d45 = C45 + bg[gid, 1]
    d90 = C90 + bg[gid, 2]; d135 = C135 + bg[gid, 3]
    _, _, r = T.xy_from_channels(d0, d45, d90, d135)
    # no-bg reference
    r_nobg = T.xy_from_channels(C0, C45, C90, C135)[2]
    h0, _ = np.histogram(r_nobg, bins=r_bins, density=True)
    axA.plot(ctr, h0, color="k", lw=2.0, ls="--", label="no background (each rod's Fourkas)")
    for g in range(0, N_GROUPS, 3):                      # ~38 individual groups
        m = gid == g
        if m.sum() < 500:
            continue
        hg, _ = np.histogram(r[m], bins=r_bins, density=True)
        axA.plot(ctr, hg, color="0.7", lw=0.5, alpha=0.5)
    hp, _ = np.histogram(r, bins=r_bins, density=True)
    axA.plot(ctr, hp, color="#d62728", lw=2.8, label="pooled = Σ over 115 recordings")
    axA.axvline(r_max, color="grey", ls=":", lw=1.0)
    axA.set_xlim(0, 1); axA.set_xlabel("r"); axA.set_ylabel("density")
    axA.set_title(f"10% background per recording (σ={SIGMA:.2f}): each grey = one recording,\n"
                  "red = their pooled sum")
    axA.legend(frameon=False, fontsize=9)

    # ---- Panel B: how the pooled p(r) develops with background level ----
    axB.plot(ctr, h0, color="k", lw=2.2, ls="--", label="no background")
    for frac, col in [(0.05, "#4c72b0"), (0.10, "#2ca02c"),
                      (0.20, "#ff7f0e"), (0.30, "#d62728")]:
        bgL = M.make_group_backgrounds(N_GROUPS, base, frac, SIGMA,
                                       np.random.default_rng(SEED + int(frac * 1000)))
        e0 = C0 + bgL[gid, 0]; e45 = C45 + bgL[gid, 1]
        e90 = C90 + bgL[gid, 2]; e135 = C135 + bgL[gid, 3]
        _, _, rr = T.xy_from_channels(e0, e45, e90, e135)
        hh, _ = np.histogram(rr, bins=r_bins, density=True)
        axB.plot(ctr, hh, color=col, lw=2.2, label=f"{int(frac*100)}% per recording")
    axB.axvline(r_max, color="grey", ls=":", lw=1.0)
    axB.set_xlim(0, 1); axB.set_xlabel("r"); axB.set_ylabel("density")
    axB.set_title("Pooled $p(r)$ vs background level\n(bell deepens as level rises; 30% ≈ the earlier bell)")
    axB.legend(frameon=False, fontsize=9)

    fig.suptitle("Summing per-recording backgrounds → bell (40×65 nm, hole-corrected, no transverse)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig3_pergroup_bell_40nm.png")
    plt.close(fig)
    print(f"  written {OUT / 'fig3_pergroup_bell_40nm.png'}")


if __name__ == "__main__":
    figure_raw_frames()
    figure_pooled()
    figure_pergroup_demo()
    print(f"\nDone. Figures in {OUT}")
