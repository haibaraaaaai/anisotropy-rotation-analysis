"""
Coherent (interference) background model — Richard Berry's proposal.

Physics
-------
The nanorod image is a point source, so (Fermat) all aperture contributions reach
the image point in phase: the rod emission has ONE common phase across the four
polarisation channels. The background at the ROI is one additional coherent field
with its OWN phase delta relative to the rod. We therefore add COMPLEX AMPLITUDES,
not intensities, in each channel:

    I_j = | sqrt(I_rod,j) + sqrt(I_bg,j) * e^{i*delta} |^2
        = I_rod,j + I_bg,j + 2*eta*sqrt(I_rod,j * I_bg,j)*cos(delta)

    eta = 1  -> fully coherent (Richard's interference model, +1 phase parameter)
    eta = 0  -> incoherent additive background (the previous model)

Key consequence: the cross (interference) term scales as sqrt(I_rod*I_bg), i.e. as
sqrt(I_bg) not I_bg. So (a) a SMALL background produces a LARGE deviation, and
(b) it perturbs the BRIGHT (near-r_max) orientations that an incoherent background
barely moves — exactly what is needed to wash out the Fourkas r_max pile-up with
the low background intensity Hugh actually measures.

Each rod gets its own background (per-channel intensity, as measured) AND its own
random relative phase delta ~ U[0, 2pi). We pool over rods and run Hugh's method.

Uses the hole-corrected Fourkas A,B,C, the transverse-mode channels (rho), and the
CSV-measured background level/variance from transverse_mode_model.py.

Output: ../coherent_background_output/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_report_figures as M
import transverse_mode_model as T

OUTDIR = Path(__file__).resolve().parent.parent / "coherent_background_output"
OUTDIR.mkdir(exist_ok=True)

N_TOTAL = 1_200_000
N_GROUPS = 115
SEED = 20260719
RHO = 0.0            # transverse ratio (0 = pure longitudinal; interference tested alone)


def _stokes(ch):
    """(C0,C45,C90,C135) -> linear Stokes S0,S1,S2 (S3 assumed 0)."""
    C0, C45, C90, C135 = ch
    return C0 + C90, C0 - C90, C45 - C135


def _jones(Ip, S1, S2):
    """Real linear-polarised Jones vector (Ex,Ey) for polarised intensity Ip."""
    Ex = np.sqrt(np.clip(0.5 * (Ip + S1), 0.0, None))
    Ey = np.sign(S2) * np.sqrt(np.clip(0.5 * (Ip - S1), 0.0, None))
    return Ex, Ey


def detect(rod, bg, delta, method):
    """Combine rod + background into detected 4 channels.

    method='incoherent'   : I_j = I_rod,j + I_bg,j  (previous additive model)
    method='coherent'     : only the POLARISED part of each field interferes
        (rod polarised fraction = degree of linear polarisation = r); the
        unpolarised parts add incoherently. Background carries relative phase delta.
    """
    C0, C45, C90, C135 = rod
    b0, b45, b90, b135 = bg
    if method == "incoherent":
        return (C0 + b0, C45 + b45, C90 + b90, C135 + b135)
    S0r, S1r, S2r = _stokes(rod)
    S0b, S1b, S2b = _stokes(bg)
    Ipr = np.sqrt(S1r ** 2 + S2r ** 2); Iur = np.clip(S0r - Ipr, 0.0, None)
    Ipb = np.sqrt(S1b ** 2 + S2b ** 2); Iub = np.clip(S0b - Ipb, 0.0, None)
    Exr, Eyr = _jones(Ipr, S1r, S2r)
    Exb, Eyb = _jones(Ipb, S1b, S2b)
    ph = np.exp(1j * delta)
    Ext = Exr + Exb * ph
    Eyt = Eyr + Eyb * ph
    incoh = 0.5 * (Iur + Iub)          # unpolarised parts add incoherently
    I0 = np.abs(Ext) ** 2 + incoh
    I90 = np.abs(Eyt) ** 2 + incoh
    I45 = 0.5 * np.abs(Ext + Eyt) ** 2 + incoh
    I135 = 0.5 * np.abs(Ext - Eyt) ** 2 + incoh
    return (I0, I45, I90, I135)


def build(theta, phi, A, B, C, rho, frac, sig, method, seed, bg_scale=1.0):
    """Return (x, y, r) for a pooled simulation with per-rod background + phase."""
    rng = np.random.default_rng(seed)
    gid = rng.integers(0, N_GROUPS, size=theta.size)
    C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, rho)
    base = float(np.max(C0))
    bg = M.make_group_backgrounds(N_GROUPS, base * bg_scale, frac, sig,
                                  np.random.default_rng(seed + 1))
    bg_pt = (bg[gid, 0], bg[gid, 1], bg[gid, 2], bg[gid, 3])
    delta_g = rng.uniform(0.0, 2.0 * np.pi, size=N_GROUPS)
    delta = delta_g[gid]
    d0, d45, d90, d135 = detect((C0, C45, C90, C135), bg_pt, delta, method)
    return T.xy_from_channels(d0, d45, d90, d135)


def main():
    A, B, C = T.effective_ABC()
    r_max = T.r_max_of_rho(A, B, C, RHO)
    curves = M.load_hugh_curves()
    rng = np.random.default_rng(SEED)
    theta, phi = M.sample_isotropic(N_TOTAL, rng)

    r_bins = np.linspace(0, 1, 140)
    ctr = 0.5 * (r_bins[:-1] + r_bins[1:])

    print(f"hole-corrected A={A:.4g} B={B:.4g} C={C:.4g}  r_max(rho={RHO})={r_max:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    for row, (size, meas) in enumerate(T.CSV_MEAS.items()):
        frac = meas["frac_med"]; sig = meas["log_sigma"]

        # no background reference (sharp pile-up)
        C0, C45, C90, C135 = T.channels_transverse(theta, phi, A, B, C, RHO)
        r_nobg = T.xy_from_channels(C0, C45, C90, C135)[2]

        # incoherent vs coherent at the MEASURED background
        xi, yi, ri = build(theta, phi, A, B, C, RHO, frac, sig, method="incoherent", seed=SEED + row)
        xc, yc, rc = build(theta, phi, A, B, C, RHO, frac, sig, method="coherent", seed=SEED + row)
        # coherent at HALF the measured background (does interference need even less?)
        xh, yh, rh = build(theta, phi, A, B, C, RHO, frac, sig, method="coherent",
                           seed=SEED + row + 50, bg_scale=0.5)

        for tag, r in (("no bg", r_nobg), ("incoh", ri), ("coh", rc)):
            print(f"  {size} {tag:6s}: frac r>0.9 r_max = {np.mean(r > 0.9 * r_max):.3f}, "
                  f"median r = {np.median(r):.3f}")

        # p(r)
        ax = axes[row, 0]
        for r, c, lab in ((r_nobg, "k", "no bg (Fourkas)"),
                          (ri, "#1f77b4", f"incoherent + bg (frac={frac:.2f})"),
                          (rc, "#d62728", f"coherent + bg (frac={frac:.2f})"),
                          (rh, "#ff7f0e", f"coherent + HALF bg")):
            h, _ = np.histogram(r, bins=r_bins, density=True)
            lw = 2.6 if c == "k" else 2.0
            ls = "-" if c != "#ff7f0e" else ":"
            ax.plot(ctr, h, color=c, lw=lw, ls=ls, label=lab)
        ax.axvline(r_max, color="grey", ls="--", lw=1.0)
        ax.set_xlim(0, 1); ax.set_xlabel("r"); ax.set_ylabel("density")
        ax.set_title(f"{size}: pooled $p(r)$  (measured bg, σ={sig:.2f}, random phase)")
        ax.legend(frameon=False, fontsize=8.0)

        # theta(r)
        ax = axes[row, 1]
        tt = np.linspace(0, np.pi / 2, 400)
        ax.plot(M.r_formula(tt, A, B, C), np.degrees(tt), "k:", lw=1.6,
                label="analytic Fourkas")
        for (x, y), c, lab in (((xi, yi), "#1f77b4", "incoherent + bg"),
                               ((xc, yc), "#d62728", "coherent + bg"),
                               ((xh, yh), "#ff7f0e", "coherent + HALF bg")):
            r_bal = M.balanced_r_sample(x, y, np.random.default_rng(hash((size, c)) % 2**31))
            cc, th, _, _ = M.hugh_theta_of_r(r_bal)
            ls = "-" if c != "#ff7f0e" else ":"
            ax.plot(cc, th, color=c, lw=2.0, ls=ls, label=lab)
        if size in curves:
            cu = curves[size]
            ax.fill_between(cu["r"], cu["lo"], cu["hi"], color="#8c564b", alpha=0.18)
            ax.plot(cu["r"], cu["theta"], color="#8c564b", lw=2.4, ls="--",
                    label=f"Hugh empirical {size}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 92)
        ax.set_xlabel("r"); ax.set_ylabel(r"$\theta$ (deg)")
        ax.set_title(f"{size}: Hugh $\\theta(r)$ — interference vs incoherent")
        ax.legend(frameon=False, fontsize=8.0, loc="lower right")

    fig.suptitle("Coherent (interference) background: I_j = |√I_rod,j + √I_bg,j·e^{iδ}|² "
                 "with measured bg + random per-rod phase δ", y=0.99, fontsize=12.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTDIR / "coherent_vs_incoherent.png", dpi=200)
    plt.close(fig)
    print(f"\nWritten {OUTDIR / 'coherent_vs_incoherent.png'}")

    figure_bg_marker_scatter(theta, phi, A, B, C, curves)


def figure_bg_marker_scatter(theta, phi, A, B, C, curves):
    """Richard's request: (x,y) scatter with a marker per recording characterising
    its background — polarisation as location in (x,y), intensity as marker size."""
    print("Scatter figure: (x,y) cloud with background markers")
    size = "25x65nm"; meas = T.CSV_MEAS[size]
    frac, sig = meas["frac_med"], meas["log_sigma"]
    rng = np.random.default_rng(SEED + 7)
    N_SHOW = 6
    idx_sub = rng.choice(theta.size, size=120_000, replace=False)
    th, ph = theta[idx_sub], phi[idx_sub]
    gid = rng.integers(0, N_SHOW, size=th.size)
    C0, C45, C90, C135 = T.channels_transverse(th, ph, A, B, C, RHO)
    base = float(np.max(C0))
    bg = M.make_group_backgrounds(N_SHOW, base, frac, sig, np.random.default_rng(SEED + 8))
    delta_g = rng.uniform(0, 2 * np.pi, size=N_SHOW)
    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    cmap = plt.get_cmap("tab10")
    for g in range(N_SHOW):
        m = gid == g
        d0, d45, d90, d135 = detect(
            (C0[m], C45[m], C90[m], C135[m]),
            (bg[g, 0], bg[g, 1], bg[g, 2], bg[g, 3]), delta_g[g], method="coherent")
        x, y, _ = T.xy_from_channels(d0, d45, d90, d135)
        take = rng.choice(x.size, size=min(1500, x.size), replace=False)
        ax.scatter(x[take], y[take], s=4, color=cmap(g), alpha=0.25, linewidths=0)
        # background marker: its own polarisation location + intensity as size
        b0, b45, b90, b135 = bg[g]
        xbg = (b0 - b90) / (b0 + b90); ybg = (b45 - b135) / (b45 + b135)
        bint = (b0 + b45 + b90 + b135) / (4 * base)     # bg intensity fraction
        ax.scatter([xbg], [ybg], s=60 + 3000 * bint, color=cmap(g),
                   edgecolor="k", marker="X", zorder=6)
    ax.add_patch(plt.Circle((0, 0), T.r_max_of_rho(A, B, C, RHO), fill=False,
                            ls="--", color="k", lw=1.2))
    ax.scatter([0], [0], s=60, marker="+", color="k", zorder=7)
    ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("(x,y) cloud per recording (coherent bg)\n"
                 "X marker = background polarisation (location) & intensity (size)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "xy_with_bg_markers.png", dpi=200)
    plt.close(fig)
    print(f"Written {OUTDIR / 'xy_with_bg_markers.png'}")


if __name__ == "__main__":
    main()
