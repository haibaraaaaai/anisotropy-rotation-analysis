#!/usr/bin/env python3
"""
Simulate Fourkas anisotropy for isotropically tumbling single dipoles / rods,
including a simple fixed per-channel background.

This version compares two source-brightness models:

    1. Itot = 1
    2. Itot ∝ sin^2(theta)

For each case, it adds a fixed background to each polarization channel:
    background_0, background_45, background_90, background_135

By default each background is set to 15% of the maximum ideal C0 value
for that Itot model.

Main output of interest:
    theta_r_relationship_flipped.png

where x-axis is r and y-axis is folded theta.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# User parameters
# -----------------------------
N = 1_000_000
RANDOM_SEED = 1

# Fourkas example: NA=1.3, n=1.5 gives alpha about 60 deg.
NA = 1.3
N_MEDIUM = 1.5

# Compare both cases.
ITOT_MODES = ["constant", "sin2theta"]

# Simple background model.
# If USE_BACKGROUND is True and AUTO_BACKGROUND_15_PERCENT_OF_MAX_C0 is True,
# each channel gets the same fixed background:
#     background_i = 0.15 * max(C0_ideal)
USE_BACKGROUND = True
AUTO_BACKGROUND_15_PERCENT_OF_MAX_C0 = True
BACKGROUND_FRACTION_OF_MAX_C0 = 0.15

# Used only if AUTO_BACKGROUND_15_PERCENT_OF_MAX_C0 = False.
BACKGROUND_0 = 0.0
BACKGROUND_45 = 0.0
BACKGROUND_90 = 0.0
BACKGROUND_135 = 0.0

OUTDIR = Path("fourkas_background_comparison_output")


# -----------------------------
# Fourkas equations
# -----------------------------
def fourkas_ABC(NA, n):
    if NA > n:
        raise ValueError("NA must be <= refractive index n.")
    alpha = np.arcsin(NA / n)
    ca = np.cos(alpha)
    A = 1/6 - 1/4 * ca + 1/12 * ca**3
    B = 1/8 * ca - 1/8 * ca**3
    C = 7/48 - 1/16 * ca - 1/16 * ca**2 - 1/48 * ca**3
    return alpha, A, B, C


def sample_isotropic(N, rng):
    # Uniform solid-angle sampling:
    # cos(theta) is uniform on [-1, 1], phi is uniform on [0, 2pi).
    cos_theta = rng.uniform(-1.0, 1.0, N)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2*np.pi, N)

    # Fourkas ideal dipole signal is symmetric under theta -> pi-theta.
    theta_folded = np.minimum(theta, np.pi - theta)
    return theta, theta_folded, phi


def get_Itot(theta_folded, mode):
    if mode == "constant":
        return np.ones_like(theta_folded)
    if mode == "sin2theta":
        return np.sin(theta_folded)**2
    raise ValueError("Unknown ITOT_MODE.")


def fourkas_channels(theta, phi, Itot, A, B, C):
    s2 = np.sin(theta)**2
    common = A + B * s2
    C0 = Itot * (common + C * s2 * np.cos(2*phi))
    C45 = Itot * (common + C * s2 * np.sin(2*phi))
    C90 = Itot * (common - C * s2 * np.cos(2*phi))
    C135 = Itot * (common - C * s2 * np.sin(2*phi))
    return C0, C45, C90, C135


def add_simple_background(C0, C45, C90, C135, b0, b45, b90, b135):
    return C0 + b0, C45 + b45, C90 + b90, C135 + b135


def anisotropy(C0, C45, C90, C135):
    x = (C0 - C90) / (C0 + C90)
    y = (C45 - C135) / (C45 + C135)
    r = np.sqrt(x*x + y*y)
    return x, y, r


def r_formula(theta, A, B, C):
    s2 = np.sin(theta)**2
    return C * s2 / (A + B * s2)


def theta_from_r(r, A, B, C):
    sin2 = r * A / (C - r * B)
    sin2 = np.clip(sin2, 0.0, 1.0)
    return np.arcsin(np.sqrt(sin2))


def percent_hist(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    perc = 100 * counts / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, perc


def run_case(theta_folded, phi, A, B, C, mode):
    Itot = get_Itot(theta_folded, mode)

    C0_ideal, C45_ideal, C90_ideal, C135_ideal = fourkas_channels(
        theta_folded, phi, Itot, A, B, C
    )

    if USE_BACKGROUND:
        if AUTO_BACKGROUND_15_PERCENT_OF_MAX_C0:
            bg = BACKGROUND_FRACTION_OF_MAX_C0 * np.max(C0_ideal)
            b0 = b45 = b90 = b135 = bg
        else:
            b0 = BACKGROUND_0
            b45 = BACKGROUND_45
            b90 = BACKGROUND_90
            b135 = BACKGROUND_135

        C0, C45, C90, C135 = add_simple_background(
            C0_ideal, C45_ideal, C90_ideal, C135_ideal,
            b0, b45, b90, b135
        )
    else:
        b0 = b45 = b90 = b135 = 0.0
        C0, C45, C90, C135 = C0_ideal, C45_ideal, C90_ideal, C135_ideal

    x, y, r = anisotropy(C0, C45, C90, C135)
    x_ideal, y_ideal, r_ideal = anisotropy(C0_ideal, C45_ideal, C90_ideal, C135_ideal)

    return {
        "mode": mode,
        "Itot": Itot,
        "background": (b0, b45, b90, b135),
        "C0_ideal": C0_ideal,
        "C45_ideal": C45_ideal,
        "C90_ideal": C90_ideal,
        "C135_ideal": C135_ideal,
        "C0": C0,
        "C45": C45,
        "C90": C90,
        "C135": C135,
        "x": x,
        "y": y,
        "r": r,
        "x_ideal": x_ideal,
        "y_ideal": y_ideal,
        "r_ideal": r_ideal,
    }


def binned_theta_vs_r(theta_folded, r, theta_bins_deg=np.linspace(0, 90, 91)):
    theta_bin_edges = np.deg2rad(theta_bins_deg)
    theta_bin_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])

    r_mean = np.full_like(theta_bin_centers, np.nan)
    r_p05 = np.full_like(theta_bin_centers, np.nan)
    r_p95 = np.full_like(theta_bin_centers, np.nan)

    for i in range(len(theta_bin_centers)):
        mask = (theta_folded >= theta_bin_edges[i]) & (theta_folded < theta_bin_edges[i+1])
        if np.any(mask):
            r_mean[i] = np.mean(r[mask])
            r_p05[i] = np.percentile(r[mask], 5)
            r_p95[i] = np.percentile(r[mask], 95)

    return theta_bin_centers, r_mean, r_p05, r_p95


def main():
    OUTDIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    alpha, A, B, C = fourkas_ABC(NA, N_MEDIUM)

    theta, theta_folded, phi = sample_isotropic(N, rng)

    results = {}
    for mode in ITOT_MODES:
        results[mode] = run_case(theta_folded, phi, A, B, C, mode)

    print("Fourkas isotropic tumbling with simple fixed background")
    print("------------------------------------------------------")
    print(f"N = {N:,}")
    print(f"NA = {NA}")
    print(f"n = {N_MEDIUM}")
    print(f"alpha = {np.rad2deg(alpha):.3f} deg")
    print(f"A = {A:.8g}")
    print(f"B = {B:.8g}")
    print(f"C = {C:.8g}")
    print(f"Use background = {USE_BACKGROUND}")
    print(f"Background setting = {BACKGROUND_FRACTION_OF_MAX_C0*100:.1f}% of max ideal C0, same in all four channels")
    print()

    for mode, res in results.items():
        b0, b45, b90, b135 = res["background"]
        print(f"Case: {mode}")
        print(f"  background_0   = {b0:.8g}")
        print(f"  background_45  = {b45:.8g}")
        print(f"  background_90  = {b90:.8g}")
        print(f"  background_135 = {b135:.8g}")
        print(f"  r min/median/max = {np.min(res['r']):.6g}, {np.median(res['r']):.6g}, {np.max(res['r']):.6g}")
        print()

    # Plot theta distribution from solid angle.
    theta_bins_deg = np.linspace(0, 90, 91)
    theta_centers_deg, theta_percent = percent_hist(np.rad2deg(theta_folded), theta_bins_deg)

    theta_grid = np.linspace(0, np.pi/2, 500)
    expected_percent_per_1deg = 100 * np.sin(theta_grid) * np.deg2rad(1)

    plt.figure(figsize=(7, 5))
    plt.bar(theta_centers_deg, theta_percent, width=0.9, alpha=0.7, label="Monte Carlo")
    plt.plot(np.rad2deg(theta_grid), expected_percent_per_1deg, linewidth=2, label="Expected sin(theta)")
    plt.xlabel("folded theta (deg)")
    plt.ylabel("points per 1-deg bin (%)")
    plt.title("Isotropic solid-angle distribution after theta folding")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "theta_distribution.png", dpi=200)

    # Flipped theta-r relationship: r on x, theta on y.
    theta_line = np.linspace(0, np.pi/2, 1000)
    r_line = r_formula(theta_line, A, B, C)

    plt.figure(figsize=(7, 6))
    plt.plot(r_line, np.rad2deg(theta_line), linewidth=2, label="analytic no-background Fourkas")

    for mode, res in results.items():
        theta_centers, r_mean, r_p05, r_p95 = binned_theta_vs_r(theta_folded, res["r"])
        theta_deg = np.rad2deg(theta_centers)

        label = "Itot = 1 + bg" if mode == "constant" else "Itot ∝ sin²θ + bg"
        plt.plot(r_mean, theta_deg, ".", label=label)
        plt.fill_betweenx(theta_deg, r_p05, r_p95, alpha=0.15)

    plt.xlabel("anisotropy radius r")
    plt.ylabel("folded theta (deg)")
    plt.title("Flipped theta-r relationship with fixed 15% channel background")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "theta_r_relationship_flipped.png", dpi=200)

    # r distributions, comparing both Itot models.
    r_max_ideal = r_formula(np.pi/2, A, B, C)
    r_bins = np.linspace(0, r_max_ideal, 120)

    plt.figure(figsize=(7, 5))
    for mode, res in results.items():
        centers, percent = percent_hist(res["r"], r_bins)
        label = "Itot = 1 + bg" if mode == "constant" else "Itot ∝ sin²θ + bg"
        plt.plot(centers, percent, label=label)
    plt.xlabel("anisotropy radius r")
    plt.ylabel("points per bin (%)")
    plt.title("r distribution with fixed 15% channel background")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "r_distribution_comparison.png", dpi=200)

    # Anisotropy clouds, separate plots.
    n_plot = min(100_000, N)
    idx = rng.choice(N, size=n_plot, replace=False)

    for mode, res in results.items():
        title = "Itot = 1 + fixed background" if mode == "constant" else "Itot ∝ sin²θ + fixed background"
        plt.figure(figsize=(6, 6))
        plt.scatter(res["x"][idx], res["y"][idx], s=1, alpha=0.15)
        plt.xlabel("x = (C0 - C90) / (C0 + C90)")
        plt.ylabel("y = (C45 - C135) / (C45 + C135)")
        plt.title(title)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"anisotropy_cloud_{mode}.png", dpi=200)

    # Save lookup table with both cases.
    theta_centers, r_mean_const, r_p05_const, r_p95_const = binned_theta_vs_r(theta_folded, results["constant"]["r"])
    _, r_mean_sin2, r_p05_sin2, r_p95_sin2 = binned_theta_vs_r(theta_folded, results["sin2theta"]["r"])

    table = np.column_stack([
        np.rad2deg(theta_centers),
        theta_percent,
        r_formula(theta_centers, A, B, C),
        r_mean_const,
        r_p05_const,
        r_p95_const,
        r_mean_sin2,
        r_p05_sin2,
        r_p95_sin2,
    ])
    header = (
        "theta_deg_center,theta_percent_per_1deg_bin,"
        "r_formula_no_bg,"
        "r_mean_Itot1_bg,r_p05_Itot1_bg,r_p95_Itot1_bg,"
        "r_mean_ItotSin2_bg,r_p05_ItotSin2_bg,r_p95_ItotSin2_bg"
    )
    np.savetxt(OUTDIR / "theta_r_table_background_comparison.csv", table, delimiter=",", header=header, comments="")

    print(f"Saved output folder: {OUTDIR.resolve()}")
    print("Files:")
    for p in sorted(OUTDIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
