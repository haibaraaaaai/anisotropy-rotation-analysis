#!/usr/bin/env python3
"""Monte Carlo comparison of three algebraically identical theta estimators.

WHAT THIS DOES (and does not do)
--------------------------------
This is a pure simulation. There is no time axis, no rotation, no trace, no experimental
data. The procedure at each point is:

    1. fix the polar angle theta;
    2. pick a true azimuth phi;
    3. compute the four ideal Fourkas channel intensities at (theta, phi) and normalise
       them to probabilities p0, p90, p45, p135 summing to one;
    4. multiply by a photon budget N (TOTAL photons summed over the four channels) and draw
       n_trials independent Poisson realisations of the four channel counts;
    5. push every realisation through ALL THREE estimators and record the s.d. and mean of
       the recovered theta;
    6. repeat for each phi.

So the x axis of the output figure is the TRUE AZIMUTH OF THE ROD, not time. Each point is
an independent single-sample noise measurement at a fixed orientation, with Poisson counting
noise as the only error source: no background, no gain mismatch, no motion blur, no
polarisation crosstalk.

WHY IT IS WORTH DOING
---------------------
The three estimators are the same function of the same four numbers. `--self-test` proves
that numerically: in the noiseless limit all three return the input theta to ~1e-11 deg at
every non-singular azimuth. Therefore any difference in output scatter is not physics and not
a modelling choice, it is arithmetic. The Fourkas original (Eqs 6-8) divides by cos(2 phi),
which vanishes at phi = 45 and 135 deg; the P-form divides by sin(2 phi)+cos(2 phi), which
vanishes at 67.5 and 157.5 deg; near those azimuths Poisson noise in the numerator is
amplified without bound. The r-form uses the anisotropy magnitude and never forms such a
quotient, so it is well-conditioned at every azimuth.

WHAT IT CANNOT TELL YOU
-----------------------
Whether the penalty matters in your data. That depends on your photon budget (see
photon_budget_from_tdms.py), on whether the rod visits the bad azimuths, and on how much
averaging you apply downstream.

A CAVEAT WORTH KNOWING BEFORE YOU QUOTE ANY NUMBER FROM THIS
------------------------------------------------------------
The advantage is not uniform in theta, and the default theta = 45 deg happens to be near its
maximum. Measured at 1e4 total photons, at the worst azimuth (67.5 deg):

    theta   ideal r   dtheta/dr    r-form s.d.   P-form s.d.   variance ratio
      30    0.268        65 deg       0.90 deg      2.08 deg        5.3
      45    0.509        62           0.82          3.02           13.6
      60    0.728        79           0.97          3.72           14.9
      70    0.836       112           1.29          4.55           12.4
      75    0.875       147           1.67          5.03            9.1
      80    0.904       218           2.72          5.44            4.0
      85    0.921       433           3.61          6.68            3.4

The reason is separate from the conditioning problem: as theta -> 90 deg the anisotropy r
saturates towards r_max = C/(A+B) = 0.9269, so d(theta)/dr blows up and BOTH estimators lose
precision in theta. The r-form still wins everywhere, but by 3-4x in variance rather than
13-15x. So run this at YOUR theta before quoting a factor in the thesis; a reviewer is
entitled to ask whether 45 deg was chosen for effect.

Usage
-----
    python3 theta_estimator_montecarlo.py --self-test
    python3 theta_estimator_montecarlo.py --plot conditioning.png
    python3 theta_estimator_montecarlo.py --theta 75 --photons 900 --plot mine.png --csv out.csv

Requires numpy; matplotlib only for --plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# the optical model
# ---------------------------------------------------------------------------


def fourkas_constants(NA=1.3, n_index=1.33):
    """The three collection constants A, B, C for a given objective and index.

    alpha is the half-angle of the collection cone, sin(alpha) = NA / n.
    """
    alpha = np.arcsin(min(NA / n_index, 1.0))
    c = np.cos(alpha)
    A = 1 / 6 - c / 4 + c ** 3 / 12
    B = c / 8 - c ** 3 / 8
    C = 7 / 48 - c / 16 - c ** 2 / 16 - c ** 3 / 48
    return A, B, C


def channel_probs(theta, phi, A, B, C):
    """The four ideal channel intensities at (theta, phi), normalised to sum to one.

    I_0   = A + B sin^2(th) + C sin^2(th) cos(2 ph)
    I_90  = A + B sin^2(th) - C sin^2(th) cos(2 ph)
    I_45  = A + B sin^2(th) + C sin^2(th) sin(2 ph)
    I_135 = A + B sin^2(th) - C sin^2(th) sin(2 ph)

    Note the two pair sums are identically equal: I_0 + I_90 == I_45 + I_135 == 2(A + B s2).
    That identity is why the two estimators below coincide exactly.
    """
    s2 = np.sin(theta) ** 2
    base = A + B * s2
    p = np.array([base + C * s2 * np.cos(2 * phi),
                  base - C * s2 * np.cos(2 * phi),
                  base + C * s2 * np.sin(2 * phi),
                  base - C * s2 * np.sin(2 * phi)])
    return p / p.sum()


# ---------------------------------------------------------------------------
# the two estimators
# ---------------------------------------------------------------------------


def theta_from_counts(counts, form, A, B, C, clamp=True):
    """Recover theta (degrees) from four channel counts.

    counts : array (4, N) of I_0, I_90, I_45, I_135.
    form   : "F" Fourkas original (Eqs 6-8), "P" the symmetrised sum/difference form,
             "r" the anisotropy-magnitude form. All three invert the same model and differ
             only in the arithmetic, hence in how Poisson noise is amplified.

    "F" form (Fourkas 2001, Eqs 6-8) -- a 3-channel azimuth, then divide by cos(2 phi):
        2 phi = atan2(I_45 - (I_0 + I_90)/2, (I_0 - I_90)/2)
        I_tot = (1/2A)[(1 - B/(C cos2phi)) I_0 + (1 + B/(C cos2phi)) I_90]
        sin^2(th) = (I_0 - I_90) / (2 I_tot C cos2phi)
    cos(2 phi) cancels analytically but vanishes at phi = 45 and 135 deg -> noise blows up.

    The "P" and "r" forms estimate the azimuth symmetrically from both numerators:
        2 phi = atan2(I_45 - I_135, I_0 - I_90)

    "P" form -- builds the totals and then divides by K = sin(2 phi) + cos(2 phi):
        Is = I_0 + I_90 + I_45 + I_135
        Id = I_0 - I_90 + I_45 - I_135
        sin^2(th) = 4 A Id / (2 K Is C - 4 B Id)
    K cancels analytically but vanishes numerically at phi = 67.5 and 157.5 deg, where it
    becomes a small denominator and amplifies whatever noise is in Id.

    "r" form -- builds the two anisotropies and their magnitude:
        x = (I_0 - I_90)/(I_0 + I_90),  y = (I_45 - I_135)/(I_45 + I_135)
        r = hypot(x, y)
        sin^2(th) = r A / (C - r B)
    No vanishing denominator: C - r B is bounded away from zero because r <= r_max = C/(A+B).

    clamp : clip sin^2(th) into [0, 1] before the arcsin, as the group code does. With
            clamp=False, out-of-range realisations become NaN instead, which changes the
            reported s.d. -- see --no-clamp.
    """
    i0, i90, i45, i135 = np.asarray(counts, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        if form == "F":
            # Fourkas original (Eqs 6-8): 3-channel azimuth, then divide by cos(2 phi).
            phi = 0.5 * np.arctan2(i45 - (i0 + i90) / 2.0, (i0 - i90) / 2.0)
            cos2 = np.cos(2 * phi)
            i_tot = (1.0 / (2 * A)) * ((1 - B / (C * cos2)) * i0
                                      + (1 + B / (C * cos2)) * i90)
            den = 2 * i_tot * C * cos2
            s2 = np.where(den != 0, (i0 - i90) / den, np.nan)
        elif form == "P":
            phi = 0.5 * np.arctan2((i45 - i135) / 2.0, (i0 - i90) / 2.0)
            Is = i0 + i90 + i45 + i135
            Id = i0 - i90 + i45 - i135
            K = np.sin(2 * phi) + np.cos(2 * phi)
            den = 2 * K * Is * C - 4 * B * Id
            s2 = np.where(den != 0, 4 * A * Id / den, np.nan)
        elif form == "r":
            phi = 0.5 * np.arctan2((i45 - i135) / 2.0, (i0 - i90) / 2.0)
            x = np.where(i0 + i90 > 0, (i0 - i90) / (i0 + i90), 0.0)
            y = np.where(i45 + i135 > 0, (i45 - i135) / (i45 + i135), 0.0)
            r = np.hypot(x, y)
            s2 = r * A / (C - r * B)
        else:
            raise ValueError(f"unknown form {form!r}")

    if clamp:
        s2 = np.clip(s2, 0.0, 1.0)
    else:
        s2 = np.where((s2 >= 0.0) & (s2 <= 1.0), s2, np.nan)
    return np.degrees(np.arcsin(np.sqrt(s2)))


FORMS = ("F", "P", "r")
FORM_LABELS = {"F": "Fourkas (Eq. 6\u20138)", "P": "P-form", "r": "r-form"}
FORM_HOLES = {"F": (45.0, 135.0), "P": (67.5, 157.5), "r": ()}


# ---------------------------------------------------------------------------
# self-test: the three forms are the same function
# ---------------------------------------------------------------------------


def self_test(NA=1.3, n_index=1.33):
    """Noiseless round trip, plus the pair-sum identity. Returns True if all checks pass."""
    A, B, C = fourkas_constants(NA, n_index)
    print(f"A = {A:.10f}   B = {B:.10f}   C = {C:.10f}", file=sys.stderr)
    print(f"r_max = C/(A+B) = {C / (A + B):.10f}\n", file=sys.stderr)

    ok = True
    worst = {f: 0.0 for f in FORMS}
    worst_pair = 0.0
    # azimuth grid offset off the singular angles (45, 67.5, 135, 157.5) so every form inverts
    for th_deg in (5.0, 20.0, 45.0, 67.5, 75.0, 89.0):
        for ph_deg in np.arange(0.3, 180.0, 1.0):
            p = channel_probs(np.deg2rad(th_deg), np.deg2rad(ph_deg), A, B, C)
            # scale by an arbitrary large "intensity": the estimators are scale free
            counts = (p * 1.0e9).reshape(4, 1)
            pair_a, pair_b = counts[0] + counts[1], counts[2] + counts[3]
            worst_pair = max(worst_pair,
                             float(abs(pair_a - pair_b)[0] / pair_a[0]))  # relative
            for form in FORMS:
                got = theta_from_counts(counts, form, A, B, C)[0]
                worst[form] = max(worst[form], abs(got - th_deg))

    print(f"noiseless round trip, worst |theta_out - theta_in| over "
          f"6 thetas x 180 azimuths:", file=sys.stderr)
    for form in FORMS:
        print(f"    {FORM_LABELS[form]:<16} : {worst[form]:.3e} deg", file=sys.stderr)
    print(f"pair-sum identity, worst relative |(I0+I90) - (I45+I135)| / (I0+I90) : "
          f"{worst_pair:.3e}", file=sys.stderr)
    for form in FORMS:
        if worst[form] > 1e-8:
            print(f"FAIL: {form}-form does not round trip", file=sys.stderr)
            ok = False
    if worst_pair > 1e-12:
        print("FAIL: pair sums are not identical", file=sys.stderr)
        ok = False

    # the three forms on identical noisy input (away from any hole): same answer to ~noise floor
    rng = np.random.default_rng(0)
    p = channel_probs(np.deg2rad(52.0), np.deg2rad(31.0), A, B, C)
    counts = rng.poisson(np.outer(1e7 * p, np.ones(2000)))
    ths = {f: theta_from_counts(counts, f, A, B, C) for f in FORMS}
    d = max(np.nanmax(np.abs(ths[a] - ths["r"])) for a in ("F", "P"))
    print(f"\nsame noisy counts through all forms, worst spread at 1e7 photons: "
          f"{d:.3e} deg", file=sys.stderr)
    print("(small but nonzero: at finite counts the shared phi estimate is itself noisy, "
          "which is\n exactly the conditioning effect this script measures)", file=sys.stderr)

    print("\nSELF-TEST " + ("PASSED" if ok else "FAILED"), file=sys.stderr)
    return ok


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def sweep(theta_deg, azimuths_deg, photon_budgets, n_trials, A, B, C,
          seed=12345, clamp=True):
    """Poisson Monte Carlo at each azimuth, for each budget, for all forms.

    Returns {budget: {form: {"sd": array, "bias": array, "nan_frac": array}}}.
    """
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(theta_deg)
    results = {}
    for n_photons in photon_budgets:
        out = {f: {"sd": [], "bias": [], "nan_frac": []} for f in FORMS}
        for ph_deg in azimuths_deg:
            p = channel_probs(theta, np.deg2rad(ph_deg), A, B, C)
            counts = rng.poisson(np.outer(n_photons * p, np.ones(n_trials)))
            for form in FORMS:
                th = theta_from_counts(counts, form, A, B, C, clamp)
                out[form]["sd"].append(np.nanstd(th))
                out[form]["bias"].append(np.nanmean(th) - theta_deg)
                out[form]["nan_frac"].append(float(np.mean(~np.isfinite(th))))
        for form in FORMS:
            for key in out[form]:
                out[form][key] = np.asarray(out[form][key])
        results[n_photons] = out
    return results


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

ORANGE, BLUE, GREY = "#E69F00", "#0072B2", "#666666"   # Okabe-Ito, colourblind safe


def make_figure(path, azimuths, results, budgets, theta_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = Path(__file__).with_name("mr_widget.mplstyle")
    if style.exists():
        plt.style.use(str(style))

    colours = {"F": "#0072b2", "P": "#d55e00", "r": "#000000"}
    linestyles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)

    for hole in (45.0, 67.5, 135.0, 157.5):
        if azimuths.min() <= hole <= azimuths.max():
            ax.axvline(hole, color="0.85", lw=0.6, zorder=0)

    for k, n in enumerate(budgets):
        ls = linestyles[k % len(linestyles)]
        for form in FORMS:
            ax.plot(azimuths, results[n][form]["sd"], ls, color=colours[form], zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("true azimuth Φ (deg)")
    ax.set_ylabel("recovered θ  s.d. (deg)")
    if azimuths.max() - azimuths.min() > 90:
        ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_title(f"θ = {theta_deg:g}°, Poisson noise only", fontsize=7)

    # colour = formula
    form_handles = [Line2D([0], [0], color=colours[f], lw=1.8) for f in FORMS]
    leg1 = ax.legend(form_handles, [FORM_LABELS[f] for f in FORMS], loc="upper left",
                     frameon=False, handlelength=1.5, borderaxespad=0.3, labelspacing=0.25)
    ax.add_artist(leg1)
    # linestyle = photon budget (only when more than one)
    if len(budgets) > 1:
        b_handles = [Line2D([0], [0], color="0.35", lw=1.4,
                            ls=linestyles[k % len(linestyles)]) for k in range(len(budgets))]
        ax.legend(b_handles, [f"{n:g} photons" for n in budgets], loc="upper right",
                  frameon=False, handlelength=2.4, borderaxespad=0.3, labelspacing=0.25)

    # panel-c-in-a-line: label each singular form's worst variance penalty vs the r-form
    nb = budgets[-1]
    for form in ("F", "P"):
        holes_in = [h for h in FORM_HOLES[form] if azimuths.min() <= h <= azimuths.max()]
        if not holes_in:
            continue
        ratios = (results[nb][form]["sd"] / results[nb]["r"]["sd"]) ** 2
        j = int(np.nanargmax(ratios))
        ax.annotate(f"×{ratios[j]:.0f}", (azimuths[j], results[nb][form]["sd"][j]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=6.5, color=colours[form])

    fig.savefig(path, dpi=400)
    for ext in (".pdf", ".svg"):
        fig.savefig(path.rsplit(".", 1)[0] + ext)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Monte Carlo conditioning comparison of three theta estimators "
                    "(Fourkas Eq 6-8, P-form, r-form). Pure simulation, Poisson noise only.")
    ap.add_argument("--self-test", action="store_true",
                    help="check that all three forms invert the model exactly, then exit")
    ap.add_argument("--theta", type=float, default=45.0,
                    help="polar angle held fixed, deg (default 45)")
    ap.add_argument("--photons", type=float, nargs="+", default=[1000.0, 10000.0],
                    help="TOTAL photons per sample summed over the four channels "
                         "(default 1000 10000)")
    ap.add_argument("--phi-min", type=float, default=0.0, help="first azimuth, deg")
    ap.add_argument("--phi-max", type=float, default=180.0, help="last azimuth, deg")
    ap.add_argument("--phi-step", type=float, default=1.0, help="azimuth step, deg")
    ap.add_argument("--trials", type=int, default=20000, help="Poisson draws per point")
    ap.add_argument("--NA", type=float, default=1.3, help="objective NA (default 1.3)")
    ap.add_argument("--n-index", type=float, default=1.33, dest="n_index",
                    help="refractive index (default 1.33)")
    ap.add_argument("--no-clamp", action="store_true",
                    help="discard out-of-range realisations as NaN instead of clipping "
                         "sin^2(theta) into [0,1]")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--plot", default=None, help="write the figure here (PNG; "
                                                "PDF and SVG siblings are written too)")
    ap.add_argument("--csv", default=None, help="write the sweep as CSV here")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test(args.NA, args.n_index) else 1

    A, B, C = fourkas_constants(args.NA, args.n_index)
    azimuths = np.arange(args.phi_min, args.phi_max + 1e-9, args.phi_step)
    budgets = list(args.photons)
    results = sweep(args.theta, azimuths, budgets, args.trials, A, B, C,
                    seed=args.seed, clamp=not args.no_clamp)

    print(f"theta = {args.theta:g} deg   NA = {args.NA:g}   n = {args.n_index:g}   "
          f"A = {A:.6f}  B = {B:.6f}  C = {C:.6f}   r_max = {C / (A + B):.6f}")
    print(f"{args.trials} Poisson trials per point, "
          f"clamp = {not args.no_clamp}, seed = {args.seed}\n")

    for n in budgets:
        print(f"  {n:g} total photons per sample:")
        for form in FORMS:
            d = results[n][form]
            print(f"    {FORM_LABELS[form]:<17} s.d. {d['sd'].min():5.2f}-{d['sd'].max():5.2f} "
                  f"deg   (bias {d['bias'].min():+.2f} to {d['bias'].max():+.2f})")
        for form in ("F", "P"):
            for hole in FORM_HOLES[form]:
                i = int(np.argmin(np.abs(azimuths - hole)))
                if abs(azimuths[i] - hole) < args.phi_step:
                    ratio = (results[n][form]["sd"][i] / results[n]["r"]["sd"][i]) ** 2
                    print(f"      -> {FORM_LABELS[form]} at its hole (Phi={hole:g} deg): "
                          f"variance x{ratio:.1f} vs r-form")
        print()

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("phi_deg,budget,form,sd_deg,bias_deg,nan_frac\n")
            for n in budgets:
                for form in FORMS:
                    d = results[n][form]
                    for k, ph in enumerate(azimuths):
                        fh.write(f"{ph:g},{n:g},{form},{d['sd'][k]:.6f},"
                                 f"{d['bias'][k]:.6f},{d['nan_frac'][k]:.6f}\n")
        print(f"wrote {args.csv}")

    if args.plot:
        make_figure(args.plot, azimuths, results, budgets, args.theta)
        print(f"wrote {args.plot} (+ .pdf, .svg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
