#!/usr/bin/env python3
"""Estimate the photon budget of a four-channel polarisation record (TDMS, 250 kHz).

The detectors are analogue: what is stored is volts, not counts. But if the volts are a
linear map of detected photons,

    V = g * N + V_off  (+ read noise)

then the *variance* of V carries the Poisson signature of N:

    Var(V) = g^2 * Var(N) + s_read^2 = g^2 * N + s_read^2 = g * (mean(V) - V_off) + s_read^2

so a straight line through (mean, variance) pairs taken at many different light levels has
slope g = volts per detected photon. This is the standard photon-transfer-curve (mean-variance)
calibration. Once g is known, photons per sample is just (V - V_off) / g, with no detector
datasheet involved.

Two things make it work on real rotation data:

  * the rod itself sweeps each channel over a wide range of intensities, so the many light
    levels the fit needs come for free;
  * the shot noise sits at 250 kHz while the rotation sits at ~1e2 Hz, so a short local
    high-pass (second differences) separates noise from signal almost perfectly.

Outputs: per-channel gain, per-channel and total photons per sample, and an independent
cross-check that compares the measured high-frequency azimuth noise against the noise a
Monte Carlo predicts from the estimated budget.

Accuracy, measured on synthetic records with a known gain (see the tests at the bottom of
this docstring's companion notes): with a --dark record supplied, the recovered budget is
within about 4 % of truth from 50 to 2e4 photons per sample. WITHOUT --dark it is biased
high, and the bias grows with intensity — about +5 % at 3e2 photons but +21 % at 2e4,
because the electronic offset then has to be inferred from the fit's own intercept and any
read noise is misattributed to shot noise. So: take a short shutter-closed record and pass
it with --dark. That single step is the difference between a 4 % number and a 20 % one.

Usage
-----
    python3 photon_budget_from_tdms.py DATA.tdms
    python3 photon_budget_from_tdms.py DATA.tdms --dark DARK.tdms --plot budget.png
    python3 photon_budget_from_tdms.py DATA_DIR/        # batch: every *.tdms in the folder

Run with --help for all options. Requires numpy, nptdms, scipy (matplotlib only for --plot).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_tdms(path, order=None, max_samples=None):
    """Return (channels_dict, sample_interval_s). channels_dict keys: '0','90','45','135'.

    max_samples : if given, read only the first N samples per channel (bounds memory/time on
    multi-GB records; the photon budget is invariant to how many representative samples are used).
    """
    from nptdms import TdmsFile

    with TdmsFile.open(str(path)) as tdms_obj:
        group = tdms_obj.groups()[0]
        channels = group.channels()
        if len(channels) < 4:
            sys.exit(f"{path}: found {len(channels)} channels, need 4")
        full = len(channels[0])
        datasize = full if max_samples is None else min(full, int(max_samples))
        sl = slice(None) if datasize == full else slice(0, datasize)

        try:
            time_inc = float(channels[0].properties["wf_increment"])
        except KeyError:
            time_inc = 4e-6
            print("Warning: no time metadata — assuming time_inc = 4e-6 s", file=sys.stderr)

        labels = ["0", "90", "45", "135"]
        if order is None:
            # try to read the polarisation from the channel names, else assume 0,90,45,135
            names = [c.name for c in channels[:4]]
            idx = _match_by_name(names, labels)
            if idx is None:
                print(f"Note: channel names {names} not recognised — assuming file order is "
                      f"0, 90, 45, 135. Override with --order if that is wrong.", file=sys.stderr)
                idx = [0, 1, 2, 3]
        else:
            idx = [labels.index(t) for t in order.split(",")]
            idx = [idx.index(k) for k in range(4)]

        data = {lab: np.asarray(channels[i][sl], dtype=float)
                for lab, i in zip(labels, idx)}
    note = "" if datasize == full else f" (first {datasize:,} of {full:,}, --max-samples)"
    print(f"Loaded {path}: {datasize:,} points{note} | {1.0 / time_inc:.0f} Hz | "
          f"{datasize * time_inc:.1f} s", file=sys.stderr)
    return data, time_inc


def _match_by_name(names, labels):
    """Map channel names like 'Pol 45' onto labels; longest-token-first to avoid 0-in-90."""
    out = [None] * 4
    for want_i, want in enumerate(sorted(labels, key=len, reverse=True)):
        for n_i, n in enumerate(names):
            if n_i in out:
                continue
            digits = "".join(ch if ch.isdigit() else " " for ch in n).split()
            if want in digits:
                out[labels.index(want)] = n_i
                break
    return out if all(v is not None for v in out) else None


# ---------------------------------------------------------------------------
# local mean / variance, high-passed so the rotation does not inflate the variance
# ---------------------------------------------------------------------------


def local_mean_var(v, block):
    """Split v into consecutive blocks; return (mean, hf_variance) per block.

    The variance is estimated from second differences,
        d[i] = v[i] - 2 v[i+1] + v[i+2],
    which annihilates any locally linear drift (i.e. the rod's rotation, which is smooth on
    the 4 us scale) and has Var(d) = 6 Var(v) for white noise. That factor 6 is exact, so
    Var(v) = Var(d) / 6 measures only the fast, sample-to-sample noise.
    """
    n = (len(v) // block) * block
    if n == 0:
        sys.exit("block size larger than the record")
    blocks = v[:n].reshape(-1, block)
    means = blocks.mean(axis=1)
    d = blocks[:, :-2] - 2 * blocks[:, 1:-1] + blocks[:, 2:]
    # robust scale: the MAD is insensitive to the occasional step or spike in a block
    mad = np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1)
    var = (mad * 1.4826) ** 2 / 6.0
    return means, var


def fit_ptc(means, var, n_bins=40, trim=(1.0, 99.0)):
    """Robust straight-line fit of variance against mean. Returns (slope, intercept, bins).

    Binning in mean before fitting keeps the dense low-intensity region from dominating, and
    taking the median variance in each bin makes the fit insensitive to blocks that contain a
    genuine transient rather than pure noise.
    """
    lo, hi = np.percentile(means, trim)
    keep = (means >= lo) & (means <= hi) & np.isfinite(var) & (var > 0)
    m, v = means[keep], var[keep]
    if m.size < 20:
        sys.exit("too few usable blocks — try a smaller --block")
    edges = np.linspace(m.min(), m.max(), n_bins + 1)
    which = np.clip(np.digitize(m, edges) - 1, 0, n_bins - 1)
    bm, bv, bn = [], [], []
    for k in range(n_bins):
        sel = which == k
        if sel.sum() >= 5:
            bm.append(np.median(m[sel]))
            bv.append(np.median(v[sel]))
            bn.append(int(sel.sum()))
    bm, bv, bn = np.asarray(bm), np.asarray(bv), np.asarray(bn)
    if bm.size < 4:
        sys.exit("too few populated intensity bins — the record may be too short or too flat")
    w = bn / bn.sum()
    slope, intercept = np.polyfit(bm, bv, 1, w=np.sqrt(w))
    # coefficient of determination, weighted
    pred = slope * bm + intercept
    ss_res = np.sum(w * (bv - pred) ** 2)
    ss_tot = np.sum(w * (bv - np.average(bv, weights=w)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, (bm, bv, bn, r2)


# ---------------------------------------------------------------------------
# cross-check: does the measured azimuth noise match the estimated budget?
# ---------------------------------------------------------------------------


def azimuth_hf_noise(data, offsets):
    """s.d. of the high-frequency part of the inferred azimuth, in degrees.

    Same second-difference trick as above: phi is computed sample by sample, unwrapped on the
    2*phi argument, and its second difference gives the fast noise with Var(d) = 6 Var(phi).
    """
    i0 = data["0"] - offsets["0"]
    i90 = data["90"] - offsets["90"]
    i45 = data["45"] - offsets["45"]
    i135 = data["135"] - offsets["135"]
    two_phi = np.unwrap(np.arctan2(i45 - i135, i0 - i90))
    phi = np.degrees(0.5 * two_phi)
    d = phi[:-2] - 2 * phi[1:-1] + phi[2:]
    mad = np.median(np.abs(d - np.median(d)))
    return (mad * 1.4826) / np.sqrt(6.0)


def measured_anisotropy(data, offsets, smooth=32):
    """Median anisotropy magnitude r = hypot(x, y) of the record, and the implied theta.

    r must be measured on *smoothed* channels: on raw 4 us samples the noise inflates r
    (it is the length of a noisy 2-vector). The window has to be short compared with one
    revolution — 32 samples is 128 us, a small fraction of a typical 100 Hz revolution.
    """
    k = np.ones(int(smooth)) / float(smooth)
    sm = {}
    for lab in LABELS:
        sm[lab] = np.convolve(data[lab] - offsets[lab], k, mode="valid")
    denom_a = sm["0"] + sm["90"]
    denom_b = sm["45"] + sm["135"]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(denom_a > 0, (sm["0"] - sm["90"]) / denom_a, np.nan)
        y = np.where(denom_b > 0, (sm["45"] - sm["135"]) / denom_b, np.nan)
    r = np.hypot(x, y)
    tot = denom_a + denom_b
    bright = tot > np.nanpercentile(tot, 50)
    return float(np.nanmedian(r[bright])), float(np.nanmedian(tot[bright]))


def predicted_noise(n_total, r, n_trials=20000, seed=0, A=None, B=None, C=None):
    """Monte Carlo noise at the given budget and anisotropy, averaged over azimuth.

    The four channel probabilities follow from r and the azimuth alone, because the pair sums
    are identically equal in the Fourkas model:
        p0, p90 = (1 +- x)/4,   p45, p135 = (1 +- y)/4,   x = r cos2phi, y = r sin2phi.
    Returns (phi s.d. in deg, r-form theta s.d. in deg, P-form theta s.d. at the worst azimuth).
    """
    rng = np.random.default_rng(seed)
    phis = np.deg2rad(np.arange(0.0, 90.0, 2.5))
    sd_phi, sd_th_r, sd_th_P = [], [], []
    for ph in phis:
        x, y = r * np.cos(2 * ph), r * np.sin(2 * ph)
        p = np.array([(1 + x) / 4, (1 - x) / 4, (1 + y) / 4, (1 - y) / 4])
        p = np.clip(p, 1e-12, None)
        counts = rng.poisson(np.outer(n_total * p, np.ones(n_trials)))
        c0, c90, c45, c135 = counts
        phi_hat = np.degrees(0.5 * np.arctan2(c45 - c135, c0 - c90))
        resid = (phi_hat - np.degrees(ph) + 45.0) % 90.0 - 45.0
        sd_phi.append(np.std(resid))
        if A is not None:
            for form, store in (("r", sd_th_r), ("P", sd_th_P)):
                th = _theta_from_counts(counts, form, A, B, C)
                store.append(np.nanstd(th))
    out_phi = float(np.mean(sd_phi))
    if A is None:
        return out_phi, np.nan, np.nan
    return out_phi, float(np.mean(sd_th_r)), float(np.max(sd_th_P))


def _theta_from_counts(counts, form, A, B, C):
    """The two algebraically identical theta estimators, for the noise cross-check."""
    i0, i90, i45, i135 = counts.astype(float)
    phi = 0.5 * np.arctan2((i45 - i135) / 2.0, (i0 - i90) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if form == "P":
            Is = i0 + i90 + i45 + i135
            Id = i0 - i90 + i45 - i135
            K = np.sin(2 * phi) + np.cos(2 * phi)
            den = 2 * K * Is * C - 4 * B * Id
            s2 = np.where(den != 0, 4 * A * Id / den, np.nan)
        else:
            x = np.where(i0 + i90 > 0, (i0 - i90) / (i0 + i90), 0.0)
            y = np.where(i45 + i135 > 0, (i45 - i135) / (i45 + i135), 0.0)
            s2 = np.hypot(x, y) * A / (C - np.hypot(x, y) * B)
    return np.degrees(np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0))))


def fourkas_constants(NA=1.3, n=1.33):
    al = np.arcsin(min(NA / n, 1.0))
    c = np.cos(al)
    A = 1 / 6 - c / 4 + c ** 3 / 12
    B = c / 8 - c ** 3 / 8
    C = 7 / 48 - c / 16 - c ** 2 / 16 - c ** 3 / 48
    return A, B, C


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

LABELS = ["0", "90", "45", "135"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Estimate photons per sample per channel from a four-channel TDMS record.")
    ap.add_argument("tdms", help="the data file (rotating rod, or any record with a wide "
                                 "range of intensities)")
    ap.add_argument("--dark", help="optional dark/background TDMS record, used for the "
                                   "electronic offset and read noise")
    ap.add_argument("--block", type=int, default=64,
                    help="samples per block for the mean/variance pairs (default 64 = 256 us)")
    ap.add_argument("--bins", type=int, default=40, help="intensity bins for the fit")
    ap.add_argument("--order", default=None,
                    help="comma-separated polarisation order of the channels in the file, "
                         "e.g. 0,90,45,135 (default: read from names, else file order)")
    ap.add_argument("--smooth", type=int, default=32,
                    help="samples to average before measuring the anisotropy r (default 32; "
                         "must stay short compared with one revolution)")
    ap.add_argument("--min-r2", type=float, default=0.7, dest="min_r2",
                    help="reject the calibration if the mean-variance fit is worse than this "
                         "(default 0.7)")
    ap.add_argument("--NA", type=float, default=1.3, help="objective NA (default 1.3)")
    ap.add_argument("--n-index", type=float, default=1.33, dest="n_index",
                    help="refractive index (default 1.33)")
    ap.add_argument("--plot", default=None, help="write a diagnostic PNG here (single-file mode)")
    ap.add_argument("--json", default=None, help="write the numbers here as JSON (single-file mode)")
    ap.add_argument("--glob", default="*.tdms",
                    help="filename pattern to match when TDMS is a directory (default *.tdms)")
    ap.add_argument("--outdir", default=None,
                    help="where to write per-file results in directory/batch mode (default: a "
                         "'photon_budget' subfolder of the input directory)")
    ap.add_argument("--plots", action="store_true",
                    help="in directory/batch mode, also write one diagnostic PNG per file")
    ap.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                    help="analyse only the first N samples per channel (default: whole record; "
                         "use to bound memory/time on multi-GB files)")
    args = ap.parse_args(argv)

    dark = load_dark(args.dark, args) if args.dark else None

    target = Path(args.tdms)
    if target.is_dir():
        run_batch(target, args, dark)
        return None

    return process_one(target, args, dark, json_path=args.json, plot_path=args.plot)


def load_dark(path, args):
    """Return (offsets, read_var) measured from a dark/background record."""
    dark, _ = load_tdms(path, args.order, args.max_samples)
    offsets, read_var = {}, {}
    for lab in LABELS:
        offsets[lab] = float(np.median(dark[lab]))
        _, dv = local_mean_var(dark[lab], args.block)
        read_var[lab] = float(np.median(dv))
    return offsets, read_var


SUMMARY_COLS = ["file", "n_total_median", "n_total_mean", "anisotropy_r", "theta_deg",
                "phi_noise_measured_deg", "phi_noise_predicted_deg", "phi_noise_ratio",
                "theta_sd_rform_deg", "theta_sd_Pform_worst_deg", "sample_interval_s",
                "n_samples", "used_dark", "error"]


def write_summary_csv(summaries, path):
    """Write the per-file summaries as a flat CSV (one row per file), UTF-8 with BOM for Excel."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            row = {k: s.get(k, "") for k in SUMMARY_COLS}
            row["file"] = Path(s["file"]).name
            w.writerow(row)


def run_batch(directory, args, dark):
    """Process every matching TDMS file in a directory; write per-file and combined results."""
    import json

    outdir = Path(args.outdir) if args.outdir else directory / "photon_budget"
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(directory.glob(args.glob))
    if not files:
        sys.exit(f"no files matching {args.glob!r} in {directory}")

    print(f"Batch: {len(files)} file(s) in {directory}  ->  {outdir}", file=sys.stderr)
    summaries = []
    for i, f in enumerate(files, 1):
        tag = f"[{i}/{len(files)}] {f.name}"
        print(f"\n=== {tag} ===", file=sys.stderr)
        try:
            s = process_one(f, args, dark,
                            json_path=outdir / (f.stem + ".json"),
                            plot_path=(outdir / (f.stem + ".png")) if args.plots else None)
        except SystemExit as e:
            s = {"file": str(f), "error": str(e)}
            print(f"  {tag}: SKIPPED — {e}", file=sys.stderr)
        except Exception as e:  # keep the batch going; record why this file failed
            s = {"file": str(f), "error": f"{type(e).__name__}: {e}"}
            print(f"  {tag}: FAILED — {type(e).__name__}: {e}", file=sys.stderr)
        summaries.append(s)

    with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=1)
    write_summary_csv(summaries, outdir / "summary.csv")

    ok = sum(1 for s in summaries if "error" not in s)
    print(f"\nBatch done: {ok}/{len(files)} file(s) succeeded. Wrote "
          f"{outdir / 'summary.csv'} and {outdir / 'summary.json'}.", file=sys.stderr)
    return summaries


def process_one(path, args, dark=None, json_path=None, plot_path=None):
    """Estimate the photon budget of one TDMS file and return its summary dict.

    dark : optional (offsets, read_var) from load_dark(); when None the electronic offset is
           taken from each channel's own mean-variance intercept (biased high — see the
           module docstring).
    """
    data, dt = load_tdms(path, args.order, args.max_samples)

    if dark is not None:
        dark_offsets, read_var = dark
        offsets = dict(dark_offsets)
        have_dark = True
    else:
        offsets = {lab: 0.0 for lab in LABELS}
        read_var = {lab: None for lab in LABELS}
        have_dark = False

    print("\n  channel   gain g        offset    photons/sample            fit", file=sys.stderr)
    print("            (V/photon)     (V)      mean      median      R^2", file=sys.stderr)

    results, bins_for_plot = {}, {}
    for lab in LABELS:
        v = data[lab]
        means, var = local_mean_var(v, args.block)
        slope, intercept, extras = fit_ptc(means, var, args.bins)
        bm, bv, bn, r2 = extras
        if slope <= 0 or r2 < args.min_r2:
            sys.exit(
                f"channel {lab}: the mean-variance fit failed (gain {slope:.3g} V/photon, "
                f"R^2 {r2:.3f} < {args.min_r2}). Variance is not rising proportionally with "
                f"intensity, so this record cannot calibrate itself. Likely causes: the record "
                f"has too little intensity contrast; the noise is dominated by something other "
                f"than shot noise (electronic pickup, drift); or the data have been filtered or "
                f"averaged before storage. Try a smaller --block, a record with more contrast, "
                f"or lower --min-r2 if you are confident.")
        # if no dark file, the fit's own intercept gives the offset in volts:
        # var = g*(mean - off) + read  ->  off = -(intercept - read)/g ; with read unknown,
        # the zero-variance intercept is the best available offset estimate.
        off = offsets[lab] if have_dark else max(0.0, -intercept / slope)
        n = (v - off) / slope
        results[lab] = dict(gain=float(slope), intercept=float(intercept), offset=float(off),
                            r2=float(r2), n_mean=float(np.mean(n)),
                            n_median=float(np.median(n)),
                            read_var=read_var[lab])
        bins_for_plot[lab] = (bm, bv, slope, intercept)
        offsets[lab] = off
        print(f"    {lab:>4}   {slope:10.3e}  {off:8.4f}  {np.mean(n):9.1f} {np.median(n):10.1f}"
              f"   {r2:6.3f}", file=sys.stderr)

    # total photons per sample: form the summed photon trace first, then take its statistics.
    # (Summing the per-channel medians is biased low, because each channel's own rotational
    # modulation pulls its median below its mean.)
    n_trace = sum((data[l] - results[l]["offset"]) / results[l]["gain"] for l in LABELS)
    n_tot_mean = float(np.mean(n_trace))
    n_tot_med = float(np.median(n_trace))
    print(f"\n  TOTAL photons per {dt * 1e6:.0f} us sample, summed over four channels:"
          f"  mean {n_tot_mean:.0f}   median {n_tot_med:.0f}", file=sys.stderr)
    print(f"  equivalent detected rate: {n_tot_mean / dt / 1e6:.2f} Mcounts/s", file=sys.stderr)

    # -- independent check: the azimuth noise the budget predicts vs the one in the data
    A, B, C = fourkas_constants(args.NA, args.n_index)
    r_meas, _ = measured_anisotropy(data, offsets, args.smooth)
    meas = azimuth_hf_noise(data, offsets)
    pred, sd_th_r, sd_th_P = predicted_noise(n_tot_med, r_meas, A=A, B=B, C=C)
    theta_meas = np.degrees(np.arcsin(np.sqrt(np.clip(r_meas * A / (C - r_meas * B), 0, 1))))

    print(f"\n  anisotropy of the record: r = {r_meas:.3f}  ->  theta = {theta_meas:.1f} deg "
          f"(r_max = {C / (A + B):.4f} at NA {args.NA}, n {args.n_index})", file=sys.stderr)
    print(f"  cross-check on the azimuth noise floor:", file=sys.stderr)
    print(f"    measured high-frequency s.d. of phi : {meas:.3f} deg", file=sys.stderr)
    print(f"    predicted at {n_tot_med:.0f} photons     : {pred:.3f} deg", file=sys.stderr)
    ratio = meas / pred if pred > 0 else np.nan
    print(f"    ratio measured/predicted           : {ratio:.2f}", file=sys.stderr)
    if not np.isfinite(ratio):
        pass
    elif ratio > 1.4:
        print("    -> excess technical noise above shot noise (or the gain is "
              "under-estimated).", file=sys.stderr)
    elif ratio < 0.7:
        print("    -> below the shot-noise prediction: the data are filtered/averaged "
              "somewhere, or the gain is over-estimated.", file=sys.stderr)
    else:
        print("    -> consistent with shot-noise-limited detection.", file=sys.stderr)

    print(f"\n  single-sample theta noise implied by this budget (simulation, theta "
          f"{theta_meas:.0f} deg):", file=sys.stderr)
    print(f"    r-form, azimuth-averaged        : {sd_th_r:.2f} deg", file=sys.stderr)
    print(f"    P-form, worst azimuth (~67.5)   : {sd_th_P:.2f} deg", file=sys.stderr)

    summary = dict(file=str(path), sample_interval_s=dt, block=args.block,
                   n_samples=int(len(data["0"])),
                   per_channel=results, n_total_mean=n_tot_mean, n_total_median=n_tot_med,
                   anisotropy_r=r_meas, theta_deg=float(theta_meas),
                   phi_noise_measured_deg=float(meas), phi_noise_predicted_deg=float(pred),
                   phi_noise_ratio=float(ratio), used_dark=bool(have_dark),
                   theta_sd_rform_deg=sd_th_r, theta_sd_Pform_worst_deg=sd_th_P)
    if json_path:
        import json
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        print(f"\n  wrote {json_path}", file=sys.stderr)

    if plot_path:
        make_plot(bins_for_plot, results, plot_path, dt)
        print(f"  wrote {plot_path}", file=sys.stderr)

    # the one number to quote
    print(f"{n_tot_med:.0f}")
    return summary


def make_plot(bins_for_plot, results, path, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"0": "#E69F00", "90": "#0072B2", "45": "#009E73", "135": "#CC79A7"}
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)

    ax = axes[0]
    for lab, (bm, bv, slope, intercept) in bins_for_plot.items():
        ax.plot(bm, bv, "o", ms=2.5, color=colours[lab], label=f"{lab}°")
        xs = np.linspace(bm.min(), bm.max(), 50)
        ax.plot(xs, slope * xs + intercept, "-", lw=0.9, color=colours[lab])
    ax.set_xlabel("block mean signal (V)")
    ax.set_ylabel("high-frequency variance (V²)")
    ax.set_title("a  photon transfer curve", loc="left", fontsize=8, fontweight="bold")
    ax.legend(frameon=False, fontsize=6, ncol=2)

    ax = axes[1]
    labs = list(bins_for_plot)
    ax.bar(range(4), [results[l]["n_median"] for l in labs],
           color=[colours[l] for l in labs], width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{l}°" for l in labs])
    ax.set_ylabel(f"photons per {dt * 1e6:.0f} µs sample")
    ax.set_title("b  median budget per channel", loc="left", fontsize=8, fontweight="bold")
    fig.savefig(path, dpi=300)


if __name__ == "__main__":
    main()
