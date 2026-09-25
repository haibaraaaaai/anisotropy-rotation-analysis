# Noise & photon-count notes — anisotropy-rotation setup

_Date: 2026-09-02_

## TL;DR
- The photon counts already measured are **good enough** for the Monte Carlo (theory side). Don't re-measure them for that.
- The real limiter is **technical noise**: detection is **not shot-noise-limited** — a coherent **~5 kHz line on all four channels**. That is the thing to fix.
- **No detector saturation** — signals use only ~10–40% of the ADC range, so there is headroom to raise power.
- Best single next step: **one shutter-closed dark record taken on batteries**, passed with `--dark`.

## Setup recap
- 4-APD polarimeter: 0 / 90 / 45 / 135 polarisation channels; Berek compensator (s/p phase); circularly-polarised excitation; drilled hole-mirror at the objective BFP.
- DAQ: 250 kHz (4 µs/sample), 16-bit **±5 V** bipolar (scale = 10/65536 V per code), stored as int16 → scaled to volts on read.
- Detector powered from 2 × 12 V sealed-lead-acid batteries (≈ ±12 V). Mentor's tip to use *charged* batteries is correct (see below).

## Photon budget — `photon_budget_from_tdms.py`
Method: photon-transfer curve (block mean vs high-frequency variance) → gain `g` (V/photon) → photons `N = (V − offset)/g`. Self-calibrating, no datasheet. Outputs written to `<dir>/photon_budget/` (per-file JSON + PNG, plus `summary.csv` / `summary.json`).

Results (40M-sample cap, block=64, min-R²=0.7, **no dark**):
- **MT_DX_MRB: 13/20 files** — ~3,800–12,900 photons per 4 µs sample (≈ 1–3 Gcount/s).
- **ox Daping: 1/6** — only `file1` (~2,938). The rest fail R² (low signal/contrast + technical noise, not saturation).
- Skips are legitimate: empty 4 KB stubs, tiny files, and one channel with a *negative* gain (misbehaving channel).

Caveats on the absolute number `N`:
- **No `--dark`** → offset inferred from the fit intercept → **biased high by ~5–20%**.
- r / θ are computed from raw channels (no crosstalk / channel-correction) → approximate.
- Numbers reflect the first ~160 s of each (long) record, due to a memory cap; representative for a stationary source.

## Noise findings (measured from the data)
- Voltages peak at 0.3–2.3 V, **0% of samples near the ±5 V rail → no clipping / no saturation**. Room to raise power (watch the **brightest channel's peak** < ~3–4 V, not its mean).
- Mains 50 Hz comb: **weak** (1.1–1.6× the floor) — not the main problem.
- **Strong coherent line at ~5.0–5.2 kHz on all four channels, 10–24× above the shot-noise floor** (plus a 54 kHz line on one channel). Common-mode → suspect **shared power supply / APD bias-converter ripple / ground loop**.
- Shot-noise cross-check (measured ÷ predicted azimuth noise): **0.5–2.8**, several > 1.4 (file10 = 2.82) → **not shot-noise-limited**.

## Why batteries help (and why "charged" matters)
- A battery is quiet DC: **no switch-mode ripple, no mains hum, and it floats (no ground loop)**. A bench/wall supply injects all three.
- APD avalanche gain `M` is **exponentially sensitive to bias**, so any ripple on the supply rail **multiplies** the signal → coherent, signal-proportional noise (exactly the 5 kHz-type lines).
- "Charged" = low internal resistance + correct voltage (correct APD bias). Don't charge while measuring (charger = mains noise back in). A low-noise linear regulator (LDO) after the battery can pin the exact bias while keeping the isolation.

## What it means for the MC — `theta_estimator_montecarlo.py`
- The MC is a **shot-noise floor** only. Floor ≈ 79°/√N at θ=45°; the r-form beats the P-form by ×7–12 in variance, robustly across the photon range.
- A ±30% error in `N` shifts the floor by only ~14% (1/√N) and never changes the conclusion → **the photon count is fine as-is.**
- Real σθ ≈ (MC floor) × (noise ratio 1.3–2.8). A more precise `N` cannot close that gap; removing the technical noise can.
- To use it: feed each file's `n_total_median` **and** `theta_deg` from the JSON, running the MC **at that file's own θ** (the r-vs-P advantage is strongly θ-dependent).

### Three-form conditioning (added 2026-09-02)
The MC now compares **three** algebraically identical θ estimators (`--plot` → single panel `mc_three_forms.png`):
- **Fourkas (Eq 6–8)** — divides by cos2Φ → singular at Φ=45°/135°; **worst** (variance ×30 at its holes, and broadly elevated because its 3-channel azimuth is noisier too).
- **P-form** (the group's `Fourkas()` function) — divides by K=sin2Φ+cos2Φ → singular at 67.5°/157.5° (variance ×11); flat elsewhere.
- **r-form** — anisotropy magnitude, no singularity; best everywhere.

**Why the r-form s.d. ripples with azimuth (period 45°) but the P-form is flat:** the r-form uses r=hypot(x,y) of the two ratios, whose Poisson noise depends on the per-channel split (σ_x² ∝ I₀·I₉₀), and that split rotates with Φ. It reduces to σ_r² = const + ½·a₁·sin²(4Φ) → a ~9% ripple, minima at Φ=0/45/90/135, maxima at 22.5/67.5/… The P-form's noise is instead dominated by Var(I_d)=N_tot (the **total** count, split-independent) → flat, except the 1/K spike. Small effect (~9%); the r-form is still "flat and low" next to the F/P holes.

## Action items (for later)
1. **Dark record on batteries** (shutter closed, ~10 s) → pass with `--dark`. One acquisition simultaneously: de-biases `N` (~+20% → ~4%), confirms the supply-noise story, and measures the 5 kHz floor. **Highest value.**
2. Compare a dark spectrum **on the bench supply vs on batteries** — if the ~5 kHz line drops on batteries, it's the supply. Keep batteries charged, charger unplugged during runs; add an LDO if needed.
3. Optionally **raise excitation power** for more photons (headroom exists) — cap the brightest channel's peak at ~3–4 V.
4. For accurate r / θ: apply the **channel crosstalk correction** (the transmission matrix in the setup docs / `_chcor.npy`; the Berek already handles the s/p phase).
5. Re-run the photon budget after fixes and drive the **noise ratio → 1** — that ratio, not the photon number, is the scorecard.
