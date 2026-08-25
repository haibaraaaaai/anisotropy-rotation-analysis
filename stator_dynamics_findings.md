# BFM stator-dynamics analysis — findings & method notes

**File analysed:** `data/Patricia/20250311/file.tdms` (gold nanorod, single BFM)
**Pipeline:** `Fourkas_processing_v4.ipynb`, Sections 12–15e
**Sample rate:** 250 kHz raw; phase tracked at 25 kHz (`PHASE_DECIMATION = 10`)
**Date:** 2026-06-16

---

## 1. Summary / headline result

On a single rotating gold-nanorod BFM we can recover an **undistorted rotation rate**
and resolve **discrete speed states whose transitions cluster at ~50 Hz — the
single-stator quantum.** The dominant noise on the speed is **mechanical (thermal
wobble of the rod on its compliant linkage), not instrumental**, and it sets a
real resolution floor of ~10 Hz at the optimal averaging time `dt ≈ 0.1 s`. Step
detection on a switching region (230–260 s) returns step sizes that pile up around
±50 Hz (and ±75 Hz ≈ 1.5 stators), consistent with catch-bond stator engage/disengage
(u↔w) events for dwells longer than ~0.1 s.

---

## 2. Why speed must come from ψ, not dφ/dt

- `φ` is the **lab-frame** azimuthal angle. When the rotation axis is tilted, equal
  increments of true mechanical rotation project onto **unequal** increments of `φ`,
  so `dφ/dt` carries a spurious once-per-revolution modulation. `φ` is still a correct
  *orientation*; its time-derivative is just not a faithful *speed*.
- Fix: as the rod turns, the anisotropy vector `(a_x, a_y)` traces a **fixed closed
  reference loop** (set by the geometry Θ, Φ, Λ). The position **along** that loop is a
  faithful, monotonic rotation coordinate `ψ` regardless of axis tilt — one lap = one
  mechanical revolution.
- Implementation (Section 12): build the Fourkas template uniformly in `ψ`
  (`N = 360` points), match each measured point to the nearest template index inside a
  hard continuity window `±W` of the previous match (prevents ψ↔ψ+π branch flips near
  self-intersections), refine to a **continuous sub-index** by projecting onto the local
  template polyline (removes 2π/N quantisation noise), and accumulate signed steps into a
  continuous unwrapped `ψ`. **Speed = dψ/dt of ψ.**
- Ideal end state: speed from the ψ projection **+** angle from phase-guided Fourkas
  inversion (angle is a first-class quantity, not distorted — only `dφ/dt`-as-speed is).

---

## 3. Noise characterisation (the part to defend carefully)

### 3a. Instrument noise is negligible (PSD overlay, Section 15b)
- Reference: mentor's **fixed (surface-immobilised) rods** (`FigS3a.npz`) — no motor, no
  linkage → pure detector/instrument floor. The plotted source data (npz) is sufficient;
  no raw tdms needed.
- Our rotating residual PSD sits **4–10 orders of magnitude above** the fixed-rod floor at
  **every frequency**. Since noise only corrupts signal at the **same** frequency, the
  comparison is purely **vertical** — instrument noise is negligible at all timescales.
- A sharp tone at ~216 Hz = the rotation frequency (a per-revolution tracking artifact),
  not a physical speed fluctuation.

### 3b. The real noise is colored wobble (Allan deviation, Section 15)
- Allan deviation of the rotation rate `σ_Ω(τ)` on a calm window (250–300 s):
  - `dt = 1 ms → 65 Hz`, `10 ms → 15.6 Hz`, `100 ms → 10.4 Hz`, `1 s → 9.5 Hz`.
  - Left branch follows `τ^-1/2` (white); a kink at ~4.6 ms = one rotation period
    (the per-rev tone averages out over a full turn); then a **flicker plateau at ~9–10 Hz**.
  - (The printed "minimum at 10 s, 3.5 Hz" is an edge artifact — too few independent
    windows at the end of a 50 s segment. The honest floor is ~10 Hz.)
- **"Colored" = power concentrated at low frequency = the noise has memory/correlation**
  (opposite of white, which is flat-spectrum and memoryless). White noise averages away as
  `τ^-1/2`; slow colored noise barely changes within the averaging window, so averaging
  longer doesn't cancel it → the plateau. The plateau height is the size of the wobble
  fluctuations too slow to average at these timescales — a **hard physical wall**, not a
  measurement limit.

### 3c. What the colored wobble says about the physical rotation
1. **There is a spring between the motor and what we measure.** The rod reports rotation
   through a compliant linkage (hook + filament stub); colored/correlated angle noise is the
   fingerprint of that elastic compliance — the rod behaves as a damped torsional relaxator
   with a relaxation time ≈ (linkage stiffness)/(rotational drag).
2. **The motor's true stepping is sharper than what we see.** The compliant linkage
   **low-pass filters** abrupt torque changes (a stator engaging) before they reach the rod.
   So the resolution limit on *real mechanical events* is set by the **linkage**, not the
   detector — this is the true reason `dt` cannot be pushed arbitrarily small.
3. **It reframes the ~10 Hz floor as a property of the motor–rod system**, not of the
   instrument. That floor is itself a reportable result: it sets the fundamental resolution
   for stator counting on this assay.

---

## 4. Choice of dt (averaging window)

- Tradeoff: small `dt` → wobble dominates (`σ_Ω ~ σ_θ/dt`, blows up); large `dt` → short
  dwells are blurred. The Allan curve has a **knee at ~0.1 s**.
- **`dt = 0.1 s` is the smallest window that already reaches the noise floor** (~10 Hz).
  Going to 1 s improves noise only 10.4→9.5 Hz (~10%) but costs 10× time resolution.
- Important subtlety for step detection: because the wobble is **colored**, pooling
  samples within a dwell does **not** give the naive `σ/√N` improvement — adjacent samples
  are correlated. Use `dt = 10 ms` as a fine **detection grid** (don't miss short dwells),
  but the **effective level discrimination still floors near ~10 Hz**. Don't threshold
  individual 10 ms samples; let a changepoint/HMM accumulate evidence.
- Detection rule of thumb: a step of size Δ is safely callable when Δ ≳ 4–6·σ_eff.
  With σ_eff ≈ 10 Hz and a single-stator Δ ≈ 50 Hz → ~5:1 contrast (comfortable).

---

## 5. Whole-file sweep (Section 15c)

- `dt = 0.1 s` speed over the full record (0–311 s), with a time-resolved speed histogram
  (per-column-normalised, so band **position** is meaningful but **brightness is not**
  comparable across time).
- Global mean 195 Hz, std 34.5 Hz — the std far exceeds the ~10 Hz floor, so most of the
  spread is **real state structure**.
- Distinct long-lived states:
  - 0–110 s: broad ~175–210 Hz with frequent brief downward spikes (transient pauses).
  - 110–140 s: jumps up to ~225–240 Hz (held ~30 s).
  - 140–230 s: ~200–215 Hz.
  - **230–260 s: descends ~200 → sustained ~150 Hz**, with a sharp dip to ~15–50 Hz near
    240–245 s, then recovers — the clearest multi-state region.
  - 265–310 s: the **cleanest single tight band**, ~215–225 Hz.

---

## 6. Allan deviation on a switching region (Section 15d)

- Red curve (switching, 225–260 s) sits clearly **above** the calm-reference floor across
  0.05–7 s and **keeps rising toward long τ** (8.4× the floor at 7 s) instead of plateauing.
- Interpretation: real switching is confirmed, but there is **no single clean hump** → the
  dwell-time distribution is **broad** (many sub-second dwells + a few multi-second states;
  the big 225 Hz state lasts 8.5 s). A broad / power-law dwell distribution gives a rising
  Allan curve, consistent with the mentor's non-Lorentzian pause statistics. Allan confirms
  *seconds-long slow states exist* but can't read a single timescale — need the step list.

---

## 7. Step detection on 230–260 s (Section 15e)

- Method: `dt = 0.1 s` speed → **exact dynamic-programming changepoint** (L2/Gaussian cost,
  BIC-like penalty from the ~10 Hz floor; one knob `STEP_PEN_K`).
- Result: 25 states / 24 transitions. **Step sizes cluster at the stator quantum:**
  - ~25–30 Hz group: +31, −30, +28, −29, −25, +26, +24 (partial / half-quantum)
  - **~45–53 Hz group (dominant): −53, +47, −44, +51, +46, +47, +49, +47 ≈ one stator**
  - ~60–80 Hz group: −81, +75, −61, −64, +79, −72 (≈ 1.5 stators, or stator + pause)
  - **median |step| = 46.1 Hz**, max 80.7 Hz.
- The ~50 Hz cluster is exactly the single-stator quantum → genuine catch-bond stator
  switching (u↔w). The final jump up to a rock-steady 225 Hz held 8.5 s is a clean
  occupancy state. Very low excursions (18, 31 Hz) are near-stalls/pauses (possibly the
  strong-binding s state or full disengagement), not simple u↔w.

### Caveats before over-claiming
1. The penalty matters: 25 states in 30 s includes single-0.1 s-sample "states" (dwells
   0.1–0.2 s) at the resolution limit — could be transient dips/backsteps, not true levels.
   Raise `STEP_PEN_K` (3–4) to keep only confident multi-sample states; the ~50 Hz cluster
   should survive while single-sample spikes drop out.
2. Dwells shorter than `dt` are censored, so the short-time end of the dwell-time histogram
   is biased. Claim only the **existence** of u↔w and the **long tail / level structure**,
   not absolute counts in the shortest bins.

---

## 8. Open items / next steps

- [ ] **Find a cleaner file** where the raw speed histogram alone shows separated peaks at
      distinct stator levels — use as the baseline for refining step detection + dwell-time
      distribution.
- [ ] Sweep `STEP_PEN_K` (≈3–4) to separate confident states from resolution-limit spikes.
- [ ] Pool detected steps across the whole file into a **step-size histogram** → show the
      ~50 Hz quantum as a clean peak (the single most convincing single-stator figure).
- [ ] Build the **dwell-time distribution** from confident states (treat <dt as censored).
- [ ] Optional: chase/remove the 216 Hz per-rev tracking tone (cosmetic; already averaged
      out at `dt = 0.1 s`).
- [ ] Optional: HMM with a slow-drift latent for the broad/power-law dwell case.

## 9. Housekeeping
- `markitdown` is **not installed** in the active env (`pip install 'markitdown[pdf]'` into
  `.venv` if the relaxation-time paper is ever needed; reportedly no specific timescale in it).
- Mentor's FigS3 source data (`FigS3a/b/c.npz`) is sufficient for the instrument-floor
  comparison; raw tdms not required.
