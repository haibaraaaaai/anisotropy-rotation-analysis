# file6 (Patricia/20250724) — autonomous analysis findings

**File:** `data/Patricia/20250724/file6.tdms`
**Pipeline:** `Fourkas_processing_v4.ipynb`, Section 16 (16a–16f). Speed = dψ/dt
(undistorted phase-loop estimator), decimated grid `_FS_EFF ≈ 25 kHz`.
**Duration:** 663.1 s · 16,576,482 phase samples.

> TL;DR — **A real resurrection is present, but the *clean* discrete evidence is
> narrower than first claimed.** The file has three regions: rotating → a long
> deliberate stop (~328 s) → restart. The single robust discrete feature is a
> **~27 Hz plateau held ~4.5 s at the very start of the restart** (one stator
> engaging); the higher "steps" during the fast climb are largely the changepoint
> over-segmenting the ~15 Hz noise. The stop is **not** a static parked state —
> it evolves coast → tightly-confined (attached) → re-loosening before restart.
>
> **Corrections to the first draft (see §2, §4, §7):** the upper restart staircase
> was over-fit; and the stopped region is **not** a clean "motor-off control" that
> proves the 15 Hz is motor noise — an attached linkage still has memory at rest.

---

## 1. Three-region structure (16a)

Auto-detected with a 25 Hz rotating/stopped threshold on 0.2 s speed estimates,
then morphological cleanup.

| Region | Time (s) | Duration | Mean speed |
|--------|----------|----------|------------|
| Rotating #1 | 0 – 181.5 | 181.5 s | 126.5 Hz |
| **STOPPED** | 181.7 – 510.1 | **328.4 s** | 1.7 Hz |
| Rotating #2 | 510.3 – 662.9 | 152.6 s | 183.4 Hz |

This matches your recollection exactly: *"for a long time in the middle we
stopped the rotation."* The stop is real and lasts ~5.5 minutes.

---

## 2. The resurrection — what is real vs over-fit (16d, 16h)

The steps-reality check (16h: raw fine-dt speed + the angle *integral*, the
least-noisy view, since a step in speed = a kink in revolutions) gives a more
honest picture than the bare changepoint fit:

**Robust (visible in the raw data):**
- A **~27 Hz plateau held ~4.5 s** right at the restart onset (~509.5–514 s).
  The raw 50/100 ms speed sits on it and the revolutions show a clean kink.
  This is the strongest single-stator signature in the file.
- The **overall** climb (stop → ~27 → fast rise to ~300+ Hz over ~3 s) is real.

**Over-fit (NOT trustworthy):**
- The fine upper "staircase" `147 → 230 → 314 → 381 → 327 → 385 → 344 Hz` lands
  inside the noisy high-speed band (±15–40 Hz wobble). The detrended revolutions
  there is a smooth bow, not piecewise-linear — the changepoint is chopping
  noise. **You were right to be skeptical of the resurrection "staircase."**

The **slow-down into the stop (155–182 s)** fit reads `135 → 103 → 74 → 38 → 21
Hz`. The raw shows rough plateau-like structure, but the detrended revolutions is
a smooth deceleration bow — so this is **mostly a continuous wind-down**, not
crisp single-stator releases. Treat the slowdown levels as coarse, not quantal.

---

## 3. Speed-level peaks (16b)

KDE + `find_peaks` on rotating-only speed (robust across dt = 50/100/200 ms):

- **Region 1 (0–181 s):** peaks at **28, 99, 158 Hz** (spacings ≈ 72, 59 Hz)
- **Region 2 (510–663 s):** peaks at **27, 146, 259 Hz** (spacings ≈ 118, 113 Hz)

Confirms your impression of *"many distinct peaks."* Notable: the **inter-peak
spacing in region 2 (~115 Hz) is about twice that of region 1 (~60 Hz)** — the
motor came back with a different effective quantum / load / stator number after
the rest.

---

## 4. Rotating vs stopped Allan deviation (16c) — the control argument

Allan deviation of the rotation rate Ω (converted to Hz):

| Window | Floor |
|--------|-------|
| Stopped (250–480 s) | **~0.4 Hz plateau** (≈1 Hz at dt = 0.1 s) |
| Rotating #1 (20–120 s) | **~15.3 Hz floor** |
| Rotating #2 (post-restart) | **~19 Hz floor** |

The stopped-region wobble is **15–40× smaller** than the rotating floor.

**Correction / caveat.** The earlier conclusion ("so the 15 Hz is motor noise,
not linkage wobble; the stop is a perfect control") was overstated. When stators
are attached but not driven, the rod is **still** a confined torsional relaxator
— the elastic linkage retains its memory *at rest* (this is exactly the confined
signature measured in §7). So the stop is a control only for the **static** wobble
floor; it cannot prove the driven 15 Hz fluctuation is "motor not linkage,"
because during rotation the linkage is loaded and twisting, and tracking error
scales with speed. Honest statement: the 15 Hz is a property of the **driven
state** (motor torque noise, driven-linkage twist, and speed-dependent tracking,
not separable here), and it is ~40× larger than the static confined wobble.

---

## 5. Pooled step statistics (16e)

Changepoint (l2 cost, penalty k=3) over both rotating regions:

- Region 1: 44 states / 43 transitions; Region 2: 64 states / 63 transitions.
- **Pooled N = 106 steps; median |step| = 66 Hz, mean = 71.6 Hz.**
- |step| histogram (15 Hz bins) has a **mode at ~38–52 Hz** (single-stator-like)
  with a broad shoulder out to ~130 Hz (multi-unit jumps).

The distribution is **broad, not crisply quantized**, because the ~15 Hz motor
noise floor blurs individual quanta and the noisy mid-regions over-segment. The
*cleanest* quantal evidence is the resurrection/de-resurrection staircases
themselves (§2), where levels are well separated.

---

## 6. Dwell-time distribution (16f)

Dwell = time spent in each detected speed level (censored below the 0.2 s
detection window; only 7% of dwells affected).

- **N = 108 dwells; median = 2.0 s, mean = 3.1 s, max = 15.8 s; 74% > 1 s.**
- The survival function P(T > t) is **close to a single exponential**
  (mean dwell ≈ 3.3 s, i.e. a transition rate ≈ 0.3 s⁻¹) — roughly Markovian
  with one dominant timescale rather than a strongly broad / power-law process.

---

---

## 7. Does the stop change character over time? (16g, 16g2) — yes, 3 phases

Test of your sodium hypothesis ("attached-but-no-Na⁺ early → all stators detach
late"). Rolling 15 s windows across the whole stop, tracking net drift, wobble
amplitude (speed SD), and the angle-MSD log-log slope (1 = free/creeping,
0 = confined = restoring force = attached):

| Phase | Time | drift | speed SD | MSD slope | reading |
|-------|------|-------|----------|-----------|---------|
| **coast** | 182–265 s | ~6 Hz → falls | ~4 Hz | ~0.86 | rod still winding down / creeping |
| **locked** | 265–470 s | ~0.03 Hz | ~0.29 Hz | ~0.15 | tightly **confined — attached relaxator at rest** |
| **loosening** | 470–510 s | ~0.2–0.8 Hz | ~0.5–1.1 Hz | ~0.45 | confinement weakens — restart precursor |

So the stop is **not** a single static state — your intuition that properties
change is correct — **but not in the predicted direction.** The rod is *most
confined in the middle*, not early. Interpretation:

- The rod stays **attached throughout** (a clear restoring force / confined well
  in the deep middle; no clean free-diffusion "detached" phase ever appears — the
  late slope only rises to ~0.45, not back to 1.0).
- What's modulated is the **drive (Na⁺ / torque)**, not stator attachment: the
  motor stops driving (rod settles into the linkage well) and then re-energizes
  ~40 s before full restart (loosening = creep returning as drive comes back).
- The early "free-diffusion-like" slope is just the **deceleration coast tail**
  (the rod loses ~37 revolutions of residual rotation before parking at ~265 s),
  not Brownian detachment.

**Caveat:** this can't exclude that one or two stators detach/rebind — it only
shows at least one stays bound (confinement never fully vanishes). If you wanted
to see clean detachment you'd look for the MSD slope going to ~1 *with* the
drift staying at zero; that combination does not occur here.

---

## How to reproduce
Run notebook Section 16 cells in order on the loaded file:
`16a` region map → `16b` histograms → `16c` Allan → `16d` resurrection zoom →
`16e` pooled steps → `16f` dwell distribution → `16g`/`16g2` stop-evolution
(attached vs detached) → `16h` steps-reality (raw + angle-integral). All cells
auto-detect the regions, so nothing is hard-wired to specific timestamps.

## Open questions / next ideas (for when you're back)
1. **The ~27 Hz restart plateau** is the cleanest single-stator number and it
   matches region-1's lowest histogram peak (28 Hz). Good internal check for
   one stator ≈ 25–28 Hz here. Worth confirming on other files.
2. **Why does region 2 spacing (~115 Hz) ≈ 2× region 1 (~60 Hz)?** Different
   stator count, viscous load change after the pause, or a calibration shift?
3. **Don't trust the high-speed "steps."** To get a real stator quantum, gate
   step detection to the low-speed onset only (stop → ~30 Hz first level), where
   SNR is good, rather than the noisy 300+ Hz band.
4. The **loosening ~40 s before restart** is a falsifiable prediction: if it's a
   Na⁺ re-energization precursor it should recur before every resurrection —
   check the other restart events / files.
