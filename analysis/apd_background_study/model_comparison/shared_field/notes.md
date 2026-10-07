# Shared stationary fields — 5 October 2026

The matrix and gains remain unchanged. This test follows the previous mask and freely varying overlap experiments. It tests whether one fixed coherent field and one fixed noninterfering background can explain both intervals. It does not assume the independently optimized overlaps from the last experiment describe a physical field.

## Observation model and assumptions

At pupil position u and analyzer j:

E_s,j(u,chi)=a0 (d_x+i d_y) [d_x F_x,j(u)+d_y F_y,j(u)+d_z F_z,j(u)].

E_b,j(u)=sum_m b_m g_m,j(u), with the same complex b_m at every trajectory point AND both intervals.

I_j=integral |E_s,j+E_b,j|² du + B_inc,j.

F are the collected dipole fields with water–oil Fresnel transmission and corrected BFP weighting. Circular annulus NA .38–1.3, water n=1.33, oil n=1.51. No amplitude/phase is adjusted independently per trajectory point. Excitation magnitude remains a0 sin(theta); complex factor d_x+i d_y also retains excitation phase. a0 is real, positive and constant within each interval. Background is shared in absolute corrected-intensity units, not independently renormalized to each interval.

Three-mode basis: uniform x and y pupil fields plus collected z-dipole (radial) field. Five-mode basis adds the two collected transverse-dipole patterns, orthogonalized against the existing modes. These are additional spatial shapes, not two new time-varying functions or claims that these are the actual sources. All modes are vector fields projected onto the four analyzers, so channel overlaps and backgrounds are physically linked. Background modes are orthonormal in full vector pupil power; their summed four-channel coherent power is 2 sum|b_m|². Noninterfering background is a positive Stokes intensity with equal pair sums, shared across intervals. A component outside the signal-overlap subspace is observationally indistinguishable from incoherent intensity; do not identify that split with source location.

Each interval has 20 parameters: cone geometry (3), ordered phase mapping (16), a0 (1). The three-mode shared background has 9 parameters; the five-mode version 13. Total 49 versus 53 jointly fitted parameters for 1024 training channel values (128 bins ×4×2). The phase mapping describes nonuniform timing/order, not background variation. Counting averaged bins as independent observations for formal statistical tests would not be justified.

The 10% cap is imposed relative to the smaller of the two training maximum totals, ensuring it holds in both windows. The relaxed cap is 200% of that reference and is inactive in the reported solutions. Independent-window models use the same reference/cap for comparability.

## Results

| Shared field | Held-out XY RMS, 30–31 s | Held-out XY RMS, 10–11 s | BG/max total, 30–31 s / 10–11 s |
|---|---:|---:|---:|
| 3 modes, relaxed | 0.0336 | 0.0509 | 53.05% / 52.76% |
| 5 modes, relaxed | 0.0281 | 0.0460 | 54.72% / 54.43% |
| 3 modes, 10% | 0.0950 | 0.1040 | 10.00% / 9.95% |
| 5 modes, 10% | 0.0930 | 0.1019 | 10.00% / 9.95% |

The additional two modes modestly improve the fit but do not enable a good fit at 10% background. The relaxed shared five-mode result uses approximately 55% background. Its coherent share is 0.9182; the rest is the fitted noninterfering component. This is a model decomposition, not a measured separation. The fitted noninterfering polarization lies at its upper bound in these solutions, indicating continuing model pressure.

## Independent-field and transfer controls

- Cap 2.0: independent fields XY RMS 0.0242 / 0.0453; field from 10–11 s frozen for 30–31 s gives 0.0408; reverse gives 0.0517. Target cone, phase map and a0 are refitted on target training cycles; target held-out cycles are used only for scoring.
- Cap 0.1: independent fields XY RMS 0.0935 / 0.1010; field from 10–11 s frozen for 30–31 s gives 0.0939; reverse gives 0.1040. Target cone, phase map and a0 are refitted on target training cycles; target held-out cycles are used only for scoring.

Sharing one field produces only a modest penalty relative to fitting it separately. This reduces concern that the earlier agreement required completely different fields in each window; it does not prove the physical field is stationary. Cone and brightness freedom can absorb some differences. A fully frozen model predicting the other interval was not tested, because the trajectory and brightness may differ.

## Validation and limits

- Retained previous alternate-cycle training/held-out averages (304 cycles at 30–31 s; 217 at 10–11 s). No new cycle selection, calibration fit or matrix changes.
- Residual objective: channel error divided by measured total at each bin, identical weighting across model comparisons. Table reports XY distance RMS, not the minimized objective. Held-out values are cycle-averaged curve repeatability, not independent orientation ground truth.
- Multiple initializations, including geometries from the prior small-background feasible constructions. Repeated runs reach the reported minima; no global optimum certificate.
- Numerical three-mode prediction reproduces the previous analytic model within 1.2e-5 of the reference total. Sum of mode Gram matrices equals 2I to 7e-14. Pair-sum balance checked at about 1e-14.
- Pupil grid radius 240 versus 400 changes predicted channel intensity by less than 4e-5 of reference maximum total. Smaller than observed residuals.
- No exact angles established. Fixed cone, dipole emission, circular excitation, calibrated channels and stationary geometry of signal fields are assumptions.
- Five modes are a physically consistent restricted family, not all possible static spatial fields or detector-path backgrounds. Failure at 10% does not prove small background impossible.
- Synthetic angle-recovery robustness and shared geometry across intervals have not been tested here.

A useful next theoretical step is to express the most general stationary field through its overlaps with the dipole basis and its positive-semidefinite Gram constraints. That could distinguish limitations of these selected modes from a broader stationary-field limitation before adding more physical sources. Independent fixed/constant-theta control traces remain needed for experimental discrimination.
