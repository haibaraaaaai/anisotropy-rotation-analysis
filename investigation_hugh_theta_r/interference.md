This directly tests Richard's proposal. Here's where it lands.

## Does our work address Richard's framing? Mostly yes

| Richard's point | Status in our work |
|---|---|
| Fourkas + reflection + **hole-in-mirror** as the theoretical base | Implemented — hole-corrected A,B,C (r_max=0.82, which matches where Hugh's data top out) |
| Each rod has a **unique background** affecting angle recovery | This is the per-recording background heterogeneity model |
| Static bg measured/subtracted, **variable component remains** | Captured by the per-recording lognormal spread (σ≈0.31) |
| **Compare modelable bg to glycerol observations** — are they explanatory? | Done, and the answer is encouraging (below) |
| Scatter plots with a **bg marker** (polarisation→location, intensity→size) | Built: xy_with_bg_markers.png |

**The key good-news result** (from the previous step): with the **measured uncorrected background** (25 nm: 21% at bright → 54% at dim), the **incoherent** model already reproduces Hugh's 25 nm θ(r) quantitatively. So for 25 nm we are in Richard's *"explanatory"* regime — we do **not** need more background than observed. 40 nm (10% bg) falls slightly short.

## The interference model — implemented and tested

I built Richard's coherent model, and I made it physically faithful in one important way: **only the polarised part of the rod light interferes**, because the rod's degree of linear polarisation *is* $r$ (the unpolarised part — the Fourkas "A" term — adds incoherently). So:

$$I_j = \big|E^{\text{pol}}_{\text{rod},j} + e^{i\delta}E^{\text{pol}}_{\text{bg},j}\big|^2 + \tfrac12(I^{\text{unpol}}_{\text{rod}}+I^{\text{unpol}}_{\text{bg}})$$

with one extra parameter, the per-rod relative phase $\delta$ (drawn uniformly). Result ([coherent_vs_incoherent.png](investigation_hugh_theta_r/coherent_background_output/coherent_vs_incoherent.png)):

- Richard's mechanism is **real**: the cross-term $\propto\sqrt{I_{\text{rod}}I_{\text{bg}}}$ does perturb the **bright** ($r_{\max}$) points that an incoherent bg barely moves, and it broadens $p(r)$ for the same bg.
- **But** with a random per-rod phase it broadens *symmetrically* (some rods brighten, some darken), so after Hugh's CDF inversion it does **not** bend θ(r) toward the empirical curve any more than the incoherent model — for 40 nm it's marginally *worse* (it leaves a small high-r tail).
- At the measured bg, **coherent ≈ incoherent** for both datasets: 25 nm matches, 40 nm slightly short. Halving the bg (orange) undershoots in both models.

So, importantly: **we do not need the interference term to explain the 25 nm data — the observed incoherent background already does it.** That's the opposite of the pessimistic scenario Richard worried about (needing more bg than observed).

## Caveats on the interference test

- I assumed the background is **fully coherent** with the rod (η=1, an upper bound). Richard's bg is "the sum of all background sources" — if those are mutually incoherent, the net interference is weaker, pushing toward the incoherent limit. Since both limits give the same θ(r) here, the conclusion is robust.
- I used a **random** relative phase. A systematic phase could shift things, but there's no physical reason to prefer one.

## Bottom line + suggestion

Our existing work (hole-corrected Fourkas + measured per-rod background) is already **quantitatively explanatory for 25 nm** and nearly so for 40 nm — which is the main thing Richard wanted to establish. The interference term is now implemented and available, but it isn't required and doesn't change the picture much under a random phase. The small 40 nm residual is more plausibly the transverse-mode contribution or Richard's unmodelled variable component "c".

Given Richard's September deadline and "write it first," I'd suggest: lead the paper with the hole-corrected Fourkas base + the measured-background explanation of the glycerol r-distributions (25 nm is the clean showcase), include the bg-marker scatter, and mention the coherent-interference term as a modelled refinement that doesn't materially change the conclusion. Want me to fit `frac_med` (and optionally the coherence fraction η) per dataset to get best-fit numbers with the measured background as a prior, so the paper can state "observed bg of X% explains the data"?