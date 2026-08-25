Updated todo list

Nice — that's the polydispersity lever, and it's free physics (no background, no noise). The spec's SPR range 575–625 nm means each rod's longitudinal resonance sits at a different distance from the 633 nm laser, giving a **different ρ per rod** (and hence a different $r_{\max}$). Let me model that directly: per-rod $\lambda_L\sim U(575,625)$, transverse fixed at 520, current hole-corrected optics, no background, no shot noise.    

Created [](file:///c%3A/Users/denny/workspace/pyqtrod/investigation_hugh_theta_r/scripts/rho_spread_only.py)

Ran terminal command: python rho_spread_only.py
Completed with input: python rho_spread_only.py

Ran terminal command: cd "c:\Users\denny\workspace\pyqtrod\investigation_hugh_theta_r\scripts"; python rho_spread_only.py
Completed with input: cd "c:\Users\denny\workspace\pyqtrod\investigation_hugh_theta_r\scripts"; python rho_spread_only.py

Viewed image [](file:///c%3A/Users/denny/workspace/pyqtrod/investigation_hugh_theta_r/transverse_mode_output/rho_spread_only.png)

Here's how ρ polydispersity alone looks (no background, no shot noise) — saved at transverse_mode_output/rho_spread_only.png:

**Setup:** $\lambda_L\sim U(575,625)$ nm → **ρ ∈ [0.07, 0.30]** (median 0.15) → **$r_{\max}$ ∈ [0.41, 0.70]** per rod.

**What happens:**
- **$p(r)$ (middle):** ρ spread *does* wash out the pile-up on its own — the sharp analytic spike at $r_{\max}=0.82$ (black) and the fixed-ρ spike at 0.58 (grey) become a **broad, flat-ish distribution spanning ~0.1–0.70** (red). So polydispersity is a genuine, background-free smearing mechanism.
- **$\theta(r)$ (right):** but it **overshoots** — the red curve bends up *more* than the empirical 40 nm (dashed), and it only reaches $r\approx0.70$ whereas the data extend to $r\approx0.87$.

**Two clear messages:**
1. ρ spread alone bends $\theta(r)$ *too much* here, and
2. it can't reach the data's high-$r$ tail because with these inputs the largest $r_{\max}$ is only 0.70.

Both point to the **transverse mode being weaker than assumed**. The culprit is the one placeholder we haven't pinned down: `AMP_RATIO_T_OVER_L = 1.0` (equal transverse/longitudinal *peak* polarizability). Since $\rho\propto(\text{amp ratio})^2$, if the longitudinal mode is stronger (ratio < 1, which is physically expected), ρ shrinks, $r_{\max}$ rises back toward 0.82 (reaching the data's extent), and the overshoot goes away. Alternatively/additionally, some rods with LSPR redder than 625 nm (toward/past the 633 laser) would give near-zero ρ and $r_{\max}\approx0.82$.

So of your knobs, the **peak-amplitude ratio inside ρ** is the key one to set next — right now it's forcing quite a lot of transverse. Want me to sweep `AMP_RATIO_T_OVER_L` (e.g. 0.2–1.0) with this same 575–625 nm spread and overlay against the empirical 40 nm curve, to see what transverse strength the data actually tolerate?

Created 3 todos

Made changes.