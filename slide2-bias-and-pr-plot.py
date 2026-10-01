"""Slide 2 (2-panel): theta bias vs the ideal model, and the isotropic p(r).

Panel 1 — theta bias relative to the IDEAL model (hole + water-oil Fresnel +
correct BFP mapping = adopted constants). For each r on the ideal r(theta)
curve, theta_true = theta_ideal(r); each curve shows theta_case(r) - theta_true
inverted with that case's constants:
    hole only      : annulus, homogeneous (no Fresnel)
    old simulation : annulus + Fresnel but the WRONG BFP mapping (the group
                     simulation's 1/cos(alpha_oil) field weight instead of
                     sqrt(cos a_oil)/cos t) - the measure that produced the
                     superseded eq-(22) set
    direct Fourkas : full cone, homogeneous (no hole, no nothing)

Panel 2 — p(r) for an isotropic ensemble (cos theta uniform), the Jacobian
pile-up at the ceiling:
    Fourkas p(r)             (full-cone homogeneous, n_water = 1.33)
    ideal p(r)               (adopted annulus + Fresnel, correct map)
    ideal p(r) + background  (5% of I_max per channel, equal in all four
                              channels, incoherent - Monte Carlo)

n_water = 1.33 throughout, consistent with the water-oil interface treatment.
(The theta_r-deviation notebook's analytic case instead uses the textbook
closed form with n = 1.47, NA_in = 0.39 and no corrections -> spike at 0.967.)

Run standalone:  FIGURE_OUT=figure.png python3 plot.py
"""
from pathlib import Path
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

N_CELL, N_CONTACT = 1.33, 1.515
NA_OUT, NA_IN = 1.3, 0.38
TMAX = np.arcsin(NA_OUT / N_CELL)
TIN = np.arcsin(NA_IN / N_CELL)


def _fresnel(t):
    ca = np.sqrt(max(0.0, 1.0 - (N_CELL * np.sin(t) / N_CONTACT) ** 2))
    ct = np.cos(t)
    ts = 2 * N_CELL * ct / (N_CELL * ct + N_CONTACT * ca)
    tp = 2 * N_CELL * ct / (N_CONTACT * ct + N_CELL * ca)
    return ts, tp


def _ABC(t_lo, t_hi, measure):
    """measure: 'corr' (corrected BFP map + Fresnel), 'wrong' (old simulation:
    geometric pixel area with 1/cos(a_oil)^2 field weight + Fresnel),
    'hom' (solid angle, no Fresnel)."""
    n = 200001
    X = Y = Z = 0.0
    for t in np.linspace(t_lo, t_hi, n):
        st, ct = np.sin(t), np.cos(t)
        sa = min(1.0, N_CELL * st / N_CONTACT)
        ca = np.cos(np.arcsin(sa))
        if measure == 'wrong':
            ts, tp = _fresnel(t)
            f = (N_CONTACT / N_CELL) * st * ct / ca**2
        elif measure == 'corr':
            ts, tp = _fresnel(t)
            f = st * (N_CONTACT * ca) / (N_CELL * ct)
        else:
            ts = tp = 1.0
            f = st
        X += f * (3/8 * (tp**2 * ct**2 + ts**2) + 1/4 * tp * ts * ct)
        Y += f * (1/8 * (tp * ct - ts)**2)
        Z += f * (1/2 * tp**2 * st**2)
    sc = (t_hi - t_lo) / (n - 1)
    X, Y, Z = X * sc, Y * sc, Z * sc
    return Z, (X + Y) / 2 - Z, (X - Y) / 2


# ---- constant sets ----------------------------------------------------------
A_AD, B_AD, C_AD = 0.097840, 0.011915, 0.101852      # ideal: hole + Fresnel + correct map
A_AN, B_AN, C_AN = _ABC(TIN, TMAX, 'hom')            # hole only
A_OW, B_OW, C_OW = _ABC(TIN, TMAX, 'wrong')          # old simulation (annular, wrong map)
A_FW, B_FW, C_FW = _ABC(0.0, TMAX, 'wrong')          # old simulation, full cone (validation)
A_HO, B_HO, C_HO = _ABC(0.0, TMAX, 'hom')            # direct Fourkas

# validation: full-cone old-simulation measure reproduces the superseded eq-(22) set
# (scale-invariant: compare ratios to C; doc set 0.126479 / 0.045525 / 0.162911)
for got, ref in [(A_FW / C_FW, 0.126479 / 0.162911), (B_FW / C_FW, 0.045525 / 0.162911)]:
    assert abs(got / ref - 1) < 0.02, (got, ref)
assert abs(C_AD / (A_AD + B_AD) - 0.927995) < 2e-6

def rmax(A, B, C):
    return C / (A + B)

def r_of_theta(th, A, B, C):
    s2 = np.sin(np.radians(th)) ** 2
    return C * s2 / (A + B * s2)

def theta_of_r(r, A, B, C):
    v = A * r / (C - B * r)
    out = np.full_like(np.asarray(r, float), np.nan, dtype=float)
    ok = (v >= 0) & (v <= 1)
    out[ok] = np.degrees(np.arcsin(np.sqrt(v[ok])))
    return out

TH = np.linspace(0, 90, 1801)
R_TRUE = r_of_theta(TH, A_AD, B_AD, C_AD)

CASES = [
    ('hole only\n(annulus, homogeneous)', A_AN, B_AN, C_AN, '#009E73', '-'),
    ('old simulation\n(hole + Fresnel, wrong BFP map)', A_OW, B_OW, C_OW, '#B45309', '--'),
    ('direct Fourkas\n(full cone, homogeneous)', A_HO, B_HO, C_HO, '#3D4753', '-'),
]
print('r_max: ideal %.4f | hole only %.4f | old sim (annular) %.4f | '
      'Fourkas %.4f' % (rmax(A_AD, B_AD, C_AD), rmax(A_AN, B_AN, C_AN),
                        rmax(A_OW, B_OW, C_OW), rmax(A_HO, B_HO, C_HO)))
for lab, A, B, C, _, _ in CASES:
    b = theta_of_r(r_of_theta(np.array([30.0, 55.0, 80.0]), A_AD, B_AD, C_AD), A, B, C) - [30, 55, 80]
    print('  bias %-24s 30/55/80: %s' % (lab.split('\n')[0], np.round(b, 2)))

# ---- panel 2: isotropic p(r), Monte Carlo for the background case -----------
rng = np.random.default_rng(11)
N = 4_000_000
cos_th = rng.uniform(-1.0, 1.0, N)
TH_RAD = np.arccos(cos_th)                  # folded below
TH_RAD = np.minimum(TH_RAD, np.pi - TH_RAD)
TH_I = np.degrees(TH_RAD)                   # degrees, for r_of_theta
PHI = rng.uniform(0.0, np.pi, N)

r_fourkas = r_of_theta(TH_I, A_HO, B_HO, C_HO)
r_ideal = r_of_theta(TH_I, A_AD, B_AD, C_AD)

# ideal + background: b = 5% of I_max in EVERY channel (incoherent, equal)
IMAX, BFRAC = 1.0, 0.05
s2 = np.sin(TH_RAD) ** 2
m = [np.cos(2 * PHI), np.sin(2 * PHI), -np.cos(2 * PHI), -np.sin(2 * PHI)]
S = IMAX / (A_AD + B_AD + C_AD)             # brightest channel at theta=90 = I_max
b = BFRAC * IMAX
I = [S * (A_AD + B_AD * s2 + C_AD * s2 * mj) + b for mj in m]
x_bg = (I[0] - I[2]) / (I[0] + I[2])
y_bg = (I[1] - I[3]) / (I[1] + I[3])
r_bg = np.hypot(x_bg, y_bg)

BINW = 0.002
EDGES = np.arange(0.0, 1.0 + BINW, BINW)
CENTERS = EDGES[:-1] + BINW / 2

def pr(x):
    cnt, _ = np.histogram(x, bins=EDGES)
    return gaussian_filter1d(cnt / (len(x) * BINW), 1.0)

p_fourkas, p_ideal, p_bg = pr(r_fourkas), pr(r_ideal), pr(r_bg)
print('p(r) edges: fourkas %.4f | ideal %.4f | bg sample max %.4f'
      % (r_fourkas.max(), r_ideal.max(), r_bg.max()))

# ------------------------------------------------------------------ figure
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11.5,
    'axes.titlesize': 12.5, 'axes.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'savefig.dpi': 300,
    'svg.fonttype': 'none', 'legend.fontsize': 10,
})
DARK, MUTED, REF = '#23313C', '#596875', '#89939C'
BLUE, GOLD = '#0072B2', '#E8A93C'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.9))

# ---------------- panel 1: theta bias vs ideal --------------------------------
for lab, A, B, C, col, ls in CASES:
    ax1.plot(TH, theta_of_r(R_TRUE, A, B, C) - TH, color=col, lw=2.2, ls=ls,
             label=lab + '\n($r_{\\max}$ = %.3f)' % rmax(A, B, C))
    over = R_TRUE > rmax(A, B, C)
    if over.any():
        ax1.axvspan(TH[over].min(), 90, color=REF, alpha=0.10, lw=0)
ax1.axhline(0, color=MUTED, lw=0.8)
ax1.set_xlim(0, 90)
ax1.set_xlabel(r'true $\theta$ (deg)')
ax1.set_ylabel(r'$\theta_{\mathrm{case}}(r) - \theta_{\mathrm{ideal}}(r)$  (deg)')
ax1.set_title('(a)  $\\theta$ bias vs the ideal model\n'
              r'(ideal = hole + water–oil Fresnel + correct BFP map)',
              loc='left', color=DARK)
ax1.legend(loc='lower left', frameon=False, handlelength=1.9,
           labelspacing=0.5, borderaxespad=0.3)
b80 = float(theta_of_r(r_of_theta(np.array([80.0]), A_AD, B_AD, C_AD),
                       A_OW, B_OW, C_OW)[0] - 80)
ax1.annotate('old simulation:\n80° reads as %.1f°' % (80 + b80),
             (80, b80), xytext=(-14, -30), textcoords='offset points',
             ha='right', fontsize=10, color='#B45309',
             arrowprops=dict(arrowstyle='-', lw=0.8, color='#B45309'))
ax1.text(0.62, 0.96, 'shaded: $r$ beyond that model\'s ceiling → clipped',
         transform=ax1.transAxes, ha='right', va='top', fontsize=9.5, color=MUTED)

# ---------------- panel 2: isotropic p(r) -------------------------------------
ax2.plot(CENTERS, p_fourkas, color=DARK, lw=1.8,
         label='Fourkas (full cone, homogeneous)  —  $r_{\\max}$ = %.3f' % rmax(A_HO, B_HO, C_HO))
ax2.plot(CENTERS, p_ideal, color=BLUE, lw=2.6,
         label='ideal (hole + Fresnel, correct map)  —  $r_{\\max}$ = %.3f' % rmax(A_AD, B_AD, C_AD))
ax2.plot(CENTERS, p_bg, color='#B45309', lw=2.0,
         label='ideal + 5% $I_{\\max}$ background per channel (equal)')
ax2.set_xlim(0.55, 0.95)
ax2.set_xlabel('anisotropy radius $r$')
ax2.set_ylabel('$p(r)$   (isotropic ensemble)')
ax2.set_title('(b)  $p(r)$: the Jacobian pile-up at the ceiling', loc='left', color=DARK)
ax2.legend(loc='upper left', frameon=False, handlelength=1.9,
           labelspacing=0.5, borderaxespad=0.3)
for rr, col, lab in [(rmax(A_HO, B_HO, C_HO), DARK, '0.927'),
                     (rmax(A_AD, B_AD, C_AD), BLUE, '0.928')]:
    ax2.axvline(rr, color=col, lw=0.9, ls=':')
edge = float(np.percentile(r_bg, 99.9))
ax2.axvline(edge, color='#B45309', lw=0.9, ls=':')
ax2.annotate('pile-up pushed to\n$\\approx$%.3f by background' % edge,
             (edge, 6.5), xytext=(6, 0), textcoords='offset points',
             fontsize=9.5, color='#B45309', va='center', ha='left')

fig.subplots_adjust(left=0.06, right=0.99, top=0.87, bottom=0.115, wspace=0.21)
out = Path(os.environ['FIGURE_OUT'])
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix('.pdf'))
fig.savefig(out.with_suffix('.svg'))
print('figure written')
