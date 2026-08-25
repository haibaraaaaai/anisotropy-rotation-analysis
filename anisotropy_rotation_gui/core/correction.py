"""Channel correction, T-matrix, and shape fitting."""

import numpy as np
from functools import partial
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import splprep, splev
from scipy.integrate import quad

from .io import get_pol_ind, anisotropy_from_channels


# ═════════════════════════════════════════════════════════════════════════════
# T-matrix (Fresnel / PBS cross-talk correction)
# ═════════════════════════════════════════════════════════════════════════════

def T_Icor_Matrix():
    """Build the raw 4×4 T-matrix for polarisation cross-talk correction.

    Order: 0, 90, 45, 135  (45 = reflected arm).
    """
    ret = np.zeros([4, 4])
    ret[0][0] = 0.152
    ret[0][1] = 0
    ret[1][0] = 0
    ret[1][1] = 0.148
    alpha = 0.455
    beta = 0.009
    t = 0.757
    r = 0.741
    ret[2][0] = -alpha * beta * r
    ret[2][1] = alpha * beta * r
    ret[2][2] = alpha * alpha * r
    ret[2][3] = beta * beta * r
    ret[3][0] = -alpha * beta * t
    ret[3][1] = alpha * beta * t
    ret[3][2] = beta * beta * t
    ret[3][3] = alpha * alpha * t
    return np.linalg.inv(ret) / 8


def build_matcor():
    """Build the permuted T-matrix for the internal stack order [c90, c45, c135, c0]."""
    matcorb = T_Icor_Matrix()
    _inds = get_pol_ind(["0", "90", "45", "135"])
    _Pmat = np.zeros([4, 4])
    for _i in range(4):
        _Pmat[_i, _inds[_i]] = 1
    matcor = np.dot(np.linalg.inv(_Pmat), np.dot(matcorb, _Pmat))
    return matcor, matcorb


# ═════════════════════════════════════════════════════════════════════════════
# Channel gain (a) and offset (o)
# ═════════════════════════════════════════════════════════════════════════════

def _chcor_objective(params, ac, mat):
    a90, a45, a135 = params
    ap = np.copy(ac)
    ap[1, :] *= a90
    ap[2, :] *= a45
    ap[3, :] *= a135
    bc = np.dot(mat, ap)
    return np.sum((bc[0] + bc[1] - bc[2] - bc[3]) ** 2)


def find_channel_gain(c0, c90, c45, c135, matcorb):
    """Find gain coefficients [a_90, a_45, a_135] (a_0 = 1 as reference).

    Uses T-matrix during optimisation.
    Returns array [a_90, a_45, a_135, a_0=1.0] in stack order [90, 45, 135, 0].
    """
    ac = np.vstack((c0, c90, c45, c135)).astype(float)
    res = minimize(partial(_chcor_objective, ac=ac, mat=matcorb),
                   [1.0, 1.0, 1.0], method='Nelder-Mead')
    l90, l45, l135 = res.x
    a = [1.0, 1.0, 1.0, 1.0]  # [a_90, a_45, a_135, a_0]
    a[get_pol_ind(["90"])[0]] = float(l90)
    a[get_pol_ind(["45"])[0]] = float(l45)
    a[get_pol_ind(["135"])[0]] = float(l135)
    return np.array(a)


def compute_offset(c0, c90, c45, c135):
    """Compute electronic offsets — ensures no channel goes negative.

    Returns [o_90, o_45, o_135, o_0] in stack order.
    """
    channels = [c90, c45, c135, c0]
    return np.array([max(0.0, -float(np.min(ch))) for ch in channels])


def apply_correction(c0, c90, c45, c135, a, o, matcor):
    """Apply gain → offset → T-matrix to raw channels.

    Parameters
    ----------
    c0, c90, c45, c135 : 1-D arrays (raw)
    a : array [a_90, a_45, a_135, a_0]  (stack order)
    o : array [o_90, o_45, o_135, o_0]  (stack order)
    matcor : 4×4 permuted T-matrix

    Returns
    -------
    tcor : (4, N) corrected channel stack in stack order [c90, c45, c135, c0]
    """
    stack = np.vstack((c90, c45, c135, c0)).astype(float)
    for i in range(4):
        stack[i] = o[i] + a[i] * stack[i]
    return np.dot(matcor, stack)


# ═════════════════════════════════════════════════════════════════════════════
# anisotropy-rotation model
# ═════════════════════════════════════════════════════════════════════════════

def fourkas_ABC(alpha):
    """Compute anisotropy-rotation A, B, C coefficients from collection half-angle alpha (rad)."""
    ca = np.cos(alpha)
    A = 1/6 - ca/4 + (ca**3)/12
    B = ca/8 - (ca**3)/8
    C = 7/48 - ca/16 - (ca**2)/16 - (ca**3)/48
    return A, B, C


def fourkas_ABC_annular(na_in, na_out, n_water=1.33):
    """Compute anisotropy-rotation A, B, C for annular (hole) collection.

    Parameters
    ----------
    na_in : float
        Inner numerical aperture (hole edge).
    na_out : float
        Outer numerical aperture (collection edge).
    n_water : float
        Refractive index of immersion medium.
    """
    if na_in < 0 or na_out <= 0:
        raise ValueError("NA_in/NA_out must be positive (NA_in can be 0).")
    if na_in >= na_out:
        raise ValueError("NA_in must be smaller than NA_out.")
    if na_out >= n_water:
        raise ValueError("NA_out must be smaller than n.")

    alpha_out = np.arcsin(na_out / n_water)
    alpha_in = np.arcsin(na_in / n_water)

    j1, _ = quad(
        lambda b: (2 * np.cos(b) ** 2 + 0.75 * np.sin(b) ** 4) * np.sin(b),
        alpha_in,
        alpha_out,
    )
    j2, _ = quad(
        lambda b: 0.25 * np.sin(b) ** 4 * np.sin(b),
        alpha_in,
        alpha_out,
    )
    j3, _ = quad(
        lambda b: np.sin(b) ** 2 * np.cos(b) ** 2 * np.sin(b),
        alpha_in,
        alpha_out,
    )

    c = j1 - j2
    a = 2 * j3
    b = j1 + j2 - 2 * j3
    return a, b, c


def fourkas_template(Theta_ax, Phi_ax, Lambda, A, B, C, n_pts=200):
    """Generate model (ax, ay) curve for a rod spinning on a cone.

    Parameters
    ----------
    Theta_ax : float — polar angle of rotation axis (rad)
    Phi_ax   : float — azimuthal angle of rotation axis (rad)
    Lambda   : float — cone half-angle (rad)
    A, B, C  : anisotropy-rotation coefficients
    n_pts    : number of points around the cone

    Returns
    -------
    ax_model, ay_model : arrays of length n_pts
    """
    # Rotation axis
    kx = np.sin(Theta_ax) * np.cos(Phi_ax)
    ky = np.sin(Theta_ax) * np.sin(Phi_ax)
    kz = np.cos(Theta_ax)
    k = np.array([kx, ky, kz])

    # Build orthonormal basis
    if abs(kz) < 0.9:
        e1 = np.cross(k, [0, 0, 1])
    else:
        e1 = np.cross(k, [1, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k, e1)

    psi = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cL, sL = np.cos(Lambda), np.sin(Lambda)

    ax_model = np.empty(n_pts)
    ay_model = np.empty(n_pts)
    for i, p in enumerate(psi):
        d = cL * k + sL * (np.cos(p) * e1 + np.sin(p) * e2)
        theta_i = np.arccos(np.clip(d[2], -1, 1))
        phi_i = np.arctan2(d[1], d[0])
        s2 = np.sin(theta_i) ** 2
        r = C * s2 / (A + B * s2)
        ax_model[i] = r * np.cos(2 * phi_i)
        ay_model[i] = r * np.sin(2 * phi_i)

    return ax_model, ay_model


# ═════════════════════════════════════════════════════════════════════════════
# Shape fitting
# ═════════════════════════════════════════════════════════════════════════════

def _arc_length_fractions(x, y):
    """Cumulative arc-length fractions along a 2-D curve."""
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.sqrt(dx**2 + dy**2)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return np.linspace(0, 1, len(x))
    return cum / total


def _reorder_from_leftmost(ax_arr, ay_arr):
    """Rotate arrays so the leftmost point is first."""
    i0 = int(np.argmin(ax_arr))
    return np.roll(ax_arr, -i0), np.roll(ay_arr, -i0)


def shape_fit_cost(params, tcor_sub, A, B, C, n_model=200):
    """Cost function for 6-parameter shape fit.

    Parameters
    ----------
    params : [Theta_ax_deg, Phi_ax_deg, Lambda_deg, dc, fa, fb]
    tcor_sub : (4, N) T-corrected channels at subsampled points (stack order)
    A, B, C : anisotropy-rotation coefficients

    Returns
    -------
    cost : float
    """
    Theta_deg, Phi_deg, Lam_deg, dc, fa, fb = params
    Theta = np.radians(Theta_deg)
    Phi = np.radians(Phi_deg)
    Lam = np.radians(Lam_deg)

    # Per-channel backgrounds
    da = fa * dc
    db = fb * dc
    b0 = (dc + da) / 2
    b90 = (dc - da) / 2
    b45 = (dc + db) / 2
    b135 = (dc - db) / 2

    # Subtract backgrounds (stack order: 90, 45, 135, 0)
    idx90 = get_pol_ind(["90"])[0]
    idx45 = get_pol_ind(["45"])[0]
    idx135 = get_pol_ind(["135"])[0]
    idx0 = get_pol_ind(["0"])[0]

    c0_cor = tcor_sub[idx0] - b0
    c90_cor = tcor_sub[idx90] - b90
    c45_cor = tcor_sub[idx45] - b45
    c135_cor = tcor_sub[idx135] - b135

    ax_data, ay_data = anisotropy_from_channels(c0_cor, c90_cor, c45_cor, c135_cor)

    # Model curve
    ax_mod, ay_mod = fourkas_template(Theta, Phi, Lam, A, B, C, n_pts=n_model)

    # Arc-length matching (try both directions)
    best = np.inf
    for ax_m, ay_m in [(ax_mod, ay_mod), (ax_mod[::-1], ay_mod[::-1])]:
        ax_d2, ay_d2 = _reorder_from_leftmost(ax_data, ay_data)
        ax_m2, ay_m2 = _reorder_from_leftmost(ax_m, ay_m)

        frac_d = _arc_length_fractions(ax_d2, ay_d2)
        frac_m = _arc_length_fractions(ax_m2, ay_m2)

        ax_interp = np.interp(frac_d, frac_m, ax_m2)
        ay_interp = np.interp(frac_d, frac_m, ay_m2)

        cost = np.mean((ax_d2 - ax_interp)**2 + (ay_d2 - ay_interp)**2)
        best = min(best, cost)

    return best


def run_shape_fit(tcor_sub, NA, n_water=1.33, n_outer=4, popsize=15,
                  A_override=None, B_override=None, C_override=None):
    """Run the alternating DE + L-BFGS-B shape fit.

    Parameters
    ----------
    tcor_sub : (4, N) T-corrected channels at subsampled points
    NA : float — numerical aperture
    n_water : float — refractive index of immersion medium
    n_outer : int — number of alternating DE iterations
    popsize : int — DE population size

    Returns
    -------
    result : dict with keys:
        Theta, Phi, Lambda : fitted angles (degrees)
        dc, fa, fb : background parameters
        b0, b90, b45, b135 : per-channel backgrounds
        cost : final cost
        ax_data, ay_data : corrected anisotropy of data
        ax_model, ay_model : best-fit model curve
        A, B, C, R_sat : anisotropy-rotation coefficients
    """
    if (A_override is not None
            and B_override is not None
            and C_override is not None):
        A = float(A_override)
        B = float(B_override)
        C = float(C_override)
    else:
        alpha = np.arcsin(NA / n_water)
        A, B, C = fourkas_ABC(alpha)
    R_sat = C / (A + B)

    # Upper bound for dc
    idx90 = get_pol_ind(["90"])[0]
    idx45 = get_pol_ind(["45"])[0]
    idx135 = get_pol_ind(["135"])[0]
    idx0 = get_pol_ind(["0"])[0]
    dc_upper = float(min(
        np.min(tcor_sub[idx0]) + np.min(tcor_sub[idx90]),
        np.min(tcor_sub[idx45]) + np.min(tcor_sub[idx135]),
    ))
    dc_upper = max(dc_upper, 0.001)

    bounds_geo = [(0, 90), (-180, 180), (0, 90)]
    bounds_bg = [(0, dc_upper), (-1, 1), (-1, 1)]

    # Start with zero background
    best_bg = [0.0, 0.0, 0.0]

    for outer in range(n_outer):
        # DE on geometry
        def cost_geo(p):
            return shape_fit_cost(
                [p[0], p[1], p[2], best_bg[0], best_bg[1], best_bg[2]],
                tcor_sub, A, B, C)
        res_geo = differential_evolution(cost_geo, bounds_geo,
                                         popsize=popsize, seed=42 + outer,
                                         maxiter=100, tol=1e-6)
        best_geo = list(res_geo.x)

        # DE on background
        def cost_bg(p):
            return shape_fit_cost(
                [best_geo[0], best_geo[1], best_geo[2], p[0], p[1], p[2]],
                tcor_sub, A, B, C)
        res_bg = differential_evolution(cost_bg, bounds_bg,
                                        popsize=popsize, seed=42 + outer,
                                        maxiter=100, tol=1e-6)
        best_bg = list(res_bg.x)

    # Final joint polish
    x0 = best_geo + best_bg
    bounds_all = bounds_geo + bounds_bg

    def cost_all(p):
        return shape_fit_cost(p, tcor_sub, A, B, C)

    res_final = minimize(cost_all, x0, method='L-BFGS-B', bounds=bounds_all)
    Theta, Phi, Lam, dc, fa, fb = res_final.x

    # Per-channel backgrounds
    da = fa * dc
    db = fb * dc
    b0 = (dc + da) / 2
    b90 = (dc - da) / 2
    b45 = (dc + db) / 2
    b135 = (dc - db) / 2

    # Corrected anisotropy for display
    c0_final = tcor_sub[idx0] - b0
    c90_final = tcor_sub[idx90] - b90
    c45_final = tcor_sub[idx45] - b45
    c135_final = tcor_sub[idx135] - b135
    ax_data, ay_data = anisotropy_from_channels(c0_final, c90_final,
                                                 c45_final, c135_final)

    # Model curve
    ax_model, ay_model = fourkas_template(
        np.radians(Theta), np.radians(Phi), np.radians(Lam),
        A, B, C, n_pts=200)

    return {
        "Theta": float(Theta),
        "Phi": float(Phi),
        "Lambda": float(Lam),
        "dc": float(dc),
        "fa": float(fa),
        "fb": float(fb),
        "b0": float(b0),
        "b90": float(b90),
        "b45": float(b45),
        "b135": float(b135),
        "cost": float(res_final.fun),
        "ax_data": ax_data,
        "ay_data": ay_data,
        "ax_model": ax_model,
        "ay_model": ay_model,
        "A": A,
        "B": B,
        "C": C,
        "R_sat": R_sat,
    }


# ── Full-trace processing ────────────────────────────────────────────────

def apply_full_trace_correction(c0_raw, c90_raw, c45_raw, c135_raw,
                                a, o, matcor, backgrounds):
    """Apply gain + offset + T-matrix + backgrounds to entire recording.

    Parameters
    ----------
    c0_raw, c90_raw, c45_raw, c135_raw : 1-D arrays
        Raw channel data for the full recording.
    a : array [a_90, a_45, a_135, a_0]
    o : array [o_90, o_45, o_135, o_0]
    matcor : (4, 4) T-correction matrix
    backgrounds : dict with keys b0, b90, b45, b135

    Returns
    -------
    c0, c90, c45, c135 : 1-D arrays — corrected, background-subtracted
    """
    # Stack in internal order [c90, c45, c135, c0]
    raw_stack = np.vstack((c90_raw, c45_raw, c135_raw, c0_raw)).astype(float)
    for i in range(4):
        raw_stack[i] = o[i] + a[i] * raw_stack[i]
    tcor_stack = matcor @ raw_stack

    idx0 = get_pol_ind(["0"])[0]
    idx90 = get_pol_ind(["90"])[0]
    idx45 = get_pol_ind(["45"])[0]
    idx135 = get_pol_ind(["135"])[0]

    c0 = tcor_stack[idx0] - backgrounds["b0"]
    c90 = tcor_stack[idx90] - backgrounds["b90"]
    c45 = tcor_stack[idx45] - backgrounds["b45"]
    c135 = tcor_stack[idx135] - backgrounds["b135"]
    return c0, c90, c45, c135


def extract_theta_phi(ax, ay, A, B, C):
    """Extract θ (degrees) and φ (radians, unwrapped period=π).

    Parameters
    ----------
    ax, ay : 1-D arrays — anisotropy
    A, B, C : anisotropy-rotation coefficients

    Returns
    -------
    theta : 1-D array — polar angle in degrees
    phi : 1-D array — azimuthal angle in radians, unwrapped (period π)
    """
    R_sat = C / (A + B)
    phi_raw = 0.5 * np.arctan2(ay, ax)
    phi = np.unwrap(phi_raw, period=np.pi)

    r_full = np.clip(np.sqrt(ax**2 + ay**2), 0.0, R_sat)
    denom = np.where(C - r_full * B > 0, C - r_full * B, np.nan)
    sinsq = np.clip((r_full * A) / denom, 0.0, 1.0)
    theta = np.degrees(np.arcsin(np.sqrt(sinsq)))
    return theta, phi


def compute_speed_windowed(phi_trace, time_s, nperseg, overlap,
                           transform=False, theta_trace=None,
                           theta_axis_deg=None, phi_axis_deg=None):
    """Windowed linear-regression rotation speed from φ.

    Parameters
    ----------
    phi_trace : array — unwrapped lab-frame φ (radians, period-π unwrapped)
    time_s : array — time axis
    nperseg : int — window size in samples
    overlap : int — overlap between windows
    transform : bool — rotate to axis frame before computing speed
    theta_trace : array — lab-frame θ in degrees (required if transform)
    theta_axis_deg : float — fitted axis polar angle (degrees)
    phi_axis_deg : float — fitted axis azimuthal angle (degrees)

    Returns
    -------
    speed_hz : array — rotation speed in Hz (signed)
    speed_time : array — window-centre times
    phi_used : array — the φ trace used for regression
    """
    if transform:
        Th = np.radians(theta_axis_deg)
        Ph = np.radians(phi_axis_deg)
        cP, sP = np.cos(Ph), np.sin(Ph)
        cT, sT = np.cos(Th), np.sin(Th)
        R = np.array([[ cT*cP,  cT*sP, -sT],
                       [-sP,     cP,      0 ],
                       [ sT*cP,  sT*sP,  cT ]])

        th_rad = np.radians(theta_trace)
        dx = np.sin(th_rad) * np.cos(phi_trace)
        dy = np.sin(th_rad) * np.sin(phi_trace)
        dz = np.cos(th_rad)
        d_rot = R @ np.vstack([dx, dy, dz])

        phi_ax_raw = np.arctan2(d_rot[1], d_rot[0])
        phi_use = np.unwrap(phi_ax_raw)
    else:
        phi_use = phi_trace

    step = nperseg - overlap
    n_windows = max(0, (len(phi_use) - nperseg) // step + 1)
    speed_hz = np.empty(n_windows)
    speed_time = np.empty(n_windows)

    for w in range(n_windows):
        s = w * step
        e = s + nperseg
        t_win = time_s[s:e]
        p_win = phi_use[s:e]
        t_c = t_win - t_win[0]
        n = len(t_c)
        sum_t = np.sum(t_c)
        sum_p = np.sum(p_win)
        sum_tt = np.sum(t_c * t_c)
        sum_tp = np.sum(t_c * p_win)
        denom_lr = n * sum_tt - sum_t * sum_t
        if abs(denom_lr) > 1e-30:
            slope = (n * sum_tp - sum_t * sum_p) / denom_lr
        else:
            slope = 0.0
        speed_hz[w] = slope / (2.0 * np.pi)
        speed_time[w] = 0.5 * (t_win[0] + t_win[-1])

    return speed_hz, speed_time, phi_use
