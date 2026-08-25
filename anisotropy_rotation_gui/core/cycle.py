"""Cycle detection and subsampling."""

import numpy as np
from scipy.spatial import KDTree
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.interpolate import splprep, splev
def detect_cycle_anisotropy(ax, ay, window=200, search_limit=None, step=5,
                            closure_factor=1.0):
    """Detect one complete closed loop in 2-D (ax, ay) anisotropy trajectory."""
    traj = np.column_stack([ax, ay])
    n = len(traj)
    if search_limit is None:
        search_limit = min(n // 2, 100_000)

    start_idx = 0
    ref_tree = KDTree(traj[start_idx: start_idx + window])

    n_sample = max(2, window // 10)
    dists_close = np.linalg.norm(
        traj[start_idx] - traj[start_idx: start_idx + n_sample], axis=1)
    dist_thresh = np.median(dists_close) * closure_factor

    i_list = np.arange(start_idx + window, start_idx + search_limit, step)
    if len(i_list) == 0:
        return start_idx, min(start_idx + search_limit, n)

    scores = []
    for i in i_list:
        end = min(i + window, n)
        tree2 = KDTree(traj[i:end])
        scores.append(tree2.count_neighbors(ref_tree, dist_thresh))
    scores = np.array(scores, dtype=float)

    prom_div = 2
    peaks = np.array([], dtype=int)
    while len(peaks) == 0 and prom_div < 128:
        peaks = find_peaks(scores, prominence=window / prom_div)[0]
        prom_div *= 2

    if len(peaks) == 0:
        end_idx = i_list[np.argmax(scores)]
        return start_idx, int(end_idx)

    _, _, _, right_ips = peak_widths(scores, peaks, rel_height=0.5)
    right_ips = np.round(right_ips).astype(int)

    peak_ind = 0
    while peak_ind < len(peaks) - 1:
        if np.min(scores[: peaks[peak_ind]]) > 2 * np.min(scores):
            peak_ind += 1
        else:
            break
    for k in range(peak_ind, min(peak_ind + 3, len(peaks))):
        if scores[peaks[k]] > 1.5 * scores[peaks[peak_ind]]:
            peak_ind = k

    end_idx = i_list[min(right_ips[peak_ind], len(i_list) - 1)]
    return int(start_idx), int(end_idx)


def _smooth_cycle_2d(ax_seg, ay_seg, smoothing=0.5, num_points=500,
                     resample_n=500):
    """Periodic B-spline smooth of a 2-D cycle."""
    pts = np.column_stack([ax_seg, ay_seg])
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_arc = np.concatenate([[0], np.cumsum(seg_len)])
    total_arc = cum_arc[-1]
    if total_arc == 0:
        return ax_seg, ay_seg

    arc_uniform = np.linspace(0, total_arc, resample_n, endpoint=False)
    rx = np.interp(arc_uniform, cum_arc, pts[:, 0], period=total_arc)
    ry = np.interp(arc_uniform, cum_arc, pts[:, 1], period=total_arc)
    pts_uniform = np.column_stack([rx, ry])
    pts_uniform = np.vstack([pts_uniform, pts_uniform[0]])

    if smoothing is None:
        smoothing = 0.01 * resample_n

    tck, _ = splprep(pts_uniform.T, per=True, s=smoothing)
    u = np.linspace(0, 1, num_points, endpoint=False)
    sx, sy = splev(u, tck)
    return sx, sy


def subsample_ref_curve(ax_cyc, ay_cyc, n_sub=360, savgol_window=201,
                        savgol_order=3):
    """Build reference curve + closest-point match -> n_sub raw indices."""
    n = len(ax_cyc)
    sg_win = min(savgol_window, n // 2 * 2 - 1)
    if sg_win < savgol_order + 2:
        sg_win = savgol_order + 2
        if sg_win % 2 == 0:
            sg_win += 1
    ax_sg = savgol_filter(ax_cyc, sg_win, savgol_order, mode='wrap')
    ay_sg = savgol_filter(ay_cyc, sg_win, savgol_order, mode='wrap')

    sx_ref, sy_ref = _smooth_cycle_2d(ax_sg, ay_sg, num_points=n_sub)
    ref_pts = np.column_stack([sx_ref, sy_ref])
    raw_pts = np.column_stack([ax_cyc, ay_cyc])

    all_dists = np.linalg.norm(
        raw_pts[None, :, :] - ref_pts[:, None, :], axis=2)
    matched_idx = np.argmin(all_dists, axis=1)
    matched_dist = all_dists[np.arange(n_sub), matched_idx]

    return matched_idx, matched_dist, sx_ref, sy_ref, ax_sg, ay_sg, sg_win


def is_monotonic(matched_idx):
    """Check whether matched indices are non-decreasing."""
    diffs = np.diff(matched_idx.astype(np.int64))
    n_violations = int(np.sum(diffs < 0))
    return n_violations == 0, n_violations


def _lis_indices(arr):
    """Longest non-decreasing subsequence — return indices into *arr*.

    O(n²) which is fine for n ≤ few hundred.
    """
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int)
    dp = np.ones(n, dtype=int)
    parent = np.full(n, -1, dtype=int)
    for i in range(1, n):
        for j in range(i):
            if arr[j] <= arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    # backtrack from best end
    k = int(np.argmax(dp))
    path = []
    while k != -1:
        path.append(k)
        k = int(parent[k])
    return np.array(path[::-1], dtype=int)


def enforce_monotonic(matched_idx, ax_cyc, ay_cyc, sx_ref, sy_ref,
                      max_iter=5):
    """LIS-anchor + constrained gap re-match to enforce monotonicity.

    No wrapping — indices must be strictly non-decreasing from
    ref[0] (start of segment) to ref[n_sub-1] (end of segment).

    Iterates until fully monotonic or *max_iter* reached.

    Returns (matched_idx_new, matched_dist_new, n_anchors, n_fixed, n_iters).
    """
    n_sub = len(matched_idx)
    n_raw = len(ax_cyc)
    raw_pts = np.column_stack([ax_cyc, ay_cyc])
    ref_pts = np.column_stack([sx_ref, sy_ref])

    current = matched_idx.copy()
    n_anchors = n_sub
    n_fixed_total = 0
    n_iters = 0

    for it in range(max_iter):
        mono, nv = is_monotonic(current)
        if mono:
            break

        n_iters = it + 1

        # LIS anchors
        anchor_pos = _lis_indices(current)
        n_anchors = len(anchor_pos)
        a_raws = current[anchor_pos]

        result = current.copy()

        # Build gap segments: (ref_lo, ref_hi, raw_lo_bound, raw_hi_bound)
        segments = []
        if anchor_pos[0] > 0:
            segments.append((0, int(anchor_pos[0]), 0, int(a_raws[0])))
        for i in range(len(anchor_pos) - 1):
            r0, r1 = int(anchor_pos[i]), int(anchor_pos[i + 1])
            if r1 - r0 > 1:
                segments.append((r0 + 1, r1, int(a_raws[i]), int(a_raws[i + 1])))
        if anchor_pos[-1] < n_sub - 1:
            segments.append((int(anchor_pos[-1]) + 1, n_sub,
                             int(a_raws[-1]), n_raw - 1))

        # Sequential constrained re-match in each gap
        n_fixed_iter = 0
        for ref_lo, ref_hi, bound_lo, bound_hi in segments:
            floor = bound_lo
            for k in range(ref_lo, ref_hi):
                lo = floor
                hi = bound_hi + 1
                if lo >= hi:
                    result[k] = min(lo, n_raw - 1)
                    floor = result[k]
                    n_fixed_iter += 1
                    continue
                dists = np.linalg.norm(raw_pts[lo:hi] - ref_pts[k], axis=1)
                best = int(np.argmin(dists))
                result[k] = lo + best
                floor = result[k]
                n_fixed_iter += 1

        n_fixed_total += n_fixed_iter
        current = result

    matched_dist_new = np.linalg.norm(raw_pts[current] - ref_pts, axis=1)
    return current, matched_dist_new, n_anchors, n_fixed_total, n_iters
