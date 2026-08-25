"""Step fitting via ruptures change-point detection."""

import numpy as np
import ruptures as rpt


def run_step_fit(trace, penalty=0.1, min_size=5, min_step=None):
    """Detect piecewise-constant steps in a 1-D trace.

    Parameters
    ----------
    trace : 1-D array — signal to fit (any units)
    penalty : float — ruptures penalty value (lower = more steps)
    min_size : int — minimum segment length in samples
    min_step : float or None — minimum step size in the trace's native
        units; smaller jumps are merged.  If None, no merging is done.

    Returns
    -------
    boundaries : 1-D int array — change-point indices into *trace*
        (includes 0 as first element, len(trace) as last)
    levels : 1-D array — mean value of each segment
        (len == len(boundaries) - 1)
    """
    algo = rpt.KernelCPD(kernel="linear", min_size=min_size).fit(trace)
    bkps = algo.predict(pen=penalty)  # list ending with len(trace)

    # Compute segment means
    segments = np.split(trace, bkps[:-1])
    levels = np.array([seg.mean() for seg in segments])

    bkps_arr = np.array([0] + bkps)  # prepend 0

    # Merge small steps if threshold is given
    if min_step is not None and min_step > 0:
        levels, bkps_arr = _merge_small_steps(trace, levels, bkps_arr, min_step)

    return bkps_arr, levels


def _merge_small_steps(trace, levels, boundaries, min_rad):
    """Iteratively merge adjacent segments whose level difference < min_rad."""
    levels = list(levels)
    bounds = list(boundaries)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(levels) - 1:
            if abs(levels[i + 1] - levels[i]) < min_rad:
                # merge: remove boundary between i and i+1
                del bounds[i + 1]
                del levels[i + 1]
                # recompute mean for merged segment
                levels[i] = float(trace[bounds[i]:bounds[i + 1]].mean()) \
                    if bounds[i + 1] > bounds[i] else levels[i]
                changed = True
                i = max(i - 1, 0)
            else:
                i += 1
    return np.array(levels), np.array(bounds)


def reconstruct_step_trace(trace_len, boundaries, levels):
    """Build a piecewise-constant array from boundaries and levels."""
    out = np.empty(trace_len)
    for k in range(len(levels)):
        out[boundaries[k]:boundaries[k + 1]] = levels[k]
    return out
