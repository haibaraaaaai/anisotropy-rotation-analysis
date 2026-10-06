"""Time-ordered pre-fit sampling; no cone or background model is fitted."""
import numpy as np
from scipy.signal import savgol_filter


def channel_xy(channels):
    """Input order 0,90,45,135; invalid pair sums remain NaN, never origin points."""
    c = np.asarray(channels, dtype=float)
    sums = np.array([c[0]+c[1], c[2]+c[3]])
    diffs = np.array([c[0]-c[1], c[2]-c[3]])
    return np.divide(diffs, sums, out=np.full_like(diffs, np.nan), where=sums>0).T


def ordered_channel_bins(channels, start, end, n_points=90, guide_window=41):
    """Partition a selected time interval into contiguous, non-overlapping bins.

    Savgol-smoothed CHANNELS provide only an approximate arc-length guide.
    All fitted channel values are means of the original samples in each bin.
    No spatial nearest-neighbour assignment, ridge snapping, periodic wrapping,
    or forced closure. Every selected sample contributes exactly once; separate
    visits to the same XY location remain separate bins. The user must still
    choose an interval containing the intended full trajectory.
    """
    c = np.asarray(channels, dtype=float)
    if c.ndim != 2 or c.shape[0] != 4 or not np.all(np.isfinite(c)):
        raise ValueError('Require finite channels with shape (4, N).')
    if not (0 <= start < end <= c.shape[1]) or end-start < n_points or n_points < 3:
        raise ValueError('Need a valid interval with at least n_points >= 3 samples.')
    if guide_window < 5 or guide_window % 2 != 1 or guide_window > c.shape[1]:
        raise ValueError('guide_window must be odd, >=5, and no longer than the data.')
    smooth = savgol_filter(c, guide_window, 3, axis=1, mode='interp')
    guide = channel_xy(smooth) [start:end]
    if not np.all(np.isfinite(guide)):
        raise ValueError('Invalid pair sums in trajectory guide; inspect offsets/signals.')
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide, axis=0), axis=1))]
    if s[-1] <= 1e-12:
        raise ValueError('No resolved trajectory; cannot distribute points by arc length.')
    n = end-start
    edges = np.r_[0, np.searchsorted(s, np.linspace(0,s[-1],n_points+1)[1:-1]), n]
    # Keep non-empty bins even when one large step spans several target arcs.
    for k in range(1,n_points):
        edges[k] = np.clip(edges[k], edges[k-1]+1, n-(n_points-k))
    bounds = np.column_stack([edges[:-1]+start, edges[1:]+start])
    means = np.column_stack([c[:,a:b].mean(axis=1) for a,b in bounds])
    centers = (bounds[:,0]+bounds[:,1]-1)/2
    guide_centers = np.column_stack([np.interp(centers-start,np.arange(n),guide[:,j]) for j in range(2)])
    xy = channel_xy(means)
    if not np.all(np.isfinite(xy)):
        raise ValueError('Invalid pair sums in channel means; inspect offsets/signals.')
    return dict(channels=means, xy=xy, bounds=bounds, centers=centers,
                guide=guide, guide_centers=guide_centers,
                endpoint_gap=float(np.linalg.norm(guide[-1]-guide[0])),
                open_arc_length=float(s[-1]), counts=np.diff(edges))


def anisotropy_axes(ax):
    """Consistent view only: values outside the frame are not clipped or discarded."""
    ax.set(xlim=(-1,1), ylim=(-1,1), aspect='equal')
    return ax
