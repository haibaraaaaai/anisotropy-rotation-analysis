"""I/O helpers — TDMS loading."""

from pathlib import Path
import numpy as np
from nptdms import TdmsFile

# Stack order used throughout the pipeline
_orientations = ["90", "45", "135", "0"]


def get_pol_ind(listpol):
    """Return stack-order indices for the requested polarisation labels."""
    return [_orientations.index(p) for p in listpol]


def load_tdms(tdms_path):
    """Load four polarisation channels from a TDMS file.

    Returns
    -------
    dict with keys:
        c0_raw, c90_raw, c45_raw, c135_raw : 1-D float arrays
        time_s : 1-D float array
        freq : float (sampling rate Hz)
        datasize : int
        time_inc, time_off : floats
    """
    tdms_path = Path(tdms_path)
    tdms_obj = TdmsFile.open(str(tdms_path))
    group = tdms_obj.groups()[0]
    channels = group.channels()
    datasize = len(channels[0])

    try:
        time_inc = float(channels[0].properties["wf_increment"])
        time_off = float(channels[0].properties["wf_start_offset"])
    except KeyError:
        time_inc = 4e-6
        time_off = 0.0

    freq = 1.0 / time_inc
    time_s = time_off + time_inc * np.arange(datasize)

    idx = get_pol_ind(["0", "90", "45", "135"])
    c0_raw = np.asarray(channels[idx[0]][:], dtype=float)
    c90_raw = np.asarray(channels[idx[1]][:], dtype=float)
    c45_raw = np.asarray(channels[idx[2]][:], dtype=float)
    c135_raw = np.asarray(channels[idx[3]][:], dtype=float)

    return {
        "c0_raw": c0_raw,
        "c90_raw": c90_raw,
        "c45_raw": c45_raw,
        "c135_raw": c135_raw,
        "time_s": time_s,
        "freq": freq,
        "datasize": datasize,
        "time_inc": time_inc,
        "time_off": time_off,
    }


def anisotropy_from_channels(c0, c90, c45, c135):
    """Compute anisotropy (ax, ay) from four polarisation channels."""
    denom_x = c0 + c90
    denom_y = c45 + c135
    ax = np.where(denom_x > 0, (c0 - c90) / denom_x, 0.0)
    ay = np.where(denom_y > 0, (c45 - c135) / denom_y, 0.0)
    return ax, ay

