"""Optional external inputs for raw-data and measured-mask reprocessing."""
from pathlib import Path
import os
STUDY_ROOT=Path(__file__).resolve().parent
REPO_ROOT=STUDY_ROOT.parents[1]
def tdms_path():
    return Path(os.environ.get("APD_TDMS_PATH", str(REPO_ROOT / "files3_first40s.tdms"))).expanduser()
def mask_path():
    value=os.environ.get("APD_MASK_PATH")
    if not value:
        raise FileNotFoundError("Set APD_MASK_PATH to the thesis image_mirror_binary.tif; private source mask is not distributed here. Cached model results do not need it.")
    return Path(value).expanduser()
