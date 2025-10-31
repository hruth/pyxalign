from typing import Optional
import dataclasses
from abc import ABC
from dataclasses import field

# Channel data could also be in /MAPS/XRF_fits
# For older data, the proper channels are:
# channel_data_path: str = "/MAPS/XRF_fits"
# channel_names_path: str = "/MAPS/channel_names"
# angle_PV_string: str = "2xfm:m58.VAL"


@dataclasses.dataclass
class XRFBaseLoadOptions:
    folder: str = ""
    "Folder containing data to load"

    scan_start: Optional[int] = None
    "Lower bound of scans to include."

    scan_end: Optional[int] = None
    "Upper bound of scans to include."

    scan_list: Optional[list[int]] = None
    "List of scans to load."


@dataclasses.dataclass
class XRFLoadOptions(ABC):
    base: XRFBaseLoadOptions = field(default_factory=XRFBaseLoadOptions)


@dataclasses.dataclass
class XRF2IDELoadOptions(XRFLoadOptions):
    """
    Loading options for XRF data from the 2-ID-E beamline
    """

    _channel_data_path: str = "/MAPS/XRF_roi"

    _channel_names_path: str = "/MAPS/channel_names"

    _angle_pv_string: str = "2xfm:m60.VAL"

    _lamino_angle_pv_string: str = "2xfm:m12.VAL"

    _mda_file_pattern: str = r"2xfm_(\d+)\.mda.h5"
