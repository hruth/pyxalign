from typing import Optional
import dataclasses


# Channel data could also be in /MAPS/XRF_fits
# For older data, the proper channels are:
# channel_data_path: str = "/MAPS/XRF_fits"
# channel_names_path: str = "/MAPS/channel_names"
# angle_PV_string: str = "2xfm:m58.VAL"
@dataclasses.dataclass
class XRFLoadOptions:
    channel_data_path: str = "/MAPS/XRF_roi"

    channel_names_path: str = "/MAPS/channel_names"

    angle_PV_string: str = "2xfm:m60.VAL"

    lamino_angle_PV_string: str = "2xfm:m12.VAL"

    file_pattern: str = r"2xfm_(\d+)\.mda.h5"
    ends_with: str = ""

    scan_start: Optional[int] = None
    "Lower bound of scans to include."

    scan_end: Optional[int] = None
    "Upper bound of scans to include."
