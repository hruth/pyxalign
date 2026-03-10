from enum import StrEnum
from typing import Optional
import dataclasses
from abc import ABC
from dataclasses import field

from pyxalign.api.options.base import BaseOptions
from pyxalign.io.loaders.enums import LaminoAnglePVStrings, MDAFilePatterns, RotationAnglePVStrings

# Channel data could also be in /MAPS/XRF_fits
# For older data, the proper channels are:
# channel_data_path: str = "/MAPS/XRF_fits"
# channel_names_path: str = "/MAPS/channel_names"
# angle_PV_string: str = "2xfm:m58.VAL"


class XRFHDF5Paths(StrEnum):
    """Different paths to the XRF image data in the analyzed XRF file.

    New files have the file extension .mda.h5 and have data at `ROI`,
    `NNLS`, and/or `MATRIX`.

    Legacy files haev the file extensions .h5 and have data at
    `LEGACY_ROI` and/or `LEGACY_MATRIX`.

    """

    ROI = "/MAPS/XRF_Analyzed/ROI/Counts_Per_Sec"
    NNLS = "/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec"
    MATRIX = " /MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec"
    LEGACY_ROI = "/MAPS/XRF_roi"
    LEGACY_MATRIX = "/MAPS/XRF_roi"


@dataclasses.dataclass
class XRFBaseLoadOptions(BaseOptions):
    folder: str = ""
    "Folder containing data to load"

    scan_start: Optional[int] = None
    "Lower bound of scans to include."

    scan_end: Optional[int] = None
    "Upper bound of scans to include."

    scan_list: Optional[list[int]] = None
    "List of scans to load."


@dataclasses.dataclass
class XRFLoadOptions(ABC, BaseOptions):
    base: XRFBaseLoadOptions = field(default_factory=XRFBaseLoadOptions)


@dataclasses.dataclass
class XRF2IDELoadOptions(XRFLoadOptions):
    """Loading options for XRF data from the 2-ID-E beamline."""

    _channel_data_path: XRFHDF5Paths = XRFHDF5Paths.NNLS
    """Path in the .h5 file to the XRF image."""

    _channel_names_path: str = "/MAPS/channel_names"

    _angle_pv_string: RotationAnglePVStrings = RotationAnglePVStrings.XFM_M60_VAL
    "String for accessing the rotation angle from the extra PVs dict."

    _lamino_angle_pv_string: LaminoAnglePVStrings = LaminoAnglePVStrings.XFM_M12_VAL
    "String for accessing the laminography angle from the extra PVs dict."

    _mda_file_pattern: MDAFilePatterns = MDAFilePatterns.XFM_MDA_H5
    "File pattern of the files in the mda file folder."