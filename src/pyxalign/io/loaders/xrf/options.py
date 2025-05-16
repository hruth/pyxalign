from abc import ABC
from typing import Optional
import numpy as np
import dataclasses
from dataclasses import field
from pyxalign.io.loaders.enums import XRFLoaderType
from pyxalign.io.loaders.lamni.options import LamniLoadOptions
from pyxalign.io.loaders.utils import select_loader_type_from_prompt


@dataclasses.dataclass
class XRFBaseLoadOptions:
    loader_type: Optional[XRFLoaderType] = dataclasses.field(default=None)

    scan_start: Optional[int] = None
    "Lower bound of scans to include."

    scan_end: Optional[int] = None
    "Upper bound of scans to include."


@dataclasses.dataclass
class XRFLoadOptions(ABC):
    @property
    def base(self) -> XRFBaseLoadOptions:
        raise NotImplementedError


@dataclasses.dataclass
class XRFV1LoadOptions(XRFLoadOptions):
    # Channel data could also be in /MAPS/XRF_fits
    # For older data, the proper channels are:
    # channel_data_path: str = "/MAPS/XRF_fits"
    # channel_names_path: str = "/MAPS/channel_names"
    # angle_PV_string: str = "2xfm:m58.VAL"
    base: XRFBaseLoadOptions = field(default_factory=XRFBaseLoadOptions)

    file_pattern: str = r"2xfm_(\d+)\.mda.h5"

    ends_with: str = ""

    channel_data_path: str = "/MAPS/XRF_roi"

    channel_names_path: str = "/MAPS/channel_names"

    angle_PV_string: str = "2xfm:m60.VAL"

    lamino_angle_PV_string: str = "2xfm:m12.VAL"


# @dataclasses.dataclass
# class XRFV2LoadOptions(XRFBaseLoadOptions):
    # base: XRFBaseLoadOptions = field(default_factory=XRFBaseLoadOptions)

# @dataclasses.dataclass
# class XRFV2LoadOptions(XRFBaseLoadOptions):
    
@dataclasses.dataclass
class XRFV2LoadOptions(LamniLoadOptions):
    base: XRFBaseLoadOptions = field(default_factory=XRFBaseLoadOptions)


# # update for xrfv1
# def factory_for_xrf_v1_base_load_options():
#     return XRFLoadOptions(
#         file_pattern=r"2xfm_(\d+)\.mda.h5",
#     )
