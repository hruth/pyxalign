import dataclasses
from dataclasses import field
from numbers import Number
from typing import Optional, Union

from matplotlib.pyplot import viridis
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.transform import DownsampleOptions
from pyxalign.api import enums
from functools import partial


@dataclasses.dataclass
class ExperimentOptions:
    laminography_angle: float = 61.1

    # tilt_angle: float = 0

    # skew_angle: float = 0

    pixel_size: float = 1.0

    sample_thickness: float = 7e-6
    "Thickness of the volume in meters"

    sample_width_type: enums.VolumeWidthType = enums.VolumeWidthType.AUTO
    "Determines if sample width is calculated manually or automatically"

    sample_width: Optional[float] = None
    "Width of the volume in meters, only used if sample_width_type is set to manual"


@dataclasses.dataclass
class MaskOptions:
    downsample: DownsampleOptions = field(
        default_factory=partial(
            DownsampleOptions, type=enums.DownsampleType.NEAREST, scale=4, enabled=True
        )
    )

    # upsample_options = UpsampleOptions(
    #     type=enums.UpsampleType.NEAREST, scale=4, enabled=True, device_options=enums.DeviceType.CPU
    # )

    binary_close_coefficient: int = 30

    binary_erode_coefficient: int = 30

    unsharp: bool = True

    fill: int = 8


@dataclasses.dataclass
class PhaseRampRemovalOptions:
    iterations: int = 5

    downsampling: int = 8


@dataclasses.dataclass
class PhaseUnwrapOptions:
    device: DeviceOptions = field(default_factory=DeviceOptions)

    iterations: int = 10

    # poly_fit_order: int = 1

    lsq_fit_ramp_removal: bool = False


@dataclasses.dataclass
class RegularizationOptions:
    enabled: bool = False

    local_TV_lambda: float = 1e-4

    iterations: int = 10
