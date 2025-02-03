import dataclasses
from dataclasses import field
from numbers import Number
from typing import Optional

from matplotlib.pyplot import viridis
from llama.api.options.device import DeviceOptions
from llama.api.options.transform import DownsampleOptions
from llama.api import enums
from functools import partial


@dataclasses.dataclass
class ExperimentOptions:
    laminography_angle: float = 61.1

    # tilt_angle: float = 0

    # skew_angle: float = 0

    sample_thickness: float = 7e-6

    pixel_size: float = 1.0


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

    poly_fit_order: int = 1


@dataclasses.dataclass
class RegularizationOptions:
    enabled: bool = False

    local_TV: bool = False

    local_TV_lambda: float = 3e-4
