from abc import ABC
import dataclasses
from dataclasses import field
import llama.api.enums as enums
from llama.api.options.device import DeviceOptions
from llama.api.options.options import RegularizationOptions, UpdatePlotOptions
from llama.api.options.reconstruct import ReconstructOptions
from llama.api.options.transform import CropOptions, DownsampleOptions


@dataclasses.dataclass
class AlignmentOptions(ABC):
    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class CrossCorrelationOptions(AlignmentOptions):
    iterations: int = 100

    binning: int = 4

    filter_position: int = 101

    filter_data: float = 0.005

    precision: float = 0.01

    device: DeviceOptions = field(default_factory=DeviceOptions)

    crop: CropOptions = field(default_factory=CropOptions)


@dataclasses.dataclass
class ProjectionMatchingOptions(AlignmentOptions):
    iterations: int = 300

    high_pass_filter: float = 0.005

    step_relax: float = 0.1

    min_step_size: float = 0.01

    max_step_size: float = 0.5

    regularization: RegularizationOptions = field(default_factory=RegularizationOptions)

    keep_on_gpu: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)

    reconstruct: ReconstructOptions = field(default_factory=ReconstructOptions)

    crop: CropOptions = field(default_factory=CropOptions)

    downsample: DownsampleOptions = field(default_factory=DownsampleOptions)

    projection_shift_type: enums.ShiftType = enums.ShiftType.FFT

    mask_shift_type: enums.ShiftType = enums.ShiftType.CIRC

    update_plot: UpdatePlotOptions = field(default_factory=UpdatePlotOptions)
