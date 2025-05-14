from abc import ABC
import dataclasses
from dataclasses import field
from functools import partial
import llama.api.enums as enums
from llama.api.options.device import DeviceOptions
from llama.api.options.options import RegularizationOptions
from llama.api.options.plotting import UpdatePlotOptions, PlotDataOptions
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

    remove_slow_variation: bool = False

    use_end_corrections: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)

    crop: CropOptions = field(default_factory=CropOptions)


@dataclasses.dataclass
class ProjectionMatchingPlotOptions:
    update: UpdatePlotOptions = field(default_factory=UpdatePlotOptions)

    reconstruction: PlotDataOptions = field(default_factory=PlotDataOptions)

    projections: PlotDataOptions = field(default_factory=PlotDataOptions)


@dataclasses.dataclass
class ReconstructionMaskOptions:
    enabled: bool = True

    rad_apod: int = 0

    radial_smooth: int = 5


@dataclasses.dataclass
class SecondaryMaskOptions:
    enabled: bool = False

    rad_apod: int = 100

    radial_smooth: int = 5


@dataclasses.dataclass
class StepMomentum:
    enabled: bool = False

    memory: int = 2

    alpha: float = 2

    gain: float = 0.5


@dataclasses.dataclass
class RefineGeometryOptions:
    enabled: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)

    step_relax: float = 0.01


@dataclasses.dataclass
class InteractiveViewerOptions:
    close_old_windows: bool = True

    update: UpdatePlotOptions = field(
        default_factory=partial(UpdatePlotOptions, enabled=True, stride=10)
    )


@dataclasses.dataclass
class ProjectionMatchingOptions(AlignmentOptions):
    iterations: int = 300

    high_pass_filter: float = 0.005

    tukey_shape_parameter: float = 0.2

    reconstruction_mask: ReconstructionMaskOptions = field(
        default_factory=ReconstructionMaskOptions
    )

    secondary_mask: SecondaryMaskOptions = field(default_factory=SecondaryMaskOptions)

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

    filter_directions: tuple[int] = (2,)

    momentum: StepMomentum = field(default_factory=StepMomentum)

    plot: ProjectionMatchingPlotOptions = field(default_factory=ProjectionMatchingPlotOptions)

    refine_geometry: RefineGeometryOptions = field(default_factory=RefineGeometryOptions)

    interactive_viewer: InteractiveViewerOptions = field(default_factory=InteractiveViewerOptions)
