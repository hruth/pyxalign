from abc import ABC
import dataclasses
from dataclasses import field
from functools import partial
import pyxalign.api.enums as enums
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.options import RegularizationOptions
from pyxalign.api.options.plotting import UpdatePlotOptions, PlotDataOptions
from pyxalign.api.options.reconstruct import ReconstructOptions
from pyxalign.api.options.transform import CropOptions, DownsampleOptions


@dataclasses.dataclass
class AlignmentOptions(ABC):
    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class CrossCorrelationOptions:
    iterations: int = 100

    binning: int = 4

    filter_position: int = 101

    filter_data: float = 0.005

    precision: float = 0.01

    remove_slow_variation: bool = False

    use_end_corrections: bool = False

    apply_optional_clamp: bool = True

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

    alpha: float = 2.0

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


def downsample_factory_for_estimate_center_options() -> DownsampleOptions:
    return DownsampleOptions(enabled=True)


@dataclasses.dataclass
class ProjectionMatchingOptions:
    device: DeviceOptions = field(default_factory=DeviceOptions)

    keep_on_gpu: bool = False

    interactive_viewer: InteractiveViewerOptions = field(default_factory=InteractiveViewerOptions)

    iterations: int = 300

    downsample: DownsampleOptions = field(
        default_factory=downsample_factory_for_estimate_center_options
    )

    crop: CropOptions = field(default_factory=CropOptions)

    high_pass_filter: float = 0.005

    step_relax: float = 0.1

    min_step_size: float = 0.01

    regularization: RegularizationOptions = field(default_factory=RegularizationOptions)

    refine_geometry: RefineGeometryOptions = field(default_factory=RefineGeometryOptions)

    momentum: StepMomentum = field(default_factory=StepMomentum)

    reconstruction_mask: ReconstructionMaskOptions = field(
        default_factory=ReconstructionMaskOptions
    )

    secondary_mask: SecondaryMaskOptions = field(default_factory=SecondaryMaskOptions)

    reconstruct: ReconstructOptions = field(default_factory=ReconstructOptions)

    tukey_shape_parameter: float = 0.2

    max_step_size: float = 0.5

    projection_shift_type: enums.ShiftType = enums.ShiftType.FFT

    mask_shift_type: enums.ShiftType = enums.ShiftType.CIRC

    filter_directions: tuple[int] = (2,)

    plot: ProjectionMatchingPlotOptions = field(default_factory=ProjectionMatchingPlotOptions)
