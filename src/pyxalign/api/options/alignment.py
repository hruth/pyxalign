from abc import ABC
import dataclasses
from dataclasses import field
from functools import partial
from typing import Optional
import pyxalign.api.enums as enums
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.options import RegularizationOptions
from pyxalign.api.options.plotting import UpdatePlotOptions, PlotDataOptions
from pyxalign.api.options.reconstruct import ReconstructOptions
from pyxalign.api.options.transform import CropOptions, DownsampleOptions
from .base import BaseOptions

@dataclasses.dataclass
class AlignmentOptions(ABC):
    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class CrossCorrelationOptions(BaseOptions):
    iterations: int = 10

    binning: int = 4

    filter_position: int = 101

    filter_data: float = 0.005

    remove_slow_variation: bool = False # change to True and update test scripts

    use_end_corrections: bool = True

    apply_optional_clamp: bool = True

    remove_ramp_artifacts: bool = False

    use_boundary_corrections: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)

    crop: CropOptions = field(default_factory=CropOptions)


@dataclasses.dataclass
class ProjectionMatchingPlotOptions(BaseOptions):
    update: UpdatePlotOptions = field(default_factory=UpdatePlotOptions)

    reconstruction: PlotDataOptions = field(default_factory=PlotDataOptions)

    projections: PlotDataOptions = field(default_factory=PlotDataOptions)


@dataclasses.dataclass
class ReconstructionMaskOptions(BaseOptions):
    enabled: bool = True

    rad_apod: int = 0

    radial_smooth: int = 5


@dataclasses.dataclass
class SecondaryMaskOptions(BaseOptions):
    enabled: bool = False

    rad_apod: int = 100

    radial_smooth: int = 5


@dataclasses.dataclass
class StepMomentum(BaseOptions):
    enabled: bool = False

    memory: int = 2

    alpha: float = 2.0

    gain: float = 0.5


@dataclasses.dataclass
class RefineGeometryOptions(BaseOptions):
    enabled: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)

    lamino_step_relax: float = 0.01

    tilt_step_relax: float = 0.01

    skew_step_relax: float = 0.01


@dataclasses.dataclass
class InteractiveViewerOptions(BaseOptions):
    close_old_windows: bool = True

    update: UpdatePlotOptions = field(
        default_factory=partial(UpdatePlotOptions, enabled=True, stride=10)
    )


def downsample_factory_for_estimate_center_options() -> DownsampleOptions:
    return DownsampleOptions(enabled=True)


@dataclasses.dataclass
class PositivityConstraint(BaseOptions):
    enabled: bool = False

    threshold: float = 0.0


@dataclasses.dataclass
class ProjectionMatchingOptions(BaseOptions):
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

    exclude_scans_from_alignment: Optional[list[int]] = None
    "These scans will not have their position updated"

    regularization: RegularizationOptions = field(default_factory=RegularizationOptions)

    refine_geometry: RefineGeometryOptions = field(default_factory=RefineGeometryOptions)

    momentum: StepMomentum = field(default_factory=StepMomentum)

    reconstruction_mask: ReconstructionMaskOptions = field(
        default_factory=ReconstructionMaskOptions
    )

    secondary_mask: SecondaryMaskOptions = field(default_factory=SecondaryMaskOptions)

    reconstruct: ReconstructOptions = field(default_factory=ReconstructOptions)

    tukey_shape_parameter: float = 0.2

    min_iterations: int = 1

    max_step_size: float = 0.5

    projection_shift_type: enums.ShiftType = enums.ShiftType.FFT

    mask_shift_type: enums.ShiftType = enums.ShiftType.CIRC

    prevent_wrapping_from_shift: bool = False

    filter_directions: tuple[int] = (2,)

    positivity_constraint: PositivityConstraint = field(default_factory=PositivityConstraint)

    plot: ProjectionMatchingPlotOptions = field(default_factory=ProjectionMatchingPlotOptions)
