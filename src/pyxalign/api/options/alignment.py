from abc import ABC
import dataclasses
from dataclasses import field
from functools import partial
from typing import Optional

from numpy import save
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

    remove_slow_variation: bool = True

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

    volume_view_type: enums.VolumeViewType = enums.VolumeViewType.AXIS_SWITCHER


def downsample_factory_for_estimate_center_options() -> DownsampleOptions:
    return DownsampleOptions(enabled=True)

def regularizaton_factory_for_pma() -> RegularizationOptions:
    return RegularizationOptions(use_gpu=True)

@dataclasses.dataclass
class PositivityConstraint(BaseOptions):
    enabled: bool = False

    threshold: float = 0.0


@dataclasses.dataclass
class PMASequenceOptions(BaseOptions):
    """
    Controls what gets recorded into the PMASequence snapshot for a
    projection matching call. Only `record_volume` does anything when
    False — the rest only matter when recording is enabled.
    """

    record_volume: bool = False
    "Whether to capture the post-PMA volume in the PMASnapshot."

    volume_crop: CropOptions = field(default_factory=CropOptions)
    "In-plane (Y, X) crop applied to each recorded volume layer."

    volume_start_layer_fractional: float = 0.0
    "Fraction (0..1) of the volume's first axis to start recording from."

    volume_end_layer_fractional: float = 1.0
    "Fraction (0..1) of the volume's first axis to stop recording at."


@dataclasses.dataclass
class SaveOptions(BaseOptions):
    enabled: bool = False

    folder: str = ""

    suffix: str = ""

    save_pma_volume: bool = False

    save_pma_projections: bool = False

    save_pma_forward_projections: bool = False


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

    regularization: RegularizationOptions = field(default_factory=regularizaton_factory_for_pma)

    refine_geometry: RefineGeometryOptions = field(default_factory=RefineGeometryOptions)

    momentum: StepMomentum = field(default_factory=StepMomentum)

    reconstruction_mask: ReconstructionMaskOptions = field(
        default_factory=ReconstructionMaskOptions
    )

    secondary_mask: SecondaryMaskOptions = field(default_factory=SecondaryMaskOptions)

    reconstruct: ReconstructOptions = field(default_factory=ReconstructOptions)

    override_projection_geometry: bool = False

    tukey_shape_parameter: float = 0.2

    min_iterations: int = 1

    max_step_size: float = 0.5

    projection_shift_type: enums.ShiftType = enums.ShiftType.FFT

    mask_shift_type: enums.ShiftType = enums.ShiftType.CIRC

    prevent_wrapping_from_shift: bool = False

    filter_directions: tuple[int] = (2,)

    positivity_constraint: PositivityConstraint = field(default_factory=PositivityConstraint)

    horizontal_offset: float = 0

    vertical_offset: float = 0

    sample_thickness: Optional[float] = None

    save: SaveOptions = field(default_factory=SaveOptions)

    pma_sequence: PMASequenceOptions = field(default_factory=PMASequenceOptions)

    # plot: ProjectionMatchingPlotOptions = field(default_factory=ProjectionMatchingPlotOptions)
