from enum import StrEnum, auto
import dataclasses
from dataclasses import field
import itertools
from typing import Optional
from pyxalign import reconstruct
from pyxalign.alignment import cross_correlation, projection_matching
from pyxalign.api.enums import PhaseUnwrapMethods
from pyxalign.api.options.alignment import CrossCorrelationOptions, ProjectionMatchingOptions
from pyxalign.api.options.base import BaseOptions
from pyxalign.api.options.options import PhaseUnwrapOptions
from pyxalign.api.options.projections import ProbePositionMaskOptions, VolumeWidthOptions
from pyxalign.api.options.reconstruct import ReconstructOptions
from pyxalign.api.options.roi import ROIOptions
from pyxalign.autorunner.enums import Checkpoints
from pyxalign.interactions import initialize_projections
from pyxalign.interactions.roi_selector import MaskFromROISelector
from pyxalign.io.loaders.enums import ExperimentType
from pyxalign.unwrap import unwrap_phase


@dataclasses.dataclass
class LoadingConfig(BaseOptions):
    experiment_type: ExperimentType = ExperimentType.LYNX_PEAR

    # initial_options_path: Optional[str] = None
    # # If none provided, or invalid path, or some error loading -- then interactivity is forced


@dataclasses.dataclass
class InitializationConfig(BaseOptions):
    pad: int = 0

    laminography_angle: float = 90.0  # tomography

    rotation_angle: float = 0

    shear_angle: float = 0

    # sample_thickness: float = 7e-6 # Move to reconstruction geometry?

    remove_scan_numbers: Optional[list] = None


@dataclasses.dataclass
class ReconstructionGeometryConfig(BaseOptions):
    center_horizontal_offset: float = 0

    center_vertical_offset: float = 0

    sample_thickness: float = 7e-6

    volume_width: VolumeWidthOptions = field(default_factory=VolumeWidthOptions)

    reconstruct: ReconstructOptions = field(default_factory=ReconstructOptions)


@dataclasses.dataclass
class InteractivityConfig(BaseOptions):
    loading: bool = True

    initialization: bool = True

    cross_correlation: bool = True

    phase_unwrapping: bool = True

    phase_unwrap_masks: bool = True

    reconstruction_tuning: bool = True

    pma_masks: bool = True

    projection_matching: bool = True

    # final_reconstruction: bool = True


@dataclasses.dataclass
class EnabledCheckpoints(BaseOptions):
    # loading: bool = True

    initialization: bool = True

    cross_correlation: bool = True

    # phase_unwrap_masks: bool = True

    phase_unwrapping: bool = True

    reconstruction_tuning: bool = True

    pma_masks: bool = True

    projection_matching: bool = True

    final_reconstruction: bool = True


@dataclasses.dataclass
class StateConfig(BaseOptions):
    # state_folder: str = ""

    use_state_file_settings: bool = True

    update_state_file: bool = True


@dataclasses.dataclass
class CheckpointConfig(BaseOptions):
    load_from_checkpoint: bool = False

    which_checkpoint: Checkpoints = Checkpoints.INITIALIZATION

    load_from_custom_task: bool = False

    custom_task_path: str = ""

    enabled_checkpoints: EnabledCheckpoints = field(default_factory=EnabledCheckpoints)


@dataclasses.dataclass
class AutorunnerConfig(BaseOptions):
    state: StateConfig = field(default_factory=StateConfig)

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    interactivity: InteractivityConfig = field(default_factory=InteractivityConfig)

    cross_correlation_enabled: bool = True

    projection_matching_enabled: bool = True

    loading: LoadingConfig = field(default_factory=LoadingConfig)

    initialize: InitializationConfig = field(default_factory=InitializationConfig)

    cross_correlation: CrossCorrelationOptions = field(default_factory=CrossCorrelationOptions)

    unwrap_phase: PhaseUnwrapOptions = field(default_factory=PhaseUnwrapOptions)

    # phase_unwrap_masks: ProbePositionMaskOptions = field(default_factory=ProbePositionMaskOptions)

    # projection_matching_masks: ProbePositionMaskOptions = field(
    #     default_factory=ProbePositionMaskOptions
    # )

    phase_unwrap_masks_from_position: ProbePositionMaskOptions = field(
        default_factory=ProbePositionMaskOptions
    )

    projection_matching_masks_from_position: ProbePositionMaskOptions = field(
        default_factory=ProbePositionMaskOptions
    )

    phase_unwrap_masks_from_roi: ROIOptions = field(default_factory=ROIOptions)

    projection_matching_masks_from_roi: ROIOptions = field(default_factory=ROIOptions)

    reconstruct: ReconstructionGeometryConfig = field(default_factory=ReconstructionGeometryConfig)

    projection_matching: ProjectionMatchingOptions = field(
        default_factory=ProjectionMatchingOptions
    )
