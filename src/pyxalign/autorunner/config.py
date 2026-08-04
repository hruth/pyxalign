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
from pyxalign.autorunner.enums import Checkpoints, LoadableCheckpoints
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
    # currently unused; will be useful later if adding
    # optional automation to code...
    loading: bool = True

    complex_projections_window: bool = True

    phase_unwrapping: bool = True

    phase_projections_window: bool = True


@dataclasses.dataclass
class EnabledCheckpoints(BaseOptions):
    # its not ideal, but these need to manually be made to match
    # the strings in the 'Checkpoints' enum

    after_loading: bool = True

    after_complex_projections_window: bool = True

    after_phase_unwrapping_window: bool = True

    final: bool = True


@dataclasses.dataclass
class StateConfig(BaseOptions):
    use_state_file_settings: bool = True
    """Load settings from the state file at the start of the run, if one exists."""

    update_state_file: bool = True
    """Write updated settings back to the state file after each step."""


@dataclasses.dataclass
class CheckpointConfig(BaseOptions):
    load_from_checkpoint: bool = False
    """Resume the run from a previously saved checkpoint instead of starting from scratch."""

    which_checkpoint: LoadableCheckpoints = LoadableCheckpoints.AFTER_LOADING
    """
    The checkpoint stage to resume from when load_from_checkpoint is enabled.

    Ex: if "after_loading" is selected, the autorunner will load the window that appears
    after loading data. 
    
    If load_from_custom_task is NOT checked, the task will be loaded from 
    checkpoints/after_loading_task.h5. If load_from_custom_task is checked,
    then the task specified in custom_task_path will be loaded.
    """

    load_from_custom_task: bool = False
    """
    Load the checkpoint from a custom task path instead of the default task path.
    """

    custom_task_path: str = ""
    """
    Path to the custom task folder to load the checkpoint from.
    """

    enabled_checkpoints: EnabledCheckpoints = field(default_factory=EnabledCheckpoints)
    """Choose the points at which a checkpoint task will be saved"""


@dataclasses.dataclass
class AutorunnerConfig(BaseOptions):
    state: StateConfig = field(default_factory=StateConfig)
    """Settings for reading from and writing to a state file."""

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Settings for saving and loading run checkpoints."""

    interactivity: InteractivityConfig = field(default_factory=InteractivityConfig)

    cross_correlation_enabled: bool = True
    """Run the cross-correlation alignment step."""

    projection_matching_enabled: bool = True
    """Run the projection-matching alignment step."""

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

    load_pma_sequence_volumes: bool = True
    """When loading a checkpoint task, also load the PMA sequence volume arrays."""
