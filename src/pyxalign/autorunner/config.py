from enum import StrEnum, auto
import dataclasses
from dataclasses import field
from typing import Optional
from pyxalign import reconstruct
from pyxalign.alignment import cross_correlation
from pyxalign.api.enums import PhaseUnwrapMethods
from pyxalign.api.options.alignment import CrossCorrelationOptions
from pyxalign.api.options.base import BaseOptions
from pyxalign.api.options.options import PhaseUnwrapOptions
from pyxalign.api.options.projections import ProbePositionMaskOptions, VolumeWidthOptions
from pyxalign.api.options.reconstruct import ReconstructOptions
from pyxalign.interactions import initialize_projections
from pyxalign.io.loaders.enums import ExperimentType
from pyxalign.unwrap import unwrap_phase


@dataclasses.dataclass
class LoadingConfig(BaseOptions):
    experiment_type: ExperimentType = ExperimentType.LYNX

    interactive: bool = False

    initial_options_path: Optional[str] = None
    # If none provided, or invalid path, or some error loading -- then interactivity is forced


@dataclasses.dataclass
class InitializationConfig(BaseOptions):
    pad: int = 0

    laminography_angle: int = 90  # tomography

    rotation_angle: float = 0

    shear_angle: float = 0

    sample_thickness: float = 7e-6 # Move to reconstruction geometry?

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
    initialization: bool = True

    cross_correlation: bool = True

    phase_unwrapping: bool = True

    phase_unwrap_masks: bool = True

    pma_masks: bool = True

    reconstruction_tuning: bool = True

@dataclasses.dataclass
class AutorunnerConfig(BaseOptions):
    state_folder: Optional[str] = None
    "Where state files get automatically saved to"

    use_state_file: bool = True

    update_state_file: bool = True

    # interactive_initialization: bool = True

    # interactive_cross_correlation: bool = True

    # interactive_phase_unwrapping: bool = True

    # interactive_phase_unwrap_masks: bool = True

    # interactive_pma_masks: bool = True

    # interactive_reconstruction_tuning: bool = True

    cross_correlation_enabled: bool = True

    interactivity: InteractivityConfig = field(default_factory=InteractivityConfig)

    loading: LoadingConfig = field(default_factory=LoadingConfig)

    initialize: InitializationConfig = field(default_factory=InitializationConfig)

    cross_correlation: CrossCorrelationOptions = field(default_factory=CrossCorrelationOptions)

    unwrap_phase: PhaseUnwrapOptions = field(default_factory=PhaseUnwrapOptions)

    phase_unwrap_masks: ProbePositionMaskOptions = field(
        default_factory=ProbePositionMaskOptions
    )

    projection_matching_masks: ProbePositionMaskOptions = field(
        default_factory=ProbePositionMaskOptions
    )

    reconstruct: ReconstructionGeometryConfig = field(default_factory=ReconstructionGeometryConfig)
