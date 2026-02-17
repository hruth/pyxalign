from enum import StrEnum, auto
import dataclasses
from dataclasses import field
from typing import Optional
from pyxalign.api.options.base import BaseOptions
from pyxalign.interactions import initialize_projections
from pyxalign.io.loaders.enums import ExperimentType


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

    sample_thickness: float = 7e-6

    remove_scan_numbers: Optional[list] = None


@dataclasses.dataclass
class AutorunnerConfig(BaseOptions):
    state_folder: Optional[str] = None
    "Where state files get automatically saved to"

    use_state_file: bool = True

    update_state_file: bool = True

    loading: LoadingConfig = field(default_factory=LoadingConfig)

    initialize: InitializationConfig = field(default_factory=InitializationConfig)
