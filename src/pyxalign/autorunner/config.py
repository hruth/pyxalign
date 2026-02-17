from enum import StrEnum, auto
import dataclasses
from dataclasses import field
from typing import Optional
from pyxalign.api.options.base import BaseOptions
from pyxalign.io.loaders.enums import ExperimentType


@dataclasses.dataclass
class LoadingConfig(BaseOptions):
    experiment_type: ExperimentType = ExperimentType.LYNX

    interactive: bool = False

    initial_options_path: Optional[str] = None
    # If none provided, or invalid path, or some error loading -- then interactivity is forced


@dataclasses.dataclass
class AutorunnerConfig(BaseOptions):
    state_folder: Optional[str] = None
    "Where state files get automatically saved to"

    use_state_file: bool = True

    update_state_file: bool = True

    loading: LoadingConfig = field(default_factory=LoadingConfig)
