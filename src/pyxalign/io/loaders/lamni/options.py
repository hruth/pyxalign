from typing import Optional, TypeVar, Union
import numpy as np
import dataclasses
from dataclasses import field
from pyxalign.io.loaders.enums import LoaderType, ExperimentInfoSourceType
from pyxalign.io.loaders.utils import select_loader_type_from_prompt


@dataclasses.dataclass
class BaseLoadOptions:
    parent_projections_folder: str

    loader_type: LoaderType = LoaderType.PEAR_V1

    # def __post_init__(self):
    #     if self.loader_type is None:  # Check if the variable is missing
    #         self.loader_type = select_loader_type_from_prompt()  # Assign a generated value

    file_pattern: str = None
    "pattern used by re to identify matching folder strings"

    scan_start: Optional[int] = None
    "Lower bound of scans to include."

    scan_end: Optional[int] = None
    "Upper bound of scans to include."

    only_include_files_with: Optional[list[str]] = None
    "Only include files with these strings in the ptycho file string."

    exclude_files_with: Optional[list[str]] = None
    "Exclude files with any of these strings in the ptycho file string."

    selected_ptycho_strings: Optional[list[str]] = None
    """
    List of ptycho file strings that are allowed to be loaded, 
    in prioritized order. The ptycho file strings are strings
    extracted from the projection file names. 
    """

    ask_for_backup_files: bool = False
    "Whether or not the UI asks for backup files if a projection file is not found."

    select_all_by_default: bool = False

    def print_selections(self):
        if np.all([v is None for v in self.__dict__.values()]):
            print("No loading options provided.", flush=True)
        else:
            print("User-provided loading options:", flush=True)
            for k, v in self.__dict__.items():
                if v is not None:
                    print(f"  {k}: {v}", flush=True)


@dataclasses.dataclass
class LYNXLoadOptions:
    dat_file_path: str

    base: BaseLoadOptions = field(default_factory=BaseLoadOptions)

    selected_experiment_name: Optional[str] = None
    """Name of the experiment to load. Use "unlabeled" to refer to
    experiments that do not have a name specified in the dat file."""

    selected_sequences: Optional[tuple[int]] = None
    """
    List of sequence numbers to load in. Each sequence corresponds
    to a set of measurements taken sequentially over a 360 degree range.
    The sequence number of a projection comes from the dat file.
    """

    is_tile_scan: bool = False

    selected_tile: Optional[int] = None

    n_tiles: Optional[int] = None


@dataclasses.dataclass
class Beamline2IDELoadOptions:
    mda_folder: str

    base: BaseLoadOptions = field(default_factory=BaseLoadOptions)
