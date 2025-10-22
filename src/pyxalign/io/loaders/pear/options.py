from abc import ABC
from typing import Optional
import numpy as np
import dataclasses
from dataclasses import field
from pyxalign.io.loaders.enums import LoaderType


@dataclasses.dataclass(kw_only=True)
class BaseLoadOptions:
    parent_projections_folder: str

    loader_type: LoaderType = LoaderType.PEAR_V1

    file_pattern: Optional[str] = None
    "pattern used by re to identify matching folder strings"

    scan_start: Optional[int] = None
    "Lower bound of scans to include."

    scan_end: Optional[int] = None
    "Upper bound of scans to include."

    scan_list: Optional[list[int]] = None
    """List of scans to load. This serves as an extra filter, meaning that 
    `scan_start`, `scan_end`, `file_pattern`, and all other settings/filters
    will still be applied."""

    file_pattern_priority_list: Optional[list[str]] = None
    """
    If multiple matching files are found, iterate through this list
    select the first file that matches the member of this list.
    """

    skip_files_not_in_priority_list: bool = True
    """
    Only applies when file_pattern_priority_list is not `None`. This 
    dictates what to do if there is a scan that has a file that matches 
    `file_pattern` but not any of the patterns in 
    `file_pattern_priority_list`.
    """

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
class PEARLoadOptions(ABC):
    base: BaseLoadOptions = field(default_factory=BaseLoadOptions)


@dataclasses.dataclass(kw_only=True)
class LYNXLoadOptions(PEARLoadOptions):
    dat_file_path: str

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
    """
    Specifies if data was taken in tile scan configuration. You can tell
    if data was taken in tile scan configuration by checking if the scan
    numbers in the tomography_scannumbers file """

    selected_tile: Optional[int] = None
    """
    The tile number to select. This is 1-indexed, so the minimum allowed
    value is 1 and the maximum allowed value is equal to the number of
    tiles.
    """


@dataclasses.dataclass(kw_only=True)
class MDAPEARLoadOptions(PEARLoadOptions):
    mda_folder: str
    """
    Folder containing MDA files, which in turn contain information
    about the measurement, including the measurment angle.
    """

    _mda_file_pattern: str

    _angle_pv_string: str


@dataclasses.dataclass(kw_only=True)
class Beamline2IDELoadOptions(MDAPEARLoadOptions):
    _mda_file_pattern: str = r"2xfm_(\d+)\.mda"

    _angle_pv_string: str = "2xfm:m60.VAL"

    channel_data_path: str = "/MAPS/XRF_roi"

    channel_names_path: str = "/MAPS/channel_names"


@dataclasses.dataclass(kw_only=True)
class Beamline2IDDLoadOptions(MDAPEARLoadOptions):
    _mda_file_pattern: str = r"bnp_fly(\d+)\.mda"

    _angle_pv_string: str = "9idbTAU:SM:ST:ActPos"