from typing import Optional
import numpy as np
import dataclasses
from llama.io.loaders.enums import LoaderType
from llama.io.loaders.utils import select_loader_type_from_prompt


@dataclasses.dataclass
class LamniLoadOptions:
    loader_type: Optional[LoaderType] = dataclasses.field(default=None)
    def __post_init__(self):
        if self.loader_type is None:  # Check if the variable is missing
            self.loader_type = select_loader_type_from_prompt()  # Assign a generated value


    selected_experiment_name: Optional[str] = None
    """Name of the experiment to load. Use "unlabeled" to refer to
    experiments that do not have a name specified in the dat file."""
    selected_metadata_list: Optional[list[str]] = None
    """        
    List of projection metadata types that are allowed to be
    loaded, in prioritized order. The metadata types are strings
    extracted from the projection file names.
    """
    selected_sequences: Optional[list[int]] = None
    """
    List of sequence numbers to load in. Each sequence corresponds
    to a set of measurements taken sequentially over a 360 degree range.
    The sequence number of a projection comes from the dat file.
    """
    scan_start: Optional[int] = None
    "Lower bound of scans to include."
    scan_end: Optional[int] = None
    "Upper bound of scans to include."
    only_include_files_with: Optional[list[str]] = None
    "Only include files with these strings in the metadata string."
    exclude_files_with: Optional[list[str]] = None
    "Exclude files with any of these strings in the metadata string."

    def print_selections(self):
        if np.all([v is None for v in self.__dict__.values()]):
            print("No loading options provided.", flush=True)
        else:
            print("User-provided loading options:", flush=True)
            for k, v in self.__dict__.items():
                if v is not None:
                    print(f"  {k}: {v}", flush=True)