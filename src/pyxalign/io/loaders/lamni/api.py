from typing import Optional, Union

from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.lamni.base_loader import BaseLoader
from pyxalign.io.loaders.lamni.options import LYNXLoadOptions
from pyxalign.io.loaders.lamni.utils import load_experiment


# This loads 2IDE data as well now, so I need to rename it somehow
def load_data_from_lamni_format(
    # parent_projections_folder: str,
    n_processes: int = 1,
    options: Optional[LYNXLoadOptions] = None, # type hint should be lynx AND 2ide
    return_loader_object: bool = False,
) -> Union[StandardData, tuple[StandardData, BaseLoader]]:
    """
    Function for loading lamni-formatted projection data and returning
    it in the standardized format.
    """
    if options is None:
        options = LYNXLoadOptions()
    options.base.print_selections()
    # Load lamni-formatted projection data
    lamni_loader = load_experiment(
        parent_projections_folder=options.base.parent_projections_folder,
        n_processes=n_processes,
        options=options,
    )
    # Load data into standard format
    standard_data = StandardData(
        lamni_loader.projections,
        lamni_loader.angles,
        lamni_loader.scan_numbers,
        lamni_loader.selected_projection_file_paths,
        lamni_loader.probe_positions,
        lamni_loader.probe,
        lamni_loader.pixel_size,
    )

    if return_loader_object:
        return standard_data, lamni_loader
    else:
        return standard_data