from typing import Union

from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.pear.base_loader import BaseLoader
from pyxalign.io.loaders.pear.options import PEARLoadOptions
from pyxalign.io.loaders.pear.utils import load_experiment


def load_data_from_pear_format(
    options: PEARLoadOptions,
    n_processes: int = 1,
    return_loader_object: bool = False,
) -> Union[StandardData, tuple[StandardData, BaseLoader]]:
    """
    Function for loading lamni-formatted projection data and returning
    it in the standardized format.
    """
    options.base.print_selections()
    # Load lamni-formatted projection data
    loader = load_experiment(
        parent_projections_folder=options.base.parent_projections_folder,
        n_processes=n_processes,
        options=options,
    )
    # Load data into standard format
    standard_data = StandardData(
        loader.projections,
        loader.angles,
        loader.scan_numbers,
        loader.selected_projection_file_paths,
        loader.probe_positions,
        loader.probe,
        loader.pixel_size,
    )

    if return_loader_object:  # for debugging purposes
        return standard_data, loader
    else:
        return standard_data
