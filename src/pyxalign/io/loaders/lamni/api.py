from typing import Optional, Union

from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.lamni.base_loader import LamniLoader
from pyxalign.io.loaders.lamni.options import LamniLoadOptions
from pyxalign.io.loaders.lamni.utils import load_experiment


def load_data_from_lamni_format(
    dat_file_path: str,
    parent_projections_folder: str,
    n_processes: int = 1,
    options: Optional[LamniLoadOptions] = None,
    return_loader_object: bool = False,
) -> Union[StandardData, tuple[StandardData, LamniLoader]]:
    """
    Function for loading lamni-formatted projection data and returning
    it in the standardized format.
    """
    if options is None:
        options = LamniLoadOptions()
    options.print_selections()
    # Load lamni-formatted projection data
    lamni_loader = load_experiment(
        dat_file_path=dat_file_path,
        parent_projections_folder=parent_projections_folder,
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