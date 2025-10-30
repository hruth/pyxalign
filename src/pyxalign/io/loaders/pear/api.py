from pyxalign.io.loaders.base import StandardData
import pyxalign.io.loaders.pear as pear
from pyxalign.io.loaders.pear.utils import load_experiment


def load_data_from_pear_format(
    options: pear.options.PEARLoadOptions,
    n_processes: int = 1,
) -> StandardData:
    """Function for loading ptychography reconstructions created by the PEAR
    Pty-Chi wrapper and returning it in the standardized format.

    Args:
        options (pear.options.PEARLoadOptions): Configuration options 
            for loading data.
        n_processes (int, optional): Number of processes to use for 
            loading. Defaults to 1.

    Returns:
        StandardData: The loaded data in a standardized format.
    """
    options.base.print_selections()
    # Load pear-formatted projection data
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

    return standard_data
