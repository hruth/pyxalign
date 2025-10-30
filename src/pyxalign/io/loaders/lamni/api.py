from pyxalign.io.loaders.base import StandardData
import pyxalign.io.loaders.pear.options as pear_options
from pyxalign.io.loaders.pear.utils import load_experiment


def load_data_from_pear_format(
    options: pear_options.PEARLoadOptions,
    n_processes: int = 1,
) -> StandardData:
    """
    Function for loading PEAR formatted datasets projection data and returning
    it in the standardized format.

    Args:
        options: The first parameter.
        n_processes: Number of processes to use when loading 
            projections.

    Returns:
        StandardData: Loaded data returned in the `StandardData` format.
    """
    options.base.print_selections()
    # Load lamni-formatted projection data
    experiment_loader = load_experiment(
        parent_projections_folder=options.base.parent_projections_folder,
        n_processes=n_processes,
        options=options,
    )
    # Load data into standard format
    standard_data = StandardData(
        experiment_loader.projections,
        experiment_loader.angles,
        experiment_loader.scan_numbers,
        experiment_loader.selected_projection_file_paths,
        experiment_loader.probe_positions,
        experiment_loader.probe,
        experiment_loader.pixel_size,
    )

    return standard_data