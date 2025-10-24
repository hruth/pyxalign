from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.pear.options import PEARLoadOptions
from pyxalign.io.loaders.pear.utils import load_experiment


def load_data_from_pear_format(
    options: PEARLoadOptions,
    n_processes: int = 1,
) -> StandardData:
    """
    Function for loading lamni-formatted projection data and returning
    it in the standardized format.
    """
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

    return standard_data