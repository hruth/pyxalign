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
            loading. Defaults to 1. Use more processes to decrease the
            loading time.

    Returns:
        StandardData: The loaded data in a standardized format.

    Example:
        Load ptycho data taken at the 2-ID-E beamline. Suppose your
        results directory looks something like this::

            pear_results_dir/pear_results_dir
                ├── S0010
                │   ├── Ndp128_LSQML_c63_m0.5_gaussian_p15
                │   │   └── recon_Niter1000.h5
                │   └── Ndp256_LSQML_c120_m0.5_gaussian_p15
                │       └── recon_Niter2000.h5
                └── S0011
                    └── Ndp128_LSQML_c50_m0.5_gaussian_p15
                    │   └── recon_Niter1000.h5
                    └── Ndp256_LSQML_c118_m0.5_gaussian_p15
                        └── recon_Niter2000.h5

        To load PEAR data, you need to specify (1) the folder containing
        the results and (2) a file-pattern that specifies which data to
        load from the scan folder. 
        In this example, specify that we want to load the data with the
        suffix 'Ndp128'. Note the inclusion of a wildcard character *::

            # specify general pear options
            base_load_options = pyxalign.io.loaders.pear.BaseLoadOptions()
            base_load_options.parent_projections_folder = "/path/to/pear/results/dir/"
            base_load_options.file_pattern = "Ndp128_LSQML_c*_m0.5_gaussian_p15/recon_Niter1000.h5"


        Additional information (like the measurement angle of each scan)
        must also be loaded. For the 2-ID-E micrprobe measurements, the
        measurement angle is pulled from the mda files::

            # specify options specific to beamline 2-ID-E data collection
            microprobe_2ide_load_options = pyxalign.io.loaders.pear.Microprobe2IDELoadOptions()
            microprobe_2ide_load_options.base = base_load_options
            microprobe_2ide_load_options.mda_folder = "/path/to/mda/folder/"

        Finally, load the data::

            standard_data = pyxalign.io.loaders.pear.load_data_from_pear_format(
                microprobe_2ide_load_options
            )

        To interactively view the loaded data, run::

            gui = pyxalign.gui.launch_standard_data_viewer(standard_data)
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
