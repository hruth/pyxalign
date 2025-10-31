# this file currently only tests the pear v3 loader
import os
import multiprocessing as mp
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.pear.options import LoaderType
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.xrf.api import load_data_from_xrf_format

import pyxalign.io.loaders.xrf.options as xrf_options
import pyxalign.io.loaders.pear.options as pear_options


ci_test_data_dir = os.environ["PYXALIGN_CI_TEST_DATA_DIR"]


def load_cSAXS_e18044_LamNI_201907_test_data(scan_start: int, scan_end: int) -> StandardData:
    parent_folder = os.path.join(ci_test_data_dir, "cSAXS_e18044_LamNI_201907", "inputs")
    # Define paths to ptycho reconstructions and the tomography_scannumbers.txt file
    dat_file_path = os.path.join(
        parent_folder, "specES1", "dat-files", "tomography_scannumbers.txt"
    )
    # parent_projection_folder = os.path.join(parent_folder, "analysis")

    # Define options for loading ptycho reconstructions
    base_load_options = pear_options.BaseLoadOptions(
        parent_projections_folder=os.path.join(parent_folder, "analysis"),
        loader_type=LoaderType.FOLD_SLICE_V1,
        file_pattern=r"*_512x512_b0_MLc_Niter500_recons.h5",
        scan_start=scan_start,
        scan_end=scan_end,
        select_all_by_default=True,
    )
    options = pear_options.LYNXLoadOptions(
        dat_file_path=dat_file_path,
        base=base_load_options,
        selected_sequences=[3, 4, 5],
    )

    # Load data
    lamni_data = load_data_from_pear_format(
        n_processes=int(mp.cpu_count() * 0.8),
        options=options,
    )

    return lamni_data


def load_2ide_xrf_test_data() -> tuple[dict[str, StandardData], dict]:
    base = xrf_options.XRFBaseLoadOptions(
        folder=os.path.join(ci_test_data_dir, "2ide", "2025-1_Lamni-4", "inputs")
    )
    xrf_load_options = xrf_options.XRF2IDELoadOptions(base=base)
    xrf_standard_data_dict, extra_PVs = load_data_from_xrf_format(xrf_load_options)
    return xrf_standard_data_dict, extra_PVs


def load_2ide_ptycho_test_data() -> StandardData:
    parent_folder = os.path.join(ci_test_data_dir, "2ide", "2025-1_Lamni-6", "inputs")
    # Define options for loading ptycho reconstructions
    base_load_options = pear_options.BaseLoadOptions(
        parent_projections_folder=os.path.join(parent_folder, "ptychi_recons"),
        loader_type=LoaderType.PEAR_V1,
        file_pattern="Ndp64_LSQML_c*_m0.5_gaussian_p10_mm_ic_pc*ul0.1/recon_Niter5000.h5",
        select_all_by_default=True,
        scan_start=115,
        scan_end=264,
    )
    options = pear_options.Microprobe2IDELoadOptions(
        mda_folder=os.path.join(parent_folder, "mda"),
        base=base_load_options,
    )

    # Load data
    standard_data = load_data_from_pear_format(
        n_processes=int(mp.cpu_count() * 0.8),
        options=options,
    )
    return standard_data
