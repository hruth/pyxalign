# this file currently only tests the pear v3 loader
import os
import argparse
import multiprocessing as mp
import cupy as cp

import matplotlib.pyplot as plt
import pyxalign
from pyxalign import options as opts
from pyxalign.api import enums
from pyxalign.api.types import r_type
from pyxalign.data_structures.task import load_task
from pyxalign.io.loaders.enums import LoaderType
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.pear.options import (
    BaseLoadOptions,
    Beamline2IDELoadOptions,
    LYNXLoadOptions,
)
from pyxalign.io.loaders.xrf.api import (
    convert_xrf_projection_dicts_to_arrays,
    load_data_from_xrf_format,
)
from pyxalign.io.loaders.xrf.options import XRFLoadOptions
from pyxalign.test_utils_2 import CITestArgumentParser, CITestHelper
from pyxalign.api.options_utils import set_all_device_options
from pyxalign.io.loaders.utils import convert_projection_dict_to_array


ci_filename_prefix = "load_test"


def run_old_lynx_load_test(update_tester_results: bool = False, save_temp_files: bool = False):
    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name="cSAXS_e18044_LamNI_201907",
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 750, 150)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    # Define paths to ptycho reconstructions and the tomography_scannumbers.txt file
    parent_folder = ci_test_helper.inputs_folder
    dat_file_path = os.path.join(
        parent_folder, "specES1", "dat-files", "tomography_scannumbers.txt"
    )
    # parent_projection_folder = os.path.join(parent_folder, "analysis")

    # Define options for loading ptycho reconstructions
    base_load_options = BaseLoadOptions(
        parent_projections_folder=os.path.join(parent_folder, "analysis"),
        loader_type=LoaderType.FOLD_SLICE_V1,
        file_pattern=r"*_512x512_b0_MLc_Niter500_recons.h5",
        scan_start=2714,
        scan_end=3465,
        select_all_by_default=True,
    )
    options = LYNXLoadOptions(
        dat_file_path=dat_file_path,
        base=base_load_options,
        selected_sequences=[3, 4, 5],
    )

    # Load data
    lamni_data = load_data_from_pear_format(
        n_processes=int(mp.cpu_count() * 0.8),
        options=options,
    )

    ci_test_helper.save_or_compare_results(
        lamni_data.projections[2730], "probe_positions_2730" + f"_{ci_filename_prefix}"
    )
    ci_test_helper.save_or_compare_results(
        lamni_data.probe, "lamni_data_probe" + f"_{ci_filename_prefix}"
    )
    ci_test_helper.save_or_compare_results(
        lamni_data.scan_numbers, "lamni_data_scan_numbers" + f"_{ci_filename_prefix}"
    )
    ci_test_helper.save_or_compare_results(
        lamni_data.angles, "lamni_data_angles" + f"_{ci_filename_prefix}"
    )
    ci_test_helper.save_or_compare_results(
        lamni_data.probe_positions[2730], "probe_positions_2730" + f"_{ci_filename_prefix}"
    )

    ci_test_helper.finish_test()


def run_2ide_ptycho_load_test(update_tester_results: bool = False, save_temp_files: bool = False):
    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name=os.path.join("2ide", "2025-1_Lamni-6"),
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 750, 150)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    # Define options for loading ptycho reconstructions
    base_load_options = BaseLoadOptions(
        parent_projections_folder=os.path.join(ci_test_helper.inputs_folder, "ptychi_recons"),
        loader_type=LoaderType.PEAR_V1,
        file_pattern="Ndp64_LSQML_c*_m0.5_gaussian_p10_mm_ic_pc*ul0.1/recon_Niter5000.h5",
        select_all_by_default=True,
        scan_start=115,
        scan_end=264,
    )
    options = Beamline2IDELoadOptions(
        mda_folder=os.path.join(ci_test_helper.inputs_folder, "mda"),
        base=base_load_options,
    )

    # Load data
    standard_data = load_data_from_pear_format(
        n_processes=int(mp.cpu_count() * 0.8),
        options=options,
    )

    w = 256
    new_shape = 576 + w, 832 + w
    projection_array = convert_projection_dict_to_array(
        standard_data.projections,
        delete_projection_dict=False,
        pad_with_mode=True,
        new_shape=new_shape,
    )

    scan_10 = standard_data.scan_numbers[10]
    ci_test_helper.save_or_compare_results(
        standard_data.probe, "standard_data_probe" + f"_{ci_filename_prefix}"
    )
    ci_test_helper.save_or_compare_results(
        standard_data.scan_numbers, "standard_data_scan_numbers" + f"_{ci_filename_prefix}"
    )
    ci_test_helper.save_or_compare_results(
        standard_data.angles, "standard_data_angles" + f"_{ci_filename_prefix}"
    )
    ci_test_helper.save_or_compare_results(
        projection_array[10], "projection_array_10" + f"_{ci_filename_prefix}"
    )
    ci_test_helper.save_or_compare_results(
        standard_data.probe_positions[scan_10], "probe_positions_10" + f"_{ci_filename_prefix}"
    )

    ci_test_helper.finish_test()


def run_2ide_xrf_load_test(update_tester_results: bool = False, save_temp_files: bool = False):
    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name=os.path.join("2ide", "2025-1_Lamni-4"),
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 750, 150)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    folder = ci_test_helper.inputs_folder
    xrf_load_options = XRFLoadOptions()
    xrf_standard_data_dict, extra_PVs = load_data_from_xrf_format(folder, xrf_load_options)

    for channel, standard_data in xrf_standard_data_dict.items():
        ci_test_helper.save_or_compare_results(
            standard_data.angles, f"standard_data_angles_{channel}" + f"_{ci_filename_prefix}"
        )
        ci_test_helper.save_or_compare_results(
            standard_data.scan_numbers,
            f"standard_data_scan_numbers_{channel}" + f"_{ci_filename_prefix}",
        )
        scan10 = standard_data.scan_numbers[10]
        ci_test_helper.save_or_compare_results(
            standard_data.projections[scan10],
            f"standard_data_projections10_{channel}" + f"_{ci_filename_prefix}",
        )

    # put data into dict of arrays
    xrf_array_dict = convert_xrf_projection_dicts_to_arrays(
        xrf_standard_data_dict,
        pad_with_mode=True,
    )

    for channel, projection_array in xrf_array_dict.items():
        ci_test_helper.save_or_compare_results(
            projection_array[:3], f"input_projections_{channel}" + f"_{ci_filename_prefix}"
        )

    ci_test_helper.finish_test()


if __name__ == "__main__":
    ci_parser = CITestArgumentParser()
    args = ci_parser.parser.parse_args()
    run_old_lynx_load_test(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
    )
    # run_2ide_ptycho_load_test(
    #     update_tester_results=args.update_results,
    #     save_temp_files=args.save_temp_results,
    # )
    # run_2ide_xrf_load_test(
    #     update_tester_results=args.update_results,
    #     save_temp_files=args.save_temp_results,
    # )
