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
from pyxalign.io.load import load_task
from pyxalign.io.loaders.enums import LoaderType
from pyxalign import gpu_utils
from pyxalign.io.loaders.lamni.options import BaseLoadOptions, Beamline2IDELoadOptions
from pyxalign.test_utils_2 import CITestArgumentParser, CITestHelper
from pyxalign.api.options_utils import set_all_device_options
import pyxalign.io.loaders
from pyxalign.io.loaders.utils import convert_projection_dict_to_array


def run_full_test_xrf_ptycho_1(
    update_tester_results: bool = False,
    save_temp_files: bool = False,
    test_start_point: enums.TestStartPoints = enums.TestStartPoints.BEGINNING,
):

    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name=os.path.join("2ide", "2025-1_Lamni-6"),
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 750, 150)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)
    # define a downscaling value for when volumes are saved to prevent
    # saving files large files
    # define a downscaling value for when volumes are saved to prevent
    # saving files large files
    s = 16

    # Setup default gpu options
    n_gpus = cp.cuda.runtime.getDeviceCount()
    gpu_list = list(range(0, n_gpus))
    single_gpu_device_options = opts.DeviceOptions()
    multi_gpu_device_options = opts.DeviceOptions(
        gpu=opts.GPUOptions(
            n_gpus=n_gpus,
            gpu_indices=gpu_list,
            chunk_length=20,
        )
    )

    checkpoint_list = [enums.TestStartPoints.BEGINNING]
    if test_start_point in checkpoint_list:
        ### Load ptycho input data ###

        # Define options for loading ptycho reconstructions
        base_load_options = BaseLoadOptions(
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
        standard_data = pyxalign.io.loaders.load_data_from_lamni_format(
            parent_projections_folder=os.path.join(ci_test_helper.inputs_folder, "ptychi_recons"),
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
        ci_test_helper.save_or_compare_results(standard_data.probe, "standard_data_probe")
        ci_test_helper.save_or_compare_results(standard_data.scan_numbers, "standard_data_scan_numbers")
        ci_test_helper.save_or_compare_results(standard_data.angles, "standard_data_angles")
        ci_test_helper.save_or_compare_results(projection_array[10], "projection_array_10")
        ci_test_helper.save_or_compare_results(
            standard_data.probe_positions[scan_10], "probe_positions_10"
        )

        ci_test_helper.finish_test()


if __name__ == "__main__":
    ci_parser = CITestArgumentParser()
    args = ci_parser.parser.parse_args()
    run_full_test_xrf_ptycho_1(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
        test_start_point=args.start_point,
    )
