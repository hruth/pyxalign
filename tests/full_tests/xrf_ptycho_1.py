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
            selected_experiment_name="",
            selected_sequences=[0],
            selected_ptycho_strings=[
                "Ndp64_LSQML_c12556_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7687_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7867_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12474_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7484_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12461_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12507_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12479_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12459_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7449_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c8104_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7735_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12482_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7688_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12463_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7885_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7762_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12508_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12518_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12562_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12541_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12561_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7717_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7707_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12331_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12545_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12435_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12504_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12488_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12468_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12493_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7256_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7743_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12559_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12476_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12531_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7946_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12488_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7808_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12484_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12554_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12479_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12469_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7842_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12460_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7849_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12186_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12523_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12502_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12487_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12314_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12544_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12525_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7381_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7608_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12480_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7804_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7773_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12515_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7561_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7716_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c8035_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12561_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12512_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12490_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12527_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7684_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12442_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12533_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12562_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7933_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7828_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12471_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12528_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12505_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12467_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7908_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12553_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c8061_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12509_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12555_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7572_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12535_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7942_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12511_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12401_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12557_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12474_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7473_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12526_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c8003_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7870_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12056_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c8067_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c8082_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12540_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12524_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c8044_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12500_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12560_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12457_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12481_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12558_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12552_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7916_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12485_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12516_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c7865_m0.5_gaussian_p10_mm_ic_pc_g_ul0.1/recon_Niter5000",
                "Ndp64_LSQML_c12492_m0.5_gaussian_p10_mm_ic_pc_ul0.1/recon_Niter5000",
            ],
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
