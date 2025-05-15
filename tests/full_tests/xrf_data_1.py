import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import pyxalign
from pyxalign import options as opts
from pyxalign.api import enums
from pyxalign.api.types import r_type
from pyxalign.data_structures.xrf_task import XRFTask
from pyxalign.io.load import load_task
from pyxalign.io.loaders.enums import LoaderType
from pyxalign import gpu_utils
from pyxalign.io.loaders.xrf.api import (
    convert_xrf_projection_dicts_to_arrays,
    load_data_from_xrf_format,
)
from pyxalign.io.loaders.xrf.options import XRFLoadOptions
from pyxalign.test_utils_2 import CITestArgumentParser, CITestHelper
from pyxalign.api.options_utils import set_all_device_options
import pyxalign.io.loaders


def run_full_test_xrf_data_type_1(
    update_tester_results: bool = False,
    save_temp_files: bool = False,
    test_start_point: enums.TestStartPoints = enums.TestStartPoints.BEGINNING,
):
    # plt.ion()

    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name="xrf_data_1",
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 750, 150)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)
    # define a downscaling value for when volumes are saved to prevent
    # saving files large files
    s = 4

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

    # if not projection_matching_only:
    checkpoint_list = [enums.TestStartPoints.BEGINNING]
    if test_start_point in checkpoint_list:
        ### Load ptycho input data ###

        folder = ci_test_helper.inputs_folder
        xrf_load_options = XRFLoadOptions()
        xrf_standard_data_dict, extra_PVs = load_data_from_xrf_format(folder, xrf_load_options)
        scan_0 = list(extra_PVs.keys())[0]
        lamino_angle = float(extra_PVs[scan_0]["2xfm:m12.VAL"])

        # Load data
        xrf_array_dict = convert_xrf_projection_dicts_to_arrays(
            xrf_standard_data_dict,
            pad_with_mode=True,
        )

        for channel, projection_array in xrf_array_dict.items():
            ci_test_helper.save_or_compare_results(projection_array[:3], f"input_projections_{channel}")

        # Insert data into an XRFTask object
        primary_channel = "Ti"
        xrf_task = XRFTask(
            xrf_array_dict=xrf_array_dict,
            angles=xrf_standard_data_dict[primary_channel].angles,
            scan_numbers=xrf_standard_data_dict[primary_channel].scan_numbers,
            task_options=opts.AlignmentTaskOptions(),
            projection_options=opts.ProjectionOptions(
                experiment=opts.ExperimentOptions(laminography_angle=90 - lamino_angle),
            ),
            primary_channel=primary_channel,
        )

        # remove bad data
        xrf_task.drop_projections_from_all_channels(remove_idx=[143])

        # Update sample thickness and center of rotation
        xrf_task.projection_options.experiment.sample_thickness = 70
        xrf_task.center_of_rotation[1] = 130
        xrf_task.center_of_rotation[0] = 30

        # create preliminary reconstructions
        for channel, proj in xrf_task.projections_dict.items():
            proj.get_3D_reconstruction(True)
            # Check/save the preliminary volume
            ci_test_helper.save_or_compare_results(
                proj.volume.data[::s, ::s, ::s], f"pre_pma_volume_{channel}"
            )

        
        # create dummy mask
        xrf_task.projections_dict[xrf_task._primary_channel].masks = np.ones_like(
            xrf_task.projections_dict[xrf_task._primary_channel].data
        )
        xrf_task.projections_dict[xrf_task._primary_channel].pin_arrays()

        # Specify which element will be used for projection-matching alignment
        xrf_task._primary_channel = "Ti"

        pma_options = xrf_task.task_options.projection_matching
        pma_options.keep_on_gpu = True
        pma_options.high_pass_filter = 0.001
        pma_options.min_step_size = 0.005
        pma_options.iterations = 1000
        pma_options.step_relax = 0.1
        pma_options.plot.update.stride = 50
        pma_options.plot.update.enabled = True
        pma_options.downsample.scale = 2
        pma_options.downsample.enabled = True
        pma_options.downsample.use_gaussian_filter = True
        # pma_options.mask_shift_type = "linear"
        # pma_options.projection_shift_type = "linear"
        pma_options.mask_shift_type = "fft"
        pma_options.projection_shift_type = "fft"
        pma_options.momentum.enabled = True
        pma_options.interactive_viewer.update.enabled = True
        pma_options.interactive_viewer.update.stride = 50


        


if __name__ == "__main__":
    ci_parser = CITestArgumentParser()
    args = ci_parser.parser.parse_args()
    run_full_test_xrf_data_type_1(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
        test_start_point=args.start_point,
    )
