import os
import argparse
import multiprocessing as mp
import cupy as cp

import matplotlib.pyplot as plt
import pyxalign
from pyxalign import options as opts
from pyxalign.api import enums
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign import gpu_utils
from pyxalign.io.loaders.utils import convert_projection_dict_to_array
from pyxalign.test_utils_2 import CITestArgumentParser, CITestHelper
from pyxalign.api.options_utils import set_all_device_options

import data_loaders
from conftest import register_processing_function


# Setup default gpu options
n_gpus = cp.cuda.runtime.getDeviceCount()
gpu_list = list(range(0, n_gpus))
multi_gpu_device_options = opts.DeviceOptions(
    gpu=opts.GPUOptions(
        n_gpus=n_gpus,
        gpu_indices=gpu_list,
        chunk_length=5,
    )
)
# define a downscaling value for when volumes are saved to prevent
# saving files large files
s = 16

@register_processing_function("cSAXS_e18044_LamNI_201907_pma")
def cSAXS_e18044_LamNI_201907_projection_matching_alignment(
    update_tester_results: bool = False,
    save_temp_files: bool = False,
    test_start_point: enums.TestStartPoints = enums.TestStartPoints.BEGINNING,
    show_gui: bool = False,
) -> dict[str, bool]:
    """
    Uses the task from cSAXS_e18044_LamNI_201907_pre_processing to do
    projection-matching alignment
    """
    ci_options = opts.CITestOptions(
        test_data_name="cSAXS_e18044_LamNI_201907",
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 750, 150)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    task = ci_test_helper.load_checkpoint_task(file_name="pre_pma_task.h5")
    ### Projection-matching alignment ####
    # Use a much smaller mask for alignment
    task.phase_projections.options.mask_from_positions.threshold = 15
    task.phase_projections.get_masks_from_probe_positions()
    task.phase_projections.pin_arrays()
    # Define projection matching options
    pma_options = task.options.projection_matching
    pma_options.downsample = opts.DownsampleOptions(
        enabled=True, scale=32, use_gaussian_filter=True
    )
    pma_options.iterations = 1000
    pma_options.high_pass_filter = 0.005
    pma_options.min_step_size = 0.01
    pma_options.step_relax = 0.1
    pma_options.reconstruct.astra.back_project_gpu_indices = gpu_list
    pma_options.reconstruct.astra.forward_project_gpu_indices = gpu_list
    pma_options.mask_shift_type = enums.ShiftType.FFT
    pma_options.keep_on_gpu = True
    pma_options.reconstruction_mask.enabled = True
    pma_options.momentum.enabled = True
    pma_options.interactive_viewer.update.enabled = show_gui
    pma_options.interactive_viewer.close_old_windows = True
    set_all_device_options(pma_options, multi_gpu_device_options)

    # define function for apropriately updating PMA options at each point
    def update_pma_options(pma_options: opts.ProjectionMatchingOptions, scale: int):
        pma_options.downsample.scale = scale
        pma_options.reconstruction_mask.radial_smooth = 5 * scale
        if scale >= 4:
            pma_options.keep_on_gpu = False
        else:
            pma_options.keep_on_gpu = False
        if scale == 1:
            pma_options.crop = opts.CropOptions(
                horizontal_range=1344,
                vertical_range=896,
                enabled=True,
            )
            pma_options.step_relax = 0.5

    # Run projection-matching alignment at successively higher resolutions
    scales = [32, 16, 8, 4, 2, 1]
    pma_shifts = {}
    for i, scale in enumerate(scales):
        update_pma_options(pma_options, scale)
        if i == 0:
            task.get_projection_matching_shift()
        else:
            task.get_projection_matching_shift(initial_shift=pma_shifts[scales[i - 1]])
        pma_shifts[scale] = task.phase_projections.shift_manager.staged_shift * 1

        # Check/save the resulting alignment shifts at each resolution
        ci_test_helper.save_or_compare_results(
            pma_shifts[scale], f"pma_shift_{scale}x_hpf_{pma_options.high_pass_filter}"
        )

    # Do one final alignment at increased high pass filter value
    pma_options.high_pass_filter = 0.01
    task.get_projection_matching_shift(initial_shift=pma_shifts[1])
    ci_test_helper.save_or_compare_results(
        pma_shifts[scale], f"pma_shift_{scale}x_hpf_{pma_options.high_pass_filter}"
    )

    # Shift the projections by the projection-matching alignment shift
    task.phase_projections.apply_staged_shift(multi_gpu_device_options)

    # Save the fully aligned task
    ci_test_helper.save_checkpoint_task(task, file_name="pma_aligned_task.h5")

    # Check/save the fully aligned task for ci testing (note: this only saves a few parts of
    # the task, as opposed to task.save_task which saves the entire task so you can reload
    # it later)
    ci_test_helper.save_or_compare_results(task, "pma_aligned_task")

    ### Generate aligned volumes ###
    task.phase_projections.volume.generate_volume(True)
    ci_test_helper.save_or_compare_results(
        task.phase_projections.volume.data[::s, ::s, ::s], "pma_aligned_volume"
    )
    ci_test_helper.save_tiff(
        task.phase_projections.volume.data,
        "pma_aligned_volume.tiff",
    )

    # Estimate optimal angles to rotate the volume by
    task.phase_projections.volume.get_optimal_rotation_of_reconstruction()
    ci_test_helper.save_or_compare_results(
        task.phase_projections.volume.optimal_rotation_angles,
        "tomogram_rotation_angles",
        atol=0.05,
        rtol=0.05,
    )

    # Rotate the volume
    task.phase_projections.volume.rotate_reconstruction(
        opts.DeviceOptions(gpu=opts.GPUOptions(chunk_length=2))
    )

    # save a tiff stack of the rotated reconstruction
    ci_test_helper.save_tiff(
        task.phase_projections.volume.data,
        "pma_aligned_rotated_volume.tiff",
    )
    # Check/save the aligned and rotated volume
    ci_test_helper.save_or_compare_results(
        task.phase_projections.volume.data[::s, ::s, ::s],
        "pma_aligned_rotated_volume",
    )

    ci_test_helper.finish_test()

    return ci_test_helper.test_result_dict


def test_single_result(test_name, result):
    """
    The conftest.pytest_generate_tests hook uses this to parameterize the
    results of the registered processing functions
    """
    assert result, f"Check '{test_name}' failed"


if __name__ == "__main__":
    ci_parser = CITestArgumentParser()
    args = ci_parser.parser.parse_args()
    cSAXS_e18044_LamNI_201907_projection_matching_alignment(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
        test_start_point=args.start_point,
        show_gui=args.show_gui,
    )
