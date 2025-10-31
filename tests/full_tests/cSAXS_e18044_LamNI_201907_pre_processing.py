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


@register_processing_function("cSAXS_e18044_LamNI_201907_pre_processing")
def cSAXS_e18044_LamNI_201907_pre_processing(
    update_tester_results: bool = False,
    save_temp_files: bool = False,
    test_start_point: enums.TestStartPoints = enums.TestStartPoints.BEGINNING,
    show_gui: bool = False,
) -> dict[str, bool]:
    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name="cSAXS_e18044_LamNI_201907",
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 750, 150)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)

    # if not projection_matching_only:
    checkpoint_list = [enums.TestStartPoints.BEGINNING]
    if test_start_point in checkpoint_list:
        ### Load ptycho input data ###

        # load data
        lamni_data = data_loaders.load_cSAXS_e18044_LamNI_201907_test_data(
            scan_start=2714, scan_end=3465
        )

        # Set experiment details
        lamino_angle = 61.108  # laminography measurement angle
        sample_thickness = 7e-6  # thickness of the sample; determines number of pixels in the depth direction of the reconstruction
        rotation_angle = 72.605  # angle that ptycho reconstructions will be rotated by
        shear_angle = -1.296

        new_shape = (2368, 1600)
        projection_array = convert_projection_dict_to_array(
            lamni_data.projections,
            pad_with_mode=True,
            new_shape=new_shape,
        )

        ci_test_helper.save_or_compare_results(lamni_data.probe, "lamni_data_probe")
        ci_test_helper.save_or_compare_results(lamni_data.scan_numbers, "lamni_data_scan_numbers")
        ci_test_helper.save_or_compare_results(lamni_data.angles, "lamni_data_angles")
        ci_test_helper.save_or_compare_results(projection_array[500], "projection_array_500")
        ci_test_helper.save_or_compare_results(
            lamni_data.probe_positions[2730], "probe_positions_2730"
        )

        # define projection options
        projection_options = opts.ProjectionOptions(
            experiment=opts.ExperimentOptions(
                laminography_angle=lamino_angle,
                sample_thickness=sample_thickness,
                pixel_size=lamni_data.pixel_size,
            ),
            input_processing=opts.ProjectionTransformOptions(
                rotation=opts.RotationOptions(
                    enabled=True,
                    angle=rotation_angle,
                ),
                shear=opts.ShearOptions(
                    enabled=True,
                    angle=shear_angle,
                ),
            ),
        )
        # Pin the projections to speed up GPU calculations
        projection_array = gpu_utils.pin_memory(projection_array)
        complex_projections = ComplexProjections(
            projections=projection_array,
            angles=lamni_data.angles,
            scan_numbers=lamni_data.scan_numbers,
            options=projection_options,
            probe_positions=list(lamni_data.probe_positions.values()),
            probe=lamni_data.probe,
            skip_pre_processing=False,
            file_paths=lamni_data.file_paths,
        )
        task = LaminographyAlignmentTask(
            options=opts.AlignmentTaskOptions(),
            complex_projections=complex_projections,
        )
        del lamni_data, projection_array

        # Check/save results after initial input processing
        ci_test_helper.save_or_compare_results(task, "input_processed_task")

        # save the task
        ci_test_helper.save_checkpoint_task(task, file_name="initial_task.h5")
    elif test_start_point == enums.TestStartPoints.INITIAL_TASK:
        # load task if initial_task is the selected test start point
        task = ci_test_helper.load_checkpoint_task(file_name="initial_task.h5")

    checkpoint_list += [enums.TestStartPoints.INITIAL_TASK]
    if test_start_point in checkpoint_list:
        ### Cross-correlation alignment ###
        width = 960
        task.options.cross_correlation = opts.CrossCorrelationOptions(
            iterations=10,
            binning=4,
            filter_position=101,
            filter_data=0.005,
            remove_slow_variation=True,
            device=multi_gpu_device_options,
            use_end_corrections=True,
            crop=opts.CropOptions(enabled=True, horizontal_range=width, vertical_range=width),
        )
        # Calculate cross-correlation shift
        task.get_cross_correlation_shift(plot_results=False)
        # Apply the cross-correlation shift to projections
        task.complex_projections.apply_staged_shift()

        # Check/save results after applying cross-correlation shift
        ci_test_helper.save_or_compare_results(task, "cross_corr_aligned_task")

        ### Get projection masks ###
        # # Use symmetric gaussian probe instead of measured probe
        # task.complex_projections.replace_probe_with_gaussian(amplitude=1, sigma=128)
        # Calculate masks from probe positions data
        task.complex_projections.options.mask_from_positions.threshold = 3
        task.complex_projections.get_masks_from_probe_positions()

        ### Unwrap phase ###
        # Unwrap phase and create phase_projections object
        # Pin data used in calculating phase projections
        task.complex_projections.options.phase_unwrap.device = multi_gpu_device_options
        task.get_unwrapped_phase()
        # Flip contrast on projections
        task.phase_projections.data[:] = -task.phase_projections.data

        # Delete arrays that are no longer needed to free up space
        task.complex_projections = None

        # Check/save results of unwrapping phase
        ci_test_helper.save_or_compare_results(task, "unwrapped_phase")
        task.phase_projections.pin_arrays()

        task.phase_projections.center_of_rotation[:] = [890, 1300]

        ### Generate preliminary volume ###
        set_all_device_options(task.phase_projections.options, multi_gpu_device_options)
        task.phase_projections.options.reconstruct.astra.back_project_gpu_indices = gpu_list
        task.phase_projections.options.reconstruct.astra.forward_project_gpu_indices = gpu_list
        task.phase_projections.options.reconstruct
        task.phase_projections.volume.generate_volume(True)

        # Check/save the preliminary volume
        ci_test_helper.save_or_compare_results(
            task.phase_projections.volume.data[::s, ::s, ::s], "pre_pma_volume"
        )

        # save the task before starting projection matching alignment
        ci_test_helper.save_checkpoint_task(task, file_name="pre_pma_task.h5")

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
    cSAXS_e18044_LamNI_201907_pre_processing(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
        test_start_point=args.start_point,
        show_gui=args.show_gui,
    )
