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
from pyxalign.test_utils_2 import CITestArgumentParser, CITestHelper
from pyxalign.api.options_utils import set_all_device_options
import pyxalign.io.loaders


def run_full_test_cSAXS_e18044_LamNi_201907(
    update_tester_results: bool = False,
    save_temp_files: bool = False,
    test_start_point: enums.TestStartPoints = enums.TestStartPoints.BEGINNING,
):
    # plt.ion()

    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name="cSAXS_e18044_LamNI_201907",
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 750, 150)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)
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

    # if not projection_matching_only:
    checkpoint_list = [enums.TestStartPoints.BEGINNING]
    if test_start_point in checkpoint_list:
        ### Load ptycho input data ###

        # Set experiment details
        lamino_angle = 61.108  # laminography measurement angle
        sample_thickness = 7e-6  # thickness of the sample; determines number of pixels in the depth direction of the reconstruction
        rotation_angle = 72.605  # angle that ptycho reconstructions will be rotated by
        shear_angle = -1.296

        # Define paths to ptycho reconstructions and the tomography_scannumbers.txt file
        parent_folder = ci_test_helper.inputs_folder
        dat_file_path = os.path.join(parent_folder, "specES1", "dat-files", "tomography_scannumbers.txt")
        parent_projection_folder = os.path.join(parent_folder, "analysis")

        # Define options for loading ptycho reconstructions
        options = pyxalign.io.loaders.LamniLoadOptions(
            loader_type=LoaderType.LAMNI_V1,
            selected_experiment_name="unlabeled",
            selected_sequences=[3, 4, 5],
            selected_metadata_list=["512x512_b0_MLc_Niter500_recons"],
            scan_start=2714,
            scan_end=3465,
        )

        # Load data
        lamni_data = pyxalign.io.loaders.load_data_from_lamni_format(
            dat_file_path=dat_file_path,
            parent_projections_folder=parent_projection_folder,
            n_processes=int(mp.cpu_count() * 0.8),
            options=options,
        )

        new_shape = (2368, 1600)
        projection_array = pyxalign.io.loaders.utils.convert_projection_dict_to_array(
            lamni_data.projections,
            delete_projection_dict=False,
            pad_with_mode=True,
            new_shape=new_shape,
        )

        ci_test_helper.save_or_compare_results(lamni_data.probe, "lamni_data_probe")
        ci_test_helper.save_or_compare_results(lamni_data.scan_numbers, "lamni_data_scan_numbers")
        ci_test_helper.save_or_compare_results(lamni_data.angles, "lamni_data_angles")
        ci_test_helper.save_or_compare_results(projection_array[500], "projection_array_500")
        ci_test_helper.save_or_compare_results(lamni_data.probe_positions[2730], "probe_positions_2730")

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
        projection_array = pyxalign.gpu_utils.pin_memory(projection_array)
        complex_projections = pyxalign.ComplexProjections(
            projections=projection_array,
            angles=lamni_data.angles,
            scan_numbers=lamni_data.scan_numbers,
            options=projection_options,
            probe_positions=list(lamni_data.probe_positions.values()),
            probe=lamni_data.probe,
            skip_pre_processing=False,
        )
        task = pyxalign.LaminographyAlignmentTask(
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
        task.get_cross_correlation_shift()
        # Apply the cross-correlation shift to projections
        task.complex_projections.apply_staged_shift()

        # Check/save results after applying cross-correlation shift
        ci_test_helper.save_or_compare_results(task, "cross_corr_aligned_task")

        ### Get projection masks ###
        # Use symmetric gaussian probe instead of measured probe
        task.complex_projections.replace_probe_with_gaussian(amplitude=1, sigma=128)
        # Calculate masks from probe positions data
        task.complex_projections.get_masks_from_probe_positions(threshold=3)

        ### Unwrap phase ###
        # Unwrap phase and create phase_projections object
        task.complex_projections.options.phase_unwrap = opts.PhaseUnwrapOptions(
            device=multi_gpu_device_options,
            iterations=10,
            lsq_fit_ramp_removal=False,
        )
        # Pin data used in calculating phase projections
        pinned_data = gpu_utils.create_empty_pinned_array(
            task.complex_projections.data.shape, dtype=r_type
        )
        task.get_unwrapped_phase(pinned_data)
        # Flip contrast on projections
        task.phase_projections.data[:] = -task.phase_projections.data

        # Delete arrays that are no longer needed to free up space
        task.complex_projections = None
        del pinned_data

        # Check/save results of unwrapping phase
        ci_test_helper.save_or_compare_results(task, "unwrapped_phase")
        task.phase_projections.show_center_of_rotation()
        task.phase_projections.pin_arrays()

        task.phase_projections.center_of_rotation[:] = [890, 1300]

        ### Generate preliminary volume ###
        pinned_data = gpu_utils.create_empty_pinned_array(
            task.phase_projections.data.shape, dtype=r_type
        )
        set_all_device_options(task.phase_projections.options, multi_gpu_device_options)
        task.phase_projections.options.reconstruct.astra.back_project_gpu_indices = gpu_list
        task.phase_projections.options.reconstruct.astra.forward_project_gpu_indices = gpu_list
        task.phase_projections.options.reconstruct
        task.phase_projections.laminogram.generate_laminogram(True, pinned_data)
        task.phase_projections.laminogram.plot_data()
        del pinned_data

        # Check/save the preliminary volume
        ci_test_helper.save_or_compare_results(
            task.phase_projections.laminogram.data[::s, ::s, ::s], "pre_pma_volume"
        )

        # save the task before starting projection matching alignment
        ci_test_helper.save_checkpoint_task(task, file_name="pre_pma_task.h5")
    elif test_start_point == enums.TestStartPoints.PRE_PMA:
        task = ci_test_helper.load_checkpoint_task(file_name="pre_pma_task.h5")

    checkpoint_list += [enums.TestStartPoints.PRE_PMA]
    if test_start_point in checkpoint_list:
        ### Projection-matching alignment ####
        # Use a much smaller mask for alignment
        task.phase_projections.get_masks_from_probe_positions(15)
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
        pma_options.interactive_viewer.update.enabled = True
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
        print(multi_gpu_device_options)
        task.phase_projections.apply_staged_shift(multi_gpu_device_options)

        # Save the fully aligned task
        ci_test_helper.save_checkpoint_task(task, file_name="pma_aligned_task.h5")

        # Check/save the fully aligned task for ci testing (note: this only saves a few parts of
        # the task, as opposed to task.save_task which saves the entire task so you can reload
        # it later)
        ci_test_helper.save_or_compare_results(task, "pma_aligned_task")

        ### Generate aligned volumes ###
        pinned_data = gpu_utils.create_empty_pinned_array(
            task.phase_projections.data.shape, dtype=r_type
        )
        task.phase_projections.laminogram.generate_laminogram(True, pinned_data)
        del pinned_data
        ci_test_helper.save_or_compare_results(
            task.phase_projections.laminogram.data[::s, ::s, ::s], "pma_aligned_volume"
        )
        ci_test_helper.save_tiff(
            task.phase_projections.laminogram.data,
            "pma_aligned_volume.tiff",
        )

        # Estimate optimal angles to rotate the volume by
        task.phase_projections.laminogram.get_optimal_rotation_of_reconstruction()
        ci_test_helper.save_or_compare_results(
            task.phase_projections.laminogram.optimal_rotation_angles,
            "tomogram_rotation_angles",
            atol=0.05,
            rtol=0.05,
        )

        # Rotate the volume
        task.phase_projections.laminogram.rotate_reconstruction()

        # save a tiff stack of the rotated reconstruction
        ci_test_helper.save_tiff(
            task.phase_projections.laminogram.data,
            "pma_aligned_rotated_volume.tiff",
        )
        # Check/save the aligned and rotated volume
        ci_test_helper.save_or_compare_results(
            task.phase_projections.laminogram.data[::s, ::s, ::s],
            "pma_aligned_rotated_volume",
        )

        ci_test_helper.finish_test()

if __name__ == "__main__":
    ci_parser = CITestArgumentParser()
    args = ci_parser.parser.parse_args()
    run_full_test_cSAXS_e18044_LamNi_201907(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
        test_start_point=args.start_point,
    )
