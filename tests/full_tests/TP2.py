import os
import multiprocessing as mp
import cupy as cp

import matplotlib.pyplot as plt
import pyxalign
from pyxalign import options as opts
from pyxalign.api import enums
from pyxalign.api.types import r_type
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.io.loaders.enums import LoaderType
from pyxalign import gpu_utils
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.utils import convert_projection_dict_to_array
from pyxalign.test_utils_2 import CITestHelper, CITestArgumentParser
from pyxalign.api.options_utils import set_all_device_options
from pyxalign.io.loaders.pear.options import LYNXLoadOptions, BaseLoadOptions


def run_full_test_TP2(
    update_tester_results: bool = False,
    save_temp_files: bool = False,
    test_start_point: enums.TestStartPoints = enums.TestStartPoints.BEGINNING,
):
    plt.ion()

    # Setup the test
    ci_options = opts.CITestOptions(
        test_data_name="TP2",
        update_tester_results=update_tester_results,
        proj_idx=list(range(0, 199, 45)),
        save_temp_files=save_temp_files,
    )
    ci_test_helper = CITestHelper(options=ci_options)
    # define a downscaling value for when volumes are saved to prevent
    # saving files large files
    s = 4

    # Setup default gpu options
    n_gpus = cp.cuda.runtime.getDeviceCount()
    gpu_list = list(range(0, n_gpus))
    multi_gpu_device_options = opts.DeviceOptions(
        gpu=opts.GPUOptions(
            n_gpus=n_gpus,
            gpu_indices=gpu_list,
            chunk_length=2,
        )
    )

    checkpoint_list = [enums.TestStartPoints.BEGINNING]
    if test_start_point in checkpoint_list:
        ### Load ptycho input data ###

        # Set experiment details
        lamino_angle = 61.3758  # laminography measurement angle
        sample_thickness = 7e-6  # thickness of the sample; determines number of pixels in the depth direction of the reconstruction
        rotation_angle = 72  # angle that ptycho reconstructions will be rotated by

        # Define paths to ptycho reconstructions and the tomography_scannumbers.txt file
        parent_folder = ci_test_helper.inputs_folder
        dat_file_path = os.path.join(
            parent_folder, "specES1", "dat-files", "tomography_scannumbers.txt"
        )

        # Define options for loading ptycho reconstructions
        base_load_options = BaseLoadOptions(
            parent_projections_folder=os.path.join(parent_folder, "ptycho_recon", "TP_2"),
            loader_type=LoaderType.FOLD_SLICE_V2,
            file_pattern=r"roi0_Ndp256/MLs_L1_p1_g50_bg0.1_vp5_vi_mm_MW10/Niter200.mat",
            select_all_by_default=True,
        )
        options = LYNXLoadOptions(
            dat_file_path=dat_file_path,
            base=base_load_options,
            selected_experiment_name="test_pattern_2",
            selected_sequences=[1, 2, 3, 4, 5, 6, 7],
        )

        # Load ptycho reconstructions, probe positions, measurement angles, and scan numbers
        lamni_data = load_data_from_pear_format(
            n_processes=int(mp.cpu_count() * 0.8),
            options=options,
        )

        # Convert projection dict to an array
        projection_array = convert_projection_dict_to_array(
            lamni_data.projections,
            delete_projection_dict=False,
            pad_with_mode=True,
        )

        # Check/save ci results for loading ptycho inputs
        scan_number_0 = lamni_data.scan_numbers[0]
        ci_test_helper.save_or_compare_results(lamni_data.probe, "lamni_data_probe")
        ci_test_helper.save_or_compare_results(lamni_data.scan_numbers, "lamni_data_scan_numbers")
        ci_test_helper.save_or_compare_results(lamni_data.angles, "lamni_data_angles")
        ci_test_helper.save_or_compare_results(projection_array[0], "projection_array_0")
        ci_test_helper.save_or_compare_results(
            lamni_data.probe_positions[scan_number_0], "probe_positions_0"
        )

        ### Create Projections object and LaminographyAlignmentTask object ###
        # Define projection options
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
                    enabled=False,
                ),
            ),
        )

        # Pin the projection array in order to speed up GPU calculations
        projection_array = gpu_utils.pin_memory(projection_array)

        complex_projections = ComplexProjections(
            projections=projection_array,
            angles=lamni_data.angles,
            scan_numbers=lamni_data.scan_numbers,
            options=projection_options,
            probe_positions=list(lamni_data.probe_positions.values()),
            probe=lamni_data.probe,
            skip_pre_processing=False,
        )
        del lamni_data

        task = LaminographyAlignmentTask(
            options=opts.AlignmentTaskOptions(),
            complex_projections=complex_projections,
        )

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
        width = 448
        task.options.cross_correlation = opts.CrossCorrelationOptions(
            iterations=10,
            binning=4,
            filter_position=9,
            filter_data=0.005,
            remove_slow_variation=True,
            crop=opts.CropOptions(enabled=True, horizontal_range=width, vertical_range=width),
            device=multi_gpu_device_options,
            use_end_corrections=False,
        )
        # Calculate cross-correlation shift
        task.get_cross_correlation_shift()
        # Apply the cross-correlation shift to projections
        task.complex_projections.apply_staged_shift()

        # Check/save results after applying cross-correlation shift
        ci_test_helper.save_or_compare_results(task, "cross_corr_aligned_task")

        ### Get projection masks ###
        # Calculate masks from probe positions data
        task.complex_projections.get_masks_from_probe_positions(0.1)

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

        # Delete arrays that are no longer needed to free up space
        task.complex_projections = None
        del pinned_data

        # Check/save results of unwrapping phase
        ci_test_helper.save_or_compare_results(task, "unwrapped_phase")
        task.phase_projections.show_center_of_rotation()
        task.phase_projections.pin_arrays()

        ### Estimate center of rotation ###
        # Define options for center estimation code -- start with coarse alignment
        estimate_center_options = opts.EstimateCenterOptions(
            horizontal_coordinate=opts.CoordinateSearchOptions(
                enabled=True,
                range=200,
                spacing=50,
            ),
            vertical_coordinate=opts.CoordinateSearchOptions(
                enabled=True,
                range=200,
                spacing=50,
            ),
        )
        # update pma options
        pma_options = estimate_center_options.projection_matching
        task.phase_projections.options.estimate_center.downsample.scale = 4
        task.phase_projections.options.estimate_center.downsample.use_gaussian_filter = True
        # update astra options to use all gpus (this makes it faster)
        astra_options = pma_options.reconstruct.astra
        astra_options.back_project_gpu_indices = gpu_list
        astra_options.forward_project_gpu_indices = gpu_list
        # update all device options in pma_options to be multi_gpu_device_options
        set_all_device_options(pma_options, multi_gpu_device_options)
        task.phase_projections.options.estimate_center = estimate_center_options

        # Run center estimation code
        center_estimate = task.phase_projections.estimate_center_of_rotation()
        task.phase_projections.center_of_rotation = center_estimate.optimal_center_of_rotation

        # Run center estimation code over a smaller area with higher precision
        estimate_center_options.vertical_coordinate.enabled = False
        estimate_center_options.horizontal_coordinate.range = 100
        estimate_center_options.horizontal_coordinate.spacing = 10
        center_estimate = task.phase_projections.estimate_center_of_rotation()
        task.phase_projections.center_of_rotation = center_estimate.optimal_center_of_rotation

        # Run center estimation code over a smaller area with even higher precision
        estimate_center_options.vertical_coordinate.enabled = False
        estimate_center_options.horizontal_coordinate.range = 20
        estimate_center_options.horizontal_coordinate.spacing = 1
        center_estimate = task.phase_projections.estimate_center_of_rotation()
        task.phase_projections.center_of_rotation = center_estimate.optimal_center_of_rotation

        # Check/save the estimated center of rotation value
        ci_test_helper.save_or_compare_results(
            task.phase_projections.center_of_rotation, "estimated_center_of_rotation"
        )

        ### Generate preliminary volume ###
        pinned_data = gpu_utils.create_empty_pinned_array(
            task.phase_projections.data.shape, dtype=r_type
        )
        set_all_device_options(task.phase_projections.options, multi_gpu_device_options)
        task.phase_projections.options.reconstruct.astra.back_project_gpu_indices = gpu_list
        task.phase_projections.options.reconstruct.astra.forward_project_gpu_indices = gpu_list
        task.phase_projections.options.reconstruct
        task.phase_projections.volume.generate_volume(True, pinned_data)
        task.phase_projections.volume.plot_data()
        del pinned_data

        # Check/save the preliminary volume
        ci_test_helper.save_or_compare_results(
            task.phase_projections.volume.data[::s, ::s, ::s], "pre_pma_volume"
        )

        # save the task before starting projection matching alignment
        ci_test_helper.save_checkpoint_task(task, file_name="pre_pma_task.h5")
    else:
        task = ci_test_helper.load_checkpoint_task(file_name="pre_pma_task.h5")

    checkpoint_list += [enums.TestStartPoints.PRE_PMA]
    if test_start_point in checkpoint_list:
        ### Projection-matching alignment ####
        # Define projection matching options
        pma_options = task.options.projection_matching
        pma_options.downsample = opts.DownsampleOptions(
            enabled=True, scale=32, use_gaussian_filter=True
        )
        pma_options.iterations = 300
        pma_options.high_pass_filter = 0.005
        pma_options.min_step_size = 0.01
        pma_options.step_relax = 0.1
        pma_options.reconstruct.astra.back_project_gpu_indices = gpu_list
        pma_options.reconstruct.astra.forward_project_gpu_indices = gpu_list
        pma_options.mask_shift_type = enums.ShiftType.FFT
        pma_options.keep_on_gpu = False
        set_all_device_options(pma_options, multi_gpu_device_options)

        # Run projection-matching alignment at successively higher resolutions
        scales = [32, 16, 8, 4, 2, 1]
        pma_shifts = {}
        for i, scale in enumerate(scales):
            pma_options.downsample.scale = scale
            if i == 0:
                task.get_projection_matching_shift()
            else:
                task.get_projection_matching_shift(initial_shift=pma_shifts[scales[i - 1]])
            pma_shifts[scale] = task.phase_projections.shift_manager.staged_shift * 1

            # Check/save the resulting alignment shifts at each resolution
            ci_test_helper.save_or_compare_results(pma_shifts[scale], f"pma_shift_{scale}x")

        # Shift the projections by the projection-matching alignment shift
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
        task.phase_projections.volume.generate_volume(True, pinned_data)
        del pinned_data
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
        task.phase_projections.volume.rotate_reconstruction()

        # save a tiff stack of the rotated reconstruction
        ci_test_helper.save_tiff(
            task.phase_projections.volume.data,
            "pma_aligned_rotated_volume.tiff",
            min=-0.022,
            max=0.025,
        )
        # Check/save the aligned and rotated volume
        ci_test_helper.save_or_compare_results(
            task.phase_projections.volume.data[::s, ::s, ::s],
            "pma_aligned_rotated_volume",
        )

        ci_test_helper.finish_test()


if __name__ == "__main__":
    ci_parser = CITestArgumentParser()
    args = ci_parser.parser.parse_args()
    run_full_test_TP2(
        update_tester_results=args.update_results,
        save_temp_files=args.save_temp_results,
        test_start_point=args.start_point,
    )
