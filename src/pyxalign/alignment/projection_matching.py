from functools import partial
from typing import Callable, Optional
import numpy as np
import cupy as cp
import copy
from scipy.optimize import minimize
from PyQt5.QtWidgets import QApplication

from pyxalign.alignment.base import Aligner
from pyxalign.api.options.projections import ProjectionTransformOptions
from pyxalign.api.options.transform import ShiftOptions
from pyxalign.api.options_utils import set_all_device_options
import pyxalign.data_structures.projections as projections
from pyxalign.plotting.interactive.projection_matching import ProjectionMatchingViewer
from pyxalign.regularization import chambolleLocalTV3D
from pyxalign.style.text import text_colors
from pyxalign.timing.timer_utils import InlineTimer, timer
from pyxalign.transformations.classes import Shifter
import pyxalign.image_processing as ip
import pyxalign.api.maps as maps
from pyxalign.api.enums import DeviceType, MemoryConfig
from pyxalign.api.options.alignment import ProjectionMatchingOptions
import pyxalign.gpu_utils as gutils
from pyxalign.api.types import ArrayType, r_type
from pyxalign.gpu_wrapper import device_handling_wrapper
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm
import astra


class ProjectionMatchingAligner(Aligner):
    @timer()
    def __init__(
        self,
        projections: "projections.PhaseProjections",
        options: ProjectionMatchingOptions,
        print_updates: bool = True,
    ):
        super().__init__(projections, options)
        self.options: ProjectionMatchingOptions = self.options
        self.print_updates = print_updates
        self.gui: ProjectionMatchingViewer = None

    @gutils.memory_releasing_error_handler
    @timer()
    def run(self, initial_shift: Optional[np.ndarray] = None) -> np.ndarray:
        if self.options.keep_on_gpu:
            cp.cuda.Device(self.options.device.gpu.gpu_indices[0]).use()
        # Create the projections object
        projection_options = copy.deepcopy(self.projections.options)
        if self.options.keep_on_gpu:
            set_all_device_options(projection_options, copy.deepcopy(self.options.device))
        projection_options.experiment.pixel_size = self.projections.pixel_size
        projection_options.input_processing = ProjectionTransformOptions(
            downsample=self.options.downsample,
            crop=self.options.crop,
            mask_downsample_use_gaussian_filter=self.options.downsample.use_gaussian_filter,
        )

        # projection_options.reconstruct = self.options.reconstruct
        projection_options.reconstruct = copy.deepcopy(self.options.reconstruct)
        self.aligned_projections = projections.PhaseProjections(
            projections=self.projections.data,
            angles=self.projections.angles,
            scan_numbers=self.projections.scan_numbers,
            options=projection_options,
            masks=self.projections.masks,
            center_of_rotation=self.projections.center_of_rotation,
        )
        if self.options.downsample.enabled:
            self.scale = self.options.downsample.scale
        else:
            self.scale = 1

        # When cropping and not downsampling, you need to make a new copy
        # of the array. I don't fully understand why this is needed, but
        # this will break if you don't have it here.
        if self.options.crop.enabled and (self.scale == 1):
            self.aligned_projections.data = self.aligned_projections.data * 1
            self.aligned_projections.masks = self.aligned_projections.masks * 1

        if initial_shift is None:
            self.initial_shift = np.zeros(
                (self.aligned_projections.n_projections, 2),
                dtype=r_type)
        else:
            self.initial_shift = initial_shift

        # Run the PMA algorithm
        self.calculate_alignment_shift()
        self.update_GUI(force_update=True) # update GUI one last time

        if self.memory_config == MemoryConfig.GPU_ONLY:
            self.move_arrays_back_to_cpu()

        # Re-scale the shift
        shift = self.total_shift * self.scale

        # Clear astra objects
        self.aligned_projections.volume.clear_astra_objects()

        return shift

    @gutils.memory_releasing_error_handler
    @timer()
    def calculate_alignment_shift(self) -> ArrayType:
        unshifted_masks, unshifted_projections = self.initialize_arrays()
        self.initialize_attributes()
        self.initialize_shifters()
        tukey_window, circulo = self.initialize_windows()

        print(f"Starting projection-matching alignment downsampling = {self.scale}...")
        if self.print_updates:
            loop_prog_bar = tqdm(range(self.options.iterations), desc="projection matching loop")
        else:
            loop_prog_bar = range(self.options.iterations)
        for self.iteration in loop_prog_bar:
            self.iterate(unshifted_projections, unshifted_masks, tukey_window, circulo)
            max_shift_step_size = self.get_step_size_update()
            if self.print_updates:
                prog_bar_update_string = (
                    f"{text_colors.HEADER}Max step size update: {max_shift_step_size:.4f} px "
                    + f"{self.momentum_update_string}{text_colors.ENDC}"
                )
                loop_prog_bar.set_description(prog_bar_update_string)
            stopping_condition_met = self.check_stopping_condition(max_shift_step_size)
            if stopping_condition_met:
                break
            self.check_for_error()

    @timer()
    def iterate(
        self,
        unshifted_projections: ArrayType,
        unshifted_masks: ArrayType,
        tukey_window: ArrayType,
        circulo: ArrayType,
    ):
        "Execute an iteration of the projection-matching aligment loop"
        xp = self.xp

        self.apply_new_shift(unshifted_projections, unshifted_masks)
        self.apply_window_to_masks(tukey_window)
        if self.options.refine_geometry:
            # When using geometry refinement, the astra vectors need
            # to be updated on every iteration.
            update_geometries = True
        else:
            update_geometries = False
        self.aligned_projections.volume.generate_volume(
            filter_inputs=True,
            pinned_filtered_sinogram=self.pinned_filtered_sinogram,
            reinitialize_astra=False,
            n_pix=self.n_pix,
            update_geometries=update_geometries,
        )
        if self.options.reconstruction_mask.enabled:
            self.aligned_projections.volume.apply_circular_window(circulo)
            # Store data for the forward projection
            astra.data3d.store(
                self.aligned_projections.volume.astra_config["ReconstructionDataId"],
                self.aligned_projections.volume.data,
            )
        self.regularize_reconstruction()
        if self.iteration == self.options.iterations:
            return
        # Get forward projection
        self.aligned_projections.volume.get_forward_projection(
            pinned_forward_projection=self.pinned_forward_projection,
            forward_projection_id=self.aligned_projections.volume.astra_config[
                "ProjectionDataId"
            ],
        )
        # Find optimal shift
        self.get_shift_update()
        self.plot_update()
        self.update_GUI()
        self.get_geometry_update()

    @timer()
    def apply_new_shift(self, unshifted_projections, unshifted_masks):
        self.projection_shifter.run(
            images=unshifted_projections,
            shift=self.total_shift,
            pinned_results=self.aligned_projections.data,
        )
        self.mask_shifter.run(
            images=unshifted_masks,
            shift=self.total_shift,
            pinned_results=self.aligned_projections.masks,
            is_binary_mask=True,
        )

    @timer()
    def initialize_attributes(self):
        xp = cp.get_array_module(self.aligned_projections.data)
        self.n_pix = self.aligned_projections.reconstructed_object_dimensions * 1
        self.mass = xp.median(xp.abs(self.aligned_projections.data).mean(axis=(1, 2)))
        # Prepare pre-allocated and pinned arrays
        if not (self.memory_config == MemoryConfig.CPU_ONLY):
            self.pinned_filtered_sinogram = gutils.pin_memory(
                np.empty(self.aligned_projections.data.shape, dtype=r_type)
            )
            self.pinned_forward_projection = gutils.pin_memory(
                np.empty(self.aligned_projections.data.shape, dtype=r_type)
            )
            n_proj = self.aligned_projections.n_projections
            if self.memory_config == MemoryConfig.MIXED:
                self.shift_update = gutils.pin_memory(np.empty((n_proj, 2), dtype=r_type))
                self.pinned_error = gutils.pin_memory(np.empty((n_proj), dtype=r_type))
                self.pinned_unfiltered_error = gutils.pin_memory(np.empty((n_proj), dtype=r_type))
                self.pinned_lamino_angle_correction = gutils.pin_memory(
                    np.empty((n_proj), dtype=r_type)
                )
                self.pinned_tilt_angle_correction = gutils.pin_memory(
                    np.empty((n_proj), dtype=r_type)
                )
                self.pinned_skew_angle_correction = gutils.pin_memory(
                    np.empty((n_proj), dtype=r_type)
                )
            elif self.memory_config == MemoryConfig.GPU_ONLY:
                self.shift_update = cp.empty((n_proj, 2), dtype=r_type)  # type: ignore
                self.pinned_error = cp.empty((n_proj), dtype=r_type)  # type: ignore
                self.pinned_unfiltered_error = cp.empty((n_proj), dtype=r_type)  # type: ignore
                self.pinned_lamino_angle_correction = cp.zeros((n_proj), dtype=r_type)
                self.pinned_tilt_angle_correction = cp.zeros((n_proj), dtype=r_type)
                self.pinned_skew_angle_correction = cp.zeros((n_proj), dtype=r_type)
        else:
            self.pinned_filtered_sinogram = None
            self.pinned_forward_projection = None
            self.shift_update = None
            self.pinned_error = None
            self.pinned_unfiltered_error = None
            self.pinned_lamino_angle_correction = None
            self.pinned_tilt_angle_correction = None
            self.pinned_skew_angle_correction = None

        shape = (self.options.iterations, *(self.pinned_error.shape))
        self.all_errors = np.zeros(shape=shape, dtype=self.pinned_error.dtype)
        self.all_unfiltered_errors = np.zeros(shape=shape, dtype=self.pinned_error.dtype)
        self.all_max_shift_step_size = np.array([], dtype=r_type)
        self.all_shifts_before_momentum = np.zeros(
            shape=(self.options.iterations, *(self.shift_update.shape)),
            dtype=self.shift_update.dtype,
        )
        self.all_shift_updates = np.zeros(
            shape=(self.options.iterations, *(self.shift_update.shape)),
            dtype=self.shift_update.dtype,
        )
        self.velocity_map = np.zeros_like(self.shift_update)
        self.all_momentum_acceleration = np.zeros((0, 2), dtype=r_type)
        self.all_friction = np.zeros(0, dtype=r_type)
        self.momentum_update_string = ""
        self.all_lamino_angle_updates = np.zeros(0, dtype=r_type)
        self.all_tilt_angle_updates = np.zeros(0, dtype=r_type)
        self.all_skew_angle_updates = np.zeros(0, dtype=r_type)

    @timer()
    def get_shift_update(self, debug=False):
        # Later, you could update the gpu wrapper
        # to move arrays to the gpu in chunks if they
        # are on the cpu. For now, just move the whole
        # array.
        if self.memory_config == MemoryConfig.GPU_ONLY:
            forward_projection_input = cp.array(
                self.aligned_projections.volume.forward_projections.data
            )
        else:
            forward_projection_input = self.aligned_projections.volume.forward_projections.data
        # if self.memory_config == MemoryConfig.GPU_ONLY:
        #     self.aligned_projections.volume.forward_projections.data = cp.array(
        #         self.aligned_projections.volume.forward_projections.data
        #     )

        inline_timer = InlineTimer("wrapped get_shift_update")
        inline_timer.start()
        wrapped_shift_calc_func = self.return_wrapped_get_shift_update()
        (
            self.shift_update,
            self.pinned_error,
            self.pinned_unfiltered_error,
        ) = wrapped_shift_calc_func(
            self.aligned_projections.data,
            self.aligned_projections.masks,
            forward_projection_input,
            # self.aligned_projections.volume.forward_projections.data,
            self.options.high_pass_filter,
            self.mass,
            self.secondary_mask,
            self.options.filter_directions,
        )
        inline_timer.end()
        self.post_process_shift()

        if self.memory_config == MemoryConfig.GPU_ONLY:
            self.all_errors[self.iteration] = self.pinned_error.get()
            self.all_unfiltered_errors[self.iteration] = self.pinned_unfiltered_error.get()
            self.all_shift_updates[self.iteration] = self.shift_update.get()
        else:
            self.all_errors[self.iteration] = self.pinned_error * 1
            self.all_unfiltered_errors[self.iteration] = self.pinned_unfiltered_error * 1
            self.all_shift_updates[self.iteration] = self.shift_update * 1
        self.total_shift += self.shift_update

    @timer()
    def post_process_shift(self):
        xp = self.xp

        # Reduce the shift on this increment by a relaxation factor
        max_allowed_step = self.options.max_step_size
        idx = xp.abs(self.shift_update) > max_allowed_step
        self.shift_update[idx] = max_allowed_step * xp.sign(self.shift_update[idx])
        self.shift_update *= self.options.step_relax

        # apply momentum
        if self.memory_config == MemoryConfig.GPU_ONLY:
            self.all_shifts_before_momentum[self.iteration] = self.shift_update.get()
        else:
            self.all_shifts_before_momentum[self.iteration] = self.shift_update * 1
        self.get_shift_momentum()

        # Center the shifts around zero in the vertical direction
        self.shift_update[:, 1] = self.shift_update[:, 1] - xp.median(self.shift_update[:, 1])

        # Prevent outliers if the result begins quickly oscillating around the solution
        max_recorded_step = xp.quantile(xp.abs(self.shift_update), 0.99, axis=0)
        max_recorded_step[max_recorded_step > max_allowed_step] = max_allowed_step

        # Do not allow more than max_allowed_step px per iteration
        idx = xp.abs(self.shift_update) > max_allowed_step
        self.shift_update[idx] = xp.min(max_recorded_step) * xp.sign(self.shift_update[idx])

        # Remove degree of freedom in the vertical dimension (avoid drifts)
        angles = xp.array(self.aligned_projections.angles)
        orthbase = xp.array([xp.sin(angles * xp.pi / 180), xp.cos(angles * xp.pi / 180)])
        A = xp.matmul(orthbase, orthbase.transpose())
        B = xp.matmul(orthbase, self.shift_update[:, 0])
        coefs = xp.matmul(xp.linalg.inv(A), B[:, None])

        # Avoid object drifts within the reconstructed field of view
        self.shift_update[:, 0] = (
            self.shift_update[:, 0] - xp.matmul(orthbase.transpose(), coefs)[:, 0]
        )

    @timer()
    def get_shift_momentum(self):
        if not self.options.momentum.enabled:
            return

        xp = cp.get_array_module(self.all_shifts_before_momentum)
        memory = self.options.momentum.memory
        if self.iteration >= memory:
            max_update = xp.quantile(xp.abs(self.shift_update), 0.995, axis=0)
            eligible_axes = np.where(max_update * self.scale < 0.5)[0]
            if self.memory_config == MemoryConfig.GPU_ONLY:
                eligible_axes = eligible_axes.get()
            self.add_momentum_to_shift_update(axes=eligible_axes)

    @staticmethod
    def fminsearch(C: ArrayType, initial_guess=[0]):
        # Define the objective function
        def objective(x, C):
            n_mem = len(C)  # Nmem should be the length of C
            # Create the array equivalent to MATLAB's [Nmem:-1:1]
            arr = np.arange(n_mem, 0, -1)
            # Ensure that the array length matches the number of elements in C
            return np.linalg.norm(C - np.exp(-x * arr))

        result = minimize(objective, initial_guess, args=(C,), method="Nelder-Mead")
        return result

    def add_momentum_to_shift_update(self, axes: list[int]):
        xp = cp.get_array_module(self.all_shifts_before_momentum)
        idx_memory = range(self.iteration - self.options.momentum.memory, self.iteration + 1)
        shift = self.shift_update
        n_mem = self.options.momentum.memory

        if len(axes) == 0:
            return

        for ax in axes:
            if np.all(self.shift_update[:, ax] == 0):
                continue
            pearson_corr_coeffs = xp.zeros(n_mem, dtype=r_type)
            for i in range(n_mem):
                pearson_corr_coeffs[i] = xp.corrcoef(
                    shift[:, ax], self.all_shifts_before_momentum[idx_memory[i], :, ax]
                )[0, 1]

            # estimate friction
            decay = self.fminsearch(pearson_corr_coeffs, [0]).x[0]
            friction = np.max([0, self.options.momentum.alpha * decay])
            friction = np.min([1, friction])
            # update velocity map
            self.velocity_map[:, ax] = (1 - friction) * self.velocity_map[:, ax] + shift[:, ax]
            # update shift estimates
            gain = self.options.momentum.gain
            shift[:, ax] = (1 - gain) * shift[:, ax] + gain * self.velocity_map[:, ax]

        # reinforce the reference, in case you make edits later and accidentally
        # remove the reference
        self.shift_update = shift

        # Save quantities to print to user
        if self.memory_config == MemoryConfig.GPU_ONLY:
            shift = shift.get()
        momentum_acceleration = np.linalg.norm(shift, ord=2, axis=0) / np.linalg.norm(
            self.all_shifts_before_momentum[idx_memory[-1]], ord=2, axis=0
        )
        for i in range(2):
            self.all_momentum_acceleration = np.append(
                self.all_momentum_acceleration, momentum_acceleration[None], axis=0
            )
        self.all_friction = np.append(self.all_friction, friction)

        self.momentum_update_string = (
            f"{text_colors.OKCYAN}Momentum acceleration: {np.array2string(momentum_acceleration, precision=2, floatmode='fixed')} "
            + f"{text_colors.OKGREEN}Friction: {friction:.2f}{text_colors.ENDC}")

    def return_wrapped_get_shift_update(self, debug: bool = False) -> Callable:
        if debug:
            pinned_results = None
        else:
            pinned_results = (
                self.shift_update,
                self.pinned_error,
                self.pinned_unfiltered_error,
            )
        wrapped_func = device_handling_wrapper(
            func=self.calculate_shift_update,
            options=self.options.device,
            chunkable_inputs_for_gpu_idx=[0, 1, 2],
            common_inputs_for_gpu_idx=[4, 5],
            pinned_results=pinned_results,
        )
        return wrapped_func

    @staticmethod
    def calculate_shift_update(
        sinogram: ArrayType,
        masks: ArrayType,
        forward_projection_model: ArrayType,
        high_pass_filter: ArrayType,
        mass: float,
        secondary_mask: ArrayType,
        filter_directions: tuple[int] = (2,),
        debug: bool = False,
    ) -> tuple[ArrayType, ArrayType, ArrayType]:
        xp = cp.get_array_module(sinogram)
        masks = secondary_mask * masks

        projections_residuals = forward_projection_model - sinogram

        # calculate error using unfiltered residual
        unfiltered_error = get_pm_error(projections_residuals, masks, mass)

        for axis in filter_directions:
            projections_residuals = ip.apply_1D_high_pass_filter(
                projections_residuals, axis, high_pass_filter
            )

        # calculate error using filtered residual
        error = get_pm_error(projections_residuals, masks, mass)

        # Calculate the alignment shift
        dX = ip.get_filtered_image_gradient(forward_projection_model, 0, high_pass_filter)
        dX = ip.apply_1D_high_pass_filter(dX, 2, high_pass_filter)
        x_shift = -(
            xp.sum(masks * dX * projections_residuals, axis=(1, 2))
            / xp.sum(masks * dX**2, axis=(1, 2))
        )
        if not debug:
            del dX

        dY = ip.get_filtered_image_gradient(forward_projection_model, 1, high_pass_filter)
        dY = ip.apply_1D_high_pass_filter(dY, 1, high_pass_filter)
        y_shift = -(
            xp.sum(masks * dY * projections_residuals, axis=(1, 2))
            / xp.sum(masks * dY**2, axis=(1, 2))
        )
        if not debug:
            del dY

        shift = xp.array([x_shift, y_shift]).transpose().astype(r_type)
        if not debug:
            return shift, error, unfiltered_error
        else:
            return shift, dX, dY, projections_residuals

    @timer()
    def apply_window_to_masks(self, tukey_window: ArrayType):
        # self.aligned_projections.masks[:] = tukey_window * self.aligned_projections.masks

        def multiply_window_and_mask(masks, tukey_window):
            return tukey_window * masks

        apply_window_wrapped = device_handling_wrapper(
            func=multiply_window_and_mask,
            options=self.options.device,
            chunkable_inputs_for_gpu_idx=[0],
            common_inputs_for_gpu_idx=[1],
            pinned_results=self.aligned_projections.masks,
        )

        return apply_window_wrapped(self.aligned_projections.masks, tukey_window)

    @timer()
    def regularize_reconstruction(self):
        if self.options.regularization.enabled:
            self.aligned_projections.volume.data[:] = chambolleLocalTV3D(
                self.aligned_projections.volume.data,
                self.options.regularization.local_TV_lambda,
                self.options.regularization.iterations,
            )
            # Store the updated reconstruction
            astra.data3d.store(
                self.aligned_projections.volume.astra_config["ReconstructionDataId"],
                self.aligned_projections.volume.data,
            )

    @timer()
    def initialize_arrays(self):
        self.memory_config = maps.get_memory_config_enum(
            self.options.keep_on_gpu, self.options.device.device_type
        )

        if self.memory_config == MemoryConfig.GPU_ONLY:
            cp.cuda.Device(self.options.device.gpu.gpu_indices[0]).use()
            initializer_function = cp.array
            self.xp = cp
            self.scipy_module = gutils.get_scipy_module(cp.array(1))
        elif self.memory_config == MemoryConfig.MIXED:
            initializer_function = partial(gutils.pin_memory, force_repin=True)
            self.xp = np
            self.scipy_module = gutils.get_scipy_module(cp.array(1))
        elif self.memory_config == MemoryConfig.CPU_ONLY:
            def initializer_function(x): return (x * 1)  # noqa: E731
            self.xp = np
            self.scipy_module = gutils.get_scipy_module(np.array(1))

        unshifted_projections = initializer_function(self.aligned_projections.data)
        unshifted_masks = initializer_function(self.aligned_projections.masks)

        if not (self.memory_config == MemoryConfig.CPU_ONLY):
            self.aligned_projections.data = initializer_function(self.aligned_projections.data)
            self.aligned_projections.masks = initializer_function(self.aligned_projections.masks)

        self.total_shift = self.xp.zeros(
            (self.aligned_projections.angles.shape[0], 2), dtype=r_type
        ) + self.xp.array(self.initial_shift / self.scale, dtype=r_type)

        return unshifted_masks, unshifted_projections

    @timer()
    def initialize_shifters(self):
        device_options = copy.deepcopy(self.options.device)
        if self.memory_config == MemoryConfig.GPU_ONLY:
            device_options.device_type = DeviceType.GPU

        projections_shift_options = ShiftOptions(
            enabled=True,
            type=self.options.projection_shift_type,
            device=device_options,
            eliminate_wrapping=self.options.prevent_wrapping_from_shift,
        )
        mask_shift_options = ShiftOptions(
            enabled=True,
            type=self.options.mask_shift_type,
            device=device_options,
            eliminate_wrapping=self.options.prevent_wrapping_from_shift,
        )

        self.projection_shifter = Shifter(projections_shift_options)
        self.mask_shifter = Shifter(mask_shift_options)

    @timer()
    def initialize_windows(self) -> tuple[ArrayType, ArrayType]:
        # Generate window for removing edge issues
        tukey_window = ip.get_tukey_window(
            self.aligned_projections.size, A=self.options.tukey_shape_parameter, xp=self.xp
        )

        # Generate circular mask for reconstruction
        if self.options.reconstruction_mask.enabled:
            circulo = self.aligned_projections.volume.get_circular_window(
                radial_smooth=self.options.reconstruction_mask.radial_smooth / self.scale,
                rad_apod=self.options.reconstruction_mask.rad_apod / self.scale,
            )
        else:
            circulo = None

        # Generate secondary masks
        if self.options.secondary_mask.enabled:
            self.secondary_mask = (
                self.aligned_projections.volume.generate_projection_masks_from_circulo(
                    radial_smooth=self.options.secondary_mask.radial_smooth / self.scale,
                    rad_apod=self.options.secondary_mask.rad_apod / self.scale,
                )
            )
        else:
            self.secondary_mask = np.ones_like(self.aligned_projections.data[0])

        if self.memory_config == MemoryConfig.MIXED:
            tukey_window = gutils.pin_memory(tukey_window)
            self.secondary_mask = gutils.pin_memory(self.secondary_mask)
        elif self.memory_config == MemoryConfig.GPU_ONLY:
            self.secondary_mask = cp.array(self.secondary_mask)

        return tukey_window, circulo

    @timer()
    def move_arrays_back_to_cpu(self):
        self.total_shift = self.total_shift.get()
        self.aligned_projections.data = self.aligned_projections.data.get()
        self.aligned_projections.masks = self.aligned_projections.masks.get()
        self.shift_update = self.shift_update.get()
        self.pinned_error = self.pinned_error.get()
        self.pinned_unfiltered_error = self.pinned_unfiltered_error.get()
        self.pinned_lamino_angle_correction = self.pinned_lamino_angle_correction.get()
        self.pinned_tilt_angle_correction = self.pinned_tilt_angle_correction.get()
        self.pinned_skew_angle_correction = self.pinned_skew_angle_correction.get()
        self.secondary_mask = self.secondary_mask.get()

    def get_step_size_update(self) -> float:
        # Check if the step size update is small enough to stop the loop
        xp = self.xp
        max_shift_step_size = xp.max(
            xp.quantile(xp.abs(self.shift_update * self.scale), 0.995, axis=0)
        )
        if self.memory_config == MemoryConfig.GPU_ONLY:
            max_shift_step_size = max_shift_step_size.get()
        self.all_max_shift_step_size = np.append(self.all_max_shift_step_size, max_shift_step_size)
        return max_shift_step_size

    def check_stopping_condition(self, max_shift_step_size: float) -> bool:
        if (
            max_shift_step_size < self.options.min_step_size
            and self.iteration > 0
            and self.iteration >= (self.options.min_iterations - 1)
        ):
            print("Minimum step size reached, stopping loop...")
            return True
        else:
            return False

    def run_with_GUI(self, initial_shift: np.ndarray):
        "Launches the PMA viewer gui and runs the PMA loop"
        app = QApplication.instance() or QApplication([])
        # Objective: make this interactive
        # I could delay starting the thread until after the user hits a button.
        # However, this makes setting up multi-resolution runs difficult.
        self.gui = ProjectionMatchingViewer(self, multi_thread_func=self.run)
        self.gui.show()
        shift = self.gui.start_thread(initial_shift=initial_shift)
        self.gui.finish_test()
        # closing the window during destroys the QThread reference
        # app.exec_() # this makes it so the user has to close the window before moving on
        return shift

    @timer()
    def update_GUI(self, force_update: bool = False):
        "Update gui plots with data from the last PMA iteration"
        gui_enabled = self.options.interactive_viewer.update.enabled
        stride_condition_met = self.iteration % self.options.interactive_viewer.update.stride == 0
        if gui_enabled and (force_update or stride_condition_met):
            # Prevent PMA thread execution until the gui plot update has
            # finished. This is probably unecessary.
            # t0 = time.time()
            self.gui.mutex.lock()
            if self.iteration == 0:
                self.gui.signals.initialize_plots.emit()
            self.gui.signals.update_plots.emit()
            self.gui.wait_cond.wait(self.gui.mutex)
            self.gui.mutex.unlock()
            # print(f"unlocked, {time.time() - t0}")

    def check_for_error(self):
        if self.options.interactive_viewer.update.enabled and self.gui.force_stop:
            # self.gui.finish_test() # unecessary?
            raise Exception("User manually stopped execution")

    def show_GUI(self):
        "Relaunch the viewer. Intended to be used after running PMA and closing the original gui."
        if self.gui is None:
            app = QApplication.instance() or QApplication([])
            self.gui = ProjectionMatchingViewer(self)
            self.gui.show()
            self.gui.initialize_plots(add_stop_button=False)
            self.gui.update_plots()
        else:
            app = QApplication.instance() or QApplication([])
            self.gui.show()

    @timer()
    def plot_update(self):
        if self.options.plot.update.enabled and (
            self.iteration and self.options.plot.update.stride == 0
        ):
            # matplotlib.use("module://matplotlib_inline.backend_inline")
            sort_idx = np.argsort(self.aligned_projections.angles)
            sorted_angles = self.aligned_projections.angles[sort_idx]
            total_shift = self.total_shift[sort_idx]
            initial_shift = self.initial_shift[sort_idx]
            if self.options.keep_on_gpu:
                total_shift = total_shift.get()

            pixel_size = self.aligned_projections.pixel_size

            clear_output(wait=True)
            fig = plt.figure(layout="compressed", figsize=(10, 10))

            gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 2, 1, 1])
            total_shift_axis = fig.add_subplot(gs[0, 0])
            new_shift_axis = fig.add_subplot(gs[1, 0])
            rec_axis = fig.add_subplot(gs[0:2, 1])
            proj_axis = fig.add_subplot(gs[2, 0])
            forward_proj_axis = fig.add_subplot(gs[2, 1])
            error_axis = fig.add_subplot(gs[3:5, 0])
            error_vs_iter_axis = fig.add_subplot(gs[3, 1])
            unfiltered_error_vs_iter_axis = fig.add_subplot(gs[4, 1])

            plt.suptitle(f"Projection-matching alignment\nIteration {self.iteration}")

            plt.sca(total_shift_axis)
            plt.title("Alignment shift")
            plt.ylabel("Shift (px)")
            plt.xlabel("Angle (deg)")
            initial_shift_colors = ((0.75, 0.75, 1), (1, 0.75, 0.75))
            for i in range(2):
                plt.plot(
                    sorted_angles, initial_shift[:, i] / self.scale, color=initial_shift_colors[i]
                )
            plt.plot(sorted_angles, total_shift)
            plt.grid()
            plt.xlim([sorted_angles[0], sorted_angles[-1]])
            ylim = plt.ylim()

            twin_ax = plt.gca().twinx()
            plt.sca(twin_ax)
            plt.ylim([y * pixel_size * 1e6 for y in ylim])
            plt.ylabel(r"Shift ($\mu m$)")
            ax_color = "palevioletred"
            twin_ax.spines["right"].set_color(ax_color)
            twin_ax.yaxis.label.set_color(ax_color)
            twin_ax.tick_params(axis="y", colors=ax_color)

            plt.sca(new_shift_axis)
            plt.title("total_shift - initial_shift")
            plt.ylabel("Shift (px)")
            plt.xlabel("Angle (deg)")
            plt.plot(sorted_angles, total_shift - initial_shift / self.scale)
            plt.grid()
            plt.xlim([sorted_angles[0], sorted_angles[-1]])

            ylim = plt.ylim()
            twin_ax = plt.gca().twinx()
            plt.sca(twin_ax)
            plt.ylim([y * pixel_size * 1e6 for y in ylim])
            plt.ylabel(r"Shift ($\mu m$)")
            ax_color = "palevioletred"
            twin_ax.spines["right"].set_color(ax_color)
            twin_ax.yaxis.label.set_color(ax_color)
            twin_ax.tick_params(axis="y", colors=ax_color)

            plt.sca(rec_axis)
            self.aligned_projections.volume.plot_data(
                self.options.plot.reconstruction, show_plot=False
            )

            plt.sca(proj_axis)
            self.aligned_projections.plot_data(self.options.plot.projections, show_plot=False)
            plt.sca(forward_proj_axis)
            self.aligned_projections.volume.forward_projections.plot_data(
                self.options.plot.projections, title_string="Forward Projection", show_plot=False
            )

            plt.sca(error_axis)
            plt.title("Error")
            # plt.plot(
            #     sorted_angles,
            #     self.all_unfiltered_errors[self.iteration, sort_idx],
            #     label="Unfiltered",
            # )
            plt.plot(
                sorted_angles, self.all_errors[self.iteration, sort_idx], ".", label="Filtered"
            )
            plt.grid()
            plt.xlabel("Angle (deg)")
            plt.ylabel("Error")
            plt.legend()

            plt.sca(error_vs_iter_axis)
            plt.title("Error vs Iteration")
            plt.plot(self.all_errors[: self.iteration].mean(axis=1), label="Filtered")
            plt.grid()
            plt.xlabel("Iteration")
            plt.ylabel("Mean Error")
            plt.legend()

            plt.sca(unfiltered_error_vs_iter_axis)
            # plt.title("Error vs Iteration")
            # plt.plot(self.all_unfiltered_errors[: self.iteration].mean(axis=1), label="Unfiltered")
            plt.title("Max step size update")
            plt.plot(self.all_max_shift_step_size, label="update")
            plt.grid()
            plt.xlabel("Iteration")
            plt.ylabel("Update size")
            plt.legend()

            # fig.tight_layout()
            plt.show()

    def debug_get_shift_update(self) -> tuple:
        wrapped_shift_calc_func = self.return_wrapped_get_shift_update(debug=True)
        (shift, dX, dY, projections_residuals) = wrapped_shift_calc_func(
            self.aligned_projections.data,
            self.aligned_projections.masks,
            self.aligned_projections.volume.forward_projections.data,
            self.options.high_pass_filter,
            self.mass,
            self.secondary_mask,
            self.options.filter_directions,
            True,
        )
        return shift, dX, dY, projections_residuals

    def plot_shift_update_arrays_for_debugging(
        self,
        i: int,
        shift: np.ndarray,
        dX: np.ndarray,
        dY: np.ndarray,
        projections_residuals: np.ndarray,
        sort: bool = True,
        arrays_clim: Optional[np.ndarray] = None,
        gradient_clim: Optional[np.ndarray] = None,
        numerator_clim: Optional[np.ndarray] = None,
        denominator_clim: Optional[np.ndarray] = None,
        use_colorbars: bool = False,
    ):
        if sort:
            sort_idx = np.argsort(self.aligned_projections.angles)
        else:
            sort_idx = np.arange(0, len(dX), dtype=int)

        proj = self.aligned_projections.data[sort_idx[i]]
        forward_proj = self.aligned_projections.volume.forward_projections.data[sort_idx[i]]
        mask = self.aligned_projections.masks[sort_idx[i]] * self.secondary_mask

        fig, ax = plt.subplots(3, 4, layout="compressed", figsize=(15, 7.5))
        plt.suptitle(f"Arrays for {i}")

        # Plot projection, forward projection, and residual
        plt.sca(ax[0, 0])
        plt.title("Projection")
        plt.imshow(proj, cmap="bone")
        plt.clim(arrays_clim)
        plt.sca(ax[0, 1])
        plt.title("Forward Projection")
        plt.imshow(forward_proj, cmap="bone")
        plt.clim(arrays_clim)
        plt.sca(ax[0, 2])
        plt.title("Residual")
        plt.imshow(proj - forward_proj, cmap="bone")
        plt.clim(arrays_clim)
        plt.sca(ax[0, 3])
        plt.title("Masked and Filtered Residual")
        plt.imshow((projections_residuals[sort_idx[i]]) * mask, cmap="bone")
        plt.clim(arrays_clim)

        # Plot image gradients
        plt.sca(ax[1, 0])
        plt.title("dX")
        plt.imshow(dX[sort_idx[i]], cmap="bone")
        plt.clim(gradient_clim)
        plt.sca(ax[1, 1])
        plt.title("dY")
        plt.imshow(dY[sort_idx[i]], cmap="bone")
        plt.clim(gradient_clim)

        # Plot numerator of shift calculation
        plt.sca(ax[1, 2])
        plt.title(r"dX $\times$ residual $\times$ mask")
        plt.imshow(dX[sort_idx[i]] * projections_residuals[sort_idx[i]] * mask, cmap="bone")
        plt.clim(numerator_clim)
        plt.sca(ax[1, 3])
        plt.title(r"dY $\times$ residual $\times$ mask")
        plt.imshow(dY[sort_idx[i]] * projections_residuals[sort_idx[i]] * mask, cmap="bone")
        plt.clim(numerator_clim)

        # Plot denominator of shift calculation
        plt.sca(ax[2, 2])
        plt.title(r"dX$^2$ $\times$ mask")
        plt.imshow(mask * dX[sort_idx[i]] ** 2, cmap="bone")
        plt.clim(denominator_clim)
        plt.sca(ax[2, 3])
        plt.title(r"dY$^2$ $\times$ mask")
        plt.imshow(mask * dY[sort_idx[i]] ** 2, cmap="bone")
        plt.clim(denominator_clim)

        for axis in ax.ravel():
            axis.axis("off")
            plt.sca(axis)
            if use_colorbars:
                plt.colorbar()
        plt.show()

        print((self.total_shift * self.scale)[sort_idx[i]])
        fig, ax = plt.subplots(2, 1, layout="compressed")
        for axis in ax.ravel():
            plt.sca(axis)
            plt.axvline(i, color="gray", label=f"Index {i}")
            plt.grid(linestyle=":")
            plt.autoscale(enable=True, axis="x", tight=True)
            plt.ylabel("Shift (px)")

        plt.sca(ax[0])
        plt.title("Total Shift")
        plt.plot((self.total_shift * self.scale)[sort_idx])
        plt.sca(ax[1])
        plt.title("Shift Update")
        plt.plot(shift[sort_idx])
        plt.legend()
        plt.show()

    @timer()
    def get_geometry_update(self):
        if not self.options.refine_geometry.enabled:
            return

        xp = cp.get_array_module(self.aligned_projections.data)
        delta = 0.01
        deltas = [-delta, delta]
        # offset_forward_projections = []

        # Get sinogram model at +/- a delta on the laminography angle
        # Get 'infinitesmal' difference of model sinogram with respect to the sinogram angle
        if self.memory_config == MemoryConfig.GPU_ONLY:
            forward_projections = cp.array(
                self.aligned_projections.volume.forward_projections.data
            )
        else:
            forward_projections = self.aligned_projections.volume.forward_projections.data * 1
        laminography_angle = self.aligned_projections.options.experiment.laminography_angle
        for i in range(2):
            self.aligned_projections.options.experiment.laminography_angle = (
                laminography_angle + deltas[i]
            )
            # get volume at new laminography angle
            self.aligned_projections.volume.generate_volume(
                filter_inputs=True,
                pinned_filtered_sinogram=self.pinned_filtered_sinogram,
                reinitialize_astra=False,
                n_pix=self.n_pix,
                update_geometries=True,
            )
            # get forward projections at new laminography angle
            self.aligned_projections.volume.get_forward_projection(
                pinned_forward_projection=self.pinned_forward_projection,
                forward_projection_id=self.aligned_projections.volume.astra_config[
                    "ProjectionDataId"
                ],
            )
            if i == 0:
                d_proj = self.aligned_projections.volume.forward_projections.data * 1
            else:
                d_proj = (d_proj - self.aligned_projections.volume.forward_projections.data) / (
                    2 * delta
                )
        # Revert the laminography angle
        self.aligned_projections.options.experiment.laminography_angle = laminography_angle

        def get_gd_update(dX, residual, weights, filter):
            xp = cp.get_array_module(dX)
            dX = ip.apply_1D_high_pass_filter(dX, 2, filter)
            optimal_shift = xp.sum(weights * residual * dX, axis=(1, 2)) / xp.sum(
                weights * dX**2, axis=(1, 2)
            )
            return optimal_shift

        # add the thing for handling forward projection models here! they need to be
        # moved to gpu in the gpu_only case.
        if self.memory_config == MemoryConfig.GPU_ONLY:
            d_proj = cp.array(d_proj)

        # gpu chunked inputs: forward_projections, projections, d_proj
        def calculate_geometry_update(
            forward_projections: ArrayType,
            projections: ArrayType,
            d_proj: ArrayType,
            weights: ArrayType,
            high_pass_filter: float,
        ):
            xp = cp.get_array_module(projections)
            projections_residuals = forward_projections - projections
            projections_residuals = ip.apply_1D_high_pass_filter(
                projections_residuals, 2, high_pass_filter
            )
            dX = ip.get_filtered_image_gradient(forward_projections, 0, 0, use_filter=False)
            dY = ip.get_filtered_image_gradient(forward_projections, 1, 0, use_filter=False)
            # get laminography angle correction
            lamino_angle_correction = get_gd_update(
                d_proj, projections_residuals, weights, high_pass_filter
            )
            # get tilt angle correction
            d_vec = dX * xp.linspace(-1, 1, dX.shape[1], dtype=r_type)[:, None] - dY * xp.linspace(
                -1, 1, dY.shape[2], dtype=r_type
            )
            tilt_angle_correction = (
                get_gd_update(
                    d_vec, projections_residuals, weights, high_pass_filter) *
                180 / xp.pi)
            # get skew angle correction
            d_vec = dY * xp.linspace(-1, 1, dY.shape[2], dtype=r_type)
            skew_angle_correction = (
                get_gd_update(
                    d_vec, projections_residuals, weights, high_pass_filter) *
                180 / xp.pi)

            return lamino_angle_correction, tilt_angle_correction, skew_angle_correction

        calculate_geometry_update_wrapped = device_handling_wrapper(
            calculate_geometry_update,
            self.options.refine_geometry.device,
            chunkable_inputs_for_gpu_idx=[0, 1, 2, 3],
            # common_inputs_for_gpu_idx=[4],
            pinned_results=(
                self.pinned_lamino_angle_correction,
                self.pinned_tilt_angle_correction,
                self.pinned_skew_angle_correction,
            ),
        )

        inline_timer = InlineTimer("calculate_geometry_update_wrapped")
        inline_timer.start()
        (
            self.pinned_lamino_angle_correction,
            self.pinned_tilt_angle_correction,
            self.pinned_skew_angle_correction,
        ) = calculate_geometry_update_wrapped(
            forward_projections,
            self.aligned_projections.data,
            d_proj,
            self.aligned_projections.masks,
            self.options.high_pass_filter,
        )
        inline_timer.end()

        # Update angles
        def get_angle_update(angle_correction: ArrayType):
            angle_update = xp.median(xp.real(angle_correction))
            if self.memory_config == MemoryConfig.GPU_ONLY:
                angle_update = angle_update.get()
            angle_update = angle_update * self.options.refine_geometry.step_relax
            return angle_update

        self.aligned_projections.options.experiment.laminography_angle += get_angle_update(
            self.pinned_lamino_angle_correction
        )
        self.aligned_projections.options.reconstruct.geometry.tilt_angle += get_angle_update(
            self.pinned_tilt_angle_correction
        )
        self.aligned_projections.options.reconstruct.geometry.skew_angle += get_angle_update(
            self.pinned_skew_angle_correction
        )

        # Store updated results for plotting
        self.all_lamino_angle_updates = np.append(
            self.all_lamino_angle_updates,
            self.aligned_projections.options.experiment.laminography_angle,
        )
        self.all_tilt_angle_updates = np.append(
            self.all_tilt_angle_updates,
            self.aligned_projections.options.reconstruct.geometry.tilt_angle,
        )
        self.all_skew_angle_updates = np.append(
            self.all_skew_angle_updates,
            self.aligned_projections.options.reconstruct.geometry.skew_angle,
        )


def get_pm_error(projections_residuals: ArrayType, masks: ArrayType, mass: float):
    xp = cp.get_array_module(projections_residuals)
    error = xp.sqrt(xp.mean((masks * projections_residuals) ** 2, axis=(1, 2))) / mass
    return error.astype(r_type)
