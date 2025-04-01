from functools import partial
import re
import traceback
from typing import Optional
import matplotlib
import numpy as np
import cupy as cp
import copy
from collections import defaultdict
from scipy.optimize import minimize

from llama.alignment.base import Aligner
from llama.api.options.projections import ProjectionTransformOptions
from llama.api.options.transform import ShiftOptions
from llama.api.options_utils import set_all_device_options
from llama.data_structures.laminogram import Laminogram

# from llama.projections import PhaseProjections
import llama.data_structures.projections as projections
from llama.timing.timer_utils import InlineTimer, timer, clear_timer_globals
from llama.transformations.classes import Shifter
import llama.image_processing as ip
import llama.api.maps as maps
from llama.api.enums import DeviceType, MemoryConfig
from llama.api.options.alignment import ProjectionMatchingOptions, ProjectionMatchingPlotOptions
import llama.gpu_utils as gutils
from llama.api.types import ArrayType, r_type
from llama.gpu_wrapper import device_handling_wrapper
from llama.timing.timer_utils import timer
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm
import astra

# To do:
# - add option for creating pinned arrays to speed up downsampling


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
        # self.plotter = ProjectionMatchingPlotter(self.options.plot)
        # clear_timer_globals()
        # self.options.reconstruct.filter.device = self.options.device

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

        projection_options.reconstruct = self.options.reconstruct
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
            self.initial_shift = np.zeros((self.aligned_projections.n_projections, 2), dtype=r_type)
        else:
            self.initial_shift = initial_shift

        # Run the PMA algorithm
        self.calculate_alignment_shift()

        if self.memory_config == MemoryConfig.GPU_ONLY:
            self.move_arrays_back_to_cpu()

        # Re-scale the shift
        shift = self.total_shift * self.scale

        # Clear astra objects
        self.aligned_projections.laminogram.clear_astra_objects()

        return shift

    @gutils.memory_releasing_error_handler
    @timer()
    def calculate_alignment_shift(self) -> ArrayType:
        # tukey_window, circulo = self.initialize_windows()
        unshifted_masks, unshifted_projections = self.initialize_arrays()
        self.initialize_attributes()
        self.initialize_shifters()
        tukey_window, circulo = self.initialize_windows()

        # for self.iteration in range(self.options.iterations):
        if self.print_updates:
            loop_over = tqdm(range(self.options.iterations), desc="projection matching loop")
        else:
            loop_over = range(self.options.iterations)
        for self.iteration in loop_over:
            # print("Iteration: ", str(self.iteration) + "/" + str(self.options.iterations))
            self.iterate(unshifted_projections, unshifted_masks, tukey_window, circulo)
            is_step_size_below_threshold = self.check_step_size_update()
            if is_step_size_below_threshold:
                break

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
        # Apply tukey window filter - convert to gpu chunked later
        self.apply_window_to_masks(tukey_window)
        # Get back projection:
        # - use method from projections class and add filtering
        # - later, add ability to re-use the same astra_config
        self.aligned_projections.get_3D_reconstruction(
            filter_inputs=True,
            pinned_filtered_sinogram=self.pinned_filtered_sinogram,
            reinitialize_astra=False,
        )
        if self.options.reconstruction_mask.enabled:
            self.aligned_projections.laminogram.apply_circular_window(circulo)
            # Store data for the forward projection
            astra.data3d.store(
                self.aligned_projections.laminogram.astra_config["ReconstructionDataId"],
                self.aligned_projections.laminogram.data,
            )
        self.regularize_reconstruction()
        if self.iteration == self.options.iterations:
            return
        # Get forward projection
        self.aligned_projections.laminogram.get_forward_projection(
            pinned_forward_projection=self.pinned_forward_projection,
            forward_projection_id=self.aligned_projections.laminogram.astra_config[
                "ProjectionDataId"
            ],
        )
        # Find optimal shift
        self.get_shift_update()
        self.plot_update()

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
        self.n_pix = self.aligned_projections.reconstructed_object_dimensions
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
            elif self.memory_config == MemoryConfig.GPU_ONLY:
                self.shift_update = cp.empty((n_proj, 2), dtype=r_type)  # type: ignore
                self.pinned_error = cp.empty((n_proj), dtype=r_type)  # type: ignore
                self.pinned_unfiltered_error = cp.empty((n_proj), dtype=r_type)  # type: ignore
        else:
            self.pinned_filtered_sinogram = None
            self.pinned_forward_projection = None
            self.shift_update = None
            self.pinned_error = None
            self.pinned_unfiltered_error = None

        shape = (self.options.iterations, *(self.pinned_error.shape))
        self.all_errors = np.zeros(shape=shape, dtype=self.pinned_error.dtype)
        self.all_unfiltered_errors = np.zeros(shape=shape, dtype=self.pinned_error.dtype)
        self.all_max_shift_step_size = np.array([], dtype=r_type)
        self.all_shifts = np.zeros(
            shape=(self.options.iterations, *(self.shift_update.shape)),
            dtype=self.shift_update.dtype,
        )
        self.velocity_map = np.zeros_like(self.shift_update)

    @timer()
    def get_shift_update(self):
        wrapped_func = device_handling_wrapper(
            func=self.calculate_shift_update,
            options=self.options.device,
            chunkable_inputs_for_gpu_idx=[0, 1, 2],
            common_inputs_for_gpu_idx=[4, 5],
            pinned_results=(
                self.shift_update,
                self.pinned_error,
                self.pinned_unfiltered_error,
            ),
        )
        # Later, you could update the gpu wrapper
        # to move arrays to the gpu in chunks if they
        # are on the cpu. For now, just move the whole
        # array.
        if self.memory_config == MemoryConfig.GPU_ONLY:
            forward_projection_input = cp.array(
                self.aligned_projections.laminogram.forward_projections.data
            )
        else:
            forward_projection_input = self.aligned_projections.laminogram.forward_projections.data

        inline_timer = InlineTimer("wrapped get_shift_update")
        inline_timer.start()
        (
            self.shift_update,
            self.pinned_error,
            self.pinned_unfiltered_error,
        ) = wrapped_func(
            self.aligned_projections.data,
            self.aligned_projections.masks,
            forward_projection_input,
            self.options.high_pass_filter,
            self.mass,
            self.secondary_mask,
            self.options.filter_directions,
        )
        inline_timer.end()
        self.post_process_shift()

        self.total_shift += self.shift_update
        if self.memory_config == MemoryConfig.GPU_ONLY:
            self.all_errors[self.iteration] = self.pinned_error.get()
            self.all_unfiltered_errors[self.iteration] = self.pinned_unfiltered_error.get()
            self.all_shifts[self.iteration] = self.total_shift.get()
        else:
            self.all_errors[self.iteration] = self.pinned_error * 1
            self.all_unfiltered_errors[self.iteration] = self.pinned_unfiltered_error * 1
            self.all_shifts[self.iteration] = self.total_shift * 1

    @timer()
    def post_process_shift(self):
        xp = self.xp

        # Reduce the shift on this increment by a relaxation factor
        max_allowed_step = self.options.max_step_size
        idx = xp.abs(self.shift_update) > max_allowed_step
        self.shift_update[idx] = max_allowed_step * xp.sign(self.shift_update[idx])
        self.shift_update *= self.options.step_relax

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
        if self.options.momentum.enabled:
            xp = cp.get_array_module(self.all_shifts)
            memory = self.options.momentum.memory
            if self.iteration >= memory:
                max_update = xp.quantile(xp.abs(self.shift_update), 0.995, axis=0)
                eligible_axes = np.where(max_update * self.scale < 0.5)[0]
                self.add_momentum_to_shift_update(axes=eligible_axes)

        # if par.momentum_acceleration
        #     momentum_memory = 2; 
        #     max_update = quantile(abs(shift_upd(valid_angles,:)), 0.995); 
        #     if ii > momentum_memory
        #         [shift_upd, shift_velocity]= add_momentum(shift_upd_all(ii-momentum_memory:ii,:,:), shift_velocity, max_update*par.binning < 0.5 );
        #         %shift_upd_all(ii,:,:) = reshape(shift_upd, [1,Nangles,2]);
        #     end
        # else
        #     if ii > 1
        #         verbose(1, 'Correlation between two last updates x:%.3f%% y:%.3f%%', 100*corr(squeeze(shift_upd_all(ii-1,:,1))', squeeze(shift_upd_all(ii,:,1))'), 100*corr(squeeze(shift_upd_all(ii-1,:,2))', squeeze(shift_upd_all(ii,:,2))'))
        #     end
        # end
    
    @staticmethod
    def fminsearch(C: ArrayType, initial_guess=[0]):
        # Define the objective function
        def objective(x, C):
            Nmem = len(C)  # Nmem should be the length of C
            # Create the array equivalent to MATLAB's [Nmem:-1:1]
            arr = np.arange(Nmem, 0, -1)
            # Ensure that the array length matches the number of elements in C
            return np.linalg.norm(C - np.exp(-x * arr))
        
        result = minimize(objective, initial_guess, args=(C,), method='Nelder-Mead')
        return result


    def add_momentum_to_shift_update(self, shift_memory: ArrayType, velocity_map: ArrayType, axes):
        xp = cp.get_array_module(self.all_shifts)
        idx_memory = range(self.iteration - self.options.momentum.memory, self.iteration)
        shift = self.all_shifts[idx_memory][-1]
        n_mem = self.options.momentum.memory - 1

        for ax in axes:
            if np.all(shift[:, ax] == 0):
                continue
            pearson_corr_coeffs = xp.zeros(n_mem, dtype=r_type)
            for i in range(n_mem):
                pearson_corr_coeffs[i] = xp.corrcoef(
                    shift[:, ax], self.all_shifts[idx_memory[i], :, ax]
                )[0, 1]
            
            # estimate friction
            decay = self.fminsearch(pearson_corr_coeffs.get(), [0])
            friction = np.max([0, self.options.momentum.alpha * decay])
            friction = np.min([1, friction])
            # update velocity map
            self.velocity_map[:, i] = (1 - friction) * self.velocity_map[:, i] + shift[:, i]
            # update shift estimates
            gain = self.options.momentum
            shift[:, i] = (1 - gain) * shift[:, i] + gain * self.velocity_map[:, i]

        # numerator = np.linalg.norm(shift[:, acc_axes])
        # denominator = np.linalg.norm(np.squeeze(shifts_memory[-1, :, acc_axes]))
        # acc = numerator / denominator

        # print(f"Momentum acceleration {acc:4.2f}x    friction {friction:4.2f}")


        # acc = math.norm2(shift(:,acc_axes)) ./ math.norm2(squeeze(shifts_memory(end,:,acc_axes))); 
        # utils.verbose(0,'Momentum acceleration %4.2fx    friction %4.2f', acc, friction )

    @staticmethod
    def calculate_shift_update(
        sinogram: ArrayType,
        masks: ArrayType,
        forward_projection_model: ArrayType,
        high_pass_filter: ArrayType,
        mass: float,
        secondary_mask: ArrayType,
        filter_directions: tuple[int] = (2,),
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
        del dX

        dY = ip.get_filtered_image_gradient(forward_projection_model, 1, high_pass_filter)
        dY = ip.apply_1D_high_pass_filter(dY, 1, high_pass_filter)
        y_shift = -(
            xp.sum(masks * dY * projections_residuals, axis=(1, 2))
            / xp.sum(masks * dY**2, axis=(1, 2))
        )
        del dY

        shift = xp.array([x_shift, y_shift]).transpose().astype(r_type)

        return shift, error, unfiltered_error

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
            pass

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
            initializer_function = lambda x: (x * 1)  # noqa: E731
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
        )
        mask_shift_options = ShiftOptions(
            enabled=True,
            type=self.options.mask_shift_type,
            device=device_options,
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
            circulo = self.aligned_projections.laminogram.get_circular_window(
                radial_smooth=self.options.reconstruction_mask.radial_smooth / self.scale,
                rad_apod=self.options.reconstruction_mask.rad_apod / self.scale,
            )
        else:
            circulo = None

        # Generate secondary masks
        if self.options.secondary_mask.enabled:
            self.secondary_mask = (
                self.aligned_projections.laminogram.generate_projection_masks_from_circulo(
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

    def check_step_size_update(self) -> bool:
        # Check if the step size update is small enough to stop the loop
        xp = self.xp
        max_shift_step_size = xp.max(
            xp.quantile(xp.abs(self.shift_update * self.scale), 0.995, axis=0)
        )
        if self.memory_config == MemoryConfig.GPU_ONLY:
            max_shift_step_size = max_shift_step_size.get()
        self.all_max_shift_step_size = np.append(self.all_max_shift_step_size, max_shift_step_size)

        if self.print_updates:
            print("Max step size update: " + "{:.4f}".format(max_shift_step_size) + " px")
        if max_shift_step_size < self.options.min_step_size and self.iteration > 0:
            print("Minimum step size reached, stopping loop...")
            return True
        else:
            return False

    @timer()
    def plot_update(self):
        if self.options.plot.update.enabled and (
            self.iteration % self.options.plot.update.stride == 0
        ):
            matplotlib.use("module://matplotlib_inline.backend_inline")
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
            self.aligned_projections.laminogram.plot_data(
                self.options.plot.reconstruction, show_plot=False
            )

            plt.sca(proj_axis)
            self.aligned_projections.plot_data(self.options.plot.projections, show_plot=False)
            plt.sca(forward_proj_axis)
            self.aligned_projections.laminogram.forward_projections.plot_data(
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


def get_pm_error(projections_residuals: ArrayType, masks: ArrayType, mass: float):
    xp = cp.get_array_module(projections_residuals)
    error = xp.sqrt(xp.mean((masks * projections_residuals) ** 2, axis=(1, 2))) / mass
    return error.astype(r_type)
