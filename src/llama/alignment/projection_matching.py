from typing import Optional
import numpy as np
import cupy as cp
import copy

from llama.alignment.base import Aligner
from llama.api.options.projections import ProjectionTransformOptions
from llama.api.options.transform import ShiftOptions

# from llama.projections import PhaseProjections
import llama.projections as projections
from llama.timing.timer_utils import timer, clear_timer_globals
from llama.transformations.classes import Shifter
import llama.image_processing as ip
import llama.api.maps as maps
from llama.api.enums import DeviceType, MemoryConfig
from llama.api.options.alignment import ProjectionMatchingOptions
import llama.gpu_utils as gutils
from llama.api.types import ArrayType, r_type
from llama.gpu_wrapper import device_handling_wrapper
from llama.timing.timer_utils import timer
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        # clear_timer_globals()
        # self.options.reconstruct.filter.device = self.options.device

    @gutils.memory_releasing_error_handler
    @timer()
    def run(self, initial_shift: Optional[np.ndarray] = None) -> np.ndarray:
        # Create the projections object
        projection_options = copy.deepcopy(self.projections.options)
        projection_options.experiment.pixel_size = self.projections.pixel_size
        # projection_options.input_processing.crop = self.options.crop
        # projection_options.input_processing.downsample = self.options.downsample
        projection_options.input_processing = ProjectionTransformOptions(
            downsample=self.options.downsample,
            crop=self.options.crop,
        )

        projection_options.reconstruct = self.options.reconstruct
        self.aligned_projections = projections.PhaseProjections(
            projections=self.projections.data,
            angles=self.projections.angles,
            options=projection_options,
            masks=self.projections.masks,
            center_of_rotation=self.projections.center_of_rotation,
        )
        if self.options.downsample.enabled:
            self.scale = self.options.downsample.scale
        else:
            self.scale = 1
        # Make sure the reference doesn't go back to the original arrays
        if self.aligned_projections.data is self.projections.data:
            self.aligned_projections.data = self.aligned_projections.data * 1
        if self.aligned_projections.masks is self.projections.masks:
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

        return shift

    @gutils.memory_releasing_error_handler
    @timer()
    def calculate_alignment_shift(self) -> ArrayType:
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
            filter_inputs=True, pinned_filtered_sinogram=self.pinned_filtered_sinogram
        )
        self.aligned_projections.laminogram.apply_circular_window(circulo)
        self.regularize_reconstruction()
        if self.iteration == self.options.iterations:
            return
        # Get forward projection
        self.aligned_projections.laminogram.get_forward_projection(self.pinned_forward_projection)
        self.plot_update()
        # Find optimal shift
        self.get_shift_update()

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

    @timer()
    def get_shift_update(self):
        wrapped_func = device_handling_wrapper(
            func=self.calculate_shift_update,
            options=self.options.device,
            chunkable_inputs_for_gpu_idx=[0, 1, 2],
            common_inputs_for_gpu_idx=[4],
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
        )
        self.post_process_shift()

        self.total_shift += self.shift_update

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

    @staticmethod
    def calculate_shift_update(
        sinogram: ArrayType,
        masks: ArrayType,
        forward_projection_model: ArrayType,
        high_pass_filter: ArrayType,
        mass: float,
    ) -> tuple[ArrayType, ArrayType, ArrayType]:
        xp = cp.get_array_module(sinogram)

        projections_residuals = forward_projection_model - sinogram

        # calculate error using unfiltered residual
        unfiltered_error = get_pm_error(projections_residuals, masks, mass)

        projections_residuals = ip.apply_1D_high_pass_filter(
            projections_residuals, 2, high_pass_filter
        )
        # projections_residuals = ip.apply_1D_high_pass_filter(
        #     projections_residuals, 2, high_pass_filter
        # )

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
        self.aligned_projections.masks[:] = tukey_window * self.aligned_projections.masks

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
            initializer_function = gutils.pin_memory
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
        tukey_window = ip.get_tukey_window(self.aligned_projections.size, A=0.2, xp=self.xp)
        # Generate circular mask for reconstruction
        circulo = self.aligned_projections.laminogram.get_circular_window()

        if self.memory_config == MemoryConfig.MIXED:
            tukey_window = gutils.pin_memory(tukey_window)

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
        if self.print_updates:
            print("Max step size update: " + "{:.4f}".format(max_shift_step_size) + " px")
        if max_shift_step_size < self.options.min_step_size and self.iteration > 0:
            print("Minimum step size reached, stopping loop...")
            return True
        else:
            return False

    @timer()
    def plot_update(self, proj_idx: int = 0):
        if self.options.update_plot.enabled and (
            self.iteration % self.options.update_plot.stride == 0
        ):
            slice_idx = int(self.aligned_projections.laminogram.data.shape[0] / 2)
            sort_idx = np.argsort(self.aligned_projections.angles)

            total_shift = self.total_shift[sort_idx]
            projection = self.aligned_projections.data[proj_idx]
            model_projection = self.aligned_projections.laminogram.forward_projections.data[
                proj_idx
            ]
            if self.options.keep_on_gpu:
                total_shift = total_shift.get()
                projection = projection.get()

            clear_output(wait=True)
            fig, ax = plt.subplots(2, 2, layout="compressed")
            plt.suptitle(f"Projection-matching alignment\nIteration {self.iteration}")
            plt.sca(ax[0, 0])
            plt.title("Alignment shift")
            plt.plot(total_shift)
            plt.grid()
            plt.xlim()
            plt.sca(ax[0, 1])
            plt.title("Middle slice of reconstruction")
            plt.imshow(self.aligned_projections.laminogram.data[slice_idx])
            plt.sca(ax[1, 0])
            plt.title(f"Input projection {proj_idx}")
            plt.imshow(projection)
            plt.sca(ax[1, 1])
            plt.title(f"Model projection {proj_idx}")
            plt.imshow(model_projection)
            plt.show()


@timer()
def get_pm_error(projections_residuals: ArrayType, masks: ArrayType, mass: float):
    xp = cp.get_array_module(projections_residuals)
    error = xp.sqrt(xp.mean((masks * projections_residuals) ** 2, axis=(1, 2))) / mass
    return error.astype(r_type)
