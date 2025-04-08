from typing import List, Optional, Union
import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
from llama.api.constants import divisor
from llama.api.options.plotting import PlotDataOptions, ImageSliderPlotOptions
from llama.estimate_center import (
    estimate_center_of_rotation,
    CenterOfRotationEstimateResults,
    format_coordinate_options,
    plot_coordinate_search_points,
)
from llama.api import enums
from llama.api.options.alignment import AlignmentOptions
from llama.api.options.device import DeviceOptions

from llama.api.options.projections import (
    EstimateCenterOptions,
    ProjectionOptions,
    ProjectionTransformOptions,
)
from llama.api.options.transform import (
    CropOptions,
    PadOptions,
    RotationOptions,
    ShearOptions,
    ShiftOptions,
    UpsampleOptions,
    DownsampleOptions,
)
import llama.gpu_utils as gpu_utils
from llama.gpu_wrapper import device_handling_wrapper
from llama.data_structures.laminogram import Laminogram
from llama.io.save import save_generic_data_structure_to_h5

from llama.mask import IlluminationMapMaskBuilder, estimate_reliability_region_mask, blur_masks

# import llama.plotting.plotters as plotters
from llama.plotting.plotters import make_image_slider_plot, plot_slice_of_3D_array, ImagePlotObject
from llama.timing.timer_utils import timer, clear_timer_globals
from llama.transformations.classes import (
    Downsampler,
    Rotator,
    Shearer,
    Shifter,
    Upsampler,
    Cropper,
    Padder,
)
from llama.transformations.functions import (
    image_shift_fft,
    rotate_positions,
    shear_positions,
    will_rotation_flip_aspect_ratio,
)
from llama.transformations.helpers import is_array_real
from llama.unwrap import unwrap_phase
from llama.data_structures.positions import ProbePositions
from llama.api.types import ArrayType, r_type
from llama.transformations.helpers import round_to_divisor


class TransformTracker:
    def __init__(
        self,
        rotation: r_type = 0,
        shear: r_type = 0,
        crop=None,
        downsample: int = 1,
    ):
        self.rotation = rotation
        self.shear = shear
        self.crop = crop
        self.scale = downsample

    def update_rotation(self, angle: r_type):
        self.rotation += angle

    def update_shear(self, angle: r_type):
        self.shear += angle

    def update_crop(self, crop_options: CropOptions):
        pass

    def update_downsample(self, scale: int):
        self.scale *= scale


class Projections:
    @timer()
    def __init__(
        self,
        *,
        projections: np.ndarray,
        angles: np.ndarray,
        options: ProjectionOptions,
        scan_numbers: Optional[np.ndarray] = None,
        probe_positions: Optional[list[np.ndarray]] = None,
        center_of_rotation: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        probe: Optional[np.ndarray] = None,
        skip_pre_processing: bool = False,
        add_center_offset_to_positions: bool = True,
        shift_manager: Optional["ShiftManager"] = None,
        transform_tracker: Optional[TransformTracker] = None,
        pin_arrays: bool = False,
    ):
        self.options = options
        self.angles = angles
        self.data = projections
        self.masks = masks
        if pin_arrays:
            self.pin_arrays()
        self.probe = probe
        if scan_numbers is not None:
            self.scan_numbers = scan_numbers
        else:
            self.scan_numbers = np.arange(0, len(projections), dtype=int)
        if probe_positions is not None:
            if add_center_offset_to_positions:
                center_pixel = np.array(self.data.shape[1:]) / 2
            else:
                center_pixel = np.array([0, 0])
            self.probe_positions = ProbePositions(
                positions=probe_positions, center_pixel=center_pixel
            )
        else:
            self.probe_positions = None
        if center_of_rotation is None:
            self.center_of_rotation = np.array(self.data.shape[1:], dtype=r_type) / 2
        else:
            self.center_of_rotation = np.array(center_of_rotation, dtype=r_type)
        # Initialize tracker for recording all transformations that have
        # been applied to the projections array
        if transform_tracker is None:
            self.transform_tracker = TransformTracker()
        else:
            self.transform_tracker = transform_tracker
        # self.pixel_size = self.options.experiment.pixel_size * self.transform_tracker.downsample
        # Apply input processing tasks (i.e. rotation, shear, downsampling, etc)
        # to projections array
        if not skip_pre_processing:
            self.transform_projections(self.options.input_processing)
        # Initialize manager for storing and applying shifts to the projections
        if shift_manager is not None:
            self.shift_manager = copy.deepcopy(shift_manager)
        else:
            self.shift_manager = ShiftManager(self.n_projections)

        # Initialize other quantities
        self.mask_builder: IlluminationMapMaskBuilder = None
        self.dropped_scan_numbers = []
        # Run initialization code specific to the projection type (i.e. PhaseProjections
        # or complex projections)
        self._post_init()

    @property
    def pixel_size(self):
        return self.options.experiment.pixel_size * self.transform_tracker.scale

    @timer()
    def transform_projections(
        self,
        options: ProjectionTransformOptions,
    ):
        # Transform the projections in the proper order
        if options.crop is not None:
            self.crop_projections(options.crop)
        if options.downsample is not None:
            self.downsample_projections(
                options.downsample,
                options.mask_downsample_type,
                options.mask_downsample_use_gaussian_filter,
            )
        if options.rotation is not None:
            self.rotate_projections(options.rotation)
        if options.shear is not None:
            self.shear_projections(options.shear)

    @timer()
    def crop_projections(self, options: CropOptions):
        if options.enabled:
            pre_crop_dims = np.array(self.data.shape[1:])
            self.data = Cropper(options).run(self.data)
            if self.masks is not None:
                self.masks = Cropper(options).run(self.masks)
            self.transform_tracker.update_crop(options)
            # To do: update this to work properly when the
            # center is shifted too!
            new_dims = np.array(self.data.shape[1:])
            shift = (new_dims - pre_crop_dims) / 2 - np.array(
                [options.vertical_offset, options.horizontal_offset]
            )
            self.center_of_rotation = self.center_of_rotation + shift
            # Shift position accordingly
            if self.probe_positions is not None:
                self.probe_positions.shift_positions(
                    shift=shift[::-1][None].repeat(self.n_projections, axis=0)
                )

    @timer()
    def pad_projections(self, options: PadOptions):
        if options.enabled:
            pre_crop_dims = np.array(self.data.shape[1:])
            self.data = Padder(options).run(self.data)
            if self.masks is not None:
                mask_options = copy.deepcopy(options)
                mask_options.pad_value = 0
                self.masks = Padder(mask_options).run(self.masks)
            # Update center of rotation and probe positions
            new_dims = np.array(self.data.shape[1:])
            shift = (new_dims - pre_crop_dims) / 2
            self.center_of_rotation = self.center_of_rotation + shift
            if self.probe_positions is not None:
                self.probe_positions.shift_positions(
                    shift=shift[::-1][None].repeat(self.n_projections, axis=0)
                )

    @timer()
    def downsample_projections(
        self,
        options: DownsampleOptions,
        mask_downsample_type: enums.DownsampleType,
        mask_downsample_use_gaussian_filter: bool,
    ):
        if options.enabled:
            self.data = Downsampler(options).run(self.data)
            if self.masks is not None:
                mask_options = copy.deepcopy(options)
                mask_options.type = mask_downsample_type
                mask_options.use_gaussian_filter = mask_downsample_use_gaussian_filter
                self.masks = Downsampler(mask_options).run(self.masks)
            # Update center of rotation
            self.center_of_rotation = self.center_of_rotation / options.scale
            # Update probe positions
            if self.probe_positions is not None:
                self.probe_positions.rescale_positions(options.scale)
            # Downsample probe
            if self.probe is not None:
                self.probe = self.probe[:: options.scale, :: options.scale]
            self.transform_tracker.update_downsample(options.scale)

    @timer()
    def rotate_projections(
        self, options: RotationOptions, apply_to_center_of_rotation: bool = True
    ):
        # Note: the fourier rotation that is used for masks will likely
        # be an issue. A new method needs to be implemented.
        if options.enabled:
            center_pixel = np.array(self.data.shape[1:]) / 2
            data_aspect_ratio_changes = will_rotation_flip_aspect_ratio(options.angle)
            if not data_aspect_ratio_changes:
                Rotator(options).run(self.data, pinned_results=self.data)
                if self.masks is not None:
                    Rotator(options).run(self.masks, pinned_results=self.masks)
                # Update probe positions
                if self.probe_positions is not None:
                    self.probe_positions.rotate_positions(
                        angle=options.angle,
                        center_pixel=center_pixel,
                    )
                if apply_to_center_of_rotation:
                    # Update center of rotation
                    self.center_of_rotation = rotate_positions(
                        self.center_of_rotation, options.angle, center_pixel
                    )
            else:
                # If projections are already pinned, create a new pinned array
                # at the new aspect ratio
                if gpu_utils.is_pinned(self.data):
                    # Note: for very large arrays where pinned memory
                    # needs to be carefully managed, this may cause
                    # problems since you are temporarily doubling the
                    # amount of pinned memory.
                    pinned_array = gpu_utils.create_empty_pinned_array(
                        shape=(self.n_projections, *self.size[::-1]),
                        dtype=self.data.dtype,
                    )
                    self.data = Rotator(options).run(self.data, pinned_results=pinned_array)
                else:
                    self.data = Rotator(options).run(self.data)
                # Do the same procedure for the masks
                if self.masks is not None:
                    if gpu_utils.is_pinned(self.masks):
                        pinned_array = gpu_utils.create_empty_pinned_array(
                            shape=(self.n_projections, *self.size),
                            dtype=self.masks.dtype,
                        )
                        self.masks = Rotator(options).run(self.masks, pinned_results=pinned_array)
                    else:
                        self.masks = Rotator(options).run(self.masks)

                new_center_pixel = np.array(self.data.shape[1:]) / 2
                # Update probe positions
                if self.probe_positions is not None:
                    self.probe_positions.rotate_positions(
                        angle=options.angle,
                        center_pixel=center_pixel,
                        new_center_pixel=new_center_pixel,
                    )
                if apply_to_center_of_rotation:
                    # Update center of rotation
                    self.center_of_rotation = rotate_positions(
                        self.center_of_rotation,
                        options.angle,
                        center_pixel,
                        new_center=new_center_pixel,
                    )
            self.transform_tracker.update_rotation(options.angle)
            # To do: insert code for updating center of rotation

    @timer()
    def shear_projections(self, options: ShearOptions, apply_to_center_of_rotation: bool = True):
        if options.enabled:
            Shearer(options).run(self.data, pinned_results=self.data)
            if self.masks is not None:
                # Will probably be wrong without fixes
                self.masks = Shearer(options).run(self.masks)
                self.masks[self.masks > 0.5] = 1
                self.masks[self.masks < 0.5] = 0

            center_pixel = np.array(self.data.shape[1:]) / 2
            if self.probe_positions is not None:
                self.probe_positions.shear_positions(
                    angle=options.angle,
                    center_pixel=center_pixel,
                )
            if apply_to_center_of_rotation:
                self.center_of_rotation = shear_positions(
                    self.center_of_rotation, options.angle, center_pixel, axis=1
                )
            self.transform_tracker.update_shear(options.angle)

    def center_projections(self):
        # Shift projections so that center of rotation is in center of image
        shift = np.round(np.array(self.size) / 2 - self.center_of_rotation)
        self.center_of_rotation = self.center_of_rotation + shift
        shift = shift[::-1][None].repeat(self.n_projections, 0)
        shifter_function = Shifter(ShiftOptions(type=enums.ShiftType.CIRC, enabled=True))
        self.data = shifter_function.run(self.data, shift)
        if self.masks is not None:
            self.masks = shifter_function.run(self.masks, shift)
        if self.probe_positions is not None:
            self.probe_positions.shift_positions(shift)

    @timer()
    def setup_masks_from_probe_positions(self):
        if self.probe is None or self.probe_positions is None:
            raise Exception(
                "The Projections object must have probe_positions and "
                + "probe attribute to run create_mask_from_probe_positions!"
            )
        self.mask_builder = IlluminationMapMaskBuilder()
        self.mask_builder.get_mask_base(self.probe, self.probe_positions.data, self.data)
        self.mask_builder.set_mask_threshold_interactively(self.data)

    @timer()
    def get_masks_from_probe_positions(
        self, threshold: Optional[float] = None, delete_mask_builder: bool = True
    ):
        """
        Do one of the following:
        1) Run `setup_masks_from_probe_positions` first
        2) Provide the threshold input, above which to set the illumination map to 1
        """
        if self.mask_builder is None and threshold is None:
            raise Exception
        if self.mask_builder is None:
            self.mask_builder = IlluminationMapMaskBuilder()
            self.mask_builder.get_mask_base(self.probe, self.probe_positions.data, self.data)
            self.mask_builder.clip_masks(threshold)
        else:
            self.mask_builder.clip_masks()
        self.masks = self.mask_builder.masks

        if delete_mask_builder:
            self.mask_builder = None

    def drop_projections(self, remove_idx: list[int], repin_array: bool = False):
        "Permanently remove specific projections from object"
        keep_idx = [i for i in range(0, self.n_projections) if i not in remove_idx]

        self.dropped_scan_numbers += self.scan_numbers[remove_idx].tolist()

        def return_modified_array(arr, repin_array: bool):
            if gpu_utils.is_pinned(self.data) and repin_array:
                # Repin data if it was already pinned
                arr = gpu_utils.pin_memory(arr[keep_idx])
            else:
                arr = arr[keep_idx]
            return arr

        self.data = return_modified_array(self.data, repin_array)
        if self.masks is not None:
            self.masks = return_modified_array(self.masks, repin_array)
        if self.probe_positions is not None:
            self.probe_positions.data = [self.probe_positions.data[i] for i in keep_idx]
        self.angles = self.angles[keep_idx]
        self.scan_numbers = self.scan_numbers[keep_idx]

    @property
    def n_projections(self) -> int:
        return self.data.__len__()

    @property
    def reconstructed_object_dimensions(self) -> np.ndarray:
        sample_thickness = self.options.experiment.sample_thickness
        if self.options.experiment.sample_width_type == enums.VolumeWidthType.AUTO:
            if not self.options.is_tomo:
                laminography_angle = self.options.experiment.laminography_angle
                n_lateral_pixels = np.ceil(
                    0.5 * self.data.shape[2] / np.cos(np.pi / 180 * (laminography_angle - 0.01))
                )
            else:
                n_lateral_pixels = np.max([self.data.shape[2], self.data.shape[1]])
                n_pix = np.array(
                    [n_lateral_pixels, n_lateral_pixels, sample_thickness / self.pixel_size]
                )
        elif self.options.experiment.sample_width_type == enums.VolumeWidthType.MANUAL:
            n_lateral_pixels = self.options.experiment.sample_width / self.pixel_size
        n_pix = np.array([n_lateral_pixels, n_lateral_pixels, sample_thickness / self.pixel_size])
        return np.ceil(n_pix).astype(int)

    @property
    def size(self):
        return self.data.shape[1:]

    def _post_init(self):
        "For running children specific code after instantiation"
        pass

    def pin_arrays(self):
        self.data = gpu_utils.pin_memory(self.data)
        if self.masks is not None:
            self.masks = gpu_utils.pin_memory(self.masks)

    def apply_staged_shift(self, device_options: Optional[DeviceOptions] = None):
        if self.probe_positions is not None:
            self.probe_positions.shift_positions(self.shift_manager.staged_shift)
        self.shift_manager.apply_staged_shift(
            images=self.data,
            masks=self.masks,
            device_options=device_options,
        )
    
    def undo_last_shift(self, device_options: Optional[DeviceOptions] = None):
        if self.probe_positions is not None:
            self.probe_positions.shift_positions(-self.shift_manager.past_shifts[-1])
        self.shift_manager.undo_last_shift(
            images=self.data,
            masks=self.masks,
            device_options=device_options,
        )

    def plot_projections(self, process_function: callable = lambda x: x):
        make_image_slider_plot(process_function(self.data))

    def plot_shifted_projections(self, shift: np.ndarray, process_function: callable = lambda x: x):
        "Plot the shifted projections using the shift that was passed in."
        make_image_slider_plot(process_function(image_shift_fft(self.data, shift)))

    def plot_sum_of_projections(
        self, process_function: callable = lambda x: x, show_cor: bool = False
    ):
        plt.imshow(process_function(self.data).sum(0))
        if show_cor:
            plt.plot(
                self.center_of_rotation[1],
                self.center_of_rotation[0],
                "*",
                color="red",
                markersize=10,
                markeredgecolor="black",
            )
        plt.show()

    def get_masks(self, enable_plotting: bool = False):
        mask_options = self.options.mask
        downsample_options = self.options.mask.downsample
        if downsample_options.enabled:
            mask_options = copy.deepcopy(mask_options)
            scale = downsample_options.scale
            mask_options.binary_close_coefficient = mask_options.binary_close_coefficient / scale
            mask_options.binary_erode_coefficient = mask_options.binary_erode_coefficient / scale
            mask_options.fill = mask_options.fill / scale
        else:
            mask_options = mask_options
        # Calculate masks
        self.masks = estimate_reliability_region_mask(
            images=Downsampler(self.options.mask.downsample).run(self.data),
            options=mask_options,
            enable_plotting=enable_plotting,
        )
        # Upsample results
        # Keep on CPU for now -- can add upsampling device options later if needed
        upsample_options = UpsampleOptions(
            scale=downsample_options.scale, enabled=downsample_options.enabled
        )
        self.masks = Upsampler(upsample_options).run(self.masks)
        # return Upsampler(upsample_options).run(self.masks)

    def blur_masks(
        self, kernel_sigma: int, use_gpu: bool = False, masks: Optional[np.ndarray] = None
    ):
        if masks is None:
            masks = self.masks
        return blur_masks(masks, kernel_sigma, use_gpu)

    def show_center_of_rotation(
        self,
        plot_sum_of_projections: bool = True,
        proj_idx: int = 0,
        center_of_rotation: Optional[tuple] = None,
        process_func: Optional[callable] = None,
        show_plot: bool = True,
    ):
        if center_of_rotation is None:
            center_of_rotation = self.center_of_rotation

        if process_func is None:
            if is_array_real(self.data):
                process_func = lambda x: x
            else:
                process_func = np.angle

        if plot_sum_of_projections:
            image = process_func(self.data).sum(0)
        else:
            image = process_func(self.data[proj_idx])

        plt.imshow(image, cmap="bone")
        plt.plot(center_of_rotation[1], center_of_rotation[0], ".m", label="Center of Rotation")
        plt.title(f"Center of Rotation\n(x={center_of_rotation[1]}, y={center_of_rotation[0]})")
        plt.legend()
        if show_plot:
            plt.show()

    def plot_data(
        self,
        options: PlotDataOptions = None,
        title_string: Optional[str] = None,
        plot_sum: bool = False,
        show_plot: bool = True,
    ):
        if options is None:
            options = PlotDataOptions()
        else:
            options = copy.deepcopy(options)

        if np.issubdtype(self.data.dtype, np.complexfloating) and options.process_func is None:
            print("process_func not provided, defaulting to plotting angle of complex projections")
            options.process_func = enums.ProcessFunc.ANGLE

        if options.index is None:
            options.index = 0

        if plot_sum:
            projection_data = self.data.sum(0, keepdims=True)
        else:
            projection_data = self.data

        full_title = f"Projection {options.index}"
        if title_string is not None:
            full_title = title_string + "\n" + full_title
        plt.title(full_title)

        plot_slice_of_3D_array(
            projection_data,
            options,
            self.pixel_size,
            show_plot=show_plot,
        )

    def plot_staged_shift(self, title: str = "", plot_kwargs: dict = {}):
        fig, ax = plt.subplots(2, layout="compressed")
        fig.suptitle(title, fontsize=17)
        sort_idx = np.argsort(self.angles)
        plt.sca(ax[0])
        plt.plot(
            self.angles[sort_idx],
            self.shift_manager.staged_shift[sort_idx],
            **plot_kwargs,
        )
        plt.legend(["horizontal", "vertical"], framealpha=0.5)
        plt.autoscale(enable=True, axis="x", tight=True)
        plt.grid(linestyle=":")
        plt.title("Shift vs Angle")
        plt.xlabel("angle (deg)")
        plt.ylabel("shift (px)")
        plt.sca(ax[1])

        plt.plot(
            self.scan_numbers,
            self.shift_manager.staged_shift,
            **plot_kwargs,
        )
        plt.grid(linestyle=":")
        plt.autoscale(enable=True, axis="x", tight=True)
        plt.title("shift vs Scan Number")
        plt.xlabel("scan number")
        plt.ylabel("shift (px)")
        plt.show()

    def save_projections_object(
        self,
        file_path: Optional[str] = None,
        h5_obj: Optional[Union[h5py.Group, h5py.File]] = None,
    ):
        if file_path is None and h5_obj is None:
            raise ValueError("Error: you must pass in either file_path or h5_obj.")
        elif file_path is not None and h5_obj is not None:
            raise ValueError("Error: you must pass in only file_path OR h5_obj, not both.")
        elif file_path is not None:
            h5_obj = h5py.File(file_path, "w")

        # Specify the data to save
        if self.probe_positions is not None:
            positions = self.probe_positions.data
        else:
            positions = None
        save_attr_dict = {
            "data": self.data,
            "angles": self.angles,
            "scan_numbers": self.scan_numbers,
            "masks": self.masks,
            "center_of_rotation": self.center_of_rotation,
            "positions": positions,
            "probe": self.probe,
            "pixel_size": self.pixel_size,
            "rotation": self.transform_tracker.rotation,
            "shear": self.transform_tracker.shear,
            "downsample": self.transform_tracker.scale,
            "applied_shifts": self.shift_manager.past_shifts,
            "staged_shift": self.shift_manager.staged_shift,
            "dropped_scan_numbers": self.dropped_scan_numbers,
        }
        # Save all elements from save_attr_dict to the .h5 file
        save_generic_data_structure_to_h5(save_attr_dict, h5_obj)
        # Save projection options
        save_generic_data_structure_to_h5(self.options, h5_obj.create_group("options"))

        print(f"projections saved to {h5_obj.file.filename}{h5_obj.name}")
        if isinstance(h5_obj, h5py.File):
            h5_obj.close()

    def get_plot_object(
        self,
        plot_options: Optional[ImageSliderPlotOptions] = None,
    ) -> ImagePlotObject:
        plot_object = ImagePlotObject(self.data, options=plot_options)
        return plot_object



class ComplexProjections(Projections):
    def unwrap_phase(self, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        unwrap_phase_wrapped = device_handling_wrapper(
            func=unwrap_phase,
            options=self.options.phase_unwrap.device,
            chunkable_inputs_for_gpu_idx=[0, 1],
            pinned_results=pinned_results,
            display_progress_bar=True,
        )
        return unwrap_phase_wrapped(self.data, self.masks, self.options.phase_unwrap)


class PhaseProjections(Projections):
    def _post_init(self):
        self.laminogram = Laminogram(self)

    def get_3D_reconstruction(
        self,
        filter_inputs: bool = False,
        pinned_filtered_sinogram: Optional[np.ndarray] = None,
        reinitialize_astra: bool = True,
    ):
        self.laminogram.generate_laminogram(
            filter_inputs=filter_inputs,
            pinned_filtered_sinogram=pinned_filtered_sinogram,
            reinitialize_astra=reinitialize_astra,
        )

    def estimate_center_of_rotation(self) -> CenterOfRotationEstimateResults:
        clear_timer_globals()
        estimate_center_options = self.return_auto_centered_search_options()
        return estimate_center_of_rotation(self, self.angles, self.masks, estimate_center_options)

    def plot_coordinate_search_points(
        self,
        proj_idx: int = 0,
        plot_projection_sum: bool = False,
    ):
        # Plots the coordinate search points
        estimate_center_options = self.return_auto_centered_search_options()
        horizontal_coordinate_array, _ = format_coordinate_options(
            estimate_center_options.horizontal_coordinate
        )
        vertical_coordinate_array, _ = format_coordinate_options(
            estimate_center_options.vertical_coordinate
        )
        plot_coordinate_search_points(
            self.data,
            horizontal_coordinate_array,
            vertical_coordinate_array,
            proj_idx=proj_idx,
            plot_projection_sum=plot_projection_sum,
        )

    def return_auto_centered_search_options(self) -> EstimateCenterOptions:
        modified_options = copy.deepcopy(self.options.estimate_center)
        if self.options.estimate_center.horizontal_coordinate.center_estimate is None:
            modified_options.horizontal_coordinate.center_estimate = self.center_of_rotation[1]
        if self.options.estimate_center.vertical_coordinate.center_estimate is None:
            modified_options.vertical_coordinate.center_estimate = self.center_of_rotation[0]
        return modified_options


class ShiftManager:
    def __init__(self, n_projections: int):
        self.staged_shift = np.zeros((n_projections, 2))
        self.past_shifts: List[np.ndarray] = []
        self.past_shift_functions: List[enums.ShiftType] = []
        self.past_shift_options: List[AlignmentOptions] = []
        self.staged_function_type = None
        self.staged_alignment_options = None

    def stage_shift(
        self,
        shift: np.ndarray,
        function_type: enums.ShiftType,
        alignment_options: Optional[AlignmentOptions] = None,
    ):
        self.staged_shift = shift
        self.staged_function_type = function_type
        self.staged_alignment_options = alignment_options

    def unstage_shift(self):
        # Store staged values
        self.past_shifts += [self.staged_shift]
        self.past_shift_functions += [self.staged_function_type]
        self.past_shift_options += [self.staged_alignment_options]
        # Clear the staged variables
        self.staged_shift = np.zeros_like(self.staged_shift)
        self.staged_function_type = None
        self.staged_alignment_options = None

    def shift_arrays(
        self,
        shift: np.ndarray,
        images: np.ndarray,
        masks: np.ndarray,
        function_type: enums.ShiftType,
        device_options: Optional[DeviceOptions] = None,
    ):
        if device_options is None:
            device_options = DeviceOptions()

        shift_options = ShiftOptions(
            enabled=True,
            type=function_type,
            device=device_options,
        )
        images[:] = Shifter(shift_options).run(
            images=images,
            shift=shift,
            pinned_results=images,
        )
        if masks is not None:
            if shift_options.type == enums.ShiftType.FFT:
                # We want to avoid doing FFT shift with binary masks
                # Use linear interpolation instead
                shift_options = copy.deepcopy(shift_options)
                shift_options.type = enums.ShiftType.LINEAR
            masks[:] = Shifter(shift_options).run(
                images=masks,
                shift=shift,
                pinned_results=masks,
            )

    def apply_staged_shift(
        self,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        device_options: Optional[DeviceOptions] = None,
    ):
        if self.is_shift_nonzero():
            self.shift_arrays(
                shift=self.staged_shift,
                images=images,
                masks=masks,
                function_type=self.staged_function_type,
                device_options=device_options,
            )        
            self.unstage_shift()
        else:
            print("There is no shift to apply!")

    def undo_last_shift(self, images: np.ndarray, masks: np.ndarray, device_options: DeviceOptions):
        self.shift_arrays(
            shift=-self.past_shifts[-1],
            images=images,
            masks=masks,
            function_type=self.past_shift_functions[-1],
            device_options=device_options,
        )
        self.past_shifts = self.past_shifts[:-1]
        self.past_shift_functions = self.past_shift_functions[:-1]
        self.past_shift_options = self.past_shift_options[:-1]

    def get_staged_shift_options(
        self,
        device_options: Optional[DeviceOptions] = None,
    ) -> ShiftOptions:
        if device_options is None:
            device_options = DeviceOptions()
        return ShiftOptions(
            enabled=True,
            type=self.staged_function_type,
            device=device_options,
        )

    def is_shift_nonzero(self):
        if self.staged_function_type is enums.ShiftType.CIRC:
            shift = np.round(self.staged_shift)
        else:
            shift = self.staged_shift
        if np.any(shift != 0):
            return True
        else:
            return False


def get_kwargs_for_copying_to_new_projections_object(
    projections: Projections, include_projections_copy: bool = True
) -> dict:
    if projections.probe_positions is not None:
        positions = projections.probe_positions.data
    else:
        positions = None
    kwargs = {
        "angles": projections.angles * 1,
        "options": copy.deepcopy(projections.options),
        "masks": copy.deepcopy(projections.masks),
        "scan_numbers": copy.deepcopy(projections.scan_numbers),
        "probe_positions": positions,
        "probe": copy.deepcopy(projections.probe),
        "transform_tracker": copy.deepcopy(projections.transform_tracker),
        "shift_manager": copy.deepcopy(projections.shift_manager),
        "center_of_rotation": projections.center_of_rotation * 1,
        "skip_pre_processing": True,
        "add_center_offset_to_positions": False,
    }
    if include_projections_copy:
        kwargs["projections"] = projections.data * 1

    return kwargs
