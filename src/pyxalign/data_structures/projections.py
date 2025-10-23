from typing import List, Optional, Sequence, Union
import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
from pyxalign.alignment.utils import (
    get_center_of_rotation_from_different_resolution_alignment,
    get_shift_from_different_resolution_alignment,
)
from pyxalign.api.options.plotting import PlotDataOptions
from pyxalign.estimate_center import (
    estimate_center_of_rotation,
    CenterOfRotationEstimateResults,
    format_coordinate_options,
    plot_coordinate_search_points,
)
from pyxalign.api import enums
from pyxalign.api.options.alignment import AlignmentOptions
from pyxalign.api.options.device import DeviceOptions

from pyxalign.api.options.projections import (
    EstimateCenterOptions,
    ProjectionOptions,
    ProjectionTransformOptions,
)
from pyxalign.api.options.transform import (
    CropOptions,
    PadOptions,
    RotationOptions,
    ShearOptions,
    ShiftOptions,
    UpsampleOptions,
    DownsampleOptions,
)
import pyxalign.gpu_utils as gpu_utils
from pyxalign.gpu_wrapper import device_handling_wrapper
from pyxalign.data_structures.volume import Volume
from pyxalign.interactions.mask import build_masks_from_threshold, launch_mask_builder
from pyxalign.io.utils import load_list_of_arrays
from pyxalign.io.save import save_generic_data_structure_to_h5

from pyxalign.mask import estimate_reliability_region_mask, blur_masks
from pyxalign.model_functions import symmetric_gaussian_2d
import pyxalign.plotting.plotters as plotters
from pyxalign.style.text import ordinal
from pyxalign.timing.timer_utils import timer, clear_timer_globals
from pyxalign.transformations.classes import (
    Downsampler,
    Rotator,
    Shearer,
    Shifter,
    Upsampler,
    Cropper,
    Padder,
)
from pyxalign.transformations.functions import (
    image_shift_fft,
    rotate_positions,
    shear_positions,
    will_rotation_flip_aspect_ratio,
)
from pyxalign.transformations.helpers import is_array_real
from pyxalign.unwrap import unwrap_phase
from pyxalign.data_structures.positions import ProbePositions
from pyxalign.api.types import ArrayType, r_type

__all__ = ["Projections", "PhaseProjections", "ComplexProjections"]

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
        file_paths: Optional[list[str]] = None,
    ):
        """
        Test description for `Projections`
        """
        self.options = options
        self.file_paths = file_paths
        self.angles = np.array(angles, dtype=r_type)
        self.data = projections
        self.masks = masks
        if pin_arrays:
            self.pin_arrays()
        self.probe = probe
        if scan_numbers is not None:
            self.scan_numbers = np.array(scan_numbers, dtype=int)
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
        # self.mask_builder: IlluminationMapMaskBuilder = None
        self.dropped_scan_numbers = []
        self.dropped_angles = {}
        self.dropped_file_paths = {}
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
            if self.probe_positions is not None:
                self.probe_positions.crop_positions(
                    x_max=self.data.shape[2], y_max=self.data.shape[1]
                )
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
        if options.enabled and options.angle != 0:
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
        if options.enabled and options.angle != 0:
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
    def get_masks_from_probe_positions(
        self, threshold: Optional[float] = None, wait_until_closed: bool = True
    ):
        if threshold is None:
            # open the window
            self.mask_gui = launch_mask_builder(
                self.data,
                self.probe,
                self.probe_positions.data,
                mask_receiver_function=self._receive_masks,
                wait_until_closed=wait_until_closed,
            )
        else:
            # bypass the GUI if the threshold is known
            self.masks = build_masks_from_threshold(
                self.data.shape, self.probe, self.probe_positions.data, threshold
            )

    def _receive_masks(self, masks: np.ndarray):
        if self.masks is None or self.masks.shape != masks.shape:
            self.masks = masks
        else:
            self.masks[:] = masks

    def drop_projections(self, remove_scans: list[int], repin_array: bool = False):
        "Permanently remove specific projections from object"
        # keep_idx = [i for i in range(0, self.n_projections) if i not in remove_idx]
        # self.dropped_scan_numbers += self.scan_numbers[remove_idx].tolist()
        if not hasattr(remove_scans, "__len__"):
            raise TypeError("Input argument `remove_scans` should be a list of integers")
        if isinstance(remove_scans, np.ndarray):
            remove_scans = list(remove_scans)
        keep_idx = [i for i, scan in enumerate(self.scan_numbers) if scan not in remove_scans]
        remove_idx = [i for i, scan in enumerate(self.scan_numbers) if scan in remove_scans]
        # update list of dropped scans
        # self.dropped_scan_numbers += [scan for scan in remove_scans if scan in self.scan_numbers]
        self.dropped_scan_numbers += [self.scan_numbers[i] for i in remove_idx]
        for i in remove_idx:
            self.dropped_angles[self.scan_numbers[i]] = self.angles[i]
            if self.file_paths is not None:
                self.dropped_file_paths[self.scan_numbers[i]] = self.file_paths[i]

        def return_modified_array(arr, repin_array: bool):
            if gpu_utils.is_pinned(self.data) and repin_array:
                # Repin data if it was already pinned
                arr = gpu_utils.pin_memory(arr[keep_idx])
            else:
                arr = arr[keep_idx]
            return arr

        # Remove projections
        self.data = return_modified_array(self.data, repin_array)
        # Update all other relevant arrays
        if self.masks is not None:
            self.masks = return_modified_array(self.masks, repin_array)
        if self.probe_positions is not None:
            self.probe_positions.data = [self.probe_positions.data[i] for i in keep_idx]
        self.angles = self.angles[keep_idx]
        if self.file_paths is not None:
            self.file_paths = [self.file_paths[i] for i in keep_idx]
        self.scan_numbers = self.scan_numbers[keep_idx]

        # Update the past shifts and staged shift
        self.shift_manager.staged_shift = self.shift_manager.staged_shift[keep_idx]
        for i, shift in enumerate(self.shift_manager.past_shifts):
            self.shift_manager.past_shifts[i] = self.shift_manager.past_shifts[i][keep_idx]

    @property
    def n_projections(self) -> int:
        return self.data.__len__()

    @property
    def reconstructed_object_dimensions(self) -> np.ndarray:
        sample_thickness = self.options.experiment.sample_thickness
        # calculate volume size
        n_lateral_pixels = self.data.shape[2]
        if self.options.volume_width.use_custom_width:
            n_lateral_pixels *= self.options.volume_width.multiplier
        n_pix = np.array([n_lateral_pixels, n_lateral_pixels, sample_thickness / self.pixel_size])
        # n_pix = round_to_divisor(n_pix, "ceil", divisor)
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
        if len(self.shift_manager.past_shifts) == 0:
            print("There is no shift to undo!")
            return
        if self.probe_positions is not None:
            self.probe_positions.shift_positions(-self.shift_manager.past_shifts[-1])
        self.shift_manager.undo_last_shift(
            images=self.data,
            masks=self.masks,
            device_options=device_options,
        )

    def plot_projections(self, process_function: callable = lambda x: x):
        plotters.make_image_slider_plot(process_function(self.data))

    def plot_shifted_projections(self, shift: np.ndarray, process_function: callable = lambda x: x):
        "Plot the shifted projections using the shift that was passed in."
        plotters.make_image_slider_plot(process_function(image_shift_fft(self.data, shift)))

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

        if (
            np.issubdtype(self.data.dtype, np.complexfloating)
            and options.process_func is enums.ProcessFunc.NONE
        ):
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

        plotters.plot_slice_of_3D_array(
            projection_data,
            options,
            self.pixel_size,
            show_plot=show_plot,
        )

    def plot_shift(
        self,
        shift_type: enums.ShiftManagerMemberType,
        title: Optional[str] = None,
        plot_kwargs: dict = {},
    ):
        """Plot different kinds of shifts stored in the ShiftManager object.

        Args:
            shift_type (ShiftManagerMemberType): The type of shift to plot. Can be entered as the
            strings `"applied_shift_total"`, `"applied_shift_seperate"`, or `"staged_shift"`.
            title (str): If included, `title` will override the default titling.

        """

        # Use the shift specified by the user
        shifts_to_plot = []
        default_titles = []
        if shift_type == enums.ShiftManagerMemberType.APPLIED_SHIFT_TOTAL:
            shifts_to_plot += [np.sum(self.shift_manager.past_shifts, 0)]
            default_titles += ["Total Applied Shift"]
        elif shift_type == enums.ShiftManagerMemberType.APPLIED_SHIFT_SEPERATE:
            for i, shift in enumerate(self.shift_manager.past_shifts):
                shifts_to_plot += [shift]
                default_titles += [f"{ordinal(i + 1)} Applied Shift"]
        elif shift_type == enums.ShiftManagerMemberType.STAGED_SHIFT:
            shifts_to_plot += [self.shift_manager.staged_shift]
            default_titles += ["Staged Shift"]

        # plot the shift(s)
        for i, shift in enumerate(shifts_to_plot):
            fig, ax = plt.subplots(2, layout="compressed")
            if title is not None:
                fig.suptitle(title, fontsize=17)
            else:
                fig.suptitle(default_titles[i], fontsize=17)
            sort_idx = np.argsort(self.angles)
            plt.sca(ax[0])
            plt.plot(
                self.angles[sort_idx],
                shift[sort_idx],
                **plot_kwargs,
            )
            plt.legend(["horizontal", "vertical"], framealpha=0.5)
            plt.title("Shift vs Angle")
            plt.xlabel("angle (deg)")
            plt.sca(ax[1])
            plt.plot(
                self.scan_numbers,
                shift,
                **plot_kwargs,
            )
            plt.title("Shift vs Scan Number")
            plt.xlabel("scan number")
            for j in range(2):
                plt.sca(ax[j])
                plt.grid(linestyle=":")
                plt.autoscale(enable=True, axis="x", tight=True)
                plt.ylabel("shift (px)")
            plt.show()

    def plot_shift_summary(self, plot_kwargs: dict = {}):
        "Show plots summarizing all of the applied shifts and the staged shift"
        for shift_type in enums.ShiftManagerMemberType:
            print(shift_type)
            self.plot_shift(shift_type, plot_kwargs=plot_kwargs)

    def replace_probe_with_gaussian(
        self, amplitude: float, sigma: float, shape: Optional[float] = None
    ):
        if shape is None:
            shape = self.probe.shape
        self.probe = symmetric_gaussian_2d(shape, amplitude, sigma)

    def save_projections_object(
        self,
        save_path: Optional[str] = None,
        h5_obj: Optional[Union[h5py.Group, h5py.File]] = None,
    ):
        if save_path is None and h5_obj is None:
            raise ValueError("Error: you must pass in either file_path or h5_obj.")
        elif save_path is not None and h5_obj is not None:
            raise ValueError("Error: you must pass in only file_path OR h5_obj, not both.")
        elif save_path is not None:
            h5_obj = h5py.File(save_path, "w")

        # Specify the data to save
        if self.probe_positions is not None:
            positions = self.probe_positions.data
        else:
            positions = None
        if self.file_paths is not None:
            file_paths = self.file_paths
        else:
            file_paths = None
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
            "staged_shift_function_type": self.shift_manager.staged_function_type,
            "file_paths": file_paths,
            "dropped_scan_numbers": self.dropped_scan_numbers,
            "dropped_angles": self.dropped_angles,
            "dropped_file_paths": self.dropped_file_paths,
        }
        # Save all elements from save_attr_dict to the .h5 file
        save_generic_data_structure_to_h5(save_attr_dict, h5_obj)
        # Save projection options
        save_generic_data_structure_to_h5(self.options, h5_obj.create_group("options"))

        if isinstance(h5_obj, h5py.File):
            h5_obj.close()
        if save_path is None:
            print(f"projections saved to {h5_obj.file.filename}{h5_obj.name}")
        else:
            print(f"projections saved to {save_path}")

    def load_and_stage_shift(
        self,
        task_file_path: str,
        staged_function_type: enums.ShiftType = enums.ShiftType.FFT,
        update_center_of_rotation: bool = True,
        drop_unshared_scans: bool = False,
    ):
        # Load data
        with h5py.File(task_file_path, "r") as F:
            if "phase_projections" in F.keys():
                group = "phase_projections"
            elif "complex_projections" in F.keys():
                group = "complex_projections"
            past_shifts = load_list_of_arrays(F[group], "applied_shifts")
            reference_scan_numbers = np.array(F[group]["scan_numbers"][()], dtype=int)
            reference_pixel_size = F[group]["pixel_size"][()]
            reference_center_of_rotation = F[group]["center_of_rotation"][()]
            reference_shape = F[group]["data"].shape
        reference_shift = np.sum(past_shifts, 0).astype(r_type)
        # get new shift and scan numbers to drop
        shared_scan_numbers, new_shift = get_shift_from_different_resolution_alignment(
            reference_shift,
            reference_scan_numbers,
            reference_pixel_size,
            self.scan_numbers,
            self.pixel_size,
        )
        remove_scans = [scan for scan in self.scan_numbers if scan not in shared_scan_numbers]
        if drop_unshared_scans:
            # remove scans that were not in reference data
            keep_idx = [
                i for i, scan in enumerate(self.scan_numbers) if scan in shared_scan_numbers
            ]
            new_shift = new_shift[keep_idx]
            self.drop_projections(remove_scans)
            if remove_scans != []:
                print(
                    f"Removed scans that were not found in reference data: {[int(x) for x in remove_scans]}"
                )
        else:
            if remove_scans != []:
                print(
                    f"Scans not found in reference data will not be shifted: {[int(x) for x in remove_scans]}"
                )
        # stage shift
        self.shift_manager.stage_shift(new_shift, staged_function_type)
        # Update center of rotation
        if update_center_of_rotation:
            self.center_of_rotation[:] = get_center_of_rotation_from_different_resolution_alignment(
                reference_shape=reference_shape[1:],
                reference_center_of_rotation=reference_center_of_rotation,
                current_shape=self.data.shape[1:],
                reference_pixel_size=reference_pixel_size,
                current_pixel_size=self.pixel_size,
            )

        return new_shift


class ComplexProjections(Projections):
    def unwrap_phase(self, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        # this method always needs a mask
        bool_1 = (
            self.options.phase_unwrap.method
            == enums.PhaseUnwrapMethods.ITERATIVE_RESIDUAL_CORRECTION
        )
        # this method does not need a mask
        bool_2 = (
            self.options.phase_unwrap.method == enums.PhaseUnwrapMethods.GRADIENT_INTEGRATION
        ) and (self.options.phase_unwrap.gradient_integration.use_masks)
        use_masks = bool_1 or bool_2
        if use_masks is True and self.masks is None:
            raise ValueError(
                "Phase unwrapping requires masks for the selected phase_unwrap settings, but masks do not exist"
            )
        # the configuration of the device_handling_wrapper depends on the number
        # of chunked arguments passed to it, so it depends on whether or not
        # we are passing in masks
        if use_masks:
            chunkable_inputs_for_gpu_idx = [0, 1]
        else:
            chunkable_inputs_for_gpu_idx = [0]
        unwrap_phase_wrapped = device_handling_wrapper(
            func=unwrap_phase,
            options=self.options.phase_unwrap.device,
            chunkable_inputs_for_gpu_idx=chunkable_inputs_for_gpu_idx,
            pinned_results=pinned_results,
            display_progress_bar=True,
        )
        if use_masks:
            return unwrap_phase_wrapped(self.data, self.masks, self.options.phase_unwrap)
        else:
            return unwrap_phase_wrapped(self.data, None, self.options.phase_unwrap)


class PhaseProjections(Projections):
    def _post_init(self):
        self.volume = Volume(self)

    def get_3D_reconstruction(
        self,
        filter_inputs: bool = True,
        pinned_filtered_sinogram: Optional[np.ndarray] = None,
        reinitialize_astra: bool = True,
        n_pix: Optional[Sequence[int]] = None,
    ):
        self.volume.generate_volume(
            filter_inputs=filter_inputs,
            pinned_filtered_sinogram=pinned_filtered_sinogram,
            reinitialize_astra=reinitialize_astra,
            n_pix=n_pix,
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
        if len(self.past_shifts) == 0:
            print("There is no shift to undo!")
            return
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
        "file_paths": copy.deepcopy(projections.file_paths),
    }
    if include_projections_copy:
        kwargs["projections"] = projections.data * 1

    return kwargs


