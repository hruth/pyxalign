from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy import stats
from tqdm import tqdm
import copy
from pyxalign.api.enums import DownsampleType, DeviceType, RoundType, ShiftType

import pyxalign.api.maps as maps
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.transform import (
    DownsampleOptions,
    PadOptions,
    ShiftOptions,
    TransformOptions,
    UpsampleOptions,
    CropOptions,
    Crop3DOptions,
    ShearOptions,
    RotationOptions,
)

from pyxalign.api.options_utils import print_options
from pyxalign.api.types import ArrayType
from pyxalign.gpu_wrapper import device_handling_wrapper
from pyxalign.transformations.functions import eliminate_wrapping_from_shift, image_crop, crop_3d, image_crop_pad
from pyxalign.timing.timer_utils import timer
from pyxalign.transformations.helpers import force_roi_parameters_into_array_bounds, round_to_divisor


class Transformation(ABC):
    """`Transformation` objects run transformation functions using the passed in device options."""

    def __init__(self, options: TransformOptions):
        self.enabled = options.enabled
        self.options = options

    @abstractmethod
    def run(self, images: ArrayType, *args, **kwargs) -> ArrayType:
        pass


class Downsampler(Transformation):
    def __init__(
        self,
        options: DownsampleOptions,
    ):
        super().__init__(options)
        self.options: DownsampleOptions = options

    @timer()
    def run(
        self,
        images: ArrayType,
        shift: Optional[ArrayType] = None,
        pinned_results: Optional[np.ndarray] = None,
    ) -> ArrayType:
        """Calls one of the image downsampling functions"""
        # Note: currently the linear downsampling function also has the option to shift
        # the inputs.
        if self.enabled and self.options.scale != 1:
            if self.options.type is DownsampleType.LINEAR and shift is not None:
                self.function = device_handling_wrapper(
                    func=maps.get_downsample_func_by_enum(self.options.type),
                    options=self.options.device,
                    chunkable_inputs_for_gpu_idx=[0, 2],
                    pinned_results=pinned_results,
                )
                return self.function(
                    images,
                    self.options.scale,
                    shift,
                    use_gaussian_filter=self.options.use_gaussian_filter,
                )
            else:
                self.function = device_handling_wrapper(
                    func=maps.get_downsample_func_by_enum(self.options.type),
                    options=self.options.device,
                    chunkable_inputs_for_gpu_idx=[0],
                    pinned_results=pinned_results,
                    display_progress_bar=True,
                )
                return self.function(
                    images,
                    self.options.scale,
                    use_gaussian_filter=self.options.use_gaussian_filter,
                )
        else:
            return images


class Upsampler(Transformation):
    def __init__(
        self,
        options: UpsampleOptions,
    ):
        super().__init__(options)
        self.options: UpsampleOptions = options

    @timer()
    def run(self, images: ArrayType, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        """Calls one of the image upsampling functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_upsample_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0],
                pinned_results=pinned_results,
                display_progress_bar=True,
            )
            return self.function(images, self.options.scale)
        else:
            return images


class Shifter(Transformation):
    def __init__(
        self,
        options: ShiftOptions,
    ):
        super().__init__(options)
        self.options: ShiftOptions = options

    @timer()
    def run(
        self,
        images: ArrayType,
        shift: np.ndarray,
        pinned_results: Optional[np.ndarray] = None,
        is_binary_mask: bool = False,
    ) -> ArrayType:
        """Calls one of the image shifting functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_shift_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0, 1],
                pinned_results=pinned_results,
            )

            if self.options.type == ShiftType.LINEAR:
                images = images * 1
                images = self.function(images, shift)
            else:
                images = self.function(
                    images, shift, eliminate_wrapping=self.options.eliminate_wrapping,
                )

            if is_binary_mask and self.options.type == ShiftType.FFT:
                idx = images > 0.5
                images[:] = 0
                images[idx] = 1

            return images
        else:
            return images


class Rotator(Transformation):
    def __init__(
        self,
        options: RotationOptions,
    ):
        super().__init__(options)
        self.options: RotationOptions = options

    @timer()
    def run(self, images: ArrayType, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        """Calls one of the image rotation functions"""
        if self.enabled:
            if self.options.device.device_type is DeviceType.CPU:
                raise NotImplementedError("This function is not supported on CPU.")
            self.function = device_handling_wrapper(
                func=maps.get_rotation_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0],
                pinned_results=pinned_results,
                display_progress_bar=True,
            )
            return self.function(images, self.options.angle)
        else:
            return images


class Shearer(Transformation):
    def __init__(
        self,
        options: ShearOptions,
    ):
        super().__init__(options)
        self.options: ShearOptions = options

    @timer()
    def run(self, images: ArrayType, pinned_results: Optional[np.ndarray] = None) -> ArrayType:
        """Calls one of the image shearing functions"""
        if self.enabled:
            self.function = device_handling_wrapper(
                func=maps.get_shear_func_by_enum(self.options.type),
                options=self.options.device,
                chunkable_inputs_for_gpu_idx=[0],
                pinned_results=pinned_results,
                display_progress_bar=True,
            )
            return self.function(images, self.options.angle)
        else:
            return images


def force_crop_options_in_bounds(crop_options: CropOptions, array_2d_size: tuple) -> tuple[CropOptions, bool]:
    horizontal_range, vertical_range = Cropper.get_ranges_from_crop_options(crop_options, array_2d_size)
    new_w_x, new_w_y, c_x, c_y, out_of_bounds = force_roi_parameters_into_array_bounds(
        horizontal_range=horizontal_range,
        vertical_range=vertical_range,
        horizontal_offset=crop_options.horizontal_offset,
        vertical_offset=crop_options.vertical_offset,
        array_2d_size=array_2d_size,
    )
    new_crop_options = copy.deepcopy(crop_options)
    new_crop_options.horizontal_range = new_w_x
    new_crop_options.vertical_range = new_w_y
    new_crop_options.horizontal_offset = c_x
    new_crop_options.vertical_offset = c_y

    return new_crop_options, out_of_bounds


class Cropper(Transformation):
    def __init__(
        self,
        options: CropOptions,
    ):
        super().__init__(options)
        self.options: CropOptions = options

    @timer()
    def run(self, images: ArrayType) -> ArrayType:
        """Calls the image cropping function"""
        if self.enabled:
            horizontal_range, vertical_range = self.get_ranges_from_crop_options(
                self.options, images.shape[1:]
            )
            cropped_images = image_crop(
                images,
                horizontal_range,
                vertical_range,
                self.options.horizontal_offset,
                self.options.vertical_offset,
            )
            if self.options.return_view:
                return cropped_images
            else:
                return cropped_images * 1
        else:
            return images
        
    @staticmethod
    def get_ranges_from_crop_options(crop_options: CropOptions, array_2d_shape: tuple) -> tuple:
        if crop_options.horizontal_range is None:
            horizontal_range = array_2d_shape[1]
        else:
            horizontal_range = crop_options.horizontal_range
        if crop_options.vertical_range is None:
            vertical_range = array_2d_shape[0]
        else:
            vertical_range = crop_options.vertical_range
        return horizontal_range, vertical_range

    @staticmethod
    def fix_crop_range(crop_options: CropOptions, multiple_of: int, array_2d_size: tuple) -> CropOptions:
        # get crop ranges
        crop_options.horizontal_range, crop_options.vertical_range = (
            Cropper.get_ranges_from_crop_options(crop_options, array_2d_size)
        )

        # check if crop options are within the bounds of the array
        new_crop_options, out_of_bounds = force_crop_options_in_bounds(crop_options, array_2d_size)

        crop_options_updated = False
        if out_of_bounds:
            crop_options_updated = True
            print("WARNING: Specified crop range is outside the bounds of the projections array.")

        # check if crop range is a multiple of the selected value
        crop_widths = new_crop_options.horizontal_range, new_crop_options.vertical_range
        if not np.all([(w % multiple_of) == 0 for w in crop_widths]):
            crop_options_updated = True
            print(f"WARNING: Specified crop widths are not a multiple of {multiple_of}")
            new_crop_options.horizontal_range = round_to_divisor(
                new_crop_options.horizontal_range,
                RoundType.FLOOR,
                divisor=int(multiple_of),
            )
            new_crop_options.vertical_range = round_to_divisor(
                new_crop_options.vertical_range,
                RoundType.FLOOR,
                divisor=int(multiple_of),
            )

        if crop_options_updated:
            print("Crop range is being updated automatically.")
            print("Original crop options:")
            print_options(crop_options, include_class_name=False)
            print("Updated crop options:")
            print_options(new_crop_options, include_class_name=False)

        return new_crop_options


class Cropper3D(Transformation):
    """
    3D cropper for volumetric data.

    This class applies 3D cropping to volumetric arrays using Crop3DOptions,
    allowing independent control over crop range and offset in all three dimensions.
    """

    def __init__(
        self,
        options: Crop3DOptions,
    ):
        super().__init__(options)
        self.options: Crop3DOptions = options

    @timer()
    def run(self, volume: ArrayType) -> ArrayType:
        """
        Apply 3D cropping to a volume.

        Args:
            volume: 3D array to crop (depth, vertical, horizontal)

        Returns:
            Cropped volume, either as a view or a copy depending on return_view option
        """
        if self.enabled:
            horizontal_range, vertical_range, depth_range = self.get_ranges_from_crop_options(
                self.options, volume.shape
            )
            cropped_volume = crop_3d(
                volume,
                horizontal_range,
                vertical_range,
                depth_range,
                self.options.horizontal_offset,
                self.options.vertical_offset,
                self.options.depth_offset,
            )
            if self.options.return_view:
                return cropped_volume
            else:
                return cropped_volume * 1
        else:
            return volume

    @staticmethod
    def get_ranges_from_crop_options(crop_options: Crop3DOptions, volume_shape: tuple) -> tuple:
        """
        Extract crop ranges from options, using volume shape for None values.

        Args:
            crop_options: Crop3DOptions instance
            volume_shape: Shape of the 3D volume (depth, vertical, horizontal)

        Returns:
            Tuple of (horizontal_range, vertical_range, depth_range)
        """
        if crop_options.horizontal_range is None:
            horizontal_range = volume_shape[2]
        else:
            horizontal_range = crop_options.horizontal_range

        if crop_options.vertical_range is None:
            vertical_range = volume_shape[1]
        else:
            vertical_range = crop_options.vertical_range

        if crop_options.depth_range is None:
            depth_range = volume_shape[0]
        else:
            depth_range = crop_options.depth_range

        return horizontal_range, vertical_range, depth_range


class Padder(Transformation):
    def __init__(
        self,
        options: PadOptions,
    ):
        super().__init__(options)
        self.options: PadOptions = options

    @timer()
    def run(self, images: ArrayType) -> ArrayType:
        """Calls the image padding function"""
        if self.enabled:
            if self.options.new_extent_x is None:
                new_extent_x = images.shape[2]
            else:
                new_extent_x = self.options.new_extent_x
            if self.options.new_extent_y is None:
                new_extent_y = images.shape[1]
            else:
                new_extent_y = self.options.new_extent_y

            padded_images = np.zeros(
                (images.shape[0], new_extent_y, new_extent_x), dtype=images.dtype
            )
            for i in tqdm(range(len(images))):
                pad_value = self.options.pad_value
                padded_images[i] = image_crop_pad(
                    images[i], new_extent_y, new_extent_x, "constant", pad_value
                )

            return padded_images
        else:
            return images


