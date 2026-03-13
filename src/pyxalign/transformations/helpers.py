from typing import Union
from functools import wraps
from numbers import Number
from typing import Sequence
import numpy as np
from pyxalign.api.enums import RoundType
from pyxalign.api.options.roi import RectangularROIOptions
from pyxalign.api.options.transform import Crop3DOptions
from pyxalign.api.types import ArrayType
# Should move all this into a different folder at some point


def preserve_complexity_or_realness():
    def inner_func(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            is_real = is_array_real(args[0])
            images = func(*args, **kwargs)
            if is_real:
                return images.real
            else:
                return images

        return wrapper

    return inner_func


def is_array_real(array: ArrayType):
    return not np.issubdtype(array.dtype, np.complexfloating)


def round_to_divisor(
    input: Union[Number, Sequence[Number], np.ndarray],
    round_type: RoundType,
    divisor: int,
) -> Union[int, np.ndarray]:
    if round_type == RoundType.CEIL:
        func = np.ceil
    elif round_type == RoundType.FLOOR:
        func = np.floor
    elif round_type == RoundType.NEAREST:
        func = np.round

    def rounding_func(x):
        return int(func(x / divisor) * divisor)

    if hasattr(input, "__len__"):
        vectorized_rounding_func = np.vectorize(rounding_func)
        return vectorized_rounding_func(input)
    else:
        return rounding_func(input)


def force_roi_parameters_into_array_bounds(
    horizontal_range: int,
    vertical_range: int,
    horizontal_offset: int,
    vertical_offset: int,
    array_2d_size: tuple,
) -> tuple:
    
    if horizontal_range is None:
        horizontal_range = array_2d_size[1]
    if vertical_range is None:
        vertical_range = array_2d_size[0]

    x0, y0 = int(np.floor(array_2d_size[1] / 2)), int(np.floor(array_2d_size[0] / 2))
    c_x, c_y, w_x, w_y = (
        horizontal_offset,
        vertical_offset,
        horizontal_range,
        vertical_range
    )
    x_start, x_end = x0 + c_x - int(np.floor(w_x / 2)), x0 + c_x + int(np.floor(w_x / 2))
    y_start, y_end = y0 + c_y - int(np.floor(w_y / 2)), y0 + c_y + int(np.floor(w_y / 2))

    out_of_bounds = False
    if x_start < 0:
        x_start = 0
        c_x = -(int(np.floor(array_2d_size[1] / 2)) - int(np.floor((x_end - x_start) / 2)))
        out_of_bounds = True
    if x_end > array_2d_size[1]:
        x_end = array_2d_size[1]
        c_x = int(np.floor(array_2d_size[1] / 2)) - int(np.floor((x_end - x_start) / 2))
        out_of_bounds = True
    if y_start < 0:
        y_start = 0
        c_y = -(int(np.floor(array_2d_size[0] / 2)) - int(np.floor((y_end - y_start) / 2)))
        out_of_bounds = True
    if y_end > array_2d_size[0]:
        y_end = array_2d_size[0]
        c_y = int(np.floor(array_2d_size[0] / 2)) - int(np.floor((y_end - y_start) / 2))
        out_of_bounds = True

    new_w_x, new_w_y = x_end - x_start, y_end - y_start

    return new_w_x, new_w_y, c_x, c_y, out_of_bounds


def force_rectangular_roi_in_bounds(
    rect_roi_options: RectangularROIOptions, array_2d_size: tuple
) -> RectangularROIOptions:
    new_w_x, new_w_y, c_x, c_y, out_of_bounds = force_roi_parameters_into_array_bounds(
        horizontal_range=rect_roi_options.horizontal_range,
        vertical_range=rect_roi_options.vertical_range,
        horizontal_offset=rect_roi_options.horizontal_offset,
        vertical_offset=rect_roi_options.vertical_offset,
        array_2d_size=array_2d_size,
    )
    new_rect_roi_options = RectangularROIOptions(
        horizontal_range=new_w_x,
        vertical_range=new_w_y,
        horizontal_offset=c_x,
        vertical_offset=c_y,
    )

    return new_rect_roi_options


def force_crop_3d_parameters_into_volume_bounds(
    horizontal_range: int,
    vertical_range: int,
    depth_range: int,
    horizontal_offset: int,
    vertical_offset: int,
    depth_offset: int,
    volume_shape: tuple,
) -> tuple:
    """
    Force 3D crop parameters to be within the bounds of a volume.

    Args:
        horizontal_range: Width of crop in horizontal dimension
        vertical_range: Height of crop in vertical dimension
        depth_range: Depth of crop in depth dimension
        horizontal_offset: Offset from center in horizontal dimension
        vertical_offset: Offset from center in vertical dimension
        depth_offset: Offset from center in depth dimension
        volume_shape: Shape of the 3D volume (depth, vertical, horizontal)

    Returns:
        Tuple of (new_horizontal_range, new_vertical_range, new_depth_range,
                  new_horizontal_offset, new_vertical_offset, new_depth_offset,
                  out_of_bounds)
    """
    # Handle None values
    if horizontal_range is None:
        horizontal_range = volume_shape[2]
    if vertical_range is None:
        vertical_range = volume_shape[1]
    if depth_range is None:
        depth_range = volume_shape[0]

    # Calculate centers of the volume
    depth_center = int(np.floor(volume_shape[0] / 2))
    vertical_center = int(np.floor(volume_shape[1] / 2))
    horizontal_center = int(np.floor(volume_shape[2] / 2))

    # Current offsets and ranges
    c_h, c_v, c_d = horizontal_offset, vertical_offset, depth_offset
    w_h, w_v, w_d = horizontal_range, vertical_range, depth_range

    # Calculate start and end indices
    h_start = horizontal_center + c_h - int(np.floor(w_h / 2))
    h_end = horizontal_center + c_h + int(np.floor(w_h / 2))
    v_start = vertical_center + c_v - int(np.floor(w_v / 2))
    v_end = vertical_center + c_v + int(np.floor(w_v / 2))
    d_start = depth_center + c_d - int(np.floor(w_d / 2))
    d_end = depth_center + c_d + int(np.floor(w_d / 2))

    out_of_bounds = False

    # Horizontal dimension bounds checking
    if h_start < 0:
        h_start = 0
        c_h = -(horizontal_center - int(np.floor((h_end - h_start) / 2)))
        out_of_bounds = True
    if h_end > volume_shape[2]:
        h_end = volume_shape[2]
        c_h = horizontal_center - int(np.floor((h_end - h_start) / 2))
        out_of_bounds = True

    # Vertical dimension bounds checking
    if v_start < 0:
        v_start = 0
        c_v = -(vertical_center - int(np.floor((v_end - v_start) / 2)))
        out_of_bounds = True
    if v_end > volume_shape[1]:
        v_end = volume_shape[1]
        c_v = vertical_center - int(np.floor((v_end - v_start) / 2))
        out_of_bounds = True

    # Depth dimension bounds checking
    if d_start < 0:
        d_start = 0
        c_d = -(depth_center - int(np.floor((d_end - d_start) / 2)))
        out_of_bounds = True
    if d_end > volume_shape[0]:
        d_end = volume_shape[0]
        c_d = depth_center - int(np.floor((d_end - d_start) / 2))
        out_of_bounds = True

    # Calculate new ranges
    new_w_h = h_end - h_start
    new_w_v = v_end - v_start
    new_w_d = d_end - d_start

    return new_w_h, new_w_v, new_w_d, c_h, c_v, c_d, out_of_bounds


def force_crop_3d_options_in_bounds(
    crop_3d_options: Crop3DOptions, volume_shape: tuple
) -> Crop3DOptions:
    """
    Force Crop3DOptions to be within the bounds of a volume.

    Args:
        crop_3d_options: Crop3DOptions instance to validate
        volume_shape: Shape of the 3D volume (depth, vertical, horizontal)

    Returns:
        New Crop3DOptions instance with parameters adjusted to be within bounds
    """
    new_w_h, new_w_v, new_w_d, c_h, c_v, c_d, out_of_bounds = (
        force_crop_3d_parameters_into_volume_bounds(
            horizontal_range=crop_3d_options.horizontal_range,
            vertical_range=crop_3d_options.vertical_range,
            depth_range=crop_3d_options.depth_range,
            horizontal_offset=crop_3d_options.horizontal_offset,
            vertical_offset=crop_3d_options.vertical_offset,
            depth_offset=crop_3d_options.depth_offset,
            volume_shape=volume_shape,
        )
    )

    new_crop_3d_options = Crop3DOptions(
        horizontal_range=new_w_h,
        vertical_range=new_w_v,
        depth_range=new_w_d,
        horizontal_offset=c_h,
        vertical_offset=c_v,
        depth_offset=c_d,
        enabled=crop_3d_options.enabled,
        return_view=crop_3d_options.return_view,
    )

    return new_crop_3d_options


