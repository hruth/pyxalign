from typing import Union
from functools import wraps
from numbers import Number
from typing import Sequence
import numpy as np
from pyxalign.api.enums import RoundType
from pyxalign.api.options.roi import RectangularROIOptions
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


