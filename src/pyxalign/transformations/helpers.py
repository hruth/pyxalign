from typing import Union
from functools import wraps
from numbers import Number
from typing import Sequence
import numpy as np
from pyxalign.api.enums import RoundType
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
