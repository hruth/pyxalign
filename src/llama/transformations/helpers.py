from functools import wraps
import numpy as np
from llama.api.types import ArrayType
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
