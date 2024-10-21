from functools import wraps
import numpy as np


def preserve_complexity_or_realness():
    def inner_func(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            is_real = not np.issubdtype(args[0].dtype, np.complexfloating)
            images = func(*args, **kwargs)
            if is_real:
                return images.real
            else:
                return images

        return wrapper

    return inner_func