from typing import Union
import numpy as np
import cupy as cp

ArrayType = Union[cp.ndarray, np.ndarray]
real_dtype = np.float32
complex_dtype = np.complex64
