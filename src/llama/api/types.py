from typing import Union
import numpy as np
import cupy as cp

ArrayType = Union[cp.ndarray, np.ndarray]
r_type = np.float32
c_type = np.complex64
