import time
from pyxalign.gpu_utils import pin_memory, is_pinned
import numpy as np
import cupy as cp
import os

if __name__ == "__main__":
    input_array = np.random.rand(100)
    pinned_array = pin_memory(input_array)
    assert is_pinned(pinned_array) is True
    assert is_pinned(input_array) is False
    print(os.path.basename(__file__) + " PASSED")

    size = (750, 750, 750)
    # input_array = np.random.rand(750, 1184, 800).astype(np.float32) + 1j*np.random.rand(750, 1184, 800).astype(np.float32) 
    input_array = np.random.rand(*size).astype(np.float32) + 1j*np.random.rand(*size).astype(np.float32) 
    pinned_array = pin_memory(input_array)

    def time_move_to_gpu(array):
        t0 = time.time()
        cupy_array = cp.array(array)
        print(time.time() - t0)


    # time_move_to_gpu(input_array)
    # cp.get_default_memory_pool().free_all_blocks()
    # time_move_to_gpu(input_array)
    # cp.get_default_memory_pool().free_all_blocks()
    # time_move_to_gpu(pinned_array)
    # cp.get_default_memory_pool().free_all_blocks()
    # time_move_to_gpu(pinned_array)
    # cp.get_default_memory_pool().free_all_blocks()

    for i in range(5):
        time_move_to_gpu(input_array)
        # cp.get_default_memory_pool().free_all_blocks()
    for i in range(5):
        time_move_to_gpu(pinned_array)
        # cp.get_default_memory_pool().free_all_blocks()
