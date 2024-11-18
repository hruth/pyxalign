from llama.gpu_utils import pin_memory, is_pinned
import numpy as np
import cupy as cp
import os

if __name__ == "__main__":
    input_array = np.random.rand(100)
    pinned_array = pin_memory(input_array)
    assert is_pinned(pinned_array) is True
    assert is_pinned(input_array) is False
    print(os.path.basename(__file__) + " PASSED")
