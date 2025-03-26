import os
from typing import Optional
import numpy as np
from llama.io.loaders.base import StandardData
from llama.io.loaders.xrf.utils import load_xrf_experiment
from llama.io.loaders.utils import convert_projection_dict_to_array


def load_data_from_xrf_format(folder: str) -> dict[str, StandardData]:
    file_names = os.listdir(folder)  # Temporary
    xrf_standard_data_dict = load_xrf_experiment(folder, file_names)
    return xrf_standard_data_dict


def convert_xrf_projection_dicts_to_arrays(
    xrf_standard_data_dict: dict[int, StandardData],
    new_shape: Optional[tuple] = None,
    repair_orientation: bool = False,
    pad_mode: str = "constant",
    pad_with_mode: bool = False,
    chunk_length: int = 250,
    delete_projection_dict: bool = False,
) -> dict[str, np.ndarray]:
    # Prepare the input for the XRFObject class

    xrf_array_dict = {}
    for channel, standard_data in xrf_standard_data_dict.items():
        xrf_array_dict[channel] = convert_projection_dict_to_array(
            standard_data.projections,
            new_shape=new_shape,
            repair_orientation=repair_orientation,
            pad_mode=pad_mode,
            pad_with_mode=pad_with_mode,
            chunk_length=chunk_length,
            delete_projection_dict=delete_projection_dict,
        )

    return xrf_array_dict