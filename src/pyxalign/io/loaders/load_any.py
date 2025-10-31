from typing import Union

import pyxalign.io.loaders.pear.options as pear_options
import pyxalign.io.loaders.xrf.options as xrf_options
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.xrf.api import load_data_from_xrf_format
from pyxalign.io.utils import OptionsClass


def load_dataset_from_arbitrary_options(
    load_options: OptionsClass, n_processes: int = 1
) -> Union[StandardData, tuple[dict[str, StandardData], dict]]:
    if isinstance(load_options, pear_options.PEARLoadOptions):
        loaded_data = load_data_from_pear_format(
            options=load_options,
            n_processes=n_processes,
        )
    elif isinstance(load_options, xrf_options.XRF2IDELoadOptions):
        loaded_data = load_data_from_xrf_format(options=load_options)
    return loaded_data