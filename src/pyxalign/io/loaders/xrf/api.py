import os
from typing import Optional
import numpy as np
from pyxalign.io.loaders.base import StandardData
import pyxalign.io.loaders.xrf.options as xrf_options
from pyxalign.io.loaders.xrf.utils import load_xrf_experiment
from pyxalign.io.loaders.utils import convert_projection_dict_to_array


def load_data_from_xrf_format(
    options: xrf_options.XRFLoadOptions,
) -> tuple[dict[str, StandardData], dict]:
    """Function for loading XRF data and returning it in the standardized
    format.

    Args:
        options (pear.options.XRFLoadOptions): Configuration options
            for loading data.

    Returns:
        A tuple containing: 1) a dict of `StandardData` objects, where
        each key is a string specifying the channel and 2) a dict
        containg a dict of the extra PVs from the MDA file.

    Example:
        Load XRF data taken at beamline 2-ID-E from the folder
        "/my/mda/data/folder/"::

            base_load_options = xrf_options.XRFBaseLoadOptions(
                "my/mda/data/folder/"
            )
            load_options = xrf_options.XRF2IDELoadOptions(base=base_load_options)
            xrf_standard_data_dict, extra_PVs = load_data_from_xrf_format(
                load_options
            )

        Once the data is loaded, you can create an XRFTask::

            # Create 3D arrays for each channel
            xrf_array_dict = (
                pyxalign.io.loaders.convert_xrf_projection_dicts_to_arrays(
                    xrf_standard_data_dict, pad_with_mode=True
                )
            )

            # specify the laminography angle, pixel size, and estimated
            # sample thickness
            projection_options = pyxalign.options.ProjectionOptions()
            projection_options.experiment.laminography_angle = 60 # degrees
            projection_options.experiment.pixel_size = 100e-9 # meters
            projection_options.experiment.sample_thickness = 1e-5 # meters

            # Insert data into an XRFTask object
            primary_channel = "Ti"  # select the channel to be used for alignment
            xrf_task = XRFTask(
                xrf_array_dict=xrf_array_dict,
                angles=xrf_standard_data_dict[primary_channel].angles,
                scan_numbers=xrf_standard_data_dict[primary_channel].scan_numbers,
                primary_channel=primary_channel,
                projection_options=projection_options,
            )
    """
    file_names = os.listdir(options.base.folder)  # Temporary?
    xrf_standard_data_dict, extra_PVs = load_xrf_experiment(file_names, options)
    return xrf_standard_data_dict, extra_PVs


def convert_xrf_projection_dicts_to_arrays(
    xrf_standard_data_dict: dict[str, StandardData],
    new_shape: Optional[tuple] = None,
    pad_mode: str = "constant",
    pad_with_mode: bool = False,
) -> dict[str, np.ndarray]:
    """Function that creates a 3D array for each XRF channel in
    xrf_standard_data_dict. Each array will be padded (or cropped) so
    that its new dimensions are equal to `new_shape`.

    Args:
        xrf_standard_data_dict (dict[str, StandardData]): standard
            formatted XRF input data.
        new_shape (Optional[tuple]): Shape of the new 3D arrays. If not
            specified, the new size will be chosen automatically.
        pad_mode (str): pad mode used by the `np.pad` function.
        pad_with_mode (bool): if `True`, the 2D arrays will be padded by
            their mode. Setting this to `True` forces the `pad_mode` to
            be `constant`.

    Returns:
        A dictionary containg the 3D projection array for each channel.
    """

    xrf_array_dict = {}
    for channel, standard_data in xrf_standard_data_dict.items():
        xrf_array_dict[channel] = convert_projection_dict_to_array(
            standard_data.projections,
            new_shape=new_shape,
            pad_mode=pad_mode,
            pad_with_mode=pad_with_mode,
        )

    return xrf_array_dict
