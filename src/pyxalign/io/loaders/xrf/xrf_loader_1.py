import numpy as np
import os
import h5py
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.xrf.options import XRFV1LoadOptions
from pyxalign.io.loaders.xrf.utils import get_scan_file_dict, remove_scans_from_dict


def load_xrf_experiment_v1(
    folder: str, options: XRFV1LoadOptions
) -> tuple[dict[str, StandardData], dict]:
    file_names = os.listdir(folder)  # Temporary?
    all_counts_dict = {}
    angles = []
    extra_PVs_dict = {}
    scan_file_dict = get_scan_file_dict(file_names, options.file_pattern)
    scan_file_dict = remove_scans_from_dict(scan_file_dict, options.scan_start, options.scan_end)

    # Load data from each file
    for scan_number, file_name in scan_file_dict.items():
        counts_dict, angle, extra_PVs = get_single_file_data(folder, file_name, options)
        all_counts_dict[scan_number] = counts_dict
        angles += [angle]
        extra_PVs_dict[scan_number] = extra_PVs

    # Make StandardData object for each xrf projection
    channels = all_counts_dict[scan_number].keys()
    scan_numbers = np.array(list(all_counts_dict.keys()))
    angles = np.array(angles)
    channel_data_objects = {}
    for channel in channels:
        channel_data_objects[channel] = StandardData(
            projections={scan_num: v[channel] for scan_num, v in all_counts_dict.items()},
            angles=angles * 1,
            scan_numbers=scan_numbers * 1,
        )
        # Drop inconsistent sizes for each channel
        remove_inconsistent_sizes(channel_data_objects[channel])
    return channel_data_objects, extra_PVs_dict


def remove_inconsistent_sizes(standard_data: StandardData):
    # input is a dict across scan numbers

    # Get the shapes of the data taken at each scan number
    shapes = [v.shape for v in standard_data.projections.values()]
    # Get the count per each shape
    n_arrays_per_shape = []
    for shape in set(shapes):
        n_arrays_per_shape += [shape == x for x in shapes]
    idx = np.argmax(n_arrays_per_shape)
    # Remove data with sizes that don't match
    scan_numbers = np.array(list(standard_data.projections.keys()), dtype=int)
    most_common_shape = shapes[idx]
    idx_keep = [x == most_common_shape for x in shapes]
    idx_remove = [not x for x in idx_keep]
    for scan in scan_numbers[idx_remove]:
        del standard_data.projections[scan]
    standard_data.angles = standard_data.angles[idx_keep]
    standard_data.scan_numbers = standard_data.scan_numbers[idx_keep]


def get_PV_value(PVs: dict, pv_name_string: str):
    # Use V9 structure
    if pv_name_string in PVs.keys():
        return PVs[pv_name_string]
    else:
        return None


def get_single_file_data(folder: str, file_name: str, options: XRFV1LoadOptions) -> tuple:
    # ) -> tuple(dict[str, np.ndarray], float, float):
    file_path = os.path.join(folder, file_name)
    with h5py.File(file_path) as F:
        counts_per_second = F[options.channel_data_path][()]
        channel_names = F[options.channel_names_path][()]
        channel_names = [name.decode() for name in channel_names]
        counts_dict = {channel: counts for channel, counts in zip(channel_names, counts_per_second)}
        # Get angle
        PVs = {
            k.decode(): v.decode()
            for k, v in zip(
                F["MAPS/Scan/Extra_PVs/"]["Names"][()], F["MAPS/Scan/Extra_PVs/"]["Values"][()]
            )
        }
        angle = float(get_PV_value(PVs, options.angle_PV_string))
        # lamino_angle = float(get_PV_value(PVs, options.lamino_angle_PV_string))

    # tomo rotation: 2xfm:m60.DESC
    return counts_dict, angle, PVs
