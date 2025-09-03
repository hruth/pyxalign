import re
from typing import TypeVar
from tqdm import tqdm
import h5py
import os
import numpy as np

from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.xrf.options import XRFLoadOptions


def load_xrf_experiment(
    folder: str, file_names: str, options: XRFLoadOptions
) -> tuple[dict[str, StandardData], dict]:
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
    if np.all([a == 0 for a in angles]):
        print("WARNING: no angle data found; enter angle data manually.")

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


def remove_scans_from_dict(scan_file_dict: dict, scan_start: int, scan_end: int):
    if scan_start is None:
        scan_start = 0
    if scan_end is None:
        scan_end = np.max(list(scan_file_dict.keys()))
    return {k: v for k, v in scan_file_dict.items() if (k >= scan_start and k <= scan_end)}


def get_scan_file_dict(file_names: list[str], file_pattern: str) -> dict:  # -> list[int]:
    scan_file_dict = {}
    for name in file_names:
        scan_number = extract_scan_number(name, file_pattern)
        if scan_number is not None:
            scan_file_dict[scan_number] = name
    return scan_file_dict


def extract_scan_number(file_name: str, file_pattern: str) -> int:
    match = re.fullmatch(file_pattern, file_name)
    if match:
        return int(match.group(1))
    else:
        return None


def remove_inconsistent_sizes(standard_data: StandardData):
    # input is a dict across scan numbers

    # Get the shapes of the data taken at each scan number
    shapes = [v.shape for v in standard_data.projections.values()]
    # Get the count per each shape
    n_arrays_per_shape = []
    for shape in set(shapes):
        n_arrays_per_shape += [np.sum([shape == x for x in shapes])]
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


# Use V9 structure
def get_single_file_data(folder: str, file_name: str, options: XRFLoadOptions) -> tuple:
    file_path = os.path.join(folder, file_name)
    with h5py.File(file_path) as F:
        counts_per_second = F[options.channel_data_path][()]
        channel_names = F[options.channel_names_path][()]
        channel_names = [name.decode() for name in channel_names]
        counts_dict = {channel: counts for channel, counts in zip(
            channel_names, counts_per_second)}
        PVs = {
            k.decode(): v.decode()
            for k, v in zip(
                F["MAPS/Scan/Extra_PVs/"]["Names"][()], F["MAPS/Scan/Extra_PVs/"]["Values"][()]
            )
        }
        try:
            # Get angle if its found correctly
            angle = float(get_PV_value(PVs, options.angle_PV_string))
        except Exception:
            angle = 0
    return counts_dict, angle, PVs


def get_PV_value(PVs: dict, pv_name_string: str):
    if pv_name_string in PVs.keys():
        return PVs[pv_name_string]
    else:
        return None
