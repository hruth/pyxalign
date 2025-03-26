import re
from typing import TypeVar
from tqdm import tqdm
import h5py
import os
import numpy as np

from llama.io.loaders.base import StandardData

T = TypeVar("T", bound=dict[int, dict[str, np.ndarray]])


def load_xrf_experiment(folder: str, file_names: str) -> T:
    all_counts_dict = {}
    angles = []
    for name in tqdm(file_names):
        scan_number = int(re.search(r"2xfm_(\d+)\.mda.h5", name).group(1))
        counts_dict, angle = get_single_file_data(folder, name)
        all_counts_dict[scan_number] = counts_dict
        angles += [angle]
    channels = all_counts_dict[scan_number].keys()
    scan_numbers = np.array(list(all_counts_dict.keys()))
    angles = np.array(angles)
    # Make StandardData object for each xrf projection
    channel_data_objects = {}
    for channel in channels:
        channel_data_objects[channel] = StandardData(
            projections={scan_num: v[channel] for scan_num, v in all_counts_dict.items()},
            angles=angles * 1,
            scan_numbers=scan_numbers * 1,
        )
        # Drop consistent sizes for each channel
        remove_inconsistent_sizes(channel_data_objects[channel])
    return channel_data_objects


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


# Use V9 structure
def get_single_file_data(folder: str, file_name: str) -> dict[str, np.ndarray]:
    file_path = os.path.join(folder, file_name)
    with h5py.File(file_path) as F:
        counts_per_second = F["/MAPS/XRF_fits"][()]
        channel_names = F["/MAPS/channel_names"][()]
        channel_names = [name.decode() for name in channel_names]
        counts_dict = {channel: counts for channel, counts in zip(channel_names, counts_per_second)}
        # Get angle

        PVs = {
            k.decode(): v.decode()
            for k, v in zip(
                F["MAPS/Scan/Extra_PVs/"]["Names"][()], F["MAPS/Scan/Extra_PVs/"]["Values"][()]
            )
        }
        angle = float(PVs["2xfm:m58.VAL"])
    return counts_dict, angle