import re
import numpy as np


def load_xrf_experiment(folder: str):
    # - auto-extract file names from folder for each loader

    pass


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
    try:
        match = re.search(file_pattern, file_name)
        return int(match.group(1))
    except Exception:
        return None
    # return int(re.search(file_pattern, file_name).group(1))