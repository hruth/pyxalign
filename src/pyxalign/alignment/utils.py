import numpy as np
from pyxalign.api.types import r_type


def get_shift_from_different_resolution_alignment(
    reference_shift: np.ndarray,
    reference_scan_numbers: np.ndarray,
    reference_pixel_size: float,
    current_scan_numbers: np.ndarray,
    current_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    new_scan_numbers = [scan for scan in current_scan_numbers if scan in reference_scan_numbers]
    idx = [i for i, scan in enumerate(reference_scan_numbers) if scan in new_scan_numbers]
    new_shift = reference_shift[idx] * (reference_pixel_size / current_pixel_size)

    return new_scan_numbers, new_shift.astype(r_type)


def get_center_of_rotation_from_different_resolution_alignment(
    reference_shape: np.ndarray,
    reference_center_of_rotation: np.ndarray,
    current_shape: np.ndarray,
    reference_pixel_size: float,
    current_pixel_size: float,
) -> np.ndarray:
    center_of_reference_array = np.array(reference_shape, dtype=int) / 2
    center_of_current_array = np.array(current_shape, dtype=int) / 2
    reference_offset = reference_center_of_rotation - center_of_reference_array
    new_center = center_of_current_array + reference_offset * (
        reference_pixel_size / current_pixel_size
    )
    return new_center.astype(r_type)
