import numpy as np
from pyxalign.api.types import r_type


def get_shift_from_different_resolution_alignment(
    reference_shift: np.ndarray,
    reference_scan_numbers: np.ndarray,
    reference_pixel_size: float,
    current_scan_numbers: np.ndarray,
    current_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    shared_scan_numbers = [scan for scan in current_scan_numbers if scan in reference_scan_numbers]
    # current_scan_numbers[idx_1] returns all scan numbers that are also in reference_scan_numbers
    idx_1 = [i for i, scan in enumerate(current_scan_numbers) if scan in reference_scan_numbers]
    # reference_scan_numbers[idx_2] returns all scan numbers that are also in current_scan_numbers
    idx_2 = [np.where(reference_scan_numbers == scan)[0][0]
             for i, scan in enumerate(current_scan_numbers) if scan in reference_scan_numbers]

    assert np.all(current_scan_numbers[idx_1] == reference_scan_numbers[idx_2])

    new_shift = np.zeros((len(current_scan_numbers), 2), dtype=r_type)
    new_shift[idx_1] = reference_shift[idx_2]
    # return shared_scan_numbers, new_shift[idx_1] * (reference_pixel_size / current_pixel_size)
    return shared_scan_numbers, new_shift * (reference_pixel_size / current_pixel_size)


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


if __name__ == "__main__":
    current_scan_nums = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    ref_scan_nums = np.array([3, 9, 2, 4, 5])
    lc, lr = len(current_scan_nums), len(ref_scan_nums)
    current_shift = np.random.rand(lc, 2)
    ref_shift = np.random.rand(lr, 2)
    get_shift_from_different_resolution_alignment(
        ref_shift, ref_scan_nums, 1, current_scan_nums, 1)
