from types import ModuleType
from typing import Union
import numpy as np
import cupy as cp
import scipy

from llama.gpu_utils import get_scipy_module
from llama.transformations.functions import image_shift_fft

from llama.api.types import ArrayType, r_type, c_type


def filtered_fft(image: ArrayType, shift: ArrayType, filter_data: float) -> ArrayType:
    # Rename filter_data to something better, maybe low_freq_cutoff
    xp = cp.get_array_module(image)
    scipy_module: scipy = get_scipy_module(image)

    [nx, ny] = image.shape[1:3]

    image = image_shift_fft(
        image, shift
    )  # Should be moved into the cross-correlation function

    spatial_filter = xp.array(
        scipy.signal.windows.tukey(nx, 0.3)[:, None] * scipy.signal.windows.tukey(ny, 0.3)[None, :],
        dtype=r_type,
    )
    image = image - xp.mean(image)
    image = image * spatial_filter
    image = scipy_module.fft.fft2(image)

    # Remove low frequencies (e.g. phase ramp issues)
    if filter_data > 0:
        X, Y = xp.meshgrid(
            xp.arange(-nx / 2, nx / 2, dtype=r_type),
            xp.arange(-ny / 2, ny / 2, dtype=r_type),
        )
        spectral_filter = xp.exp(
            -((0.5 * (nx + ny) * filter_data) ** 2) / (X**2 + Y**2 + 1e-10)
        )
        spectral_filter = scipy_module.fft.fftshift(spectral_filter).transpose()
        image = image * spectral_filter

    return image


def get_cross_correlation_shift(image: ArrayType, image_ref: ArrayType) -> ArrayType:
    "Fast subpixel cross correlation"
    xp = cp.get_array_module(image)
    scipy_module: scipy = get_scipy_module(image)

    cross_corr_matrix = scipy_module.fft.fftshift(
        xp.abs(scipy_module.fft.ifft2(image * xp.conj(image_ref))), (1, 2)
    )

    # Get a mask for a small region around the maximum
    kernel_width = 5
    mask = cross_corr_matrix == np.max(cross_corr_matrix, axis=(1, 2))[:, None, None]
    mask = scipy_module.signal.fftconvolve(
        mask,
        xp.ones([kernel_width, kernel_width], dtype=r_type)[None],
        "same",
    )
    mask = mask > 0.1

    cross_corr_matrix[~mask] = np.inf
    cross_corr_matrix = (
        cross_corr_matrix - np.min(cross_corr_matrix, (1, 2))[:, None, None]
    )
    cross_corr_matrix[cross_corr_matrix < 0] = 0
    cross_corr_matrix[~mask] = 0
    cross_corr_matrix = (
        cross_corr_matrix / np.max(cross_corr_matrix, axis=(1, 2))[:, None, None]
    ) ** 2
    # Get center of mass of the central peak
    cross_corr_matrix = cross_corr_matrix - 0.5
    cross_corr_matrix[cross_corr_matrix < 0] = 0
    cross_corr_matrix = cross_corr_matrix**2

    def find_center_fast(cross_corr_matrix):
        mass = np.sum(cross_corr_matrix, (1, 2))
        N, M = cross_corr_matrix.shape[1:3]
        x = xp.sum(
            xp.sum(cross_corr_matrix, 1) * xp.arange(0, M, dtype=r_type), 1
        ) / mass - np.floor(M / 2)
        y = xp.sum(
            xp.sum(cross_corr_matrix, 2) * xp.arange(0, N, dtype=r_type), 1
        ) / mass - np.floor(N / 2)

        return x, y, mass

    x, y, mass = find_center_fast(cross_corr_matrix)
    relative_shifts = xp.array([x, y]).transpose()
    # if xp == cp:
    # relative_shifts = relative_shifts.get()

    return relative_shifts


def apply_3D_apodization(image: ArrayType, rad_apod: float, radial_smooth: float):
    n_z = image.shape[0]
    xt = np.arange(-n_z / 2, n_z / 2, dtype=r_type)
    X, Y = np.meshgrid(xt, xt)
    if len(np.shape(radial_smooth)) > 0:
        radial_smooth[radial_smooth < 1] = 1
    circulo = 1 - rad_tap(X, Y, radial_smooth, np.round(n_z / 2 - rad_apod - radial_smooth))
    image = image * circulo[:, :, None]

    return circulo


def rad_tap(X, Y, tap_pix, zero_rad):
    tau = 2 * tap_pix
    R = np.sqrt(X**2 + Y**2)
    taper_func = 0.5 * (1 + np.cos(2 * np.pi * (R - zero_rad - tau / 2) / tau))
    taper_func = (R > zero_rad + tau / 2) * 1.0 + taper_func * (R <= zero_rad + tau / 2)
    taper_func = taper_func * (R >= zero_rad)

    return taper_func


def get_tukey_window(
    size: Union[list, np.ndarray, tuple], A: float, xp: ModuleType = np
) -> ArrayType:
    tukey_window = (
        scipy.signal.windows.tukey(size[0], A)[:, None] * scipy.signal.windows.tukey(size[1], A)
    ).astype(r_type)

    tukey_window = xp.array(tukey_window)

    return tukey_window