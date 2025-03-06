from typing import Union
import cupy as cp
import numpy as np
import scipy
import cupyx
from llama.api.options.options import PhaseUnwrapOptions
from llama.api.types import r_type, c_type, ArrayType
from llama.gpu_utils import memory_releasing_error_handler, get_scipy_module
from llama.timing.timer_utils import timer, InlineTimer


@memory_releasing_error_handler
@timer()
def unwrap_phase(
    images: ArrayType,
    weights: ArrayType,
    options: PhaseUnwrapOptions,
    empty_region=[],
) -> ArrayType:
    xp = cp.get_array_module(images)
    # Ensure the weights are all between 0 and 1
    weights[weights < 0] = 0
    weights = weights / weights.max()

    bool_weights = weights.astype(bool)

    phase_block = 0
    for i in range(options.iterations):
        if i == 0:
            images_resid = images
        else:
            images_resid = images * xp.exp(-1j * phase_block)
        phase_block = phase_block + weights * phase_unwrap_2D(images_resid, weights)
        # if empty_region != []:
        # raise NotImplementedError
        # phase_block = remove_sinogram_ramp(phase_block, empty_region, options.poly_fit_order)
        # Remove phase ramp
        if options.lsq_fit_ramp_removal:
            for j in range(len(phase_block)):
                phase_block[j] = remove_phase_ramp(phase_block[j], bool_weights[j])
    return phase_block


@timer()
def remove_phase_ramp(phase: ArrayType, mask: np.ndarray):
    """
    Removes the phase ramp from a 2D phase array, using only the masked region for estimation.

    Parameters:
        phase (np.ndarray): A 2D numpy array representing the phase (in radians).
        mask (np.ndarray): A 2D boolean numpy array (same shape as `phase`), where True indicates
                           the region to use for phase ramp estimation.

    Returns:
        np.ndarray: The phase-corrected 2D numpy array.
    """
    if phase.shape != mask.shape:
        raise ValueError("Phase and mask arrays must have the same shape.")

    xp = cp.get_array_module(phase)

    inline_timer = InlineTimer("meshgrid")
    inline_timer.start()
    ny, nx = phase.shape
    x, y = xp.meshgrid(xp.arange(nx), xp.arange(ny))
    inline_timer.end()

    inline_timer = InlineTimer("extract masked data")
    inline_timer.start()
    # Extract only masked data
    x_masked = x[mask]
    y_masked = y[mask]
    phase_masked = phase[mask]
    inline_timer.end()

    inline_timer = InlineTimer("get design matrix")
    inline_timer.start()
    # Construct the design matrix A for Ax = b (where A contains x, y, and constant terms)
    A = xp.column_stack((x_masked, y_masked, xp.ones_like(x_masked)))
    b = phase_masked
    inline_timer.end()

    inline_timer = InlineTimer("lsq fit")
    inline_timer.start()
    # Solve the least-squares problem using the normal equation: x = (A^T A)^(-1) A^T b
    inv_timer = InlineTimer("inverse")
    inv_timer.start()
    AtA_inv = cp.array(np.linalg.inv(A.get().T @ A.get()))  # (A^T A)^(-1)
    inv_timer.end()
    atb_timer = InlineTimer("atb")
    atb_timer.start()
    Atb = A.T @ b  # A^T b
    atb_timer.end()
    solve_timer = InlineTimer("solve")
    solve_timer.start()
    params_opt = AtA_inv @ Atb  # Solve for [a, b, c]
    solve_timer.end()
    inline_timer.end()

    inline_timer = InlineTimer("compute ramp over full grid")
    inline_timer.start()
    # Compute the phase ramp over the full grid
    phase_ramp = params_opt[0] * x + params_opt[1] * y + params_opt[2]
    inline_timer.end()

    # Remove the phase ramp
    return phase - phase_ramp


def phase_unwrap_2D(images: ArrayType, weights: ArrayType, padding: int = 64):
    xp = cp.get_array_module(images)

    images = weights * images / (xp.abs(images) + 1e-10)
    del weights

    pad_shape = xp.pad(images[0], padding, "symmetric").shape
    padded_images = cp.zeros((len(images), pad_shape[0], pad_shape[1]), dtype=c_type)
    if np.any(padding) > 0:
        for i in range(len(images)):
            padded_images[i] = xp.pad(images[i], padding, "symmetric")

    dX, dY = get_phase_gradient(padded_images)
    phase = xp.real(get_images_int_2D(dX, dY))

    start_idx_1, stop_idx_1 = padding, phase.shape[1] - padding
    start_idx_2, stop_idx_2 = padding - 1, phase.shape[2] - padding - 1

    return phase[:, start_idx_1:stop_idx_1, start_idx_2:stop_idx_2]


def get_phase_gradient(images: ArrayType):
    xp = cp.get_array_module(images)
    dX, dY = get_image_grad(images)
    dX = xp.imag(xp.conj(images) * dX)
    dY = xp.imag(xp.conj(images) * dY)

    return dX, dY


def get_image_grad(images: ArrayType):
    xp = cp.get_array_module(images)
    scipy_module: scipy = get_scipy_module(images)

    n_z, n_y, n_x = images.shape

    X = scipy_module.fft.ifftshift(xp.arange(-np.fix(n_x / 2), np.ceil(n_x / 2), dtype=c_type))
    X *= 2j * xp.pi / n_x
    dX = scipy_module.fft.fft(images, axis=2) * X
    dX = scipy_module.fft.ifft(dX, axis=2)

    Y = scipy_module.fft.ifftshift(xp.arange(-np.fix(n_y / 2), np.ceil(n_y / 2), dtype=c_type))
    Y *= 2j * xp.pi / n_y
    dY = scipy_module.fft.fft(images, axis=1) * Y[:, None]
    dY = scipy_module.fft.ifft(dY, axis=1)

    return dX, dY


def get_images_int_2D(dX: ArrayType, dY: ArrayType):
    xp = cp.get_array_module(dX)
    scipy_module: scipy = get_scipy_module(dX)

    n_z, n_y, n_x = dX.shape

    fD = scipy_module.fft.fft2(dX + 1j * dY, axes=(1, 2))
    x_grid = scipy_module.fft.ifftshift(xp.arange(-np.fix(n_x / 2), np.ceil(n_x / 2), dtype=r_type))
    x_grid /= n_x
    y_grid = scipy_module.fft.ifftshift(xp.arange(-np.fix(n_y / 2), np.ceil(n_y / 2), dtype=r_type))
    y_grid /= n_y

    X = xp.exp((2j * xp.pi) * x_grid + y_grid[:, None])
    # apply integration filter
    X /= 2j * xp.pi * (x_grid + 1j * y_grid[:, None])
    X[0, 0] = 0
    integral = fD * X
    integral = scipy_module.fft.ifft2(integral, axes=(1, 2))

    return integral
