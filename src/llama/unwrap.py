import cupy as cp
import numpy as np
import scipy
from llama.api.options.options import PhaseUnwrapOptions
from llama.api.types import r_type, c_type, ArrayType
from llama.gpu_utils import memory_releasing_error_handler, get_scipy_module


@memory_releasing_error_handler
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

    phase_block = 0
    for i in range(options.iterations):
        if i == 0:
            images_resid = images
        else:
            images_resid = images * xp.exp(-1j * phase_block)
        phase_block = phase_block + weights * phase_unwrap_2D(images_resid, weights)
        if empty_region != []:
            raise NotImplementedError
            phase_block = remove_sinogram_ramp(phase_block, empty_region, options.poly_fit_order)
    return phase_block


def remove_sinogram_ramp():
    pass


def phase_unwrap_2D(images: ArrayType, weights: ArrayType):
    xp = cp.get_array_module(images)

    images = weights * images / (xp.abs(images) + 1e-10)
    del weights

    padding = 64
    padShape = xp.pad(images[0], padding, "symmetric").shape
    padded_images = cp.zeros((len(images), padShape[0], padShape[1]), dtype=c_type)
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
    dX = xp.imag(xp.conj(images)*dX)
    dY = xp.imag(xp.conj(images)*dY)

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

