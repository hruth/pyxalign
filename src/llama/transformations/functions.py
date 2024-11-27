from typing import Optional
import numpy as np
import cupy as cp
import scipy

# import functools.partial as partial
import functools
from llama.gpu_utils import get_fft_backend, get_scipy_module
from llama.transformations.helpers import preserve_complexity_or_realness

from llama.api.types import ArrayType, r_type


def image_crop(
    images: ArrayType,
    horizontal_range: int,
    vertical_range: int,
    horizontal_offset: int = 0,
    vertical_offset: int = 0,
) -> ArrayType:
    """Returns a view of the specified region of the image."""

    if len(images.shape) == 3:
        image_dims = images.shape[1:]
    else:
        image_dims = images.shape

    image_center = np.array(image_dims) / 2 + [vertical_offset, horizontal_offset]
    vertical_index_start, vertical_index_end = (
        int(image_center[0] - vertical_range / 2),
        int(image_center[0] + vertical_range / 2),
    )
    horizontal_index_start, horizontal_index_end = (
        int(image_center[1] - horizontal_range / 2),
        int(image_center[1] + horizontal_range / 2),
    )
    if (
        horizontal_index_start < 0
        or horizontal_index_end >= image_dims[1]
        or vertical_index_start < 0
        or vertical_index_end >= image_dims[0]
    ):
        raise ValueError("Invalid values entered for cropping.")

    if len(images.shape) == 3:
        return images[
            :,
            vertical_index_start:vertical_index_end,
            horizontal_index_start:horizontal_index_end,
        ]
    else:
        return images[
            vertical_index_start:vertical_index_end,
            horizontal_index_start:horizontal_index_end,
        ]


def image_crop_pad(
    images: ArrayType, new_extent_y: int, new_extent_x: int, pad_mode: str = "constant"
):
    [new_extent_y, new_extent_x] = images.shape[1:]

    # Crop
    if new_extent_y > new_extent_y:
        w = new_extent_y - new_extent_y
        a, b = int(w / 2), int(new_extent_y - w / 2)
        images = images[:, a:b]
    if new_extent_x > new_extent_x:
        w = new_extent_x - new_extent_x
        a, b = int(w / 2), int(new_extent_x - w / 2)
        images = images[:, :, a:b]

    # Pad
    if new_extent_y < new_extent_y:
        w = new_extent_y - new_extent_y
        pad_width = ((0, 0), (int(np.ceil(w / 2)), int(np.floor(w / 2))), (0, 0))
        images = np.pad(images, pad_width, mode=pad_mode)
    if new_extent_x < new_extent_x:
        w = new_extent_x - new_extent_x
        pad_width = ((0, 0), (0, 0), (int(np.ceil(w / 2)), int(np.floor(w / 2))))
        images = np.pad(images, pad_width, mode=pad_mode)

    return images


def image_shift_fft(images: ArrayType, shift: ArrayType) -> ArrayType:
    xp = cp.get_array_module(images)
    scipy_module: scipy = get_scipy_module(images)
    is_real = not xp.issubdtype(images.dtype, xp.complexfloating)

    x = shift[:, 0][:, None]
    y = shift[:, 1][:, None]

    shape = images.shape

    images = scipy_module.fft.fft2(images)

    x_grid = (
        scipy_module.fft.ifftshift(
            xp.arange(-np.fix(shape[2] / 2), np.ceil(shape[2] / 2), dtype=r_type)
        )
        / shape[2]
    )
    X = (x * x_grid)[:, None, :]
    X = xp.exp(-2j * xp.pi * X)

    y_grid = (
        scipy_module.fft.ifftshift(
            xp.arange(-np.fix(shape[1] / 2), np.ceil(shape[1] / 2), dtype=r_type)
        )
        / shape[1]
    )
    Y = (y * y_grid)[:, :, None]
    Y = xp.exp(-2j * xp.pi * Y)

    images = images * X
    images = images * Y

    images = scipy_module.fft.ifft2(images)

    if is_real:
        images = xp.real(images)

    return images


def image_shift_circ(images: ArrayType, shift: ArrayType) -> ArrayType:
    xp = cp.get_array_module(images)

    x = shift[:, 0]
    y = shift[:, 1]

    n_z, n_x, n_y = images.shape
    X = xp.arange(0, n_x, dtype=int)
    Y = xp.arange(0, n_y, dtype=int)

    # Shift the image
    for proj_idx in range(n_z):
        shift_1 = int(np.round(y[proj_idx]))
        shift_2 = int(np.round(x[proj_idx]))
        idx_1 = xp.roll(X, shift_1)
        idx_2 = xp.roll(Y, shift_2)
        images[proj_idx] = images[proj_idx, idx_1[:, None], idx_2[None]]

    return images


def image_shift_linear(images: ArrayType, shift: Optional[ArrayType] = None) -> ArrayType:
    return functools.partial(image_downsample_linear, scale=1)


@preserve_complexity_or_realness()
def image_downsample_fft(images: ArrayType, scale: int) -> ArrayType:
    xp = cp.get_array_module(images)
    fft_backend = get_fft_backend(images)

    # Pad the array to prevent boundary issues
    pad_by = 2
    image_size = xp.array(images.shape, dtype=int)[1:]
    image_size_new = xp.round(xp.ceil(image_size / scale / 2) * 2) + pad_by
    scale = xp.prod(image_size_new - pad_by) / xp.prod(image_size)
    downsample = int(xp.ceil(xp.sqrt(1 / scale)))
    pad_width = int(downsample * pad_by / 2)
    pad_shape = xp.pad(images[0], pad_width, "symmetric").shape
    padded_images = xp.zeros((len(images), pad_shape[0], pad_shape[1]), dtype=images.dtype)
    for i in range(len(images)):
        padded_images[i] = xp.pad(images[i], pad_width, "symmetric")
    images = padded_images
    del padded_images

    # Downsample the image
    with scipy.fft.set_backend(fft_backend):
        images = scipy.fft.fft2(images)
        # apply +/-0.5 px shift
        images = image_shift_fft(images, xp.array([[-0.5, -0.5]], dtype=r_type), applyFFT=False)
        # crop in the Fourier space
        images = scipy.fft.ifftshift(
            image_crop_pad(
                scipy.fft.fftshift(images, axes=(1, 2)),
                image_size_new[0],
                image_size_new[1],
            ),
            axes=(1, 2),
        )
        # apply -/+0.5 px shift in the cropped space
        images = image_shift_fft(images, xp.array([[0.5, 0.5]]), applyFFT=False)
        images = scipy.fft.ifft2(images)
        # scale to keep the average constant
        images = images * scale
        # remove the padding
        a = int(pad_by / 2)
        images = images[:, a : (image_size_new[0] - a), a : (image_size_new[1] - a)]

    return images


def image_downsample_linear(
    images: ArrayType, scale: int, shift: Optional[ArrayType] = None
) -> ArrayType:
    # Note: this function also is used to shift the data if the scale is set to 0.
    # This function should not be used with complex data
    # If memory serves, parallelizing this on the gpus doesn't improve speed
    xp = cp.get_array_module(images)
    scipy_module = get_scipy_module(images)
    interpolator = scipy_module.interpolate.RegularGridInterpolator

    n_z, n_x, n_y = images.shape
    X = xp.arange(0, n_x, dtype=int)
    Y = xp.arange(0, n_y, dtype=int)

    if scale != 1:
        new_n_x = int(round(n_x / scale))
        new_n_y = int(round(n_y / scale))
        new_images = xp.zeros((n_z, new_n_x, new_n_y), dtype=images.dtype)
    else:
        new_n_x = n_x
        new_n_y = n_y
        new_images = images
    # Create the interpolation function
    x0 = xp.arange(0, n_x, dtype=images.dtype)
    y0 = xp.arange(0, n_y, dtype=images.dtype)
    z0 = xp.arange(0, n_z, dtype=images.dtype)
    interp_function = interpolator(
        (z0, x0, y0),
        images,
        bounds_error=False,
        fill_value=0,  # fill_value=images.dtype(0)
    )
    # Define the new coordinates
    x0 = np.linspace(x0[0], x0[-1], new_n_x, dtype=images.dtype)
    y0 = np.linspace(y0[0], y0[-1], new_n_y, dtype=images.dtype)

    Z, X, Y = xp.meshgrid(z0, x0, y0, indexing="ij")
    X = X + xp.array(-shift[:, 1], dtype=images.dtype, ndmin=3).transpose([2, 0, 1])
    Y = Y + xp.array(-shift[:, 0], dtype=images.dtype, ndmin=3).transpose([2, 0, 1])

    # Get the interpolated function at the new coordinates
    # Would be better to find a way to do this that doesn't require
    # recasting the float64 to float32!
    new_images[:] = interp_function((Z, X, Y))
    return new_images


def image_downsample_nearest(images: ArrayType, scale: int) -> ArrayType:
    return images[:, ::scale, ::scale]


def image_upsample_nearest(images: ArrayType, scale: int) -> ArrayType:
    return images.repeat(scale, axis=1).repeat(scale, axis=2)
