from typing import Optional
import numpy as np
import cupy as cp
import scipy

# import functools.partial as partial
import functools
from llama.timing.timer_utils import timer
from llama.gpu_utils import get_fft_backend, get_scipy_module
from llama.transformations.helpers import preserve_complexity_or_realness
from llama.timing.timer_utils import timer

from llama.api.types import ArrayType, r_type, c_type


@timer()
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


@timer()
def image_crop_pad(
    images: ArrayType,
    new_extent_y: int,
    new_extent_x: int,
    pad_mode: str = "constant",
    constant_values=None,
):
    if len(images.shape) == 2:
        added_extra_dim = True
        images = images[None]
    else:
        added_extra_dim = False

    [extent_y, extent_x] = images.shape[1:]

    # Crop
    if extent_y > new_extent_y:
        a, b = (
            int(extent_y / 2 - new_extent_y / 2),
            int(extent_y / 2 + new_extent_y / 2),
        )
        images = images[:, a:b]
    if extent_x > new_extent_x:
        a, b = (
            int(extent_x / 2 - new_extent_x / 2),
            int(extent_x / 2 + new_extent_x / 2),
        )
        images = images[:, :, a:b]

    # Pad
    if extent_y < new_extent_y:
        w = new_extent_y - extent_y
        pad_width = ((0, 0), (int(np.ceil(w / 2)), int(np.floor(w / 2))), (0, 0))
        images = np.pad(images, pad_width, mode=pad_mode, constant_values=constant_values)
    if extent_x < new_extent_x:
        w = new_extent_x - extent_x
        pad_width = ((0, 0), (0, 0), (int(np.ceil(w / 2)), int(np.floor(w / 2))))
        images = np.pad(images, pad_width, mode=pad_mode, constant_values=constant_values)

    if added_extra_dim:
        images = images[0]

    return images


@timer()
@preserve_complexity_or_realness()
def image_shift_fft(images: ArrayType, shift: ArrayType, apply_FFT: bool = True) -> ArrayType:
    xp = cp.get_array_module(images)
    scipy_module: scipy = get_scipy_module(images)

    x = shift[:, 0][:, None]
    y = shift[:, 1][:, None]

    shape = images.shape

    if apply_FFT:
        images = scipy_module.fft.fft2(images)

    x_grid = xp.arange(-np.fix(shape[2] / 2), np.ceil(shape[2] / 2), dtype=r_type)
    x_grid = scipy_module.fft.ifftshift(x_grid) / shape[2]
    X = (x * x_grid)[:, None, :]
    X = xp.exp(-2j * np.pi * X)

    y_grid = xp.arange(-np.fix(shape[1] / 2), np.ceil(shape[1] / 2), dtype=r_type)
    y_grid = scipy_module.fft.ifftshift(y_grid) / shape[1]
    Y = (y * y_grid)[:, :, None]
    Y = xp.exp(-2j * xp.pi * Y)

    images = images * X
    images = images * Y

    if apply_FFT:
        images = scipy_module.fft.ifft2(images)

    return images


@timer()
def image_shift_circ(images: ArrayType, shift: ArrayType, in_place=False) -> ArrayType:
    xp = cp.get_array_module(images)

    if not in_place:
        images = images * 1

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


@timer()
def image_shift_linear(images: ArrayType, shift: ArrayType) -> ArrayType:
    return image_downsample_linear(images, 1, shift)


@timer()
def apply_gaussian_filter(images: ArrayType, scale: int) -> ArrayType:
    xp = cp.get_array_module(images)
    scipy_module: scipy = get_scipy_module(images)

    # May not always be necessary
    images = images * 1

    gaussian_filter = scipy_module.ndimage.gaussian_filter
    for i in range(len(images)):
        # in place replacement is required here for matching the old
        # version, but it might be more correct to make it not in-place
        # later
        images[i] = gaussian_filter(images[i], scale)

    images = images / gaussian_filter(xp.ones(images.shape[1:], dtype=r_type), scale)

    return images


@timer()
@preserve_complexity_or_realness()
def image_downsample_fft(images: ArrayType, scale: int, use_gaussian_filter=False) -> ArrayType:
    xp = cp.get_array_module(images)
    scipy_module: scipy = get_scipy_module(images)
    interp_sign = -1

    if use_gaussian_filter:
        images = apply_gaussian_filter(images, scale)
    image_size = xp.array(images.shape, dtype=int)[1:]
    pad_by = 2
    image_size_new = (xp.round(xp.ceil(image_size / scale / 2) * 2) + pad_by).astype(int)
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
    images = scipy_module.fft.fft2(images)
    # apply +/-0.5 px shift
    images = image_shift_fft(
        images, interp_sign * xp.array([[-0.5, -0.5]], dtype=r_type), apply_FFT=False
    )
    # crop in the Fourier space
    images = scipy_module.fft.ifftshift(
        image_crop_pad(
            scipy.fft.fftshift(images, axes=(1, 2)),
            image_size_new[0],
            image_size_new[1],
        ),
        axes=(1, 2),
    )
    # apply -/+0.5 px shift in the cropped space
    images = image_shift_fft(
        images, interp_sign * xp.array([[0.5, 0.5]], dtype=r_type), apply_FFT=False
    )
    images = scipy_module.fft.ifft2(images)
    # scale to keep the average constant
    images = images * scale.astype(r_type)
    # remove the padding
    a = int(pad_by / 2)
    images = images[:, a : (image_size_new[0] - a), a : (image_size_new[1] - a)]

    return images


@timer()
def image_downsample_linear(
    images: ArrayType,
    scale: int,
    shift: Optional[ArrayType] = None,
    use_gaussian_filter: bool = False,
) -> ArrayType:
    # Note: this function also is used to shift the data if the scale is set to 0.
    # This function should not be used with complex data
    # If memory serves, parallelizing this on the gpus doesn't improve speed
    xp = cp.get_array_module(images)
    scipy_module = get_scipy_module(images)
    interpolator = scipy_module.interpolate.RegularGridInterpolator

    if shift is None:
        shift = xp.zeros((len(images), 2), dtype=r_type)

    if use_gaussian_filter:
        images = apply_gaussian_filter(images, scale)

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


@timer()
def image_downsample_nearest(
    images: ArrayType, scale: int, use_gaussian_filter: bool = False
) -> ArrayType:
    if use_gaussian_filter:
        images = apply_gaussian_filter(images, scale)
    return images[:, ::scale, ::scale]


@timer()
def image_upsample_nearest(images: ArrayType, scale: int) -> ArrayType:
    return images.repeat(scale, axis=1).repeat(scale, axis=2)


def will_rotation_flip_aspect_ratio(theta: float) -> bool:
    n_rotations = round(theta / 90)
    is_even = n_rotations % 2 == 0
    return not is_even


def rotateStackMod90(img, theta) -> ArrayType:
    xp = cp.get_array_module(img)

    numRotations = round(theta / 90)
    img = xp.rot90(img, numRotations, axes=(1, 2))
    theta = theta - 90 * numRotations

    return img


@timer()
@preserve_complexity_or_realness()
def image_rotate_fft(
    images: ArrayType, theta: float, preserve_aspect_ratio: bool = False
) -> ArrayType:
    """Rotates the image around the z-axis (0th axis) of the input images"""
    xp = cp.get_array_module(images)
    scipy_module = get_scipy_module(images)

    if not preserve_aspect_ratio:
        # imageStack = lam.utils.rotateStackMod90(imageStack, tiltAngle)
        images = rotateStackMod90(images, theta)
        n_rotations = round(theta / 90)
        theta = theta - 90 * n_rotations

    M, N = images.shape[1:]

    # x_grid = xp.matrix(xp.arange(-np.fix(M / 2), np.ceil(M / 2), dtype=r_type)).transpose()
    x_grid = xp.arange(-np.fix(M / 2), np.ceil(M / 2), dtype=r_type)[:, None]
    x_grid = scipy_module.fft.ifftshift(x_grid) / M
    # y_grid = xp.matrix(xp.arange(-np.fix(N / 2), np.ceil(N / 2), dtype=r_type))
    y_grid = xp.arange(-np.fix(N / 2), np.ceil(N / 2), dtype=r_type)[None, :]
    y_grid = scipy_module.fft.ifftshift(y_grid) / N

    # m_grid = xp.matrix(xp.arange(1, M + 1, dtype=r_type)).transpose() - xp.floor(M / 2) - 0.5
    # n_grid = xp.matrix(xp.arange(1, N + 1, dtype=r_type)) - xp.floor(N / 2) - 0.5
    m_grid = xp.arange(1, M + 1, dtype=r_type)[:, None] - xp.floor(M / 2) - 0.5
    n_grid = xp.arange(1, N + 1, dtype=r_type)[None, :] - xp.floor(N / 2) - 0.5

    n_x = -xp.sin(theta * xp.pi / 180) * x_grid
    n_y = xp.tan(theta / 2 * xp.pi / 180) * y_grid

    m_1 = xp.array(xp.exp(-2j * xp.pi * m_grid * n_y)).astype(c_type)
    m_2 = xp.array(xp.exp(-2j * xp.pi * xp.multiply(n_grid, n_x))).astype(c_type)

    images = scipy_module.fft.ifft(xp.multiply(scipy_module.fft.fft(images, axis=2), m_1), axis=2)
    images = scipy_module.fft.ifft(xp.multiply(scipy_module.fft.fft(images, axis=1), m_2), axis=1)
    images = scipy_module.fft.ifft(xp.multiply(scipy_module.fft.fft(images, axis=2), m_1), axis=2)

    return images


@timer()
@preserve_complexity_or_realness()
def image_shear_fft(images: ArrayType, theta: float) -> ArrayType:
    """Shears the image about the z-axis (0th axis) of the input images"""
    xp = cp.get_array_module(images)
    scipy_module = get_scipy_module(images)

    M, N = images.shape[1:]

    # y_grid = xp.matrix(xp.arange(-np.fix(N / 2), np.ceil(N / 2), dtype=r_type))
    y_grid = xp.arange(-np.fix(N / 2), np.ceil(N / 2), dtype=r_type)[None, :]
    y_grid = scipy_module.fft.ifftshift(y_grid) / N
    n_y = xp.tan(theta / 2 * xp.pi / 180).astype(xp.float32) * y_grid
    # m_grid = xp.array(
    #     (xp.matrix(xp.arange(1, M + 1) - np.floor(M / 2)).transpose() * 2j * xp.pi), dtype=c_type
    # )
    m_grid = xp.array(((xp.arange(1, M + 1) - np.floor(M / 2))[:, None] * 2j * xp.pi), dtype=c_type)
    images = scipy_module.fft.ifft(
        xp.multiply(scipy_module.fft.fft(images, axis=2), xp.exp(-xp.multiply(m_grid, n_y))), axis=2
    )

    return images
