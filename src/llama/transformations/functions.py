import numpy as np
import cupy as cp
import scipy
from llama.gpu_utils import get_fft_backend, get_scipy_module
from llama.transformations.helpers import preserve_complexity_or_realness

from llama.api.types import ArrayType


def image_crop(
    images: ArrayType,
    horizontal_range: int,
    vertical_range: int,
    horizontal_offset: int = 0,
    vertical_offset: int = 0,
) -> ArrayType:
    """Returns a view of the specified region of the image."""
    image_center = np.array(images.shape[1:]) / 2 + [vertical_offset, horizontal_offset]
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
        or horizontal_index_end >= images.shape[2]
        or vertical_index_start < 0
        or vertical_index_end >= images.shape[1]
    ):
        raise ValueError("Invalid values entered for cropping.")

    return images[
        :,
        vertical_index_start:vertical_index_end,
        horizontal_index_start:horizontal_index_end,
    ]


def image_crop_pad(
    images: ArrayType, new_extent_y: int, new_extent_x: int, pad_mode: str = "constant"
):
    xp = cp.get_array_module(images)
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

    x_grid = scipy_module.fft.ifftshift(xp.arange(-np.fix(shape[2] / 2), np.ceil(shape[2] / 2))) / shape[2]
    X = (x * x_grid)[:, None, :]
    X = xp.exp(-2j * xp.pi * X)

    y_grid = scipy_module.fft.ifftshift(xp.arange(-np.fix(shape[1] / 2), np.ceil(shape[1] / 2))) / shape[1]
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


@preserve_complexity_or_realness()
def image_downsample_fft(images: ArrayType, scale: int) -> ArrayType:
    xp = cp.get_array_module(images)
    fft_backend = get_fft_backend(images)
    # is_real = not xp.issubdtype(images.dtype, xp.complexfloating)

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
        images = image_shift_fft(images, xp.array([[-0.5, -0.5]]), applyFFT=False)

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

        # if is_real:
        #     images = xp.real(images)

    return images


def image_downsample_linear(images: ArrayType, scale: int) -> ArrayType:
    pass
