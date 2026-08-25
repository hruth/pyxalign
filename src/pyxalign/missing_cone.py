import numpy as np
import cupy as cp
import scipy
import cupyx.scipy.fft as cufft
from tqdm import tqdm
from pyxalign.gpu_utils import memory_releasing_error_handler
from pyxalign.regularization import chambolleLocalTV3D
from pyxalign.timing.timer_utils import timer, InlineTimer
from pyxalign.api.types import r_type


@memory_releasing_error_handler
@timer()
def fill_missing_cone(
    rec: np.ndarray,
    lamino_angle: float,
    delta_background: float,
    delta_maximal: float,
    mask_relax: float = 0.05,
    max_scale: int = 16,
    n_iter: int = 10,
    tv_lambda: float = 1e-7,
):
    # Process the reconstruction using multiscale approach, start at 0.5^max_scale
    delta_background = r_type(delta_background)
    delta_maximal = r_type(delta_maximal)
    tv_lambda = r_type(tv_lambda)

    n_pix = np.array(rec.shape, dtype=int)

    # weakly suppress nonzero values in empty regions (along vertical axis)
    mask_vert = np.mean(np.abs(rec), axis=(1, 2))
    mask_vert = 1 - mask_relax + mask_relax * mask_vert / np.max(mask_vert)

    border_size = np.array([32, 32], dtype=int)
    block_size = np.array([512, 512], dtype=int) - 2 * border_size

    scales = (2 ** np.arange(np.log2(max_scale), -1, -1)).astype(int)
    for scale in tqdm(scales):
        inline_timer = InlineTimer(f"scale_{scale}")
        inline_timer.start()
        print("Running scale", scale, "...")

        if scale > 1:
            rec_small = interp_ft_3d(rec, np.ceil(n_pix / scale))
            x_points = np.arange(0, len(mask_vert), dtype=int)
            x_interp = np.linspace(0, len(mask_vert), len(rec_small))
            mask_vert_small = np.interp(x=x_interp, xp=x_points, fp=mask_vert)[:, None, None]
        else:
            rec_small = rec
            mask_vert_small = mask_vert[:, np.newaxis, np.newaxis]
        low_freq_protection = scale < max_scale

        rec_regularized = apply_lamino_constraints(
            rec_small,
            mask_vert_small,
            lamino_angle,
            low_freq_protection,
            delta_maximal,
            delta_background,
            n_iter,
            tv_lambda,
            border_size=border_size,
            block_size=block_size,
        )

        rec = rec + interp_ft_3d(rec_regularized - rec_small, n_pix)

        del rec_regularized, rec_small
        inline_timer.end()

    return rec


def block_proc(func):
    """Decorator for splitting the input into blocks, so that the GPU
    memory is not exceeded. Similar to matlab's blockproc."""

    def wrapped_func(*args, **kwargs):
        args = list(args)

        block_size = kwargs["block_size"]

        h, w = args[0].shape[1:]
        m, n = block_size

        img = args[0] * 1
        for x in range(0, h, m):
            for y in range(0, w, n):
                print("block shape:", args[0][:, x : x + m, y : y + n].shape)
                block = img[:, x : x + m, y : y + n]
                block[:, :, :] = func(block, *args[1:], **kwargs)
        return img

    return wrapped_func


@memory_releasing_error_handler
@timer()
def pad_inputs(func):
    """Decorator for padding the input volume before a function call and
    removing the padding after the function call"""

    def wrapped_func(*args, **kwargs):
        args = list(args)

        border_size = kwargs["border_size"]

        # Pad on each side
        args[0] = np.pad(args[0], ([0, 0], border_size, border_size), "symmetric")
        # Call the function
        results = func(*args, **kwargs)
        # Remove padding
        results = results[
            :,
            border_size[0] : results.shape[1] - border_size[0],
            border_size[1] : results.shape[2] - border_size[1],
        ]
        return results

    return wrapped_func


@memory_releasing_error_handler
@block_proc
@pad_inputs
@timer()
def apply_lamino_constraints(
    volume,
    mask,
    lamino_angle,
    low_freq_protection,
    value_max,
    value_min,
    n_iter,
    tv_lambda,
    border_size,
    block_size,
):
    n_pix = volume.shape
    fft_mask = get_lamino_fourier_mask(n_pix, lamino_angle, True)
    fft_mask = cp.array(fft_mask)
    mask = cp.array(mask)

    if low_freq_protection:
        # Avoid modifying low spatial frequencies that were already
        # refined
        fft_mask = scipy.fft.fftshift(fft_mask)
        pts = []
        for i in range(3):
            pts = pts + [
                [
                    int(np.ceil(n_pix[i] / 2) - np.ceil(n_pix[i] / 8)),
                    int(np.ceil(n_pix[i] / 2) + np.floor(n_pix[i] / 8)),
                ]
            ]
        pts = np.array(pts, dtype=int)
        fft_mask = fft_mask.astype(int)
        fft_mask[pts[0][0] - 1 : pts[0][1], pts[1][0] - 1 : pts[1][1], pts[2][0] - 1 : pts[2][1]] = 0
        fft_mask = scipy.fft.fftshift(fft_mask)

    volume = cp.array(volume)
    volume_new = volume * 1

    for i in tqdm(range(n_iter)):
        volume_new = chambolleLocalTV3D(volume_new, tv_lambda, 10)

        # Positivity constraint
        volume_new[volume_new < value_min] = value_min
        volume_new[volume_new > value_max] = value_max
        volume_new = volume_new * mask

        # Go to the Fourier space
        with scipy.fft.set_backend(cufft):
            fft_volume = scipy.fft.fftn(volume)
            fft_volume_new = scipy.fft.fftn(volume_new)

        # Merge updated and original dataset in the Fourier space
        # Use overrelaxation of the constraint to get faster convergence
        relax = r_type(1.5)
        regularize = 0
        fft_volume = fft_volume * (1 - relax * fft_mask) + fft_volume_new * relax * fft_mask
        fft_volume = fft_volume * (1 - regularize * fft_mask).astype(np.float32)
        del fft_volume_new

        # Go back to real space
        with scipy.fft.set_backend(cufft):
            volume_new = np.real(scipy.fft.ifftn(fft_volume))
        del fft_volume

        volume = volume_new

    volume_new = volume_new.get()

    return volume_new


@memory_releasing_error_handler
@timer()
def get_lamino_fourier_mask(n_pix, lamino_angle, keep_on_gpu=False):
    grid = []
    for i in range(3):
        grid = grid + [scipy.fft.fftshift(np.linspace([[-1]], [[1]], n_pix[i], axis=i))]
    fft_mask = get_mask(grid[1], grid[2], grid[0], lamino_angle)

    return fft_mask


@memory_releasing_error_handler
@timer()
def get_mask(x_grid, y_grid, z_grid, lamino_angle):
    fft_mask = (
        np.ceil(180 / np.pi * np.arctan(np.abs(z_grid) / np.sqrt(x_grid**2 + y_grid**2))) > lamino_angle
    )

    return fft_mask


## Utils


@memory_releasing_error_handler
@timer()
def interp_ft_3d(img, n_out):
    # Functionality is questionable -- needs to be tested

    n_in = np.array(img.shape, dtype=int)
    n_out = np.array(n_out, dtype=int)

    im_ft = scipy.fft.fftshift(scipy.fft.fftn(img))
    im_out = crop_pad_3d(im_ft, n_out)
    im_out = scipy.fft.ifftn(scipy.fft.ifftshift(im_out)) * n_out.prod() / n_in.prod()

    is_real = img.dtype != np.complexfloating
    if is_real:
        im_out = im_out.astype(img.dtype)

    return im_out


@memory_releasing_error_handler
@timer()
def crop_pad_3d(img, n_out):
    # Functionality is questionable -- needs to be tested

    n_in = img.shape
    center = np.floor(np.array(img.shape) / 2)
    im_out = np.zeros(n_out, dtype=img.dtype)
    center_out = np.floor(n_out / 2)
    offset = center_out - center

    idx_out = {}
    idx_in = {}
    for i in range(3):
        idx_out[i] = np.arange(
            np.append(offset[i], 0).max(), np.append(offset[i] + n_in[i], n_out[i]).min(), dtype=int
        )
        idx_in[i] = np.arange(
            np.append(-offset[i], 0).max(), np.append(-offset[i] + n_out[i], n_in[i]).min(), dtype=int
        )

    im_out[
        idx_out[0][0] : idx_out[0][-1], idx_out[1][0] : idx_out[1][-1], idx_out[2][0] : idx_out[2][-1]
    ] = img[idx_in[0][0] : idx_in[0][-1], idx_in[1][0] : idx_in[1][-1], idx_in[2][0] : idx_in[2][-1]]

    is_complex = img.dtype == np.complexfloating
    if is_complex:
        im_out = im_out.astype(img.dtype)

    return im_out
