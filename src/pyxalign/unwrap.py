from typing import Literal, Optional
import cupy as cp
import numpy as np
import scipy
import math
from pyxalign.api.enums import ImageGradientMethods, ImageIntegrationMethods, PhaseUnwrapMethods
from pyxalign.api.options.options import PhaseUnwrapOptions
from pyxalign.api.types import r_type, c_type, ArrayType
from pyxalign.gpu_utils import memory_releasing_error_handler, get_scipy_module
from pyxalign.timing.timer_utils import timer, InlineTimer
from pyxalign.transformations.functions import image_shift_fft


@memory_releasing_error_handler
@timer()
def unwrap_phase(
    images: ArrayType,
    weights: ArrayType,
    options: PhaseUnwrapOptions,
) -> ArrayType:
    """Unwrap phase from complex images using specified method.

    This function serves as the main entry point for phase unwrapping operations.
    It supports two different unwrapping methods: iterative residual correction
    and gradient integration.

    Args:
        images: Complex-valued images to unwrap. Shape should be (N, H, W) where
            N is the number of images, H and W are spatial dimensions.
        weights: Weight arrays corresponding to each image. Same shape as images.
            Values should be between 0 and 1, where 1 indicates full confidence.
        options: Configuration object containing phase unwrapping parameters
            including method selection, iterations, and other algorithm settings.

    Returns:
        Unwrapped phase arrays with the same shape as input images but with
        real-valued dtype (r_type).
    """
    xp = cp.get_array_module(images)

    if options.method == PhaseUnwrapMethods.IterativeResidualCorrection:
        unwrapped_phase = unwrap_phase_iterative_residual_correction(
            images,
            weights,
            options.iterative_residual.iterations,
            options.iterative_residual.lsq_fit_ramp_removal,
        )
    elif options.method == PhaseUnwrapMethods.GradientIntegration:
        unwrapped_phase = xp.zeros(shape=images.shape, dtype=r_type)
        for i in range(len(images)):
            if options.gradient_integration.use_masks:
                weight_map = weights[i]
            else:
                weight_map = None
            unwrapped_phase[i] = unwrap_phase_gradient_integration(
                images[i],
                image_grad_method=options.gradient_integration.gradient_method,
                image_integration_method=options.gradient_integration.integration_method,
                fourier_shift_step=options.gradient_integration.fourier_shift_step,
                weight_map=weight_map,
                deramp_polyfit_order=options.gradient_integration.deramp_polyfit_order,
            )
    return unwrapped_phase


#### Functions for unwrap_phase_iterative_residual_correction ####
@timer()
def unwrap_phase_iterative_residual_correction(
    images: ArrayType,
    weights: ArrayType,
    iterations: int,
    lsq_fit_ramp_removal: Optional[bool] = False,
):
    """Unwrap phase using iterative residual correction method.

    This method iteratively corrects phase residuals by applying phase unwrapping
    to the residual images after removing the current phase estimate.

    Args:
        images: Complex-valued images to unwrap. Shape should be (N, H, W).
        weights: Weight arrays for each image. Values should be between 0 and 1.
            Negative values are clipped to 0 and weights are normalized.
        iterations: Number of iterative correction steps to perform.
        lsq_fit_ramp_removal: Whether to remove phase ramps using least-squares
            fitting after unwrapping. Defaults to False.

    Returns:
        Unwrapped phase arrays with the same shape as input images.

    Note:
        The algorithm normalizes weights and converts them to boolean masks for
        the optional ramp removal step.
    """
    xp = cp.get_array_module(images)
    # Ensure the weights are all between 0 and 1
    weights[weights < 0] = 0
    weights = weights / weights.max()
    bool_weights = weights.astype(bool)
    phase_block = 0
    for i in range(iterations):
        if i == 0:
            images_resid = images
        else:
            images_resid = images * xp.exp(-1j * phase_block)
        phase_block = phase_block + weights * phase_unwrap_2D(images_resid, weights)
        # if empty_region != []:
        #   raise NotImplementedError
        # phase_block = remove_sinogram_ramp(phase_block, empty_region, options.poly_fit_order)
    # Remove phase ramp
    if lsq_fit_ramp_removal:
        for j in range(len(phase_block)):
            phase_block[j] = remove_phase_ramp(phase_block[j], bool_weights[j])
    return phase_block


def phase_unwrap_2D(images: ArrayType, weights: ArrayType, padding: int = 64):
    """Perform 2D phase unwrapping using Fourier gradient integration.

    This function unwraps phase by computing phase gradients in Fourier domain
    and then integrating them back to recover the unwrapped phase. Padding is
    applied to reduce boundary artifacts.

    Args:
        images: Complex-valued images to unwrap. Shape should be (N, H, W).
        weights: Weight arrays for each image. Values should be between 0 and 1.
        padding: Number of pixels to pad on each side. Defaults to 64.

    Returns:
        Unwrapped phase arrays with padding removed.

    Note:
        The function normalizes the input images by their magnitude and applies
        symmetric padding before processing.
    """
    xp = cp.get_array_module(images)

    images = weights * images / (xp.abs(images) + 1e-10)
    del weights

    pad_shape = xp.pad(images[0], padding, "symmetric").shape
    padded_images = cp.zeros((len(images), pad_shape[0], pad_shape[1]), dtype=c_type)
    if np.any(padding) > 0:
        for i in range(len(images)):
            padded_images[i] = xp.pad(images[i], padding, "symmetric")

    dX, dY = get_phase_gradient_fourier(padded_images)
    phase = xp.real(get_images_int_2D(dX, dY))

    start_idx_1, stop_idx_1 = padding, phase.shape[1] - padding
    start_idx_2, stop_idx_2 = padding - 1, phase.shape[2] - padding - 1

    return phase[:, start_idx_1:stop_idx_1, start_idx_2:stop_idx_2]



def get_images_int_2D(dX: ArrayType, dY: ArrayType):
    """Integrate 2D phase gradients using Fourier domain integration.

    This function performs 2D integration of phase gradients by applying
    an integration filter in the Fourier domain. The method is based on
    the relationship between differentiation and multiplication by frequency
    in Fourier space.

    Args:
        dX: Phase gradients in X direction. Shape should be (N, H, W).
        dY: Phase gradients in Y direction. Shape should be (N, H, W).

    Returns:
        Integrated phase arrays with the same shape as input gradients.

    Note:
        The DC component (0,0) of the integration filter is set to zero
        to avoid division by zero and ensure proper integration.
    """
    xp = cp.get_array_module(dX)
    scipy_module: scipy = get_scipy_module(dX)

    n_z, n_y, n_x = dX.shape

    fD = scipy_module.fft.fft2(dX + 1j * dY, axes=(1, 2))
    x_grid = scipy_module.fft.ifftshift(
        xp.arange(-np.fix(n_x / 2), np.ceil(n_x / 2), dtype=r_type)
    )
    x_grid /= n_x
    y_grid = scipy_module.fft.ifftshift(
        xp.arange(-np.fix(n_y / 2), np.ceil(n_y / 2), dtype=r_type)
    )
    y_grid /= n_y

    X = xp.exp((2j * xp.pi) * x_grid + y_grid[:, None])
    # apply integration filter
    X /= 2j * xp.pi * (x_grid + 1j * y_grid[:, None])
    X[0, 0] = 0
    integral = fD * X
    integral = scipy_module.fft.ifft2(integral, axes=(1, 2))

    return integral


#### Functions for unwrap_phase_gradient_integration ####
# this method is pulled from ptychi and was modified to work with
# cupy/numpy instead of torch


@timer()
def unwrap_phase_gradient_integration(
    image: ArrayType,
    fourier_shift_step: float = 0.5,
    image_grad_method: ImageGradientMethods = ImageGradientMethods.FOURIER_DIFFERENTIATION,
    image_integration_method: ImageIntegrationMethods = ImageIntegrationMethods.FOURIER,
    weight_map: Optional[ArrayType] = None,
    flat_region_mask: Optional[ArrayType] = None,
    deramp_polyfit_order: int = 1,
    return_phase_grads: bool = False,
    eps: float = 1e-9,
):
    """Phase unwrapping function adapted from pty-chi.

    This function unwraps phase using gradient integration methods. It supports
    multiple gradient computation and integration approaches, with optional
    weight mapping and polynomial background removal.

    Args:
        image: A complex 2D array giving the image.
        fourier_shift_step: The finite-difference step size used to calculate
            the gradient, if the Fourier shift method is used.
        image_grad_method: The method used to calculate the phase gradient.
            - "fourier_shift": Use Fourier shift to perform shift.
            - "nearest": Use nearest neighbor to perform shift.
            - "fourier_differentiation": Use Fourier differentiation.
        image_integration_method: The method used to integrate the image back
            from gradients.
            - ImageIntegrationMethods.FOURIER: Use Fourier integration as implemented in PtychoShelves.
            - "deconvolution": Deconvolve ramp filter.
        weight_map: A weight map multiplied to the input image. Optional.
        flat_region_mask: A boolean mask with the same shape as `image` that
            specifies the region of the image that should be flat. This is used
            to remove unrealistic phase ramps. If None, de-ramping will not be done.
        deramp_polyfit_order: The order of the polynomial fit used to de-ramp
            the phase.
        return_phase_grads: Whether to return the phase gradient.
        eps: A small number to avoid division by zero.

    Returns:
        The phase of the original image after unwrapping. If return_phase_grads
        is True, returns a tuple of (phase, (gy, gx)).

    Raises:
        ValueError: If input array is not complex.
    """
    # unwraps a single frame
    xp = cp.get_array_module(image)

    if not np.iscomplexobj(image):
        raise ValueError("Input array must be complex.")

    if weight_map is not None:
        weight_map = xp.clip(weight_map, 0.0, 1.0)
    else:
        weight_map = 1

    image = weight_map * image / (xp.abs(image) + eps)
    bc_center = xp.angle(image[image.shape[0] // 2, image.shape[1] // 2])

    # Pad image to avoid FFT boundary artifacts.
    padding = [64, 64]
    if np.any(np.array(padding) > 0):
        if image.shape[-2] > padding[0] and image.shape[-1] > padding[1]:
            padding_mode = "reflect"
        else:
            padding_mode = "replicate"
        image = xp.pad(
            image, (padding[1], padding[1], padding[0], padding[0]), mode=padding_mode
        )
        image = vignette(image, margin=10, sigma=2.5)

    gy, gx = get_phase_gradient(
        image,
        fourier_shift_step=fourier_shift_step,
        image_grad_method=image_grad_method,
    )

    # if image_integration_method == ImageIntegrationMethods.DISCRETE and np.any(np.array(padding) > 0):
    #     gy = gy[padding[0] : -padding[0], padding[1] : -padding[1]]
    #     gx = gx[padding[0] : -padding[0], padding[1] : -padding[1]]
    # if image_integration_method == ImageIntegrationMethods.DISCRETE:
    #     phase = xp.real(integrate_image_2d(gy, gx, bc_center=bc_center))
    if image_integration_method == ImageIntegrationMethods.FOURIER:
        phase = xp.real(integrate_image_2d_fourier(gy, gx))
    elif image_integration_method == ImageIntegrationMethods.DECONVOLUTION:
        phase = xp.real(integrate_image_2d_deconvolution(gy, gx, bc_center=bc_center))
    else:
        raise ValueError(f"Unknown integration method: {image_integration_method}")

    # if image_integration_method != ImageIntegrationMethods.DISCRETE and np.any(np.array(padding) > 0):
    if np.any(np.array(padding) > 0):
        gy = gy[padding[0] : -padding[0], padding[1] : -padding[1]]
        gx = gx[padding[0] : -padding[0], padding[1] : -padding[1]]
        phase = phase[padding[0] : -padding[0], padding[1] : -padding[1]]

    if flat_region_mask is not None:
        phase = remove_polynomial_background(
            phase, flat_region_mask, polyfit_order=deramp_polyfit_order
        )
    if return_phase_grads:
        return phase, (gy, gx)
    return phase


@timer()
def remove_polynomial_background(
    images: ArrayType,
    flat_region_mask: ArrayType,
    polyfit_order: int = 1,
) -> ArrayType:
    """Fit a 2D polynomial to flat regions and subtract from image.

    This function fits a 2D polynomial to the region that is supposed to be flat
    in an image, and subtracts the fitted function from the image to remove
    polynomial background variations.

    Args:
        images: The input image.
        flat_region_mask: A boolean mask with the same shape as `images` that
            specifies the region of the image that should be flat.
        polyfit_order: The order of the polynomial to fit. Should be an integer
            >= 0. If 0, just subtract the average. Defaults to 1.

    Returns:
        The image with the polynomial background subtracted.

    Note:
        For polyfit_order=0, only the mean is subtracted. For higher orders,
        a full 2D polynomial fit is performed using least squares.
    """
    xp = cp.get_array_module(images)

    if polyfit_order == 0:
        return images - images[flat_region_mask].mean()
    ys, xs = xp.where(flat_region_mask)
    y_full, x_full = xp.meshgrid(
        xp.arange(images.shape[0]), xp.arange(images.shape[1]), indexing="ij"
    )
    y_full = y_full.reshape(-1)
    x_full = x_full.reshape(-1)

    y_all_orders = []
    x_all_orders = []
    y_full_all_orders = []
    x_full_all_orders = []
    for order in range(polyfit_order + 1):
        y_all_orders.append(ys**order)
        x_all_orders.append(xs**order)
        y_full_all_orders.append(y_full**order)
        x_full_all_orders.append(x_full**order)
    const_basis = xp.ones(len(ys))
    const_basis_full = xp.ones(len(y_full))

    a_mat = xp.stack(y_all_orders + x_all_orders + [const_basis], dim=1)
    b_vec = images[flat_region_mask].reshape(-1, 1)
    x_vec = xp.linalg.solve(a_mat, b_vec)
    a_mat_full = xp.stack(
        y_full_all_orders + x_full_all_orders + [const_basis_full], dim=1
    )
    bg = a_mat_full @ x_vec
    bg = bg.reshape(images.shape)
    return images - bg


@timer()
def vignette(
    images: ArrayType,
    margin: int = 20,
    sigma: float = 1.0,
    method: Literal["gaussian", "linear", "window"] = "gaussian",
    dim=(-2, -1),
):
    """Apply vignetting to gradually decay image near boundaries.

    For each dimension of the image, a mask with a width of `2 * margin`
    and with half of it filled with 0s and half with 1s is generated and
    convolved with a Gaussian kernel. The blurred mask is cropped and
    multiplied to the near-edge regions of the image.

    Args:
        images: The input image.
        margin: The margin of image where the decay takes place.
            Only used if `method` is "gaussian" or "linear".
        sigma: The standard deviation of the Gaussian kernel.
            Only used if `method` is "gaussian".
        method: The method to use to generate the vignette mask.
            "window" is a Hann window.
        dim: Dimensions along which to apply vignetting. Defaults to (-2, -1).

    Returns:
        The vignetted image with gradual decay near boundaries.

    Note:
        This function is not differentiable because of the slice-assignment
        operation. The "window" method applies a Hamming window function.
    """
    xp = cp.get_array_module(images)
    scipy_module = get_scipy_module(images)

    dims = [d % images.ndim for d in dim]
    # images = images.clone()
    images = images * 1
    for i_dim in dims:
        if images.shape[i_dim] <= 2 * margin:
            continue
        mask_shape = (
            [images.shape[i] for i in range(i_dim)]
            + [2 * margin]
            + [images.shape[i] for i in range(i_dim + 1, images.ndim)]
        )
        if method == "gaussian":
            mask = xp.zeros(mask_shape, dtype=r_type)  ##, device=images.device)
            mask_slicer = [slice(None)] * i_dim + [slice(margin, None)]
            mask[tuple(mask_slicer)] = 1.0
            # gauss_win = torch.signal.windows.gaussian(margin // 2, std=sigma)
            gauss_win = scipy_module.signal.windows.gaussian(M=margin // 2, std=sigma)
            gauss_win = gauss_win / xp.sum(gauss_win)
            mask = convolve1d(mask, gauss_win, dim=i_dim, padding="same")
            mask_final_slicer = [slice(None)] * i_dim + [
                slice(len(gauss_win), len(gauss_win) + margin)
            ]
            mask = mask[tuple(mask_final_slicer)]
            mask = xp.where(mask < 1e-3, 0, mask)
        elif method == "linear":
            ramp = xp.linspace(0, 1, margin)
            new_shape = [1] * images.ndim
            new_shape[i_dim] = margin
            ramp = ramp.reshape(new_shape)
            rep = list(images.shape)
            rep[i_dim] = 1
            mask = ramp.repeat(rep)
        elif method == "window":
            # window_func = torch.hamming_window(
            #     images.shape[i_dim], periodic=True, alpha=0.5, beta=0.5
            # )
            window_func = scipy_module.signal.windows.general_hamming(
                images.shape[i_dim], periodic=True, alpha=0.5, beta=0.5
            )
            images = images * window_func.reshape(
                [-1] + [1] * (images.ndim - i_dim - 1)
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        if method in ["gaussian", "linear"]:
            slicer = [slice(None)] * i_dim + [slice(0, margin)]
            # images[slicer] = images[slicer] * mask
            images[tuple(slicer)] = images[tuple(slicer)] * mask

            slicer = [slice(None)] * i_dim + [slice(-margin, None)]
            # images[slicer] = images[slicer] * mask.flip(i_dim)
            images[tuple(slicer)] = images[tuple(slicer)] * np.flip(mask, axis=i_dim)
    return images


@timer()
def integrate_image_2d_fourier(grad_y: ArrayType, grad_x: ArrayType) -> ArrayType:
    """Integrate image gradients using Fourier differentiation.

    This function integrates an image with the gradient in y and x directions
    using Fourier domain methods. The integration is performed by applying
    the inverse of the differentiation operator in frequency space.

    Args:
        grad_y: Gradients in Y direction. Shape should be (H, W).
        grad_x: Gradients in X direction. Shape should be (H, W).

    Returns:
        The integrated image with the same shape as input gradients.

    Note:
        The DC component (0,0) is set to zero to avoid division by zero.
        This implementation follows PtychoShelves conventions.
    """
    xp = cp.get_array_module(grad_y)
    scipy_module = get_scipy_module(grad_y)

    shape = grad_y.shape
    # f = pmath.fft2_precise(grad_x + 1j * grad_y) # I am not using precise fft
    f = scipy_module.fft.fft2(grad_x + 1j * grad_y)
    y, x = scipy_module.fft.fftfreq(shape[0]), scipy_module.fft.fftfreq(shape[1])

    # In PtychoShelves' get_img_int_2D.m, they set the numerator of r to be
    # exp(2j * pi * (x + y[:, None])) to shift it by 1 pixel. We should NOT
    # do this in order to get the same result as PtychoShelves.
    r = 1.0
    r = r / (2j * xp.pi * (x + 1j * y[:, None]))
    r[0, 0] = 0
    integrated_image = f * r
    # integrated_image = pmath.ifft2_precise(integrated_image)
    integrated_image = scipy_module.fft.ifft2(integrated_image)
    if not xp.iscomplexobj(grad_x):
        integrated_image = xp.real(integrated_image)
    return integrated_image


@timer()
def integrate_image_2d(
    grad_y: ArrayType, grad_x: ArrayType, bc_center: float = 0
) -> ArrayType:
    """Integrate an image with the gradient in y and x directions.

    Args:
        grad_y: The gradient in y direction.
        grad_x: The gradient in x direction.
        bc_center: The boundary condition at the center of the image. Defaults to 0.

    Returns:
        The integrated image.
    """
    xp = cp.get_array_module(grad_y)

    left_boundary = xp.cumsum(grad_y[:, 0], axis=0)
    int_img = xp.cumsum(grad_x, axis=1) + left_boundary[:, None]
    int_img = (
        int_img + bc_center - int_img[int_img.shape[0] // 2, int_img.shape[1] // 2]
    )
    return int_img


@timer()
def integrate_image_2d_deconvolution(
    grad_y: ArrayType,
    grad_x: ArrayType,
    tf_y: Optional[ArrayType] = None,
    tf_x: Optional[ArrayType] = None,
    bc_center: float = 0,
) -> ArrayType:
    """Integrate an image with gradients by deconvolving the differentiation kernel.

    The transfer function is assumed to be a ramp function. This method is adapted
    from Tripathi et al. (2016) for single-view phase retrieval.

    Args:
        grad_y: A (H, W) array of gradients in y direction.
        grad_x: A (H, W) array of gradients in x direction.
        tf_y: A (H, W) array of transfer functions in y direction. If not
            provided, assumed to be 2i * pi * u, which are the effective
            transfer functions in Fourier differentiation. Optional.
        tf_x: A (H, W) array of transfer functions in x direction. If not
            provided, assumed to be 2i * pi * v, which are the effective
            transfer functions in Fourier differentiation. Optional.
        bc_center: The value of the boundary condition at the center of the image.
            Defaults to 0.

    Returns:
        The integrated image.

    Note:
        Adapted from Tripathi, A., McNulty, I., Munson, T., & Wild, S. M. (2016).
        Single-view phase retrieval of an extended sample by exploiting edge detection
        and sparsity. Optics Express, 24(21), 24719–24738. doi:10.1364/OE.24.024719
    """
    xp = cp.get_array_module(grad_y)
    scipy_module = get_scipy_module(grad_x)

    u, v = (
        scipy_module.fft.fftfreq(grad_x.shape[0]),
        scipy_module.fft.fftfreq(grad_x.shape[1]),
    )
    u, v = xp.meshgrid(u, v, indexing="ij")
    if tf_y is None or tf_x is None:
        tf_y = 2j * xp.pi * u
        tf_x = 2j * xp.pi * v
    # f_grad_y = pmath.fft2_precise(grad_y)
    # f_grad_x = pmath.fft2_precise(grad_x)
    f_grad_y = scipy_module.fft.fft2(grad_y)
    f_grad_x = scipy_module.fft.fft2(grad_x)
    images = (f_grad_y * tf_y + f_grad_x * tf_x) / (
        xp.abs(tf_y) ** 2 + xp.abs(tf_x) ** 2 + 1e-5
    )
    # images = -pmath.ifft2_precise(images)
    images = -scipy_module.fft.ifft2(images)
    images = images + bc_center - images[images.shape[0] // 2, images.shape[1] // 2]
    return images


@timer()
def convolve1d(
    input: ArrayType,
    kernel: ArrayType,
    padding: Literal["same", "valid"] = "same",
    padding_mode: Literal["replicate", "constant"] = "replicate",
    dim: int = -1,
) -> ArrayType:
    """1D convolution with an explicitly given kernel.

    This routine flips the kernel to adhere with the textbook definition of convolution.
    The original implementation was adapted from torch.nn.functional.conv1d.

    Args:
        input: A (... d) array of signals.
        kernel: A (d,) array of kernel.
        padding: Padding mode for convolution. Defaults to "same".
        padding_mode: Mode for padding operation. Defaults to "replicate".
        dim: Dimension along which to perform convolution. Defaults to -1.

    Returns:
        A (... d) array of convolved signals.
    """
    xp = cp.get_array_module(input)
    scipy_module = get_scipy_module(input)
    if not input.ndim >= 1:
        raise ValueError("Image must have at least 1 dimensions.")
    if not kernel.ndim == 1:
        raise ValueError("Kernel must have exactly 1 dimensions.")

    if xp.iscomplexobj(input):
        kernel = kernel.type(input.dtype)

    dim = dim % input.ndim
    orig_shape = input.shape
    # Move dim to the end.
    if dim != input.ndim - 1:
        input = xp.moveaxis(input, dim, input.ndim - 1)
    bcast_shape = input.shape[:-1]
    # Reshape image to (N, 1, d).
    if input.ndim == 1:
        input = input.reshape(1, 1, input.shape[-1])
    else:
        input = input.reshape(-1, 1, input.shape[-1])

    # Reshape kernel to (1, 1, d).
    # kernel = kernel.flip((0,))
    # kernel = xp.flip(kernel, axis=0) # not needed since not using torch convolution
    kernel = kernel.reshape(1, 1, kernel.shape[-1])

    if padding == "same":
        pad_lengths = [
            kernel.shape[-1] // 2,
            kernel.shape[-1] // 2,
        ]
        if kernel.shape[-1] % 2 == 0:
            pad_lengths[-1] -= 1
        # input = xp.pad(input, pad_lengths, mode=padding_mode)
        input = torch_pad_to_numpy_pad(input, pad_lengths, mode=padding_mode)

    # result = torch.nn.functional.conv1d(input, kernel, padding="valid")
    # result = xp.convolve(input, kernel, mode="valid") # here
    result = scipy_module.ndimage.convolve1d(
        input[:, 0, :], kernel[0, 0, :], mode="constant", origin=0
    )

    # Restore shape.
    if len(orig_shape) == 1:
        result = result.reshape(orig_shape[0])
    else:
        result = result.reshape([*bcast_shape, result.shape[-1]])
        if dim != input.ndim - 1:
            result = xp.moveaxis(result, result.ndim - 1, dim)
    return result


@timer()
def remove_phase_ramp(phase: ArrayType, mask: np.ndarray):
    """Remove the phase ramp from a 2D phase array using masked region estimation.

    This function removes linear phase ramps from a 2D phase array by fitting
    a plane to the masked region and subtracting it from the entire array.

    Args:
        phase: A 2D array representing the phase (in radians).
        mask: A 2D boolean array (same shape as `phase`), where True indicates
            the region to use for phase ramp estimation.

    Returns:
        The phase-corrected 2D array.

    Raises:
        ValueError: If phase and mask arrays have different shapes.
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


@timer()
def get_phase_gradient(
    image: ArrayType,
    fourier_shift_step: float = 0,
    image_grad_method: ImageGradientMethods = ImageGradientMethods.FOURIER_DIFFERENTIATION,
    eps: float = 1e-6,
) -> ArrayType:
    """Get the gradient of the phase of a complex 2D image.

    This function calculates the spatial gradient of the complex image first, then
    takes the phase of the complex gradient. This approach avoids sharp gradients
    due to phase wrapping when directly taking the gradient of the phase.

    Args:
        image: A [N, H, W] or [H, W] array giving a batch of images or a single image.
        fourier_shift_step: The finite-difference step size used to calculate the
            gradient, if the Fourier shift method is used.
        image_grad_method: The method used to calculate the phase gradient.
            - "fourier_shift": Use Fourier shift to perform shift.
            - "nearest": Use nearest neighbor to perform shift.
            - "fourier_differentiation": Use Fourier differentiation.
        eps: A stabilizing constant.

    Returns:
        A tuple of 2 arrays with the gradient in y and x directions.

    Raises:
        ValueError: If step is not positive when using fourier_shift method.
        ValueError: If unknown finite-difference method is specified.
    """
    xp = cp.get_array_module(image)
    if fourier_shift_step <= 0 and image_grad_method == ImageGradientMethods.FOURIER_SHIFT:
        raise ValueError("Step must be positive.")

    if image_grad_method == ImageGradientMethods.FOURIER_DIFFERENTIATION:
        gx, gy = get_phase_gradient_fourier(image[None])
        gx, gy = gx[0], gy[0]
    else:
        # Use finite difference.
        if image.ndim == 2:
            image = image[None]
        pad = int(math.ceil(fourier_shift_step)) + 1
        image = torch_pad_to_numpy_pad(image, (pad, pad, pad, pad), mode="constant")

        sy1 = xp.array([[-fourier_shift_step, 0]]).repeat(image.shape[0], 1)
        sy2 = xp.array([[fourier_shift_step, 0]]).repeat(image.shape[0], 1)
        if image_grad_method == ImageGradientMethods.FOURIER_SHIFT:
            # If the image contains zero-valued pixels, Fourier shift can result in small
            # non-zero values that dangles around 0. This can cause the phase
            # of the shifted image to dangle between pi and -pi. In that case, use
            # `finite_diff_method="nearest" instead`, or use `step=1`.
            complex_prod = (
                image_shift_fft(image, sy1) * image_shift_fft(image, sy2).conj()
            )
        elif image_grad_method == ImageGradientMethods.NEAREST:
            complex_prod = (
                image
                * xp.concatenate([image[:, :1, :], image[:, :-1, :]], axis=1).conj()
            )
        else:
            raise ValueError(f"Unknown finite-difference method: {image_grad_method}")
        complex_prod = xp.where(
            xp.abs(complex_prod) < xp.abs(complex_prod).max() * 1e-6, 0, complex_prod
        )
        # gy = pmath.angle(complex_prod, eps=eps) / (2 * fourier_shift_step)
        gy = xp.angle(complex_prod) / (2 * fourier_shift_step)
        gy = gy[0, pad:-pad, pad:-pad]

        sx1 = xp.array([[0, -fourier_shift_step]]).repeat(image.shape[0], 1)
        sx2 = xp.array([[0, fourier_shift_step]]).repeat(image.shape[0], 1)
        if image_grad_method == ImageGradientMethods.FOURIER_SHIFT:
            complex_prod = (
                image_shift_fft(image, sx1) * image_shift_fft(image, sx2).conj()
            )
        elif image_grad_method == ImageGradientMethods.NEAREST:
            complex_prod = (
                image
                * xp.concatenate([image[:, :, :1], image[:, :, :-1]], axis=2).conj()
            )
        complex_prod = xp.where(
            xp.abs(complex_prod) < xp.abs(complex_prod).max() * 1e-6, 0, complex_prod
        )
        # gx = pmath.angle(complex_prod, eps=eps) / (2 * fourier_shift_step)
        gx = xp.angle(complex_prod) / (2 * fourier_shift_step)
        gx = gx[0, pad:-pad, pad:-pad]
    return gy, gx


def torch_pad_to_numpy_pad(input, pad_lengths, mode="replicate"):
    """Convert PyTorch-style padding to NumPy padding.

    This function converts padding specifications from PyTorch format to NumPy format
    and applies the padding to the input array.

    Args:
        input: Array to pad.
        pad_lengths: Tuple in PyTorch format (left, right, top, bottom, ...)
            specified in reverse order starting from last dimension.
        mode: Padding mode string. Defaults to "replicate".

    Returns:
        Padded array.
    """
    # Map PyTorch modes to NumPy modes
    mode_map = {
        "replicate": "edge",
        "constant": "constant",
        "reflect": "reflect",
        "circular": "wrap",
    }
    numpy_mode = mode_map.get(mode, mode)

    # Convert pad_lengths from PyTorch format to NumPy format
    # PyTorch: (left, right, top, bottom, front, back, ...) - pairs in reverse dimension order
    # NumPy: ((dim0_before, dim0_after), (dim1_before, dim1_after), ...)

    # Group into pairs: [(left, right), (top, bottom), (front, back), ...]
    pad_pairs = [
        (pad_lengths[i], pad_lengths[i + 1]) for i in range(0, len(pad_lengths), 2)
    ]

    # Reverse to get forward dimension order
    pad_pairs = pad_pairs[::-1]

    # Add (0, 0) for dimensions that aren't being padded
    ndim = input.ndim
    num_padded_dims = len(pad_pairs)
    pad_width = [(0, 0)] * (ndim - num_padded_dims) + pad_pairs

    # Apply padding
    return np.pad(input, pad_width=pad_width, mode=numpy_mode)


#### shared functions #### 
def get_phase_gradient_fourier(images: ArrayType):
    """Compute phase gradients using Fourier differentiation.

    This function calculates phase gradients by first computing spatial gradients
    of the complex images using Fourier differentiation, then extracting the
    imaginary part of the product with the conjugate of the original images.

    Args:
        images: Complex-valued images. Shape should be (N, H, W).

    Returns:
        Tuple of phase gradients (dX, dY) in X and Y directions with the same
        shape as input images.

    Note:
        This method avoids phase wrapping issues by working with the complex
        gradients rather than taking gradients of the phase directly.
    """
    xp = cp.get_array_module(images)
    dX, dY = get_image_grad(images)
    dX = xp.imag(xp.conj(images) * dX)
    dY = xp.imag(xp.conj(images) * dY)

    return dX, dY


def get_image_grad(images: ArrayType):
    """Compute spatial gradients of images using Fourier differentiation.

    This function calculates spatial gradients in X and Y directions using
    Fourier domain differentiation. The method multiplies the Fourier transform
    of the images by the appropriate frequency grids to compute derivatives.

    Args:
        images: Complex-valued images. Shape should be (N, H, W).

    Returns:
        Tuple of spatial gradients (dX, dY) in X and Y directions with the same
        shape as input images.

    Note:
        This is a core function used by phase gradient computation routines.
        The gradients are computed in Fourier domain for accuracy and efficiency.
    """
    xp = cp.get_array_module(images)
    scipy_module: scipy = get_scipy_module(images)

    n_z, n_y, n_x = images.shape

    X = scipy_module.fft.ifftshift(
        xp.arange(-np.fix(n_x / 2), np.ceil(n_x / 2), dtype=c_type)
    )
    X *= 2j * xp.pi / n_x
    dX = scipy_module.fft.fft(images, axis=2) * X
    dX = scipy_module.fft.ifft(dX, axis=2)

    Y = scipy_module.fft.ifftshift(
        xp.arange(-np.fix(n_y / 2), np.ceil(n_y / 2), dtype=c_type)
    )
    Y *= 2j * xp.pi / n_y
    dY = scipy_module.fft.fft(images, axis=1) * Y[:, None]
    dY = scipy_module.fft.ifft(dY, axis=1)

    return dX, dY
