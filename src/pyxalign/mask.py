from typing import Optional
import cupy as cp
import numpy as np
import scipy
import cupyx.scipy
import cupyx.scipy.ndimage
import scipy.ndimage
import skimage
import scipy.fft
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from contextlib import nullcontext
from pyxalign import gpu_utils
from pyxalign.api.options.device import DeviceOptions
from pyxalign.gpu_wrapper import device_handling_wrapper
from pyxalign.interactions.mask import illum_map_threshold_plotter
from pyxalign.transformations.helpers import is_array_real
from IPython.display import display
from PyQt5.QtWidgets import QApplication
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyxalign.api.options.options import MaskOptions
from pyxalign.gpu_utils import get_scipy_module, memory_releasing_error_handler
from pyxalign.timing.timer_utils import timer, InlineTimer
from pyxalign.api.types import ArrayType, r_type


@memory_releasing_error_handler
@timer()
def estimate_reliability_region_mask(
    images: np.ndarray, options: MaskOptions, enable_plotting=False
):
    """Use flood-fill to get a mask for the actual object region in each projection"""
    # xp = cp.get_array_module(images)
    xp = cp  # need to generalize later for machines without gpu
    scipy_module: scipy = cupyx.scipy

    # masks = gpu_utils.pin_memory(np.zeros(images.shape))
    masks = gpu_utils.create_empty_pinned_array(images.shape, dtype=r_type)
    # masks = gpu_utils.create_empty_pinned_array(images.shape, dtype=np.uint8)

    # images = np.angle(images) # slow
    close_structure = xp.array(skimage.morphology.diamond(options.binary_close_coefficient))
    erode_structure = xp.array(skimage.morphology.diamond(options.binary_erode_coefficient))

    unsharp_structure = xp.array(
        [[-0.1667, -0.6667, -0.1667], [-0.6667, 4.3333, -0.6667], [-0.1667, -0.6667, -0.1667]]
    )

    if enable_plotting:
        # Make a plot that updates on each iteration
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Sobel filtered",
                "flood fill",
                "otsu threshold",
                "binary fill",
                "binary close",
                "binary erode",
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.10,
        )
        heatmaps = []
        for i in range(6):
            heatmap = go.Heatmap(
                z=np.zeros(images[0].shape),
                colorscale="Viridis",
                showscale=False,
            )
            fig.add_trace(heatmap, row=(i // 3) + 1, col=(i % 3) + 1)
            heatmaps.append(heatmap)
            fig.update_xaxes(
                scaleanchor="y",
                scaleratio=1,
            )
        # Tighten layout
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),  # Minimize margins
            # height=500,
            # width=500,
        )
        fig_widget = go.FigureWidget(fig)

    for i in tqdm(range(len(images)), desc="calculate mask"):
        # time.sleep(0.01)
        if enable_plotting:
            context_manager = fig_widget.batch_update()
        else:
            context_manager = nullcontext()
        with context_manager:
            temp_sino = xp.array(images[i])
            ignore_idx = xp.abs(temp_sino) < 0.1
            temp_sino[ignore_idx] = 0

            if not is_array_real(temp_sino):
                temp_sino = np.angle(temp_sino)  # * np.abs(temp_sino)

            inline_timer = InlineTimer("scipy.ndimage.correlate")
            inline_timer.start()
            if options.unsharp:
                temp_sino[:] = scipy_module.ndimage.correlate(temp_sino, unsharp_structure)
            inline_timer.end()

            inline_timer = InlineTimer("sobel filter")
            inline_timer.start()
            sobelx = scipy_module.ndimage.sobel(temp_sino, 1)
            sobely = scipy_module.ndimage.sobel(temp_sino, 0)
            temp_sino = xp.sqrt(sobelx**2 + sobely**2)
            temp_sino[temp_sino > 1] = 0
            inline_timer.end()

            if enable_plotting:
                # Sobel filtered plotting
                fig_widget.data[0].z = temp_sino.get()

            center_point = (round(temp_sino.shape[0] / 2), round(temp_sino.shape[1] / 2))
            if isinstance(temp_sino, cp.ndarray):
                temp_sino = temp_sino.get()
            inline_timer = InlineTimer("flood_fill")
            inline_timer.start()
            temp_sino = skimage.segmentation.flood_fill(temp_sino, center_point, 1)
            # temp_sino = skimage.segmentation.flood_fill(temp_sino.astype(bool), center_point, 1)
            inline_timer.end()

            if enable_plotting:
                # Flood fill plotting
                fig_widget.data[1].z = temp_sino

            inline_timer = InlineTimer("skimage.filters.threshold_otsu")
            inline_timer.start()
            level = skimage.filters.threshold_otsu(temp_sino)
            temp_sino = temp_sino > level
            inline_timer.end()
            if enable_plotting:
                fig_widget.data[2].z = temp_sino.astype(int)

            inline_timer = InlineTimer("scipy.ndimage.binary_fill_holes")
            inline_timer.start()
            if options.fill > 0:
                # can be put on GPU, but much slower for some reason
                temp_sino = scipy.ndimage.binary_fill_holes(temp_sino)
            inline_timer.end()
            if enable_plotting:
                fig_widget.data[3].z = temp_sino.astype(int)

            temp_sino = xp.array(temp_sino)
            inline_timer = InlineTimer("ndimage.binary_close")
            inline_timer.start()
            temp_sino = scipy_module.ndimage.binary_closing(temp_sino, close_structure)
            inline_timer.end()
            if enable_plotting:
                fig_widget.data[4].z = temp_sino.get().astype(int)

            inline_timer = InlineTimer("ndimage.binary_erosion")
            inline_timer.start()
            temp_sino = scipy_module.ndimage.binary_erosion(temp_sino, erode_structure)
            inline_timer.end()
            if isinstance(temp_sino, cp.ndarray):
                temp_sino = temp_sino.get()
            if enable_plotting:
                fig_widget.data[5].z = temp_sino.astype(int)
                if i == 0:
                    display(fig_widget)

            masks[i] = temp_sino

    return masks


def close_images(
    images: ArrayType,
    binary_close_coefficient: int,
    device_options: DeviceOptions,
    pinned_results: Optional[np.ndarray] = None,
):
    def close_images_function(images, close_structure):
        xp = cp.get_array_module(images)
        scipy_module = gpu_utils.get_scipy_module(images)
        new_images = xp.empty_like(images)
        for i in range(len(new_images)):
            new_images[i] = scipy_module.ndimage.binary_closing(images[i], close_structure)
        return new_images

    wrapped_func = device_handling_wrapper(
        func=close_images_function,
        options=device_options,
        pinned_results=pinned_results,
        common_inputs_for_gpu_idx=[1],
        display_progress_bar=True,
    )
    close_structure = skimage.morphology.diamond(binary_close_coefficient)
    return wrapped_func(images, close_structure)


def erode_images(
    images: ArrayType,
    binary_erode_coefficient: int,
    device_options: DeviceOptions,
    pinned_results: Optional[np.ndarray] = None,
):
    def erode_images_function(images, erode_structure):
        xp = cp.get_array_module(images)
        scipy_module = gpu_utils.get_scipy_module(images)
        new_images = xp.empty_like(images)
        for i in range(len(new_images)):
            new_images[i] = scipy_module.ndimage.binary_erosion(images[i], erode_structure)
        return new_images

    wrapped_func = device_handling_wrapper(
        func=erode_images_function,
        options=device_options,
        pinned_results=pinned_results,
        common_inputs_for_gpu_idx=[1],
        display_progress_bar=True,
    )
    erode_structure = skimage.morphology.diamond(binary_erode_coefficient)
    return wrapped_func(images, erode_structure)


@memory_releasing_error_handler
def blur_masks(masks: np.ndarray, kernel_sigma: int, use_gpu: bool = False):
    blurred_masks = np.zeros_like(masks)
    for i in tqdm(range(len(masks)), desc="Apply gaussian blur to masks"):
        if use_gpu:
            blurred_masks[i] = cupyx.scipy.ndimage.gaussian_filter(
                cp.array(masks[i]), kernel_sigma
            ).get()
        else:
            blurred_masks[i] = scipy.ndimage.gaussian_filter(masks[i], kernel_sigma)
    return blurred_masks


def get_illumination_map(empty_array: np.ndarray, probe: np.ndarray, positions: np.ndarray):
    """
    Get illumination map from probe and probe positions

    Parameters:
        large_array (np.ndarray): The large 2D array where patches will be added.
        patch (np.ndarray): The smaller 2D patch to be added.
        positions (np.ndarray): A 2D array of shape (N, 2) containing x and y positions.

    Returns:
        np.ndarray: The updated large array with patches added.
    """
    patch_h, patch_w = probe.shape
    large_h, large_w = empty_array.shape

    # Compute the half size of the patch to center it
    half_patch_h = patch_h // 2
    half_patch_w = patch_w // 2

    for x, y in positions:
        x, y = int(x), int(y)
        # Compute start and end indices, ensuring they remain within bounds
        x_start = max(x - half_patch_h, 0)
        x_end = min(x + half_patch_h + (patch_h % 2), large_h)
        y_start = max(y - half_patch_w, 0)
        y_end = min(y + half_patch_w + (patch_w % 2), large_w)

        # Compute corresponding valid region of the patch
        patch_x_start = max(half_patch_h - x, 0)
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_start = max(half_patch_w - y, 0)
        patch_y_end = patch_y_start + (y_end - y_start)

        # Add the patch to the large array
        empty_array[x_start:x_end, y_start:y_end] += probe[
            patch_x_start:patch_x_end, patch_y_start:patch_y_end
        ]

    return empty_array


def shrink_binary_mask(mask: np.ndarray, shrink_radius: int):
    """
    Shrinks a binary circular mask by a given radius.

    Parameters
    ----------
    mask : 2D numpy array
        Binary mask with circular region (1 inside, 0 outside).
    shrink_radius : float
        Number of pixels to shrink the mask inward.

    Returns
    -------
    shrunken_mask : 2D numpy array
        Binary mask after shrinking.
    """
    if shrink_radius <= 0:
        return mask.copy()

    # Compute distance from the edge inward (from inside the object)
    distance_inside = distance_transform_edt(mask)

    # Keep only areas where the distance from the edge is greater than shrink_radius
    shrunken_mask = distance_inside > shrink_radius

    return shrunken_mask.astype(mask.dtype)


def place_patches_fourier_batch(
    input_shape: tuple, patch: np.ndarray, positions: list[np.ndarray], pad_edges: bool = True
) -> np.ndarray:
    patch = cp.array(patch)
    xp = cp.get_array_module(patch)
    scipy_module = get_scipy_module(patch)

    # pad by the patch size to prevent wrapping
    if pad_edges:
        padding = int(patch.shape[1])
        input_shape = (input_shape[0], input_shape[1] + padding, input_shape[2] + padding)

    impulse_mask = xp.zeros(input_shape[1:], dtype=r_type)
    padded_patch = xp.zeros_like(impulse_mask)
    masks_out = gpu_utils.create_empty_pinned_array(input_shape, r_type)
    for i in tqdm(range(len(positions))):
        # reset the impulse mask
        impulse_mask[:] = 0
        padded_patch[:] = 0

        # mark impulse locations
        locations = positions[i]
        if pad_edges:
            locations += int(padding / 2)
        offset = np.array(patch.shape, dtype=r_type) / 2
        coords = locations - offset
        indices = np.round(coords).astype(int)
        valid = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < input_shape[1])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < input_shape[2])
        )
        indices = indices[valid]
        impulse_mask[indices[:, 0], indices[:, 1]] = 1.0

        # pad patch to the same size
        ph, pw = patch.shape
        padded_patch[:ph, :pw] = patch

        # Shift patch to center the kernel
        padded_patch = scipy_module.fft.fftshift(padded_patch)

        # Convolution via FFT (using multiplication in Fourier domain)
        fft_mask = scipy_module.fft.fft2(impulse_mask)
        fft_patch = scipy_module.fft.fft2(padded_patch)
        result = xp.real(scipy_module.fft.ifft2(fft_mask * fft_patch))
        result = scipy_module.fft.fftshift(result)
        result.get(out=masks_out[i])
    # undo initial padding
    if pad_edges:
        a = int(padding / 2)
        masks_out = masks_out[:, a:-a, a:-a]

    return masks_out


class IlluminationMapMaskBuilder:
    """
    Class for building mask from the illumination map.
    """

    def get_mask_base(
        self,
        probe: np.ndarray,
        positions: list[np.ndarray],
        projections: np.ndarray,
        use_fourier: bool = True,
    ):
        # The base for building the mask is the illumination map
        if use_fourier:
            self.masks = place_patches_fourier_batch(projections.shape, probe, positions)
        else:
            for i in range(len(positions)):
                self.masks = np.zeros_like(projections, dtype=r_type)
                get_illumination_map(self.masks[i], probe, positions[i])

    def set_mask_threshold_interactively(self, projections: np.ndarray) -> float:
        # temporary bugfix: all windows need to be closed or else app.exec_() will 
        # hang indefinitely. I am putting this temporary solution (which I don't like
        # very much) in place, because any changes will be overwritten once merged with
        # interactive_pma_gui anyway.
        app = QApplication.instance() or QApplication([])
        app.closeAllWindows()

        # Use interactivity to decide mask threshold"
        self.threshold_selector = illum_map_threshold_plotter(
            self.masks, projections, init_thresh=0.01
        )
        self.threshold_selector.show()

        app.exec_()
        threshold = self.threshold_selector.threshold
        return threshold

    def clip_masks(self, thresh: Optional[float] = None):
        clip_idx = self.masks > thresh
        self.masks[:] = 0
        self.masks[clip_idx] = 1
