import cupy as cp
import numpy as np
import scipy
import cupyx.scipy
import cupyx.scipy.ndimage
import scipy.ndimage
import skimage
import scipy.fft
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from llama.transformations.helpers import is_array_real
from IPython.display import clear_output, display

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from llama.api.options.options import MaskOptions
from llama.gpu_utils import memory_releasing_error_handler, get_scipy_module


@memory_releasing_error_handler
def estimate_reliability_region_mask(
    images: np.ndarray, options: MaskOptions, enable_plotting=False
):
    """Use flood-fill to get a mask for the actual object region in each projection"""
    # xp = cp.get_array_module(images)
    xp = cp  # need to generalize later for machines without gpu
    scipy_module: scipy = cupyx.scipy

    masks = np.zeros(images.shape)

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

    for i in tqdm(range(len(images))):
        # time.sleep(0.01)
        with fig_widget.batch_update():
            temp_sino = xp.array(images[i])
            ignore_idx = xp.abs(temp_sino) < 0.1
            temp_sino[ignore_idx] = 0

            if not is_array_real(temp_sino):
                temp_sino = np.angle(temp_sino)  # * np.abs(temp_sino)

            if options.unsharp:
                temp_sino = scipy_module.ndimage.correlate(temp_sino, unsharp_structure)

            sobelx = scipy_module.ndimage.sobel(temp_sino, 1)
            sobely = scipy_module.ndimage.sobel(temp_sino, 0)
            temp_sino = xp.sqrt(sobelx**2 + sobely**2)
            temp_sino[temp_sino > 1] = 0

            if enable_plotting:
                # Sobel filtered plotting
                fig_widget.data[0].z = temp_sino.get()

            center_point = (round(temp_sino.shape[0] / 2), round(temp_sino.shape[1] / 2))
            if isinstance(temp_sino, cp.ndarray):
                temp_sino = temp_sino.get()
            temp_sino = skimage.segmentation.flood_fill(temp_sino, center_point, 1)

            if enable_plotting:
                # Flood fill plotting
                fig_widget.data[1].z = temp_sino

            level = skimage.filters.threshold_otsu(temp_sino)
            temp_sino = temp_sino > level
            if enable_plotting:
                fig_widget.data[2].z = temp_sino.astype(int)

            if options.fill > 0:
                # can be put on GPU, but much slower for some reason
                temp_sino = scipy.ndimage.binary_fill_holes(temp_sino)
            if enable_plotting:
                fig_widget.data[3].z = temp_sino.astype(int)

            temp_sino = xp.array(temp_sino)
            temp_sino = scipy_module.ndimage.binary_closing(temp_sino, close_structure)
            if enable_plotting:
                fig_widget.data[4].z = temp_sino.get().astype(int)

            temp_sino = scipy_module.ndimage.binary_erosion(temp_sino, erode_structure)
            if isinstance(temp_sino, cp.ndarray):
                temp_sino = temp_sino.get()
            if enable_plotting:
                fig_widget.data[5].z = temp_sino.astype(int)
                if i == 0:
                    display(fig_widget)

        masks[i] = temp_sino

    return masks


@memory_releasing_error_handler
def blur_masks(masks: np.ndarray, kernel_sigma: int, use_gpu: bool = False):
    blurred_masks = np.zeros_like(masks)
    for i in range(len(masks)):
        if use_gpu:
            blurred_masks[i] = cupyx.scipy.ndimage.gaussian_filter(
                cp.array(masks[i]), kernel_sigma
            ).get()
        else:
            blurred_masks[i] = scipy.ndimage.gaussian_filter(masks[i], kernel_sigma)
    return blurred_masks
