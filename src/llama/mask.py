import cupy as cp
import numpy as np
import scipy
import cupyx.scipy
import skimage
import scipy.fft
import time
import matplotlib.pyplot as plt
from llama.transformations.helpers import is_array_real
from IPython.display import clear_output

from llama.api.options.options import MaskOptions


def estimate_reliability_region_mask(images: np.ndarray, options: MaskOptions, enable_plotting=False):
    """Use flood-fill to get a mask for the actual object region in each projection"""
    # xp = cp.get_array_module(images)
    xp = cp
    scipy_module: scipy = cupyx.scipy

    masks = np.zeros(images.shape)

    # images = np.angle(images) # slow
    close_structure = xp.array(skimage.morphology.diamond(options.binary_close_coefficient))
    erode_structure = xp.array(skimage.morphology.diamond(options.binary_erode_coefficient))

    unsharp_structure = xp.array(
        [[-0.1667, -0.6667, -0.1667], [-0.6667, 4.3333, -0.6667], [-0.1667, -0.6667, -0.1667]]
    )

    for i in range(len(images)):
        if i % 4 == 0:
            print("iteration " + str(i + 1) + "/" + str(len(images)))

        if enable_plotting:
            fig, ax = plt.subplots(2, 3)
            fig.tight_layout()

        temp_sino = xp.array(images[i])

        if not is_array_real(temp_sino):
            temp_sino = np.angle(temp_sino)

        if options.unsharp:
            temp_sino = scipy_module.ndimage.correlate(temp_sino, unsharp_structure)

        sobelx = scipy_module.ndimage.sobel(temp_sino, 1)
        sobely = scipy_module.ndimage.sobel(temp_sino, 0)
        temp_sino = xp.sqrt(sobelx**2 + sobely**2)
        temp_sino[temp_sino > 1] = 0

        if enable_plotting:
            r, c = 0, 0
            ax[r, c].imshow(temp_sino.get())
            ax[r, c].set_title("sobel")

        center_point = (round(temp_sino.shape[0] / 2), round(temp_sino.shape[1] / 2))
        if isinstance(temp_sino, cp.ndarray):
            temp_sino = temp_sino.get()
        temp_sino = skimage.segmentation.flood_fill(temp_sino, center_point, 1)

        if enable_plotting:
            r, c = 0, 1
            ax[r, c].imshow(temp_sino)
            ax[r, c].set_title("flood fill")

        level = skimage.filters.threshold_otsu(temp_sino)
        temp_sino = temp_sino > level
        if enable_plotting:
            r, c = 0, 2
            ax[r, c].imshow(temp_sino)
            ax[r, c].set_title("otsu threshold")

        if options.fill > 0:
            # can be put on GPU, but much slower for some reason
            temp_sino = scipy.ndimage.binary_fill_holes(temp_sino)
        if enable_plotting:
            r, c = 1, 0
            ax[r, c].imshow(temp_sino)
            ax[r, c].set_title("binary fill")

        temp_sino = xp.array(temp_sino)
        temp_sino = scipy_module.ndimage.binary_closing(temp_sino, close_structure)
        if enable_plotting:
            r, c = 1, 1
            ax[r, c].imshow(temp_sino.get())
            ax[r, c].set_title("binary closed")

        temp_sino = scipy_module.ndimage.binary_erosion(temp_sino, erode_structure)
        if isinstance(temp_sino, cp.ndarray):
            temp_sino = temp_sino.get()
        if enable_plotting:
            r, c = 1, 2
            ax[r, c].imshow(temp_sino)
            ax[r, c].set_title("binary erode")
            plt.show()

        masks[i] = temp_sino

        clear_output(wait=True)

    return masks
