from numbers import Number
from typing import Optional
from ipywidgets import interact
import ipywidgets as widgets

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact
import cupy as cp
from llama.api.options.plotting import PlotDataOptions

from llama.api.types import ArrayType


def make_image_slider_plot(images: ArrayType):
    # Create the play button and slider (slider will not be displayed)
    play = widgets.Play(
        value=0,
        min=0,
        max=len(images) - 1,
        interval=500,
        description="Play",
        disabled=False,
    )

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(images) - 1,
        description="index",
        visible=False,  # Hide the slider
    )

    # Link the play button and slider
    widgets.jslink((play, "value"), (slider, "value"))

    # Function to update the plot
    def update_plot(idx):
        plt.imshow(images[idx])
        plt.show()

    # Use interact with the slider (without displaying the slider)
    interact(update_plot, idx=slider)

    # Display only the play button
    display(play)


def plot_sum_of_images(images: ArrayType):
    plt.imshow(images.sum(0))
    plt.show()


def plot_slice_of_3D_array(
    images: ArrayType,
    options: PlotDataOptions,
    pixel_size: Optional[Number] = None,
    show_plot: bool = True,
):
    if options.process_func is None:
        options.process_func = lambda x: x
    if options.index is None:
        index = 0
    else:
        index = options.index
    image = options.process_func(images[index])

    if options.widths is None:
        widths = np.array(image.shape)
    else:
        widths = options.widths[::-1] * 1

    if cp.get_array_module(images) is cp:
        image = image.get()

    centers = np.array(image.shape) / 2 + options.center_offsets[::-1]

    x_idx = (centers[1] + np.array([-widths[1] / 2, widths[1] / 2])).astype(int)
    x_idx = np.clip(x_idx, 0, image.shape[1])
    y_idx = (centers[0] + np.array([-widths[0] / 2, widths[0] / 2])).astype(int)
    y_idx = np.clip(y_idx, 0, image.shape[0])

    image = image[y_idx[0] : y_idx[1], x_idx[0] : x_idx[1]]

    plt.imshow(image, cmap=options.cmap)

    if options.scalebar.enabled:
        add_scalebar(pixel_size, image.shape[1], options.scalebar.fractional_width)

    if show_plot:
        plt.show()


def add_scalebar(pixel_size: Number, image_width: int, scalebar_fractional_width: float = 0.15):
    # Update the scalebar to be length in microns with exactly 2 decimals of precision
    round_to = 1e-8
    display_units = 1e-6
    m = pixel_size / round_to
    scale_string = r"$\mu m$"
    scalebar_width_px = scalebar_fractional_width * image_width
    scalebar_width_px = int(scalebar_width_px * m) / m  # Round to round_to
    scalebar_width_si = scalebar_width_px * pixel_size / display_units  # Convert to display_units

    scalebar = AnchoredSizeBar(
        transform=plt.gca().transData,
        size=scalebar_width_px,
        label=f"{round(scalebar_width_si, 2)} {scale_string}",
        loc="lower right",
        color="sandybrown",
        frameon=False,
        # size_vertical=2,
        fontproperties=fm.FontProperties(size=10),#, weight="bold"),
    )
    plt.gca().add_artist(scalebar)
