from abc import ABC
from numbers import Number
from typing import Optional, Sequence
from ipywidgets import interact
import ipywidgets as widgets
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np

from IPython.display import display, clear_output
import cupy as cp
from llama.api.maps import get_process_func_by_enum
from llama.api.options.plotting import PlotDataOptions
from matplotlib.image import AxesImage
import copy
from llama.api.types import ArrayType
from llama.transformations.classes import Cropper


class PlotObject(ABC):
    array: np.ndarray

    def __init__(
        self,
        array: np.ndarray,
        title: str = "slice",
        sort_idx=None,
        subplot_idx=None,
    ):
        self.array = array
        self.title = title
        self.sort_idx = sort_idx
        self.subplot_idx = subplot_idx

    def plot_callback(self):
        raise NotImplementedError


class ImagePlotObject(PlotObject):
    axes_object: AxesImage = None

    def __init__(
        self,
        array: np.ndarray,
        title: str = "slice",
        sort_idx=None,
        subplot_idx=None,
        *,
        options: Optional[PlotDataOptions] = None,
    ):
        super().__init__(array, title, sort_idx, subplot_idx)
        if options is None:
            self.options = PlotDataOptions()
        else:
            self.options = copy.deepcopy(options)

    def plot_callback(self, idx) -> callable:
        plt.title(f"{self.title} {idx}")
        self.options.index = idx
        self.axes_object = plot_slice_of_3D_array(
            images=self.array,
            options=self.options,
            show_plot=False,
            axis_image=self.axes_object,
        )


class LinePlotObject(PlotObject):
    lines = None

    def __init__(
        self,
        array: np.ndarray,
        title: str = "index",
        sort_idx=None,
        subplot_idx=None,
        *,
        label=None,
        ylim=None,
    ):
        super().__init__(array, title, sort_idx, subplot_idx)
        if len(self.array.shape) == 2:
            self.array = self.array[:, :, None]
        self.ylim = ylim
        self.label = label

    def plot_callback(self, idx) -> callable:
        plt.title(f"{self.title} {idx}")
        if self.sort_idx is not None:
            idx = int(self.sort_idx[idx])
        if self.lines is None:
            self.lines = plt.plot(self.array[idx], label=self.label)
            self.set_ylim()
            plt.grid(linestyle=":")
            plt.xlim(0, self.array.shape[1])
            if self.ylim is not None:
                plt.ylim(self.ylim)
            plt.legend()
        else:
            for i, line in enumerate(self.lines):
                line.set_ydata(self.array[idx, :, i])

    def set_ylim(self):
        plt.ylim([np.min(self.array), np.max(self.array)])


def make_image_slider_plot(
    plot_objects: Sequence[PlotObject],
    subplot_dims: Optional[Sequence] = None,
    interval: int = 500,
):
    if isinstance(plot_objects, PlotObject):
        plot_objects = [plot_objects]

    matplotlib_backend = matplotlib.get_backend()

    if subplot_dims is None:
        n_rows = 1
        n_cols = len(plot_objects)
    else:
        n_rows, n_cols = subplot_dims

    n_frames = len(plot_objects[0].array)
    # Create the play button and slider
    play = widgets.Play(
        value=0,
        min=0,
        max=n_frames - 1,
        interval=interval,
        description="Play",
    )
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_frames - 1,
        description="index",
        visible=False,  # The slider will be shown via the interact link
    )

    # Link the play button and slider
    widgets.jslink((play, "value"), (slider, "value"))

    fig, ax = plt.subplots(n_rows, n_cols)
    if n_rows == 1 and n_cols == 1:
        ax = [ax]
    plt.tight_layout()

    def update_plot(idx):
        for i in range(n_rows * n_cols):
            if n_rows > 1 and n_cols > 1:
                plot_idx = np.unravel_index(i, (n_rows, n_cols))
            else:
                plot_idx = i

            if len(plot_objects) > i:
                plot_object = plot_objects[i]
            else:
                ax[plot_idx].axis("off")
                continue

            if plot_object.subplot_idx is None:
                plt.sca(ax[plot_idx])
            else:
                # Use pre-set subplot location
                ax[plot_idx].axis("off")
                plt.subplot(*plot_object.subplot_idx)
            plot_object.plot_callback(idx)
        # fig.canvas.draw_idle()
        if matplotlib_backend == "module://ipympl.backend_nbagg":
            pass
        else:
            plt.show()

    interact(update_plot, idx=slider)

    display(play)


def plot_sum_of_images(images: ArrayType):
    plt.imshow(images.sum(0))
    plt.show()


def plot_slice_of_3D_array(
    images: ArrayType,
    options: PlotDataOptions,
    pixel_size: Optional[Number] = None,
    show_plot: bool = True,
    axis_image: Optional[AxesImage] = None,
):
    process_func = get_process_func_by_enum(options.process_func)

    if options.index is None:
        index = 0
    else:
        index = options.index
    image = process_func(images[index])

    if cp.get_array_module(images) is cp:
        image = image.get()

    crop_options = copy.deepcopy(options.crop)
    crop_options.return_view = True
    image = Cropper(options.crop).run(image[None])[0]

    if axis_image is not None:
        axis_image = axis_image.set_data(image)
    else:
        axis_image = plt.imshow(image, cmap=options.cmap)

    if options.scalebar.enabled and pixel_size is not None:
        add_scalebar(pixel_size, image.shape[1], options.scalebar.fractional_width)

    plt.clim(options.clim)

    if show_plot:
        plt.show()

    return axis_image


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
        fontproperties=fm.FontProperties(size=10, weight="bold"),
    )
    plt.gca().add_artist(scalebar)
