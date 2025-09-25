from typing import Optional
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from pyxalign.plotting.plotters import add_scalebar
from pyxalign.transformations.functions import image_crop


def plot_slice_comparison_with_insets(
    volume_paths: list[str],
    layer_indices: list[int],
    pixel_sizes: list[float],
    outer_crop_width_m: float = 20e-6,
    zoom_width_m: float = 2e-6,
    inset_zoom: int = 4,
    plot_center_frac=(0.75, 0.4),
    inset_loc: str = "upper right",
    scalebar_fractional_width: float = 0.15,
    append_title_list: Optional[list[str]] = None,
    clim_mult_list: Optional[list[tuple[float]]] = None,
    n_rows: int = 1,
    figsize: Optional[tuple[int]] = None,
    show_plot: bool = True,
):
    """
    Plot a comparison of image slices from multiple TIFF volumes with
    zoomed insets.

    This function reads a specified layer (slice) from each TIFF volume
    on disk, applies an outer crop to focus on a larger region, and then
    adds a zoomed inset showing a smaller subregion centered at a
    fractional position. Each subplot displays its own scalebar and
    optional contrast adjustment, and titles can be automatically
    generated or appended to.

    Args:
        volume_paths (list[str]):
            Paths to the TIFF volume files.
        layer_indices (list[int]):
            Slice indices to extract from each volume.
        pixel_sizes (list[float]):
            Physical pixel sizes (in meters) corresponding to each volume.
        outer_crop_width_m (float):
            Side length (in meters) of the outer crop applied to each
            slice. Defaults to 20e-6.
        zoom_width_m (float):
            Side length (in meters) of the zoomed inset region. Defaults
            to 2e-6.
        inset_zoom (int):
            Zoom factor for the inset axes. Defaults to 4.
        plot_center_frac (tuple[float, float]):
            Fractional (y, x) coordinates in [0, 1] on the cropped image
            that define the center of the inset region. Defaults to (0.75, 0.4).
        inset_loc (str):
            Matplotlib location code for placing the inset axes (e.g.,
            "upper right"). Defaults to "upper right".
        scalebar_fractional_width (float):
            Fraction of the inset width occupied by the scalebar. Defaults
            to 0.15.
        append_title_list (Optional[list[str]]):
            Additional strings to append to each subplot title. If None, no
            extra text is appended. Defaults to None.
        clim_mult_list (Optional[list[tuple[float, float]]]):
            Multipliers for adjusting the colormap intensity range per
            subplot as (low_mul, high_mul). If None, uses default
            intensities. Defaults to None.
        n_rows (int):
            Number of rows in the subplot grid. Defaults to 1.
        figsize (Optional[tuple[int, int]]):
            Figure size in inches as (width, height). If None,
            Matplotlib default is used. Defaults to None.
        show_plot (bool):
            If True, calls plt.show() to display the figure. Defaults
            to True.

    Returns:
        matplotlib.figure.Figure:
            The figure object containing the grid of subplots, each with its inset.
    """
    if clim_mult_list is None:
        clim_mult_list = [None] * (len(volume_paths))

    n_cols = int(np.ceil(len(volume_paths) / n_rows))
    fig, ax = plt.subplots(n_rows, n_cols, layout="compressed", figsize=figsize)
    ax = ax.ravel()
    for i in range(n_cols * n_rows):
        plt.sca(ax[i])
        plot_title = rf"$\nu$ = {pixel_sizes[i] * 1e9:.1f} nm"
        if append_title_list is not None:
            plot_title += f"{append_title_list[i]}"
        plt.title(plot_title)
        # pick exact pixel center on the shown (large-cropped) image
        outer_crop_width_px = int(outer_crop_width_m / pixel_sizes[i])
        zoom_width_px = int(zoom_width_m / pixel_sizes[i])
        plot_tiff_layer(
            ax[i],
            volume_paths[i],
            layer_indices[i],
            zoom_width_px,
            pixel_size=pixel_sizes[i],
            inset_loc=inset_loc,
            inset_zoom=inset_zoom,
            large_crop_size=outer_crop_width_px,
            scalebar_fractional_width=scalebar_fractional_width,
            clim_mult=clim_mult_list[i],
            plot_center_frac=plot_center_frac,
        )
    if show_plot:
        plt.show()

    return fig


def plot_tiff_layer(
    ax,
    filepath: str,
    layer_index: int,
    plot_crop_width: int,
    pixel_size: float,
    inset_loc="upper right",
    inset_zoom=4,
    large_crop_size=1200,
    scalebar_fractional_width=0.15,
    clim_mult=None,
    return_layer=False,
    *,
    plot_center=None,  # (y, x) in pixels on the (possibly) cropped image
    plot_center_frac=None,  # (fy, fx) in [0,1]; ignored if plot_center is provided
):
    """
    Show the (optionally large-cropped) slice on `ax` and add a zoomed inset of side `plot_crop_width`.

    Choose inset source region by:
      - plot_center=(y, x) in pixels (relative to shown image after large_crop_size crop), OR
      - plot_center_frac=(fy, fx) in [0,1].
    If neither is provided, center of the image is used.
    """
    with tifffile.TiffFile(filepath) as tif:
        layer = tif.pages[layer_index].asarray()

    # Apply your larger background crop first (your image_crop)
    layer = image_crop(layer, large_crop_size, large_crop_size)

    # Parent image
    im = ax.imshow(layer, cmap="bone")
    if clim_mult is not None:
        default_clim = im.get_clim()
        clim_mean = np.mean(default_clim)
        clim_range = default_clim[1] - default_clim[0]
        new_clim = (
            clim_mean - clim_mult[0] * clim_range / 2,
            clim_mean + clim_mult[1] * clim_range / 2,
        )
        im.set_clim(new_clim)
    ax.set_xticks([])
    ax.set_yticks([])

    h, w = layer.shape[:2]

    # Determine inset center
    if plot_center is not None:
        cy, cx = plot_center
    elif plot_center_frac is not None:
        fy, fx = plot_center_frac
        # clamp fractions just in case
        fy = float(np.clip(fy, 0.0, 1.0))
        fx = float(np.clip(fx, 0.0, 1.0))
        cy = int(round(fy * (h - 1)))
        cx = int(round(fx * (w - 1)))
    else:
        cy, cx = h // 2, w // 2

    # Inset crop bounds
    x1, x2, y1, y2 = _clamped_crop_bounds(h, w, plot_crop_width, cy, cx)

    # Inset axes & view
    inset_color = "sandybrown"
    ls = "-"
    inset_linewidth = 1.1
    axins = zoomed_inset_axes(ax, zoom=inset_zoom, loc=inset_loc, borderpad=0.3)
    im = axins.imshow(layer, cmap="bone")
    if clim_mult is not None:
        im.set_clim(new_clim)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)  # origin='upper' default -> invert y to show correctly
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor(inset_color)
        spine.set_linewidth(inset_linewidth)  # optional, to match thickness
        spine.set_linestyle(ls)
    plt.sca(axins)
    add_scalebar(pixel_size, plot_crop_width, scalebar_fractional_width=scalebar_fractional_width)
    # Connect and outline the marked region on the parent
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec=inset_color, lw=inset_linewidth, ls=ls)
    if not return_layer:
        return im, axins
    else:
        return im, axins, layer


def _clamped_crop_bounds(h: int, w: int, crop: int, cy: int, cx: int):
    """Return (x1, x2, y1, y2) for a square crop of side `crop` centered at (cy, cx)."""
    half = crop // 2
    x1 = max(0, cx - half)
    x2 = min(w, cx - half + crop)
    y1 = max(0, cy - half)
    y2 = min(h, cy - half + crop)
    # ensure non-empty in degenerate cases
    if x2 - x1 < 1:
        x2 = min(w, x1 + 1)
    if y2 - y1 < 1:
        y2 = min(h, y1 + 1)
    return x1, x2, y1, y2
