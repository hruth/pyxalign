import copy
import numpy as np
import pyxalign.alignment.projection_matching as pm
from pyxalign.api.options import transform
import pyxalign.data_structures.projections as proj
from pyxalign.api.options.projections import (
    CoordinateSearchOptions,
    EstimateCenterOptions,
    ProjectionOptions,
    ProjectionTransformOptions,
)
from pyxalign.api import enums
from pyxalign.api.types import r_type
from pyxalign.timing.timer_utils import InlineTimer
import matplotlib.pyplot as plt
from tqdm import tqdm


class CenterOfRotationEstimateResults:
    def __init__(self):
        self.error = {}
        self.mean_error = {}

    def update_errors(self, error, vertical_coordinate, horizontal_coordinate):
        key = (vertical_coordinate, horizontal_coordinate)
        self.error[key] = error
        self.mean_error[key] = error.mean()

    @property
    def optimal_center_of_rotation(self) -> np.ndarray:
        y, x, Z = convert_dict_to_arrays(self.mean_error)
        min_idx = np.unravel_index(Z.argmin(), Z.shape)
        min_h = x[min_idx[1]]
        min_v = y[min_idx[0]]
        return np.array((min_v, min_h), dtype=r_type)

    @property
    def horizontal_coordinates(self) -> np.ndarray:
        return [k[1] for k in self.error.keys()]

    @property
    def vertical_coordinates(self) -> np.ndarray:
        return [k[0] for k in self.error.keys()]


def estimate_center_of_rotation(
    # projections: np.ndarray,
    projections: "proj.Projections",
    angles: np.ndarray,
    masks: np.ndarray,
    options: EstimateCenterOptions,
) -> CenterOfRotationEstimateResults:
    # Still need to update this for different croppings

    # Override projection matching options
    options = copy.deepcopy(options)
    options.projection_matching.downsample.enabled = False
    options.projection_matching.crop.enabled = False
    if options.downsample.enabled:
        scale = options.downsample.scale
    else:
        scale = 1
    options.projection_matching.secondary_mask.rad_apod /= scale
    options.projection_matching.secondary_mask.radial_smooth /= scale
    options.projection_matching.reconstruction_mask.rad_apod /= scale
    options.projection_matching.reconstruction_mask.radial_smooth /= scale

    # Create a new projections object
    # downsampled_projections = proj.PhaseProjections(
    #     projections=projections,
    #     angles=angles,
    #     masks=masks,
    #     options=ProjectionOptions(
    #         input_processing=ProjectionTransformOptions(
    #             downsample=options.downsample,
    #             crop=options.crop,
    #         )
    #     ),
    # )

    # Copy the passed in projection options -- at the moment
    # of writing this option, this is necessary to preserve
    # the pixel size and laminography angle
    projection_options = copy.deepcopy(projections.options)
    projection_options.input_processing = ProjectionTransformOptions(
        downsample=options.downsample,
        crop=options.crop,
    )

    downsampled_projections = proj.PhaseProjections(
        projections=projections.data,
        angles=angles,
        scan_numbers=projections.scan_numbers,
        masks=masks,
        options=projection_options,
        transform_tracker=copy.deepcopy(projections.transform_tracker),
    )

    vertical_coordinates, num_v_pts = format_coordinate_options(
        options.vertical_coordinate,
    )
    horizontal_coordinates, num_h_pts = format_coordinate_options(
        options.horizontal_coordinate,
    )

    center_of_rotation_estimate_results = CenterOfRotationEstimateResults()

    # Run projection-matching alignment at different center of rotation values
    inline_timer = InlineTimer("estimate center loop")
    inline_timer.start()
    with tqdm(total=num_v_pts * num_h_pts, desc="Estimate center of rotation") as progress_bar:
        for i in range(num_v_pts):
            for j in range(num_h_pts):
                downsampled_projections.center_of_rotation = (
                    np.array([vertical_coordinates[i], horizontal_coordinates[j]], dtype=r_type)
                    / scale
                )
                pma_object = pm.ProjectionMatchingAligner(
                    downsampled_projections, options.projection_matching, print_updates=False
                )
                pma_object.run()
                center_of_rotation_estimate_results.update_errors(
                    pma_object.pinned_error,
                    vertical_coordinates[i],
                    horizontal_coordinates[j],
                )
                progress_bar.update(1)
    inline_timer.end()

    return center_of_rotation_estimate_results


def format_coordinate_options(
    options: CoordinateSearchOptions,
) -> np.ndarray:
    center = options.center_estimate

    if options.enabled is False:
        coordinate_array = np.array([center], dtype=r_type)
        return coordinate_array, 1

    if options.range is None or options.spacing is None:
        raise ValueError("Invalid values are set in coordinate array options!")

    # Generate the coordinate array
    coordinate_array = center + np.arange(
        -options.range / 2,
        options.range / 2 + 1e-10,
        options.spacing,
    )
    return coordinate_array, len(coordinate_array)


def convert_dict_to_arrays(data):
    # Extract unique x and y coordinates
    v_coords = sorted(set(key[0] for key in data.keys()))
    h_coords = sorted(set(key[1] for key in data.keys()))

    # Create a 2D array
    array_2d = np.zeros((len(v_coords), len(h_coords)))

    # Fill the 2D array with the values from the dictionary
    for (v, h), value in data.items():
        v_idx = v_coords.index(v)
        h_idx = h_coords.index(h)
        array_2d[v_idx, h_idx] = value
    return np.array(v_coords), np.array(h_coords), array_2d


def plot_center_of_rotation_estimate_results(
    center_of_rotation_estimate_results: CenterOfRotationEstimateResults,
    projections: np.ndarray,
    proj_idx: int = 0,
    plot_projection_sum: bool = False,
):
    y, x, Z = convert_dict_to_arrays(center_of_rotation_estimate_results.mean_error)
    min_idx = np.unravel_index(Z.argmin(), Z.shape)
    min_x = x[min_idx[1]]
    min_y = y[min_idx[0]]

    # Create a 2x2 subplot grid
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # 2D Heatmap (Top-left)
    heatmap = ax[0, 0].imshow(
        Z[::-1, :],
        extent=[x.min(), x.max(), y.max(), y.min()],
        # extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    ax[0, 0].set_title("2D Heatmap of Mean Error")
    ax[0, 0].set_xlabel("Horizontal Coordinates")
    ax[0, 0].set_ylabel("Vertical Coordinates")
    ax[0, 0].scatter(
        min_x,
        min_y,
        color="red",
        marker="*",
        s=500,
        edgecolor="black",
        label=f"Min: ({min_x:.2f}, {min_y:.2f})",
        # label=f"Min: ({min_y:.2f}, {min_x:.2f})",
    )
    ax[0, 0].legend()
    fig.colorbar(heatmap, ax=ax[0, 0], shrink=0.8, aspect=10)

    # Plot all measured points
    plt.sca(ax[0, 1])
    plot_coordinate_search_points(projections, x, y, proj_idx, plot_projection_sum, show_plot=False)
    plt.plot(min_x, min_y, "*", color="red", markersize=10, markeredgecolor="black")

    # Slice in the X direction (row corresponding to min_y) (Bottom-left)
    z_slice_x = Z[min_idx[0], :]
    ax[1, 0].axvline(min_x, color="gray", linestyle=":")
    # ax[1, 0].plot(x, z_slice_x, ".-", label=f"Slice at Y={min_y:.2f}")
    ax[1, 0].plot(x, z_slice_x, ".-", label=f"Slice at X={min_x:.2f}")
    ax[1, 0].set_title("Slice along horizontal direction")  # "Slice in X direction")
    ax[1, 0].set_xlabel("Horizontal Coordinates")
    ax[1, 0].set_ylabel("Mean Error")
    ax[1, 0].legend()
    ax[1, 0].grid(linestyle=":")

    # Slice in the Y direction (column corresponding to min_x) (Bottom-right)
    z_slice_y = Z[:, min_idx[1]]
    ax[1, 1].axvline(min_y, color="gray", linestyle=":")
    ax[1, 1].plot(y, z_slice_y, ".-", label=f"Slice at Y={min_y:.2f}")
    ax[1, 1].set_title("Slice along vertical direction")
    ax[1, 1].set_xlabel("Vertical Coordinates")
    ax[1, 1].set_ylabel("Mean Error")
    ax[1, 1].legend()
    ax[1, 1].grid(linestyle=":")

    y_ranges = (np.diff(ax[1, 0].get_ylim()), np.diff(ax[1, 1].get_ylim()))
    idx_max = np.argmax(y_ranges)
    shared_range = y_ranges[idx_max] / 2 * np.array([-1, 1])
    if idx_max == 0:
        ax[1, 1].set_ylim(np.mean(ax[1, 1].get_ylim()) + shared_range)
    elif idx_max == 1:
        ax[1, 0].set_ylim(np.mean(ax[1, 0].get_ylim()) + shared_range)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_coordinate_search_points(
    projections: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    proj_idx: int = 0,
    plot_projection_sum: bool = False,
    show_plot: bool = True,
):
    # Plot all measured points
    if plot_projection_sum:
        image = projections.sum(0)
    else:
        image = projections[proj_idx]
    plt.imshow(image, cmap="bone")
    plt.plot(*np.meshgrid(x, y), ".", color="gray", markeredgecolor="black")
    plt.title("Projection")
    plt.xlabel("Horizontal Direction")
    plt.ylabel("Vertical Direction")
    if show_plot:
        plt.show()
