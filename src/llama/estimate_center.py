import numpy as np
import llama.alignment.projection_matching as pm
import llama.projections as proj
from llama.api.options.projections import (
    CoordinateSearchOptions,
    EstimateCenterOptions,
    ProjectionOptions,
)
from llama.api.types import r_type
import matplotlib.pyplot as plt


class CenterOfRotationEstimateResults:
    def __init__(self, vertical_coordinates, horizontal_coordinates, n_projections):
        self.vertical_coordinates = vertical_coordinates
        self.horizontal_coordinates = horizontal_coordinates

        # Initialize arrays for holding error
        num_v_pts = len(vertical_coordinates)
        num_h_pts = len(horizontal_coordinates)
        self.error = np.zeros((n_projections, num_v_pts, num_h_pts), dtype=r_type)
        self.mean_error = np.zeros((num_v_pts, num_h_pts), dtype=r_type)

    def update_errors(self, error, i, j):
        self.error[:, i, j] = error
        self.mean_error[i, j] = error.mean()

    @property
    def optimal_center_of_rotation(self) -> np.ndarray:
        h_min_idx, v_min_idx = np.unravel_index(np.argmin(self.mean_error), self.mean_error.shape)
        return np.array(
            [self.vertical_coordinates[h_min_idx], self.horizontal_coordinates[v_min_idx]]
        )


def estimate_center_of_rotation(
    projections: np.ndarray, angles: np.ndarray, masks: np.ndarray, options: EstimateCenterOptions
) -> CenterOfRotationEstimateResults:
    # Still need to update this for different croppings

    # Override projection matching options
    options.projection_matching.downsample.enabled = False
    options.projection_matching.crop.enabled = False
    if options.downsample.enabled:
        scale = options.downsample.scale
    else:
        scale = 1

    # Create a new projections object
    downsampled_projections = proj.PhaseProjections(
        projections=projections,
        angles=angles,
        masks=masks,
        options=ProjectionOptions(
            downsample=options.downsample,
            crop=options.crop,
        ),
    )

    # Get arrays of coordinates used in search
    def generate_coordinate_array(options: CoordinateSearchOptions) -> np.ndarray:
        coordinate_array = options.center_estimate + np.arange(
            -options.range / 2,
            options.range / 2 + 1e-10,
            options.spacing,
        )
        return coordinate_array, len(coordinate_array)

    vertical_coordinates, num_v_pts = generate_coordinate_array(options.vertical_coordinate)
    horizontal_coordinates, num_h_pts = generate_coordinate_array(options.horizontal_coordinate)
    center_of_rotation_estimate_results = CenterOfRotationEstimateResults(
        vertical_coordinates, horizontal_coordinates, len(projections)
    )

    # Run projection-matching alignment at different center of rotation values
    for i in range(num_v_pts):
        for j in range(num_h_pts):
            downsampled_projections.center_of_rotation = (
                np.array([vertical_coordinates[i], horizontal_coordinates[j]]) / scale
            )
            pma_object = pm.ProjectionMatchingAligner(
                downsampled_projections, options.projection_matching
            )
            pma_object.run()
            center_of_rotation_estimate_results.update_errors(pma_object.pinned_error, i, j)

    return center_of_rotation_estimate_results


def plot_center_of_rotation_estimate_results(
    center_of_rotation_estimate_results: CenterOfRotationEstimateResults,
    projections: np.ndarray,
    proj_idx: int = 0,
):
    # Assuming center_of_rotation_estimate_results contains the data
    x = center_of_rotation_estimate_results.horizontal_coordinates
    y = center_of_rotation_estimate_results.vertical_coordinates
    Z = center_of_rotation_estimate_results.mean_error

    # Find the minimum of Z and its indices
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = x[min_idx[1]]
    min_y = y[min_idx[0]]

    # Create a 2x2 subplot grid
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # 2D Heatmap (Top-left)
    heatmap = ax[0, 0].imshow(
        Z,
        extent=[x.min(), x.max(), y.max(), y.min()],
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
    )
    ax[0, 0].legend()
    fig.colorbar(heatmap, ax=ax[0, 0], shrink=0.8, aspect=10)

    # Plot all measured points
    ax[0, 1].imshow(projections[proj_idx], cmap="bone")
    ax[0, 1].plot(*np.meshgrid(x, y), ".", color="gray", markeredgecolor="black")
    ax[0, 1].plot(min_x, min_y, "*", color="red", markersize=10, markeredgecolor="black")
    ax[0, 1].set_title("Projection")
    ax[0, 1].set_xlabel("Horizontal Direction")
    ax[0, 1].set_ylabel("Vertical Direction")

    # Slice in the X direction (row corresponding to min_y) (Bottom-left)
    z_slice_x = Z[min_idx[0], :]
    ax[1, 0].plot(x, z_slice_x, ".-", label=f"Slice at Y={min_y:.2f}")
    ax[1, 0].set_title("Slice in X direction")
    ax[1, 0].set_xlabel("Horizontal Coordinates")
    ax[1, 0].set_ylabel("Mean Error")
    ax[1, 0].legend()
    ax[1, 0].grid(linestyle=":")

    # Slice in the Y direction (column corresponding to min_x) (Bottom-right)
    z_slice_y = Z[:, min_idx[1]]
    ax[1, 1].plot(y, z_slice_y, ".-", label=f"Slice at X={min_x:.2f}")
    ax[1, 1].set_title("Slice in Y direction")
    ax[1, 1].set_xlabel("Vertical Coordinates")
    ax[1, 1].set_ylabel("Mean Error")
    ax[1, 1].legend()
    ax[1, 1].grid(linestyle=":")

    # ax[0, 1].axis('off')  # Turn off the axis for the empty subplot

    # Adjust layout
    plt.tight_layout()
    plt.show()
