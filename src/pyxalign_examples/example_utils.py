import numpy as np
import astra
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pyxalign
from pyxalign.reconstruct import (
    get_astra_reconstructor_geometry,
    get_object_geometries,
    get_forward_projection,
)
from pyxalign.transformations.helpers import round_to_divisor
from pyxalign.api import enums
from pyxalign.api.constants import divisor


def get_simulated_phantom_task(angles: np.ndarray, shift: np.ndarray):
    width = 250
    n_slices = 100
    pad_width = 0
    lamino_angle = 60
    volume_data, projection_data = get_simulated_phantom(
        volume_size=(width, width, n_slices),
        lamino_angle=lamino_angle,
        proj_padding=pad_width,
        angles=angles,
    )

    # normalize projection_data to 1
    projection_data /= projection_data.max()

    # manually generate masks
    masks = np.zeros_like(projection_data)
    # cutoff = int(pad_width / 2)
    cutoff = 32
    masks[:, cutoff:-cutoff, cutoff:-cutoff] = 1

    # create pyxalign projections object
    options = pyxalign.options.ProjectionOptions()
    options.experiment.laminography_angle = lamino_angle
    # use a sample thickness that is slightly larger than the actual volume
    options.experiment.sample_thickness = int(n_slices * 1.1)
    options.experiment.pixel_size = 1
    projections = pyxalign.data_structures.PhaseProjections(
        projections=projection_data,
        angles=angles,
        masks=masks,
        options=options,
    )
    projections.pin_arrays()
    # create pyxalign alignment task object
    task = pyxalign.data_structures.LaminographyAlignmentTask(
        pyxalign.options.AlignmentTaskOptions(),
        phase_projections=projections,
    )

    # shift the projections
    projections.shift_manager.stage_shift(
        shift, function_type=pyxalign.enums.ShiftType.FFT, eliminate_wrapping=True
    )
    projections.apply_staged_shift()

    # # crop to make the projections look more realistic
    # projections.crop_projections(
    #     pyxalign.options.CropOptions(
    #         enabled=True, vertical_range=160, horizontal_range=192
    #     )
    # )
    # projections.pad_projections(
    #     pyxalign.options.PadOptions(
    #         enabled=True,
    #         new_extent_x=projections.data.shape[2] + 128,
    #         new_extent_y=projections.data.shape[1] + 128,
    #     )
    # )

    return task


def calculate_projection_size(
    lamino_angle: float, volume_width: int, volume_thickness: int, proj_padding: int
) -> tuple[int, int]:
    x_extent = volume_width * np.sqrt(2)
    theta = lamino_angle * np.pi / 180
    y_extent = np.sqrt(2) * volume_width * np.cos(theta) + volume_thickness * np.cos(
        theta - np.pi / 2
    )
    projection_size = (y_extent + proj_padding, x_extent + proj_padding)
    projection_size = round_to_divisor(
        projection_size, enums.RoundType.CEIL, divisor * 2
    )
    print("projection size", projection_size)
    return projection_size


def get_simulated_phantom(
    volume_size: tuple, lamino_angle: float, proj_padding: int, angles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    volume_width = volume_size[1]
    volume_thickness = volume_size[2]
    # y, x dimensions of projections
    projection_size = calculate_projection_size(
        lamino_angle, volume_width, volume_thickness, proj_padding
    )
    # y, x center of rotation
    center_of_rotation = (projection_size[0] / 2, projection_size[1] / 2)
    # get geometry for doing astra computations
    scan_geometry_config, vectors = get_astra_reconstructor_geometry(
        projection_size,
        angles,
        volume_size,
        center_of_rotation,
        lamino_angle,
    )
    object_geometries = get_object_geometries(scan_geometry_config, vectors)
    # get phatnom volume
    id, volume = astra.data3d.shepp_logan(object_geometries["vol_geom"])
    print(volume.shape)
    # get forard projections of volume
    forward_projections = np.array(
        get_forward_projection(volume, object_geometries), dtype=np.float32
    )
    astra.data3d.clear()
    return volume, forward_projections


def get_simulated_displacement_curve(
    angles: np.ndarray,
    x_mult: float = 10,
    y_mult: float = 8,
    noise_coeff: float = 5,
) -> np.ndarray:
    f_x = [3, 2.2, -0.3]
    f_y = [0.3, 1.2]

    shift = np.zeros((len(angles), 2), dtype=np.float32)
    for i, f in enumerate(f_x):
        shift[:, 0] += np.sin(f * angles * np.pi / 180) * x_mult
    for i, f in enumerate(f_y):
        shift[:, 1] += np.sin(f * angles * np.pi / 180 + 0.3) * y_mult
    # stitch ends together
    m = (shift[:1].mean(0) - shift[-1:].mean(0)) / len(shift)
    linear_offset = m * np.arange(0, len(shift))[:, None]
    shift += linear_offset
    # add noise
    shift += np.random.normal(size=shift.shape) * noise_coeff
    # center around 0
    shift -= shift.mean(0)

    return shift


def get_true_displacement_residual(
    true_displacement: np.ndarray,
    estimated_displacement: np.ndarray,
    angles: np.ndarray,
):
    def model_function(
        angles: np.ndarray, a_h: float, phase: float, offset: float
    ) -> np.ndarray:
        # y is size (n_angles, 2)
        h_curve = np.sin(angles * np.pi / 180 + phase) * a_h
        v_curve = (
            np.cos(angles * np.pi / 180 + phase) * a_h * np.cos(60 * np.pi / 180)
        ) + offset
        return np.append(h_curve, v_curve)

    alignment_shift_diff = estimated_displacement - true_displacement

    y = np.append(alignment_shift_diff[:, 0], alignment_shift_diff[:, 1])
    angles = angles
    # Fit to the expected oscillatory function
    fit_vals, cov = curve_fit(model_function, angles, y)
    # print(f"Amplitude: {fit_vals[0]:.2f}, Phase {fit_vals[1]:.2f}, Y Offset {fit_vals[2]:.2f}")
    # Create fit result array
    h_curve_fit = model_function(angles, *fit_vals)[: len(angles)]
    v_curve_fit = model_function(angles, *fit_vals)[-len(angles) :]
    underlying_osc_shift = (
        np.array([h_curve_fit, v_curve_fit]).transpose().astype(np.float32)
    )

    true_residuals = alignment_shift_diff - underlying_osc_shift

    fig, ax = plt.subplots(2, 1, layout="compressed")
    colors = ["tab:blue", "tab:orange"]
    for i in range(2):
        plt.sca(ax[i])
        plt.title(["Horizontal", "Vertical"][i] + " Shift")
        plt.plot(
            angles,
            true_displacement[:, i],
            ".",
            label="True Displacement",
            color="black",
        )
        plt.plot(
            angles,
            estimated_displacement[:, i],
            ".",
            label="Estimated Displacement",
            color=colors[i],
        )
        plt.ylabel("displacement (px)")
        plt.grid(linestyle=":")
        plt.autoscale(True, "x", True)
        plt.legend(fontsize=8)
    plt.xlabel("angle (deg)")
    plt.show()

    fig, ax = plt.subplots(layout="compressed")
    plt.title("True Displacement - Estimated Displacement")
    # plt.plot(angles, true_residuals, ".")
    plt.plot(angles, true_displacement - estimated_displacement)

    plt.grid(linestyle=":")
    plt.autoscale(True, "x", True)
    plt.ylabel("displacement (px)")
    plt.xlabel("angle (deg)")
    plt.legend(["Horizontal", "Vertical"])
    plt.show()

    fig, ax = plt.subplots(layout="compressed")
    plt.title("Residuals")
    plt.plot(angles, true_residuals, ".")
    plt.grid(linestyle=":")
    plt.autoscale(True, "x", True)
    plt.ylabel("displacement (px)")
    plt.xlabel("angle (deg)")
    plt.legend(["Horizontal", "Vertical"])
    plt.show()

    return true_residuals
