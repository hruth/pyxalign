import numpy as np
import astra
from pyxalign.reconstruct import (
    get_astra_reconstructor_geometry,
    get_object_geometries,
    get_forward_projection,
)
from pyxalign.transformations.helpers import round_to_divisor
from pyxalign.api import enums
from pyxalign.api.constants import divisor


def calculate_projection_size(
    lamino_angle: float, volume_width: int, volume_thickness: int, proj_padding: int
) -> tuple[int, int]:
    x_extent = volume_width * np.sqrt(2)
    theta = lamino_angle * np.pi / 180
    y_extent = np.sqrt(2) * volume_width * np.cos(theta) + volume_thickness * np.cos(
        theta - np.pi / 2
    )
    projection_size = (y_extent + proj_padding, x_extent + proj_padding)
    projection_size = round_to_divisor(projection_size, enums.RoundType.CEIL, divisor * 2)
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
