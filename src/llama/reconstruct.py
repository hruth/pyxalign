from typing import Optional
import numpy as np
import astra

import llama.projections as projections


# def get_astra_reconstructor_geometry(
#     projections: "projections.PhaseProjections",
# ) -> tuple[dict, np.ndarray]:
def get_astra_reconstructor_geometry(
    sinogram: np.ndarray,
    angles: np.ndarray,
    n_pix: np.ndarray,
    center_of_rotation: np.ndarray,
    lamino_angle: float,
    tilt_angle: float = 0.0,
    skew_angle: float = 0.0,
) -> tuple[dict, np.ndarray]:
    pixel_scale = [1, 1]
    scan_geometry_config = {}
    scan_geometry_config["iVolX"] = n_pix[0]
    scan_geometry_config["iVolY"] = n_pix[1]
    scan_geometry_config["iVolZ"] = n_pix[2]
    scan_geometry_config["iProj_angles"] = len(angles)
    scan_geometry_config["iProjU"] = sinogram.shape[2]
    scan_geometry_config["iProjV"] = sinogram.shape[1]
    scan_geometry_config["iRaysPerDet"] = 1
    scan_geometry_config["iRaysPerDetDim"] = 1
    scan_geometry_config["iRaysPerVoxelDim"] = 1
    sourceDistance = 1

    # Get projection geometry
    angles = (angles + 90) * np.pi / 180
    lamino_angle = (np.pi / 2 - lamino_angle * np.pi / 180) * np.ones((len(angles), 1))
    tilt_angle = np.pi / 180 * tilt_angle * np.ones((len(angles), 1))
    skew_angle = np.pi / 180 * skew_angle * np.ones((len(angles), 1))
    pixel_scale = pixel_scale * np.ones((len(angles), 2))
    rotation_center = np.array(
        [center_of_rotation[1], center_of_rotation[0]], dtype=np.float32
    )
    CoR_offset = rotation_center - np.array(sinogram.shape[1:][::-1]) / 2 # Might need the sign flipped!

    # We generate the same geometry as the circular one above.
    vectors = np.zeros((len(angles), 12))

    lamino_angle = lamino_angle.transpose()

    # https://www.astra-toolbox.com/docs/geom3d.html
    # Vectors: (rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ)

    # ray direction
    vectors[:, 0] = np.sin(angles) * np.cos(lamino_angle)
    vectors[:, 1] = -np.cos(angles) * np.cos(lamino_angle)
    vectors[:, 2] = np.sin(lamino_angle)
    vectors[:, [0, 1, 2]] = vectors[:, [0, 1, 2]] * sourceDistance

    # center of detector
    vectors[:, [3, 4, 5]] = 0

    # vector from detector pixel (0,0) to (0,1)
    vectors[:, 6] = np.cos(angles) / pixel_scale[:, 0]
    vectors[:, 7] = np.sin(angles) / pixel_scale[:, 0]
    vectors[:, 8] = 0 / pixel_scale[:, 0]

    # vector from detector pixel (0,0) to (1,0)
    vectors[:, 9] = -np.sin(lamino_angle) * np.sin(angles) / pixel_scale[:, 1]
    vectors[:, 10] = np.sin(lamino_angle) * np.cos(angles) / pixel_scale[:, 1]
    vectors[:, 11] = np.cos(lamino_angle) / pixel_scale[:, 1]

    # Center offset alignment
    vectors[:, 3:6] = vectors[:, 3:6] - (
        vectors[:, 9:12] * (CoR_offset[0]) + vectors[:, 6:9] * (CoR_offset[1])
    )

    # Apply Rodrigues' rotation formula to rotate and skew detector
    # Apply tilt angle: rotate detector in plane perpendicular to the beam axis
    for i in range(len(angles)):
        vectors[i, 6:9] = (
            vectors[i, 6:9] * np.cos(tilt_angle[i])
            + np.cross(vectors[i, 0:3], vectors[i, 6:9]) * np.sin(tilt_angle[i])
            + vectors[i, 0:3]
            * np.dot(vectors[i, 0:3], vectors[i, 6:9])
            * (1 - np.cos(tilt_angle[i]))
        )
        vectors[i, 9:12] = (
            vectors[i, 9:12] * np.cos(tilt_angle[i])
            + np.cross(vectors[i, 0:3], vectors[i, 9:12]) * np.sin(tilt_angle[i])
            + vectors[i, 0:3]
            * np.dot(vectors[i, 0:3], vectors[i, 9:12])
            * (1 - np.cos(tilt_angle[i]))
        )

    # Apply skew angle: rotate only one axis of the detector
    for i in range(len(angles)):
        vectors[i, 9:12] = (
            vectors[i, 9:12] * np.cos(skew_angle[i] / 2)
            + np.cross(vectors[i, 0:3], vectors[i, 9:12]) * np.sin(skew_angle[i] / 2)
            + vectors[i, 0:3]
            * np.dot(vectors[i, 0:3], vectors[i, 9:12])
            * (1 - np.cos(skew_angle[i] / 2))
        )

    return scan_geometry_config, vectors


def create_astra_reconstructor_config(
    sinogram: np.ndarray, scan_geometry_config: dict, vectors: np.ndarray
):
    geometries = get_geometries(scan_geometry_config, vectors)
    astra_config = astra.astra_dict("BP3D_CUDA")  # update this for cpu option later
    astra_config["ReconstructionDataId"] = astra.data3d.create("-vol", geometries["vol_geom"])
    astra_config["ProjectionDataId"] = astra.data3d.create(
        "-sino", geometries["proj_geom"], sinogram.transpose([1, 0, 2])
    )
    return astra_config


def get_geometries(scan_geometry_config: dict, vectors: np.ndarray) -> dict:
    geometries = {}
    geometries["vol_geom"] = astra.create_vol_geom(
        scan_geometry_config["iVolX"], scan_geometry_config["iVolY"], scan_geometry_config["iVolZ"]
    )
    geometries["proj_geom"] = astra.create_proj_geom(
        "parallel3d_vec",
        scan_geometry_config["iProjV"],  # detector rows
        scan_geometry_config["iProjU"],  # detector columns
        vectors,
    )

    return geometries


def update_astra_reconstructor_sinogram(sinogram: np.ndarray, astra_config: dict):
    # # may be unecessary
    # astra.data3d.change_geometry(astra_config["ReconstructionDataId"], geometries["vol_geom"])
    # # may be unecessary
    # astra.data3d.change_geometry(astra_config["ProjectionDataId"], geometries["proj_geom"])
    astra.data3d.store(astra_config["ProjectionDataId"], sinogram.transpose([1, 0, 2]))


def get_3D_reconstruction(astra_config: Optional[dict] = None) -> tuple[np.ndarray, dict, dict]:
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(astra_config)

    # Run the reconstruction algorithm
    astra.algorithm.run(alg_id)
    astra.algorithm.clear()

    # Retrieve the reconstruction
    # rec = astra.data3d.get_shared(astra_config['ReconstructionDataId'])
    reconstruction = astra.data3d.get(astra_config["ReconstructionDataId"])

    # Delete the stored astra data # Is this made null by the clear action?
    # astra.data3d.delete(astra_config["ProjectionDataId"])

    return reconstruction
