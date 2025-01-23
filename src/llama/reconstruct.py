from typing import Optional
import numpy as np
import cupy as cp
import scipy
import astra

from llama.api.options.device import DeviceOptions
from llama.gpu_utils import get_scipy_module
from llama.api.types import r_type, ArrayType
from llama.gpu_wrapper import device_handling_wrapper
from llama.timer import timer


@timer()
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
    scan_geometry_config_inputs = {}
    scan_geometry_config_inputs["iVolX"] = n_pix[0]
    scan_geometry_config_inputs["iVolY"] = n_pix[1]
    scan_geometry_config_inputs["iVolZ"] = n_pix[2]
    scan_geometry_config_inputs["iProj_angles"] = len(angles)
    scan_geometry_config_inputs["iProjU"] = sinogram.shape[2]
    scan_geometry_config_inputs["iProjV"] = sinogram.shape[1]
    scan_geometry_config_inputs["iRaysPerDet"] = 1
    scan_geometry_config_inputs["iRaysPerDetDim"] = 1
    scan_geometry_config_inputs["iRaysPerVoxelDim"] = 1
    sourceDistance = 1

    # Get projection geometry
    angles = (angles + 90) * np.pi / 180
    lamino_angle = (np.pi / 2 - lamino_angle * np.pi / 180) * np.ones((len(angles), 1))
    tilt_angle = np.pi / 180 * tilt_angle * np.ones((len(angles), 1))
    skew_angle = np.pi / 180 * skew_angle * np.ones((len(angles), 1))
    pixel_scale = pixel_scale * np.ones((len(angles), 2))
    rotation_center = np.array([center_of_rotation[1], center_of_rotation[0]], dtype=r_type)
    CoR_offset = (rotation_center - np.array(sinogram.shape[1:][::-1]) / 2)[::-1]

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

    return scan_geometry_config_inputs, vectors


@timer()
def create_astra_reconstructor_config(
    sinogram: np.ndarray,
    scan_geometry_config: dict,
    vectors: np.ndarray,
    astra_config: Optional[dict] = None,
) -> tuple[dict, dict]:
    geometries = get_geometries(scan_geometry_config, vectors)
    if astra_config is None:
        astra_config = astra.astra_dict("BP3D_CUDA")  # update this for cpu option later
        astra_config["ReconstructionDataId"] = astra.data3d.create("-vol", geometries["vol_geom"])
        astra_config["ProjectionDataId"] = astra.data3d.create(
            "-sino", geometries["proj_geom"], sinogram.transpose([1, 0, 2])
        )
    else:
        astra.data3d.change_geometry(astra_config["ReconstructionDataId"], geometries["vol_geom"])
        astra.data3d.change_geometry(astra_config["ProjectionDataId"], geometries["proj_geom"])
        astra.data3d.store(astra_config["ProjectionDataId"], sinogram.transpose([1, 0, 2]))

    return astra_config, geometries


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


@timer()
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


@timer()
def filter_sinogram(
    sinogram: ArrayType,
    vectors: np.ndarray,
    device_options: DeviceOptions,
    pinned_results: Optional[np.ndarray] = None,
) -> ArrayType:
    xp = cp.get_array_module(sinogram)
    # calculate the original angles
    # can these lines be replaced?
    theta = np.pi - np.arctan2(vectors[:, 1], -vectors[:, 0])
    lamino_angle = np.pi / 2 - np.arctan2(vectors[:, 2], vectors[:, 0] / np.cos(theta))

    # determine weights in case of irregular fourier space sampling
    theta = np.mod(theta - theta[0], np.pi)
    sort_idx = np.argsort(theta)
    sorted_theta = theta[sort_idx]
    n_proj = len(sinogram)

    weights = np.zeros(n_proj)
    weights[1 : n_proj - 1] = -sorted_theta[0 : n_proj - 2] / 2 + sorted_theta[2:] / 2
    weights[0] = sorted_theta[1] - sorted_theta[0]
    weights[-1] = sorted_theta[-1] - sorted_theta[-2]
    weights[sort_idx] = weights  # unsort
    if np.any(weights > 2 * np.median(weights)):
        weights[weights > 2 * np.median(weights)] = np.median(weights)
    weights = weights / np.mean(weights)
    weights = weights * (np.pi / 2 / n_proj) * np.sin(lamino_angle)
    weights = weights.astype(r_type)

    # The size here doesn't make a ton of sense, should look into it later.
    filter = design_filter(sinogram.shape[2])

    # account for laminography tilt + unequal spacing of the tomo
    # angles
    filter = weights[:, None] * filter

    apply_filter_wrapped = device_handling_wrapper(
        func=apply_filter,
        options=device_options,
        chunkable_inputs_for_gpu_idx=[0],
        chunkable_inputs_for_cpu_idx=[1],
        pinned_results=pinned_results,
    )

    filtered_sinogram = apply_filter_wrapped(sinogram, filter, sinogram.shape[2])

    # Check this is true for the mixed memory config
    # assert filtered_sinogram is sinogram

    # RESUME HERE -- NEED TO SEE HOW TO DEAL WITH PINNED RESULTS IN THE
    # GET_3D_RECONSTRUCTION METHOD IN THE PROJECTIONS

    return filtered_sinogram


def apply_filter(sinogram: ArrayType, filter: ArrayType, n_pix_width: int) -> ArrayType:
    xp = cp.get_array_module(sinogram)
    scipy_module: scipy = get_scipy_module(sinogram)

    m = filter.shape[1]
    pad_width = int((filter.shape[1] - n_pix_width) / 2)
    if n_pix_width % 2 == 0:
        array_padder = ([0, 0], [0, 0], [pad_width, pad_width])
    else:
        array_padder = ([0, 0], [0, 0], [pad_width, pad_width + 1])

    sinogram = xp.pad(sinogram, array_padder, "symmetric")

    # sinogram holds fft of projections
    sinogram = scipy_module.fft.fft(sinogram, axis=2)

    sinogram = sinogram * xp.array(filter[:, None])

    sinogram = scipy_module.fft.ifft(sinogram, axis=2)
    sinogram = xp.real(sinogram)

    # Truncate the filtered projections
    truncate_idx = np.arange(m / 2 - n_pix_width / 2, m / 2 + n_pix_width / 2, dtype=int)
    sinogram = sinogram[:, :, truncate_idx]

    return sinogram


def design_filter(width: int, d: float = 1.0) -> ArrayType:
    order = np.max([64, 2 ** (np.ceil(np.log2(2 * width)))])
    filter = np.linspace(0, 1, int(order / 2), dtype=r_type)
    # Frequency axis up to Nyquist
    w = np.linspace(0, np.pi, int(order / 2), dtype=r_type)
    # Crop the frequency response
    filter[w > np.pi * d] = 0
    # Make filter symmetric
    filter = np.append(filter, filter[::-1])

    return filter


@timer()
def get_forward_projection(
    reconstruction: np.ndarray,
    geometries: dict,
    pinned_forward_projection: Optional[np.ndarray] = None,
) -> np.ndarray:
    forward_projection_ID = astra.create_sino3d_gpu(
        reconstruction,
        geometries["proj_geom"],
        geometries["vol_geom"],
        returnData=False,
    )

    # if you are having issues with segmentation faults, it may be due to the
    # use of get_shared. Using get_shared is slightly faster.
    if pinned_forward_projection is None:
        pinned_forward_projection = astra.data3d.get_shared(forward_projection_ID).transpose(
            [1, 0, 2]
        )
    else:
        # pinned_forward_projection[:] = astra.data3d.get(forward_projection_ID).transpose([1, 0, 2])
        pinned_forward_projection[:] = astra.data3d.get_shared(forward_projection_ID).transpose(
            [1, 0, 2]
        )
    astra.data3d.delete(forward_projection_ID)

    return pinned_forward_projection
