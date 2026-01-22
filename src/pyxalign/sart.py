from typing import Optional
import numpy as np
import cupy as cp
import astra
import copy
import tqdm
import scipy

from pyxalign import reconstruct
from pyxalign.api.options.device import DeviceOptions
from pyxalign.gpu_utils import create_empty_pinned_array_like, pin_memory
from pyxalign.timing.timer_utils import InlineTimer, timer

from pyxalign.api.types import r_type


def sart_prepare(
    volume: np.ndarray,
    projection_size: np.ndarray,
    angles: np.ndarray,
    reconstruction_size: tuple,
    center_of_rotation: np.ndarray,
    laminography_angle: float,
    tilt_angle: float,
    skew_angle: float,
):
    scan_geometry_config, vectors = reconstruct.get_astra_reconstructor_geometry(
        size=projection_size,
        angles=angles,
        n_pix=reconstruction_size,
        center_of_rotation=center_of_rotation,
        lamino_angle=laminography_angle,
        tilt_angle=tilt_angle,
        skew_angle=skew_angle,
    )
    object_geometries = reconstruct.get_object_geometries(scan_geometry_config, vectors)
    volume_of_ones = np.ones_like(volume)
    r, sino_id = reconstruct.get_forward_projection(
        reconstruction=volume_of_ones,
        object_geometries=object_geometries,
        return_id=True,
    )
    astra.data3d.delete(sino_id)
    r_val = r[r.shape[0] // 2, r.shape[1] // 2, r.shape[2] // 2]
    r = (r > 0).astype(r_type) / np.sqrt(r**2 + (r_val * 1e-2) ** 2)

    return r, scan_geometry_config, vectors


@timer()
def sart(
    volume: np.ndarray,
    sinogram: np.ndarray,
    scan_geometry_config: dict,
    vectors: np.ndarray,
    r: np.ndarray,
    iterations: int = 1,
    relaxation: float = 0.0,
    constraint: Optional[callable] = None,
    n_sets: int = 1,
):
    n_angles = len(vectors)
    # optimize group sizes
    g = int(np.ceil(n_angles / n_sets))
    n_sets = int(np.ceil(n_angles / g))

    # make a config copy
    subset_geometry_config = copy.deepcopy(scan_geometry_config)
    err = np.zeros(shape=(iterations, n_angles), dtype=r_type)
    tukey_window = scipy.signal.windows.tukey(sinogram.shape[1], 0.2)[:, None].astype(r_type)

    if n_sets == 1:
        # create astra config to use (only works for 1-subtomogram)
        object_geometries = reconstruct.get_object_geometries(subset_geometry_config, vectors)
        astra_config = reconstruct.create_astra_reconstructor_config(
            sinogram, object_geometries, "BP3D_CUDA"
        )
        # p = create_empty_pinned_array_like(sinogram) # didn't make things faster for test case
    for iter in tqdm.tqdm(range(iterations)):
        for i in range(n_sets):
            # indicate projections to use in subtomogram
            a, b = g * i, np.min([n_angles, (i + 1) * g])
            if n_sets == 1:
                subset_vectors = vectors
                sinogram_subset = sinogram
            else:
                subset_vectors = vectors[a:b] * 1
                sinogram_subset = sinogram[a:b]
                # update scan geometry config
                subset_geometry_config["iProj_angles"] = len(sinogram_subset)

            if iter == 0 or n_sets != 1:
                # re-make geometries for each subtomogram
                object_geometries = reconstruct.get_object_geometries(
                    subset_geometry_config, subset_vectors
                )
            if n_sets == 1:
                # re-use astra config when there is only 1 subtomogram
                astra.data3d.store(
                    astra_config["ReconstructionDataId"],
                    volume,
                )
                p, _ = reconstruct.get_forward_projection(
                    volume_id=astra_config["ReconstructionDataId"],
                    object_geometries=object_geometries,
                    return_id=True,
                    forward_projection_id=astra_config["ProjectionDataId"],
                    # pinned_forward_projection=p, # didn't make things faster for test case
                )
            else:
                # re-make sino_ids for each subtomogram
                p, sino_id = reconstruct.get_forward_projection(
                    reconstruction=volume,
                    object_geometries=object_geometries,
                    return_id=True,
                    # pinned_forward_projection=p, # didn't make things faster for test case
                )
                astra.data3d.delete(sino_id)

            inline_timer = InlineTimer("get subset difference")
            inline_timer.start()
            p[:] = sinogram_subset - p
            p *= r[a:b] * tukey_window
            # get error before update
            err[iter, a:b] = np.sqrt(np.mean(p**2, (1, 2)))
            inline_timer.end()

            if n_sets == 1:
                # update projections manually
                reconstruct.update_stored_sinogram(p, astra_config)
            else:
                astra_config = reconstruct.create_astra_reconstructor_config(
                    p, object_geometries, "BP3D_CUDA"
                )
            rec_upd = reconstruct.get_3D_reconstruction(astra_config)
            if n_sets != 1:
                astra.data3d.delete(astra_config["ReconstructionDataId"])
                astra.data3d.delete(astra_config["ProjectionDataId"])

            inline_timer = InlineTimer("update sart volume")
            inline_timer.start()
            rec_upd *= (1 - relaxation) / len(sinogram_subset)
            # update volume
            volume += rec_upd
            # apply circulo constraint
            if constraint is not None:
                volume = constraint(volume)
            inline_timer.end()
    astra.data3d.delete(astra_config["ReconstructionDataId"])
    astra.data3d.delete(astra_config["ProjectionDataId"])
    return volume, err
