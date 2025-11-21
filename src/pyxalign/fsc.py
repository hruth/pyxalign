import h5py
import numpy as np
from pyxalign import options
from pyxalign.api.options_utils import set_all_device_options
from pyxalign.data_structures.task import load_task
from pyxalign.gpu_utils import free_blocks_on_all_gpus

gpu_list = (0,)


def get_fsc(task_path: str, cropped_vol_width: int, pixel_size_str: str, save_results: bool = True):
    print("get_fsc")
    n_splits = 2
    volumes = {}
    for split_num in range(n_splits):
        task = load_task(task_path)
        drop_scans = task.phase_projections.scan_numbers[split_num::2]
        task.phase_projections.drop_projections(drop_scans)
        task.phase_projections.pin_arrays()
        task.phase_projections.options.reconstruct.astra.back_project_gpu_indices = gpu_list
        set_all_device_options(
            task.phase_projections.options,
            options.DeviceOptions(gpu=options.GPUOptions(chunk_length=2, gpu_indices=(split_num,))),
        )
        print(f"get 3D recon {split_num}")
        task.phase_projections.get_3D_reconstruction()
        volumes[split_num] = task.phase_projections.volume.data * 1
        free_blocks_on_all_gpus(gpu_list)

    if pixel_size_str == "22nm":
        cropped_vol_width *= 2
    elif pixel_size_str == "11nm":
        cropped_vol_width *= 4

    try:
        volume_width = volumes[0].shape[1]
        crop_by = (volume_width - cropped_vol_width) / 2
        print("getting fsc...")
        f, fsc, n_shell = fsc_version_2(
            volumes[0],
            volumes[1],
            nbins=int(volume_width - crop_by * 2),
            crop_by=int(crop_by),
            rmax=0.5,
        )
        print("fsc calculation finished")
    except Exception as ex:
        print("error when getiting fsc")
        print("crop_by:", crop_by)
        print("volume_width:", volume_width)
        f, fsc, n_shell = 0, 0, 0
    try:
        if save_results:
            with h5py.File(os.path.join(fsc_results_folder, f"fsc_{pixel_size_str}.h5"), "w") as F:
                F["f"] = f
                F["fsc"] = fsc
                F["n_shell"] = n_shell
                F["one_bit_curve"] = one_bit_threshold(n_shell, D_over_L=1)
                F["pixel_size"] = task.phase_projections.pixel_size
    except Exception as ex:
        print("error when saving results")

    return f, fsc, n_shell, volumes


def _freq_radius(shape, voxel_size=1.0):
    """3D radial frequency grid (cycles per physical unit)."""
    nz, ny, nx = shape
    fx = np.fft.fftfreq(nx, d=voxel_size)
    fy = np.fft.fftfreq(ny, d=voxel_size)
    fz = np.fft.fftfreq(nz, d=voxel_size)
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="xy", sparse=False)
    # meshgrid -> (ny, nx, nz); move axes to (nz, ny, nx) to match volumes
    FX = np.moveaxis(FX, -1, 0)
    FY = np.moveaxis(FY, -1, 0)
    FZ = np.moveaxis(FZ, -1, 0)
    return np.sqrt(FX**2 + FY**2 + FZ**2)


def fsc_version_2(vol1, vol2, nbins=200, rmax=None, voxel_size=1.0, crop_by: int = 0):
    """
    Returns:
      f:      per-shell frequency (cycles per unit)
      fsc:    Fourier shell correlation per shell
      n_shell: voxel count per shell
    """
    if crop_by:
        vol1 = vol1[:, crop_by:-crop_by, crop_by:-crop_by]
        vol2 = vol2[:, crop_by:-crop_by, crop_by:-crop_by]

    # FFTs
    V1 = np.fft.fftn(np.asarray(vol1, dtype=np.float32))
    V2 = np.fft.fftn(np.asarray(vol2, dtype=np.float32))

    # Radial frequency
    r = _freq_radius(V1.shape, voxel_size=voxel_size)
    if rmax is None:
        # Up to the smallest-axis Nyquist radius is fine; r already reflects that
        rmax = r.max()

    # Bin edges and indices
    edges = np.linspace(0.0, rmax, nbins + 1, dtype=np.float64)
    # digitize returns 1..nbins; convert to 0..nbins-1 and mask out-of-range
    shell = np.digitize(r.ravel(), edges) - 1
    valid = (shell >= 0) & (shell < nbins)

    # Values to accumulate
    v1 = V1.ravel()[valid]
    v2 = V2.ravel()[valid]
    sh = shell[valid]

    num_vals = np.real(v1 * np.conj(v2))
    den1_vals = np.abs(v1) ** 2
    den2_vals = np.abs(v2) ** 2
    fr_vals = r.ravel()[valid]

    # Bin sums via bincount
    n_shell = np.bincount(sh, minlength=nbins)
    num_sum = np.bincount(sh, weights=num_vals, minlength=nbins)
    den1_sum = np.bincount(sh, weights=den1_vals, minlength=nbins)
    den2_sum = np.bincount(sh, weights=den2_vals, minlength=nbins)

    # Prefer the mean frequency in each shell (instead of mid-edge)
    f_sum = np.bincount(sh, weights=fr_vals, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n_shell > 0, f_sum / n_shell, np.nan)
        denom = np.sqrt(den1_sum * den2_sum)
        fsc = np.where(denom > 0, num_sum / denom, np.nan)

    # Optional: drop DC bin (first) if desired
    # f[0], fsc[0] = np.nan, np.nan

    return f, fsc, n_shell


def one_bit_threshold(
    n_shell: np.ndarray,
    D_over_L: float = 1,
    n_asym: int = 1,
) -> np.ndarray:
    """
    Compute the 1-bit FSC threshold curve T_1bit per shell.

    Parameters
    ----------
    n_shell : array
        Raw number of Fourier voxels in each shell i (before symmetry/size corrections).
    D_over_L : float
        Object-to-box linear size ratio (D/L). If unknown, 2/3 is often assumed.
    n_asym : int
        Number of asymmetric units (symmetry correction). 1 for no symmetry.

    Returns
    -------
    T1bit : array of same length as n_shell
        The 1-bit threshold value for each shell.
    """
    n_shell = np.asarray(n_shell, dtype=np.float64)
    # Effective number of independent voxels per shell
    size_factor = (1.5 * D_over_L) ** 2  # (3/2 * D/L)^2
    ne = n_shell * size_factor / (2.0 * max(n_asym, 1))
    # ne = np.clip(ne, 1.0, np.inf)

    # 1-bit threshold curve (van Heel & Schatz 2005, Eq. 14)
    # ne=n_shell
    T = (0.5 + 2.414213562373095 / np.sqrt(ne)) / (1.5 + 1.4142135623730951 / np.sqrt(ne))
    return T


def one_half_bit_threshold(
    n_shell: np.ndarray, D_over_L: float = 2 / 3, n_asym: int = 1, eps: float = 1e-12
):
    n_shell = np.asarray(n_shell, dtype=np.float64)
    # Effective number of independent voxels per shell
    size_factor = (1.5 * D_over_L) ** 2  # (3/2 * D/L)^2
    ne = n_shell * size_factor / (2.0 * max(n_asym, 1))
    # ne = np.clip(ne, 1.0, np.inf)

    # 1-bit threshold curve (van Heel & Schatz 2005, Eq. 14)
    T = (0.2071 + 1.9102 / (np.sqrt(ne) + eps)) / (1.2071 + 0.9102 / (np.sqrt(ne) + eps))
    return T


def resolution_from_curve(freqs, curve, n_shell, threshold_curve_type="one-bit"):
    # find first index where curve falls below level
    if threshold_curve_type == "one-bit":
        level = one_bit_threshold(n_shell, 1, 1)
    elif threshold_curve_type == "half-bit":
        level = one_half_bit_threshold(n_shell, 1, 1)
    idx = np.where(curve < level)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    f = freqs[i]
    return (1.0 / f) if f > 0 else None  # resolution in length units


def get_resolution_crossing(fsc: np.ndarray, x_bit_curve: np.ndarray, f:np.ndarray):
    idx = np.where((fsc < x_bit_curve))# and [not np.isnan(x) for x in one_bit_curve])
    if len(idx[0]) == 0:
        f_crossing = f[-1]
    else:
        f_crossing = f[idx[0][0]]
    resolution = 1/f_crossing
    return f_crossing, resolution