from typing import Optional
import matplotlib.pyplot as plt
import h5py
import numpy as np
from pyxalign.timing.timer_utils import timer


class FourierShellCorrelation:
    def __init__(self, pixel_size: float, laminography_angle: Optional[float] = None):
        self.pixel_size = pixel_size
        self.laminography_angle = laminography_angle

    def get_fourier_shell_correlation(
        self,
        volume_1: np.ndarray,
        volume_2: np.ndarray,
        crop_width: Optional[int] = None,
        n_bins: Optional[int] = None,
    ):
        """Calculate FSC of two volumes"""
        if volume_1.shape != volume_2.shape:
            raise ValueError("Volume arrays must have the same shape")
        volume_width = volume_1[0].shape[1]
        if crop_width is None:
            crop_width = volume_width
        if n_bins is None:
            self.n_bins = crop_width
        else:
            self.n_bins = n_bins
        self.f, self.fsc, self.n_shell = calculate_fourier_shell_correlation(
            volume_1,
            volume_2,
            n_bins=self.n_bins,
            crop_by=int((volume_width - crop_width) / 2),
            rmax=0.5,
            voxel_size=1,
            laminography_angle=self.laminography_angle,
        )
        # convert to meters
        self.f = self.f / self.pixel_size

    def plot_fsc(
        self,
        plot_half_bit_curve: bool = True,
        plot_freq_crossing: bool = True,
        label: Optional[str] = None,
        show_plot: bool = True,
    ):
        plot_f = self.f * 1e-6
        plt.title("Fourier Shell Correlation")
        (ln,) = plt.plot(plot_f, self.fsc, label=label)
        if plot_half_bit_curve:
            half_bit_threshold = one_half_bit_threshold(self.n_shell, 1, 1)
            plt.plot(
                plot_f,
                half_bit_threshold,
                "k:",
            )
            if plot_freq_crossing:
                f_crossing, resolution = get_resolution_crossing(
                    self.fsc, half_bit_threshold, self.f
                )
                plt.axvline(f_crossing * 1e-6, color=ln.get_color(), ls="--")
                res_string = f"resolution crossing: {resolution * 1e9:0.2f} nm"
                if label is not None:
                    res_string = label + " - " + res_string
                print(res_string)
        plt.xlabel("spatial frequency $\mu m ^{-1}$")
        plt.ylabel("spatial frequency")
        plt.grid(ls=":")
        plt.autoscale(True, "x", True)
        plt.ylim([0, 1.01])
        if label is not None:
            plt.legend()
        if show_plot:
            plt.show()

    def save_fsc(self, file_path: str):
        with h5py.File(file_path, "w") as F:
            F.create_dataset(name="fsc", data=self.fsc)
            F.create_dataset(name="f", data=self.f)
            F.create_dataset(name="n_shell", data=self.n_shell)
            F.create_dataset(name="pixel_size", data=self.pixel_size)
            if self.laminography_angle is not None:
                F.create_dataset(name="laminography_angle", data=self.laminography_angle)


def load_fsc_object(file_path: str) -> FourierShellCorrelation:
    with h5py.File(file_path, "r") as F:
        if "laminography_angle" in F.keys():
            lamino_angle = F["laminography_angle"][()]
        fsc = FourierShellCorrelation(
            pixel_size=F["pixel_size"][()], laminography_angle=lamino_angle
        )
        fsc.fsc = F["fsc"][()]
        fsc.f = F["f"][()]
        fsc.n_shell = F["n_shell"][()]

    return fsc


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
    return np.sqrt(FX**2 + FY**2 + FZ**2), FX, FY, FZ


@timer()
def calculate_fourier_shell_correlation(
    vol1,
    vol2,
    n_bins=200,
    rmax=None,
    voxel_size=1.0,
    crop_by: int = 0,
    laminography_angle: Optional[float] = None,
):
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
    r, FX, FY, FZ = _freq_radius(V1.shape, voxel_size=voxel_size)
    if rmax is None:
        # Up to the smallest-axis Nyquist radius is fine; r already reflects that
        rmax = r.max()

    # Bin edges and indices
    edges = np.linspace(0.0, rmax, n_bins + 1, dtype=np.float64)
    # digitize returns 1..n_bins; convert to 0..n_bins-1 and mask out-of-range
    shell = np.digitize(r.ravel(), edges) - 1
    valid = (shell >= 0) & (shell < n_bins)

    # only include values not in the missing cone
    if laminography_angle is not None:
        print("removing missing cone data")
        missing_cone_mask = create_missing_cone_mask(FX, FY, FZ, 90 - laminography_angle)
        print(valid.sum())
        valid = valid & missing_cone_mask.ravel()
        print(valid.sum())

    # Values to accumulate
    v1 = V1.ravel()[valid]
    v2 = V2.ravel()[valid]
    sh = shell[valid]

    num_vals = np.real(v1 * np.conj(v2))
    den1_vals = np.abs(v1) ** 2
    den2_vals = np.abs(v2) ** 2
    fr_vals = r.ravel()[valid]

    # Bin sums via bincount
    n_shell = np.bincount(sh, minlength=n_bins)
    num_sum = np.bincount(sh, weights=num_vals, minlength=n_bins)
    den1_sum = np.bincount(sh, weights=den1_vals, minlength=n_bins)
    den2_sum = np.bincount(sh, weights=den2_vals, minlength=n_bins)

    # Prefer the mean frequency in each shell (instead of mid-edge)
    f_sum = np.bincount(sh, weights=fr_vals, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n_shell > 0, f_sum / n_shell, np.nan)
        denom = np.sqrt(den1_sum * den2_sum)
        fsc = np.where(denom > 0, num_sum / denom, np.nan)

    # Optional: drop DC bin (first) if desired
    # f[0], fsc[0] = np.nan, np.nan

    return f, fsc, n_shell


def create_missing_cone_mask(FX, FY, FZ, cone_angle_degrees) -> np.ndarray:
    """
    Create a 3D boolean mask for points inside a double cone (both directions).

    Parameters are the same as create_cone_mask.
    """

    cone_angle = np.radians(cone_angle_degrees)
    tan_angle = np.tan(cone_angle)

    radial_distance = np.sqrt(FX**2 + FY**2)
    axial_distance = np.abs(FZ)  # Already taking absolute value
    # The abs() already makes this a double cone
    cone_mask = radial_distance >= axial_distance * tan_angle

    if cone_angle_degrees >= 90:
        cone_mask = np.ones_like(FX, dtype=bool)

    return cone_mask


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


def get_resolution_crossing(fsc: np.ndarray, x_bit_curve: np.ndarray, f: np.ndarray):
    idx = np.where((fsc < x_bit_curve))  # and [not np.isnan(x) for x in one_bit_curve])
    if len(idx[0]) == 0:
        f_crossing = f[-1]
    else:
        f_crossing = f[idx[0][0]]
    resolution = 1 / f_crossing
    return f_crossing, resolution
