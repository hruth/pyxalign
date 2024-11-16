import numpy as np
import cupy as cp
import scipy
import statsmodels.robust
from scipy.signal import savgol_filter
import pandas as pd

from llama.alignment.base import Aligner
import llama.image_processing as ip
from llama.projections import Projections
from llama.api.options.alignment import CrossCorrelationOptions
from llama.gpu_utils import function_compute_device_manager, get_scipy_module, pin_memory
from llama.api.options.device import DeviceOptions
from llama.api import enums
from llama.transformations.classes import PreProcess
from llama.transformations.functions import image_shift_circ
from llama.api.types import ArrayType

# from cupyx.scipy.signal import fftconvolve


class CrossCorrelationAligner(Aligner):
    def __init__(self, projections: Projections, options: CrossCorrelationOptions):
        super().__init__(projections, options)
        self.options: CrossCorrelationOptions = self.options  # for static checker and type checking

    def run(self, illum_sum: np.ndarray) -> np.ndarray:
        # This could be moved to its own thing
        pre_processed_projections = self.pre_process_projections()
        shift = self.calculate_alignment_shift(
            projections=pre_processed_projections,
            angles=self.projections.angles,
            illum_sum=illum_sum,
        )
        return shift

    def calculate_alignment_shift(
        self, projections: np.ndarray, angles: np.ndarray, illum_sum: np.ndarray
    ) -> np.ndarray:
        weights = illum_sum / (illum_sum + 0.1 * np.max(illum_sum))

        # Should only be pinning memory when doing calculations on GPU
        # I think I should somehow bundle this with device options in the
        # input to the function_compute_device_manager
        # variation_shape = (
        #     projections.shape / np.array([1, self.downsampling, self.downsampling])
        # ).astype(int)
        # variation = pin_memory(np.empty(variation_shape))

        # This has improved speed when run on multiple GPUs # Is it the actual time sink though?
        variation = self.get_variation(
            projections, weights, device_options=self.options.device_options
        )

        # Ensure the array is on a single device for the rest of the calculations
        if self.options.device_options is enums.DeviceType.CPU:
            variation = np.array(variation)
        elif self.options.device_options is enums.DeviceType.GPU:
            variation = cp.array(variation)

        idx_sort = np.argsort(angles)
        idx_sort_inverse = np.argsort(idx_sort)
        n_angles = len(angles)
        shift_total = np.zeros((n_angles, 2))
        shift_saved = np.zeros((self.options.iterations, n_angles, 2))

        for i in range(self.options.iterations):
            variation_fft = ip.filtered_fft(variation, shift_total, self.options.filter_data)

            shift_relative = ip.get_cross_correlation_shift(
                image=variation_fft[idx_sort],
                image_ref=variation_fft[np.roll(idx_sort, -1)],
            )
            shift_relative = np.roll(shift_relative, 1, 0)
            shift_relative = self.clamp_shift(shift_relative, 3)
            # RESUME HERE -- move this to a subtract_smooth function
            # Subtract the underlying slow variation
            cumulative_shift = np.cumsum(shift_relative, axis=0)
            cumulative_shift = cumulative_shift - np.mean(cumulative_shift, axis=0)
            for i in range(2):
                df = pd.DataFrame(dict(x=cumulative_shift[:, i]))
                smoothed_shift = (
                    df[["x"]]
                    .apply(savgol_filter, window_length=self.options.filter_position, polyorder=2)
                    .to_numpy()[:, 0]
                )
                cumulative_shift[:, i] = cumulative_shift[:, i] - smoothed_shift

            shift_total = shift_total + cumulative_shift[idx_sort_inverse, :]
            shift_total = self.clamp_shift(shift_total, 6)

        shift_total = np.round(shift_total * self.options.binning)

        return shift_total

    def clamp_shift(self, shift: ArrayType, max_std_variation: float):
        # RESUME HERE
        # how does this work?
        shift_max = max_std_variation * statsmodels.robust.mad(
            shift, axis=0
        )  # 3 is number of std for the outliers?
        shift_max[shift_max < 10] = 10  # why is this here
        for i in range(2):
            sign = np.sign(shift[:, i])
            idx = shift_max[i] < np.abs(shift[:, i])
            shift[idx, i] = shift_max[i] * sign[idx]
        return shift

    # @function_compute_device_manager(array_to_move_indices=[1], single_array_input=[2])
    def get_variation(
        self,
        projections: ArrayType,
        weights: ArrayType,
        *,
        device_options: DeviceOptions = None,  # will need to add pinned array too
    ) -> ArrayType:  # pick a more clear name later
        xp = cp.get_array_module(projections)
        scipy_module: scipy = get_scipy_module(projections)

        dX = scipy_module.signal.fftconvolve(projections, xp.array([[[1, -1]]]), "same")
        dY = scipy_module.signal.fftconvolve(
            projections, xp.array([[[1, -1]]]).transpose(0, 2, 1), "same"
        )
        variation = xp.sqrt(xp.abs(dX) ** 2 + xp.abs(dY) ** 2)
        # Ignore regions with low amplitude
        variation = variation * xp.abs(projections)
        # Remove high values in the noisy areas
        variation_mean = xp.mean(variation * weights, axis=(1, 2)) / xp.mean(weights)
        # This could probably be replaced with a simpler calculation.
        variation_std = xp.sqrt(
            xp.mean(
                (variation - variation_mean[:, xp.newaxis, xp.newaxis]) ** 2 * weights, axis=(1, 2)
            )
            / xp.mean(weights)
        )
        for i in range(len(variation_mean)):
            cutoff = variation_mean[i] + variation_std[i]
            variation[i, (variation[i, :, :] > cutoff)] = cutoff
            variation[i, :, :] = scipy_module.ndimage.gaussian_filter(
                variation[i, :, :], 2 * self.options.binning
            )

        variation = xp.real(variation[:, 0 :: self.options.binning, 0 :: self.options.binning])

        return variation

    def find_shift():  # pick a better name
        pass
