from typing import Optional
import numpy as np
import cupy as cp
import astra
import copy

import llama.image_processing as ip
from llama import reconstruct
import llama.projections as projections
from llama.timing.timer_utils import timer
import matplotlib.pyplot as plt

from llama.api.types import ArrayType, r_type


class Laminogram:
    def __init__(
        self,
        projections: "projections.PhaseProjections",
    ):
        # Store a reference to the projections
        self.projections = projections
        self.astra_config = None

    @timer()
    def generate_laminogram(
        self,
        filter_inputs: bool = False,
        pinned_filtered_sinogram: Optional[np.ndarray] = None,
        regenerate_reconstructor_geometry: bool = False,
    ):
        # Copy the settings used at the time of the reconstruction
        self.options = copy.deepcopy(self.projections.options.reconstruct)
        self.experiment_options = copy.deepcopy(self.projections.options.experiment)
        astra.set_gpu_index(self.options.astra.back_project_gpu_indices)
        if not hasattr(self, "scan_geometry_config") or regenerate_reconstructor_geometry:
            self.scan_geometry_config, self.vectors = reconstruct.get_astra_reconstructor_geometry(
                size=self.projections.size,
                angles=self.projections.angles,
                n_pix=self.projections.reconstructed_object_dimensions,
                center_of_rotation=self.projections.center_of_rotation,
                lamino_angle=self.experiment_options.laminography_angle,
                tilt_angle=self.options.geometry.tilt_angle,
                skew_angle=self.options.geometry.skew_angle,
                # tilt_angle=self.experiment_options.tilt_angle,
                # skew_angle=self.experiment_options.skew_angle,
            )
        if filter_inputs:
            sinogram = reconstruct.filter_sinogram(
                sinogram=self.projections.data,
                vectors=self.vectors,
                device_options=self.options.filter.device,
                pinned_results=pinned_filtered_sinogram,
            )
        else:
            sinogram = self.projections.data
        # # Temporary fix for keep_on_gpu case. The fix I need to do is 
        # # to filter_sinogram and/or gpu wrapper to have an option to insert
        # # a cupy array and get a numpy array at output
        # if type(sinogram) is cp.ndarray:
        #     sinogram.get(out=pinned_filtered_sinogram)
        #     sinogram = pinned_filtered_sinogram
        self.astra_config, self.geometries = (
            reconstruct.create_or_update_astra_reconstructor_config(
                sinogram=sinogram,
                scan_geometry_config=self.scan_geometry_config,
                vectors=self.vectors,
                astra_config=self.astra_config,
            )
        )
        self.data: np.ndarray = reconstruct.get_3D_reconstruction(self.astra_config)

    def get_forward_projection(self, pinned_forward_projection: Optional[np.ndarray] = None):
        astra.set_gpu_index(self.options.astra.forward_project_gpu_indices)
        forward_projections = reconstruct.get_forward_projection(
            reconstruction=self.data,
            geometries=self.geometries,
            pinned_forward_projection=pinned_forward_projection,
        )
        # Create a projections object in case I want to use any of the
        # projections object methods
        self.forward_projections = projections.PhaseProjections(
            projections=forward_projections,
            options=self.projections.options,
            angles=self.projections.angles,
            center_of_rotation=self.projections.center_of_rotation,
            skip_pre_processing=True,
        )
        # astra.clear()

    @timer()
    def apply_circular_window(self, circulo: Optional[ArrayType] = None):
        if circulo is None:
            self.data[:] = self.data * self.get_circular_window()
        else:
            self.data[:] = self.data * circulo

    def get_circular_window(self, radial_smooth: int = 5):
        # Generate circular mask for reconstruction
        return ip.apply_3D_apodization(
            image=np.zeros(self.projections.reconstructed_object_dimensions),
            rad_apod=0,
            radial_smooth=radial_smooth,
        ).astype(r_type)

    def plot_data(
        self,
        slice_idx: Optional[int] = None,
        plot_sum: bool = False,
        ax_lim_width: Optional[tuple] = None,
        ax_lim_center: Optional[tuple] = None,
        show_plot: bool = True,
    ):
        if plot_sum is True:
            laminogram = self.data.sum(0)
        else:
            if slice_idx is None:
                slice_idx = int(self.data.shape[0] / 2)
            laminogram = self.data[slice_idx]

        plt.imshow(self.data[slice_idx])
        if ax_lim_center is None:
            ax_lim_center = laminogram.shape[1:] / 2
        if ax_lim_width is not None:
            range_array = np.array([-ax_lim_width / 2, ax_lim_width / 2])
            plt.xlim(ax_lim_center[0] + range_array)
            plt.ylim(ax_lim_center[1] + range_array)
        if plot_sum:
            plt.title(f"3D Reconstruction: sum along z-axis")
        else:
            plt.title(f"3D Reconstruction: slice {slice_idx}")

        if show_plot:
            plt.show()
