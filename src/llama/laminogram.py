from typing import Optional
import numpy as np
import cupy as cp
import astra
import copy
from llama.api.options.plotting import PlotDataOptions

import llama.image_processing as ip
from llama import reconstruct
from llama.plotting.plotters import plot_slice_of_3D_array
import llama.projections as projections
from llama.timing.timer_utils import timer
import matplotlib.pyplot as plt

from llama.api.types import ArrayType, r_type


class Laminogram:
    astra_config: dict = None
    object_geometries: dict = None
    scan_geometry_config: dict = None
    vectors: np.ndarray = None
    forward_projection_id: int = None

    def __init__(
        self,
        projections: "projections.PhaseProjections",
    ):
        # Store a reference to the projections
        self.projections = projections
        # self.options = copy.deepcopy(projections.options.reconstruct)
        # self.experiment_options = copy.deepcopy(projections.options.experiment)
        self.options = projections.options.reconstruct
        self.experiment_options = projections.options.experiment
        self.scan_geometry_config, self.vectors, self.object_geometries = (
            self.intialize_astra_reconstructor_inputs()
        )

    @property
    def n_layers(self):
        return self.projections.reconstructed_object_dimensions[2]

    def intialize_astra_reconstructor_inputs(self):
        scan_geometry_config, vectors = reconstruct.get_astra_reconstructor_geometry(
            size=self.projections.size,
            angles=self.projections.angles,
            n_pix=self.projections.reconstructed_object_dimensions,
            center_of_rotation=self.projections.center_of_rotation,
            lamino_angle=self.experiment_options.laminography_angle,
            tilt_angle=self.options.geometry.tilt_angle,
            skew_angle=self.options.geometry.skew_angle,
        )
        object_geometries = reconstruct.get_object_geometries(scan_geometry_config, vectors)
        return scan_geometry_config, vectors, object_geometries

    @timer()
    def generate_laminogram(
        self,
        filter_inputs: bool = False,
        pinned_filtered_sinogram: Optional[np.ndarray] = None,
        reinitialize_astra: bool = True,
    ):
        # Copy the settings used at the time of the reconstruction
        # self.options = copy.deepcopy(self.projections.options.reconstruct)
        # self.experiment_options = copy.deepcopy(self.projections.options.experiment)

        # Re-initialize the inputs and clear outputs
        if reinitialize_astra:
            self.clear_astra_objects()
            self.scan_geometry_config, self.vectors, self.object_geometries = (
                self.intialize_astra_reconstructor_inputs()
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

        if self.astra_config is None:
            self.astra_config = reconstruct.create_astra_reconstructor_config(
                sinogram,
                self.object_geometries,
            )
        else:
            self.astra_config = reconstruct.update_astra_reconstructor_config(
                sinogram,
                self.object_geometries,
                self.astra_config,
            )

        astra.set_gpu_index(self.options.astra.back_project_gpu_indices)
        self.data: np.ndarray = reconstruct.get_3D_reconstruction(self.astra_config)

    def get_forward_projection(
        self,
        pinned_forward_projection: Optional[np.ndarray] = None,
        forward_projection_id: Optional[int] = None,
    ):
        # If no forward_projection_id was passed in, a new projection
        # id will be generated. If that is the case, save that ID so
        # you can skip the creation of a new 3D astra object next time
        # this is run
        is_new_sino_id_created = (forward_projection_id is None) and (
            self.forward_projection_id is None
        )

        if forward_projection_id is None:
            forward_projection_id = self.forward_projection_id

        astra.set_gpu_index(self.options.astra.forward_project_gpu_indices)
        forward_projections, forward_projection_id = reconstruct.get_forward_projection(
            reconstruction=self.data,
            object_geometries=self.object_geometries,
            pinned_forward_projection=pinned_forward_projection,
            volume_id=self.astra_config["ReconstructionDataId"],
            forward_projection_id=forward_projection_id,
            return_id=True,
        )

        if is_new_sino_id_created:
            self.forward_projection_id = forward_projection_id

        # Create a projections object in case I want to use any of the
        # projections object methods
        forward_projection_options = copy.deepcopy(self.projections.options)
        forward_projection_options.experiment.pixel_size = self.projections.pixel_size
        self.forward_projections = projections.PhaseProjections(
            projections=forward_projections,
            options=forward_projection_options,
            angles=self.projections.angles,
            center_of_rotation=self.projections.center_of_rotation,
            skip_pre_processing=True,
        )

    def generate_projection_masks_from_circulo(
        self,
        forward_project_gpu_indices: Optional[tuple] = None,
        radial_smooth: int = 0,
        rad_apod: int = 0,
    ) -> np.ndarray:
        if forward_project_gpu_indices is None:
            astra.set_gpu_index(self.options.astra.forward_project_gpu_indices)
        else:
            astra.set_gpu_index(forward_project_gpu_indices)

        reconstruction_mask = self.get_circular_window(radial_smooth, rad_apod)
        reconstruction_mask = np.repeat(reconstruction_mask[None], self.n_layers, axis=0)

        _, _, object_geometries = self.intialize_astra_reconstructor_inputs()
        mask, sino_id = reconstruct.get_forward_projection(
            reconstruction=reconstruction_mask,
            object_geometries=object_geometries,
            return_id=True,
        )
        mask = mask[0]/mask[0].max()
        astra.data3d.delete(sino_id)

        return mask

    def clear_astra_objects(self):
        astra_objects = []
        if self.astra_config is not None:
            astra_objects += [
                self.astra_config["ReconstructionDataId"],
                self.astra_config["ProjectionDataId"],
            ]
        if self.forward_projection_id is not None:
            astra_objects += [self.forward_projection_id]
        astra.data3d.delete(astra_objects)

        self.astra_config = None
        self.object_geometries = None
        self.scan_geometry_config = None
        self.vectors = None
        self.forward_projection_id = None

    @timer()
    def apply_circular_window(self, circulo: Optional[ArrayType] = None):
        if circulo is None:
            self.data[:] = self.data * self.get_circular_window()
        else:
            self.data[:] = self.data * circulo

    def get_circular_window(self, radial_smooth: int = 5, rad_apod: int = 0):
        # Generate circular mask for reconstruction
        return ip.apply_3D_apodization(
            image=np.zeros(self.projections.reconstructed_object_dimensions),
            rad_apod=rad_apod,
            radial_smooth=radial_smooth,
        ).astype(r_type)

    def plot_data(
        self,
        options: Optional[PlotDataOptions] = None,
        show_plot: bool = True,
    ):
        if options is None:
            options = PlotDataOptions()
        else:
            options = copy.deepcopy(options)

        if options.index is None:
            options.index = int(self.data.shape[0] / 2)

        plt.title(f"Reconstruction Slice {options.index}")
        plot_slice_of_3D_array(
            self.data,
            options,
            self.projections.pixel_size,
            show_plot=show_plot,
        )
