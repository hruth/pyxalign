from typing import List, Optional, Union
import numpy as np
import astra
import copy
from time import time
from llama.api import enums, maps
from llama.api.options.alignment import AlignmentOptions
from llama.api.options.device import DeviceOptions

from llama.api.options.projections import ProjectionOptions
from llama.api.options.transform import ShiftOptions, UpsampleOptions
import llama.gpu_utils as gpu_utils
from llama.gpu_wrapper import device_handling_wrapper
from llama.mask import estimate_reliability_region_mask, blur_masks
import llama.image_processing as ip

import llama.plotting.plotters as plotters
from llama import reconstruct

# from llama.projections import PhaseProjections
import llama.projections as projections
from llama.transformations.classes import Downsampler, Shifter, Upsampler, Cropper
from llama.transformations.functions import image_shift_fft
from llama.unwrap import unwrap_phase
from llama.api.types import ArrayType, r_type


class Laminogram:
    def __init__(
        self,
        projections: "projections.PhaseProjections",
    ):
        # Store a reference to the projections
        self.projections = projections

    def generate_laminogram(
        self,
        filter_inputs: bool = False,
        pinned_filtered_sinogram: Optional[np.ndarray] = None,
    ):
        # Copy the settings used at the time of the reconstruction
        self.options = copy.deepcopy(self.projections.options.reconstruct)
        self.experiment_options = copy.deepcopy(self.projections.options.experiment)
        astra.set_gpu_index(self.options.astra.back_project_gpu_indices)
        scan_geometry_config, vectors = reconstruct.get_astra_reconstructor_geometry(
            sinogram=self.projections.data,
            angles=self.projections.angles,
            n_pix=self.projections.reconstructed_object_dimensions,
            center_of_rotation=self.projections.center_of_rotation,
            lamino_angle=self.experiment_options.laminography_angle,
            tilt_angle=self.experiment_options.tilt_angle,
            skew_angle=self.experiment_options.skew_angle,
        )
        if filter_inputs:
            sinogram = reconstruct.filter_sinogram(
                sinogram=self.projections.data,
                vectors=vectors,
                device_options=self.options.filter.device,
                pinned_results=pinned_filtered_sinogram,
            )
        else:
            sinogram = self.projections.data
        astra_config, self.geometries = reconstruct.create_astra_reconstructor_config(
            sinogram=sinogram,
            scan_geometry_config=scan_geometry_config,
            vectors=vectors,
        )
        self.data: np.ndarray = reconstruct.get_3D_reconstruction(astra_config)
    
    def get_forward_projection(self, pinned_forward_projection: Optional[np.ndarray] = None):
        astra.set_gpu_index(self.options.astra.forward_project_gpu_indices)
        forward_projections = reconstruct.get_forward_projection(
            reconstruction=self.data,
            geometries=self.geometries,
            pinned_forward_projection=pinned_forward_projection,
        )
        # Create a projections object in case I want to use any of the
        # projections object methods
        self.model_forward_projections = projections.PhaseProjections(
            projections=forward_projections,
            options=self.projections.options,
            angles=self.projections.angles,
            center_of_rotation=self.projections.center_of_rotation,
        )

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
