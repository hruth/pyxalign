from typing import Union, Optional
import numpy as np
import cupy as cp

from llama.api.options.projections import ProjectionOptions
from llama.api.options.projections import ProjectionDeviceOptions
import llama.gpu_utils as gpu_utils
import llama.api.enums as enums
from llama.mask import estimate_reliability_region_mask

# from llama.plotting.plotters import make_image_slider_plot
import llama.plotting.plotters as plotters
from llama.transformations.classes import PreProcess

from llama.api.types import ArrayType, r_type, c_type
from llama.transformations.functions import image_shift_fft


class Projections:
    def __init__(
        self,
        projections: np.ndarray,
        angles: np.ndarray,
        options: ProjectionOptions,
    ):
        self.data = projections
        self.options = options
        # self.data = PreProcess(options.pre_processing_options).run(projections)
        self.angles = angles
        # # device management will need work!
        # if options.projection_device_options.pin_memory:
        #     projections = gpu_utils.pin_memory(self.data)
        # self.data = gpu_utils.move_to_device(
        #     self.data, options.projection_device_options.device_type
        # )

        self.center_of_rotation = np.array(projections.shape[1:]) / 2

    @property
    def n_projections(self) -> int:
        return self.data.__len__()

    @property
    def reconstructed_object_dimensions(self) -> np.ndarray:
        # function for calculating n_pix_align
        pass

    def pin_projections(self):
        self.data = gpu_utils.pin_memory(self.data)

    # def set_data(self, data: ArrayType):
        # self.data = data  # Maybe need to change to views later when this is pinned

    # def shift_projections(self, shift):
    #     self.projections = image_shift_circ(self.data, shift)

    def plot_projections(self, process_function: callable = lambda x: x):
        plotters.make_image_slider_plot(process_function(self.data))

    def plot_shifted_projections(self, shift: np.ndarray, process_function: callable = lambda x: x):
        "Plot the shifted projections using the shift that was passed in."
        plotters.make_image_slider_plot(process_function(image_shift_fft(self.data, shift)))

    def plot_sum_of_projections(self, process_function: callable = lambda x: x):
        plotters.plot_sum_of_images(process_function(self.data))

    def get_masks(self, enable_plotting: bool = False):
        self.masks = estimate_reliability_region_mask(
            self.data, self.options.mask_options, enable_plotting
        )


class ComplexProjections(Projections):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PhaseProjections(Projections):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
