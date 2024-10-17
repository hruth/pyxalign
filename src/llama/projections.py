import numpy as np
from llama.api.options.projections import ProjectionOptions
from llama.api.options.projections import ProjectionDeviceOptions
import llama.gpu_utils as gpu_utils
# from llama.transformations import downsample_fft
import llama.api.enums as enums
from llama.transformations import image_pre_process


class Projections:
    def __init__(
        self,
        projections: np.ndarray,
        options: ProjectionOptions = None,
    ):
        self.projections = image_pre_process(projections)
        if options.projection_device_options.pin_memory:
            projections = gpu_utils.pin_memory(projections)

        # projections = gpu_utils.move_to_device(
        #     projections, options.projection_device_options.device_type, return_copy=True
        # )
        self.data = projections

        self.center_of_rotation = np.array(projections.shape[1:]) / 2

    @property
    def n_projections(self) -> int:
        return self.data.shape[0]

    @property
    def reconstructed_object_dimensions(self) -> np.ndarray:
        # function for calculating n_pix_align
        pass

    # def shift_projections(self, shift):
    #     self.projections = image_shift_circ(self.data, shift)


class ComplexProjections(Projections):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PhaseProjections(Projections):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
