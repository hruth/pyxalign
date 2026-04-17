import dataclasses
from dataclasses import field
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.roi import ROIOptions, RectangularROIOptions
from pyxalign.api.options.transform import CropOptions, DownsampleOptions
from pyxalign.api import enums
from functools import partial
from .base import BaseOptions


@dataclasses.dataclass
class ExperimentOptions(BaseOptions):
    """Options related to the experimental configuration."""

    laminography_angle: float = 61.1
    """
    Laminography angle in degrees. For tomography data, set this to 90.
    """

    sample_thickness: float = 7e-6
    """
    Estimate of the sample thickness in meters.
    """

    pixel_size: float = 1.0
    """"
    Estimate of the pixels size in meters.
    """


@dataclasses.dataclass
class MorphologicalMaskOptions(BaseOptions):
    downsample: DownsampleOptions = field(
        default_factory=partial(
            DownsampleOptions,
            type=enums.DownsampleType.NEAREST,
            scale=4,
            enabled=True,
        )
    )
    """
    Options for downsampling data before doing any of the other
    morphological transformations.
    """

    binary_close_coefficient: int = 30

    binary_erode_coefficient: int = 30

    unsharp: bool = True

    fill: int = 8


@dataclasses.dataclass
class PhaseRampRemovalOptions(BaseOptions):
    iterations: int = 5

    downsampling: int = 8


@dataclasses.dataclass
class GradientIntegrationUnwrapOptions(BaseOptions):
    gradient_method: enums.ImageGradientMethods = enums.ImageGradientMethods.FOURIER_DIFFERENTIATION
    "The method used to calculate the phase gradient"

    integration_method: enums.ImageIntegrationMethods = enums.ImageIntegrationMethods.FOURIER
    "The method used to integrate the image back from gradients"

    fourier_shift_step: float = 0.5
    """
    The finite-difference step size used to calculate the gradient, 
    if the Fourier shift method is selected
    """

    use_masks: bool = True
    """
    Determines if the projection masks should be multiplied with the 
    projections before unwrapping
    """

    # deramp_polyfit_order: int = 1
    # "The order of the polynomial fit used to de-ramp the phase"


@dataclasses.dataclass
class IterativeResidualUnwrapOptions(BaseOptions):
    iterations: int = 10
    "Number of iterative correction steps to perform"

    # lsq_fit_ramp_removal: bool = False
    # """
    # Whether to remove phase ramps using least-squares fitting after 
    # unwrapping
    # """

@dataclasses.dataclass
class AirGapRampRemovalOptions(BaseOptions):
    enabled: bool = False

    # air_region: CropOptions = field(default_factory=CropOptions)
    air_region: RectangularROIOptions = field(default_factory=RectangularROIOptions)
    "ROI for defining the air region"

    polyfit_order: int = 1


@dataclasses.dataclass
class PhaseUnwrapOptions(BaseOptions):
    method: enums.PhaseUnwrapMethods = enums.PhaseUnwrapMethods.ITERATIVE_RESIDUAL_CORRECTION
    """
    Phase unwrapping method to use

    Options(BaseOptions):
    - PhaseUnwrapMethods.IterativeResidualCorrection
        - default choice; typically performs better
    - PhaseUnwrapMethods.GradientIntegration
        - can perform better if the IterativeResidualCorrection 
        unwrapping is producing large phase ramps
        - same unwrapping method that is used by pty-chi
    """

    gradient_integration: GradientIntegrationUnwrapOptions = field(
        default_factory=GradientIntegrationUnwrapOptions
    )
    "Options for GradientIntegration unwrapping"

    iterative_residual: IterativeResidualUnwrapOptions = field(
        default_factory=IterativeResidualUnwrapOptions
    )
    "Options for IterativeResidualCorrection unwrapping"

    remove_ramp_using_air_gap: AirGapRampRemovalOptions = field(
        default_factory=AirGapRampRemovalOptions
    )

    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class RegularizationOptions(BaseOptions):
    enabled: bool = False

    local_TV_lambda: float = 1e-4

    iterations: int = 10

    use_gpu: bool = False
