import dataclasses
from dataclasses import field
from numbers import Number
from typing import Optional

from matplotlib.pyplot import viridis
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.transform import DownsampleOptions
from pyxalign.api import enums
from functools import partial


@dataclasses.dataclass
class ExperimentOptions:
    laminography_angle: float = 61.1

    sample_thickness: float = 7e-6

    pixel_size: float = 1.0


@dataclasses.dataclass
class MaskOptions:
    downsample: DownsampleOptions = field(
        default_factory=partial(
            DownsampleOptions, type=enums.DownsampleType.NEAREST, scale=4, enabled=True
        )
    )

    # upsample_options = UpsampleOptions(
    #     type=enums.UpsampleType.NEAREST, scale=4, enabled=True, device_options=enums.DeviceType.CPU
    # )

    binary_close_coefficient: int = 30

    binary_erode_coefficient: int = 30

    unsharp: bool = True

    fill: int = 8


@dataclasses.dataclass
class PhaseRampRemovalOptions:
    iterations: int = 5

    downsampling: int = 8


@dataclasses.dataclass
class GradientIntegrationUnwrapOptions:
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

    deramp_polyfit_order: int = 1
    "The order of the polynomial fit used to de-ramp the phase"


@dataclasses.dataclass
class IterativeResidualUnwrapOptions:
    iterations: int = 10
    "Number of iterative correction steps to perform"

    lsq_fit_ramp_removal: bool = False
    """
    Whether to remove phase ramps using least-squares fitting after 
    unwrapping
    """


@dataclasses.dataclass
class PhaseUnwrapOptions:
    device: DeviceOptions = field(default_factory=DeviceOptions)

    method: enums.PhaseUnwrapMethods = enums.PhaseUnwrapMethods.ITERATIVE_RESIDUAL_CORRECTION
    """
    Phase unwrapping method to use

    Options:
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


@dataclasses.dataclass
class RegularizationOptions:
    enabled: bool = False

    local_TV_lambda: float = 1e-4

    iterations: int = 10
