from dataclasses import field
import dataclasses
from typing import Optional
from pyxalign.api import enums
from pyxalign.api.options.alignment import InteractiveViewerOptions, ProjectionMatchingOptions
from pyxalign.api.options.options import (
    ExperimentOptions,
    MorphologicalMaskOptions,
    PhaseUnwrapOptions,
)
from pyxalign.api.options.plotting import UpdatePlotOptions
from pyxalign.api.options.reconstruct import ReconstructOptions
from pyxalign.api.options.transform import CropOptions, DownsampleOptions, RotationOptions
from functools import partial


def pma_factory_for_estimate_center_options() -> ProjectionMatchingOptions:
    return ProjectionMatchingOptions(
        iterations=1,
        interactive_viewer=InteractiveViewerOptions(update=UpdatePlotOptions(enabled=False)),
    )


@dataclasses.dataclass
class CoordinateSearchOptions:
    center_estimate: Optional[int] = None

    range: Optional[int] = None

    spacing: Optional[int] = None

    enabled: bool = False


@dataclasses.dataclass
class EstimateCenterOptions:
    downsample: DownsampleOptions = field(
        default_factory=partial(
            DownsampleOptions, type=enums.DownsampleType.FFT, scale=4, enabled=True
        )
    )

    crop: CropOptions = field(default_factory=CropOptions)

    projection_matching: ProjectionMatchingOptions = field(
        default_factory=pma_factory_for_estimate_center_options
    )

    horizontal_coordinate: CoordinateSearchOptions = field(default_factory=CoordinateSearchOptions)

    vertical_coordinate: CoordinateSearchOptions = field(default_factory=CoordinateSearchOptions)


@dataclasses.dataclass
class ProjectionTransformOptions:
    crop: CropOptions = field(default_factory=CropOptions)

    downsample: DownsampleOptions = field(default_factory=DownsampleOptions)

    mask_downsample_type: enums.DownsampleType = enums.DownsampleType.LINEAR

    mask_downsample_use_gaussian_filter: bool = False

    rotation: RotationOptions = field(default_factory=RotationOptions)

    shear: RotationOptions = field(default_factory=RotationOptions)


@dataclasses.dataclass
class VolumeWidthOptions:
    """
    Options for determining the reconstructed volume size. The
    reconstructed volume defaults to the width of the projection
    when `use_custom_width` is `False`.
    """

    use_custom_width: bool = False
    """
    Determines if `multiplier` is used to change the reconstructed 
    volume width or not.
    """

    multiplier: float = 1
    """
    If `use_custom_width` is `True`, the reconstructed volume size is 
    equal to the projection width multiplied by `multiplier`.
    """


@dataclasses.dataclass
class SimulatedProbe:
    """
    Parameters for creating a gaussian probe
    """

    fractional_width: float = 0.25
    "Probe FWHM as a fraction of its total window size"


@dataclasses.dataclass
class ProbePositionMaskOptions:
    """
    Options for building projection masks from probe positions
    """

    threshold: bool = 0.1
    """
    Masks are set to 1 above the threshold and 0 below the threshold
    """

    use_simulated_probe: bool = True
    """
    Whether or not to use a model gaussian probe. If false, it will
    use the probe in the Projections object.
    """

    probe: SimulatedProbe = field(default_factory=SimulatedProbe)


@dataclasses.dataclass
class ProjectionOptions:
    experiment: ExperimentOptions = field(default_factory=ExperimentOptions)
    """
    Options related to the experimental configuration.
    """

    reconstruct: ReconstructOptions = field(default_factory=ReconstructOptions)
    """
    Options used by the `PhaseProjections` method `get_3D_reconstruction`.
    """

    mask_from_positions: ProbePositionMaskOptions = field(default_factory=ProbePositionMaskOptions)
    """
    Options used by the `Projections` method `get_masks_from_probe_positions`.
    These options are also used by the GUI tools for building masks from probe
    positions.
    """

    phase_unwrap: PhaseUnwrapOptions = field(default_factory=PhaseUnwrapOptions)
    "Options used by the `ComplexProjections` method `unwrap_phase`."

    estimate_center: EstimateCenterOptions = field(default_factory=EstimateCenterOptions)
    "Options used by the `PhaseProjections` method `estimate_center_of_rotation`"

    input_processing: ProjectionTransformOptions = field(default_factory=ProjectionTransformOptions)
    """
    Options for the image transformations applied to the 
    projection array upon the initialization of a `Projections`
    object. These options are passed to the `Projections` method
    `transform_projections` during initialization.
    """

    volume_width: VolumeWidthOptions = field(default_factory=VolumeWidthOptions)
    "Determines reconstructed volume size"

    masks_from_morphology: MorphologicalMaskOptions = field(
        default_factory=MorphologicalMaskOptions
    )
    """
    Options for getting masks by using morphological operations on the
    input projections. This usually is not used since it is better to 
    get masks from the probe positions, the functions are slow, and the
    results are often unsatisfactory.
    """
