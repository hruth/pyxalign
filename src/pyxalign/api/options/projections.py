from dataclasses import field
import dataclasses
from typing import Optional
from pyxalign.api import enums
from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.api.options.options import (
    ExperimentOptions,
    MaskOptions,
    PhaseUnwrapOptions,
)
from pyxalign.api.options.reconstruct import ReconstructOptions
from pyxalign.api.options.transform import CropOptions, DownsampleOptions, RotationOptions
from functools import partial


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
        default_factory=partial(ProjectionMatchingOptions, iterations=1)
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
class ProjectionOptions:
    experiment: ExperimentOptions = field(default_factory=ExperimentOptions)

    reconstruct: ReconstructOptions = field(default_factory=ReconstructOptions)

    mask: MaskOptions = field(default_factory=MaskOptions)

    # Technically this should really only be here for complex projections
    phase_unwrap: PhaseUnwrapOptions = field(default_factory=PhaseUnwrapOptions)

    estimate_center: EstimateCenterOptions = field(default_factory=EstimateCenterOptions)

    input_processing: ProjectionTransformOptions = field(default_factory=ProjectionTransformOptions)

    is_tomo: bool = False

    # crop: CropOptions = field(default_factory=CropOptions)

    # downsample: DownsampleOptions = field(default_factory=DownsampleOptions)

    # mask_downsample_type: enums.DownsampleType = enums.DownsampleType.LINEAR

    # mask_downsample_use_gaussian_filter: bool = False

    # phase_ramp_removal: PhaseRampRemovalOptions = field(default_factory=PhaseRampRemovalOptions)

