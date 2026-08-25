import dataclasses
from dataclasses import field

from pyxalign.api.options.transform import Crop3DOptions
from pyxalign.api.options.base import BaseOptions


@dataclasses.dataclass
class FillMissingConeOptions(BaseOptions):
    """Options for the fill_missing_cone reconstruction regularization function.

    Attributes:
        delta_background: Lower bound on voxel values (soft positivity constraint).
        delta_maximal: Upper bound on voxel values.
        mask_relax: Relaxation factor for the vertical mask that suppresses
            nonzero values in empty regions.
        max_scale: Maximum downscaling factor used in the multiscale approach.
            The algorithm starts at 1/max_scale and refines down to full resolution.
        n_iter: Number of regularization iterations per scale level.
        tv_lambda: Regularization strength for the total variation term.
        crop_3d: Options for cropping the input volume before processing.
            The crop is applied to the volume before it is passed to
            fill_missing_cone.
    """

    delta_background: float = 0.02
    delta_maximal: float = 0.4
    mask_relax: float = 0.05
    max_scale: int = 16
    n_iter: int = 10
    tv_lambda: float = 1e-7
    crop_3d: Crop3DOptions = field(default_factory=Crop3DOptions)
