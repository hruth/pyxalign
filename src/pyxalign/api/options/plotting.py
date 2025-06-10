import dataclasses
from dataclasses import field
from numbers import Number
from typing import Optional, Sequence
from pyxalign.api import enums
from pyxalign.api.options.transform import CropOptions


@dataclasses.dataclass
class UpdatePlotOptions:
    enabled: bool = False

    stride: int = 1


@dataclasses.dataclass
class ScalebarOptions:
    enabled: bool = True

    fractional_width: float = 0.15
    "Must be between 0 and 1"


@dataclasses.dataclass
class PlotDataOptions:
    cmap: Optional[str] = "bone"

    # widths: Optional[tuple[Number, Number]] = None

    # center_offsets: tuple[Number, Number] = (0, 0)

    crop: CropOptions = field(default_factory=CropOptions)

    scalebar: ScalebarOptions = field(default_factory=ScalebarOptions)

    process_func: enums.ProcessFunc = enums.ProcessFunc.NONE

    index: Optional[int] = 0

    clim: Optional[tuple[float]] = None


@dataclasses.dataclass
class ArrayViewerOptions:
    slider_axis: int = 0

    start_index: int = 0

    auto_adjust_clim: bool = False


@dataclasses.dataclass
class ProjectionViewerOptions:
    show_mask: bool = False

    process_func: enums.ProcessFunc = enums.ProcessFunc.NONE

    sort: bool = True
