import dataclasses
from dataclasses import field
from numbers import Number
from typing import Optional, Sequence
from llama.api import enums


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
class ImagePlotOptions:
    cmap: Optional[str] = "bone"

    widths: Optional[tuple[Number, Number]] = None

    center_offsets: tuple[Number, Number] = (0, 0)

    scalebar: ScalebarOptions = field(default_factory=ScalebarOptions)

    # process_func: callable = lambda x: x
    process_func: Optional[enums.ProcessFunc] = None

    index: Optional[int] = None

    clim: Optional[Sequence] = None

    colorbar: bool = False


@dataclasses.dataclass
class LinePlotOptions:
    ylabel: Optional[str] = None

    ylim: Optional[Sequence[int]] = None


@dataclasses.dataclass
class SliderPlotOptions:
    title: Optional[str] = None

    indexed_titles: Optional[list[str]] = None

    sort_idx: Optional[Sequence[int]] = None

    subplot_idx: Optional[Sequence[int]] = None


@dataclasses.dataclass
class ImageSliderPlotOptions:
    slider: SliderPlotOptions = field(default_factory=SliderPlotOptions)

    image: ImagePlotOptions = field(default_factory=ImagePlotOptions)


@dataclasses.dataclass
class LineSliderPlotOptions:
    slider: SliderPlotOptions = field(default_factory=SliderPlotOptions)

    image: LinePlotOptions = field(default_factory=ImagePlotOptions)
