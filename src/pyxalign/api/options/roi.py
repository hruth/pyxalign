import dataclasses
from dataclasses import field
from enum import StrEnum, auto
from typing import Optional
from .base import BaseOptions


class ROIType(StrEnum):
    RECTANGULAR = auto()
    ELLIPTICAL = auto()


@dataclasses.dataclass
class RectangularROIOptions(BaseOptions):
    horizontal_range: Optional[int] = None

    vertical_range: Optional[int] = None

    horizontal_offset: int = 0

    vertical_offset: int = 0


@dataclasses.dataclass
class EllipticalROIOptions(BaseOptions):
    pass


@dataclasses.dataclass
class ROIOptions(BaseOptions):
    shape: ROIType = ROIType.RECTANGULAR

    rectangle: RectangularROIOptions = field(default_factory=RectangularROIOptions)