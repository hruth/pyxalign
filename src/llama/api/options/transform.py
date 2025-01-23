from abc import ABC, abstractmethod
import dataclasses
from dataclasses import field
from typing import Union
import numpy as np
from llama.api.enums import ShiftType, DownsampleType, UpsampleType, RotationType, ShearType
from llama.api.options.device import DeviceOptions


# @dataclasses.dataclass
class TransformOptions(ABC):
    @property
    @abstractmethod
    def enabled(self) -> bool:
        pass

    enabled: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class ShiftOptions(TransformOptions):
    type: ShiftType = ShiftType.FFT

    enabled: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class RotationOptions(TransformOptions):
    type: ShiftType = RotationType.FFT

    enabled: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class ShearOptions(TransformOptions):
    type: ShiftType = ShearType.FFT

    enabled: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class DownsampleOptions(TransformOptions):
    type: DownsampleType = DownsampleType.FFT

    scale: int = 1

    enabled: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)

    use_gaussian_filter: bool = False


@dataclasses.dataclass
class UpsampleOptions(TransformOptions):
    type: UpsampleType = UpsampleType.NEAREST

    scale: float = 1.0

    enabled: bool = False

    device: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class CropOptions(TransformOptions):
    horizontal_range: int = 0

    vertical_range: int = 0

    horizontal_offset: int = 0

    vertical_offset: int = 0

    enabled: bool = False


if __name__ == "__main__":
    shift_options = ShiftOptions()