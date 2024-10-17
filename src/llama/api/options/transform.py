from abc import ABC, abstractmethod
import dataclasses
from dataclasses import field
from typing import Union
import numpy as np
from llama.api.enums import ShiftType, DownsampleType
from llama.api.options.device import DeviceOptions


@dataclasses.dataclass
class TransformOptions(ABC):
    @property
    @abstractmethod
    def enabled(self) -> bool:
        pass


@dataclasses.dataclass
class ShiftOptions(TransformOptions):
    type: ShiftType = ShiftType.FFT

    shift: np.ndarray = 0

    enabled: bool = False

    device_options: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class DownsampleOptions(TransformOptions):
    type: DownsampleType = DownsampleType.FFT

    scale: int = 1

    enabled: bool = False

    device_options: DeviceOptions = field(default_factory=DeviceOptions)


@dataclasses.dataclass
class CropOptions(TransformOptions):
    horizontal_range: int = 0

    vertical_range: int = 0

    horizontal_offset: int = 0

    vertical_offset: int = 0

    enabled: bool = False


@dataclasses.dataclass
class PreProcessingOptions:
    shift_options: ShiftOptions = field(default_factory=ShiftOptions)

    crop_options: CropOptions = field(default_factory=CropOptions)

    downsample_options: DownsampleOptions = field(default_factory=DownsampleOptions)
