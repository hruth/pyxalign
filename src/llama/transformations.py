from abc import ABC, abstractmethod
from array import ArrayType
import numpy as np
import llama.api.maps as maps
from llama.api.options.device import DeviceOptions, GPUOptions
from llama.api.options.transform import CropOptions, DownsampleOptions, PreProcessingOptions

class Transformation(ABC):
    def __init__(self, device_options: DeviceOptions = DeviceOptions):
        pass

    @abstractmethod
    def run(self, images: ArrayType, *args, **kwargs) -> ArrayType:
        pass


class Downsample(Transformation):
    def __init__(
        self,
        downsample_options: DownsampleOptions,
    ):
        super().__init__(device_options=DownsampleOptions.device_options)
        self.scale = downsample_options.scale
        self.function_type = maps.get_downsample_func_by_enum(downsample_options.type)

    def run(self, images: ArrayType) -> ArrayType:
        """Calls one of the image_downsample functions"""
        return self.function_type(images, self.scale)


def image_crop(images: ArrayType, crop_options: CropOptions) -> ArrayType:
    pass


def image_shift_fft(images: ArrayType, shift: np.ndarray) -> ArrayType:
    pass


def image_shift_circ(images: ArrayType, shift: np.ndarray) -> ArrayType:
    pass


def image_downsample_fft(images: ArrayType, scale: int) -> ArrayType:
    pass


def image_downsample_linear(images: ArrayType, scale: int) -> ArrayType:
    pass


def image_pre_process(images: ArrayType, pre_processing_options: PreProcessingOptions) -> ArrayType:
    # shift
    # crop
    images = Downsample(pre_processing_options).run(images)
    pass