from abc import ABC, abstractmethod
from array import ArrayType
import llama.api.maps as maps
from llama.api.options.device import DeviceOptions
from llama.api.options.transform import DownsampleOptions, PreProcessingOptions, TransformOptions


class Transformation(ABC):
    """`Transformation` objects run transformation functions using the passed in device options."""

    def __init__(self, options: TransformOptions):
        self.enabled = options.enabled

    @abstractmethod
    def run(self, images: ArrayType, *args, **kwargs) -> ArrayType:
        pass


class Downsample(Transformation):
    def __init__(
        self,
        options: DownsampleOptions,
    ):
        super().__init__(options)
        self.scale = options.scale
        self.function_type = maps.get_downsample_func_by_enum(options.type)

    def run(self, images: ArrayType) -> ArrayType:
        """Calls one of the image_downsample functions"""
        if self.enabled:
            return self.function_type(images, self.scale)
        else:
            return images


class PreProcess(Transformation):
    def __init__(
        self,
        options: PreProcessingOptions,
    ):
        super().__init__(device_options=options.device_options)
        self.enabled = options.enabled

    def run(images: ArrayType, pre_processing_options: PreProcessingOptions) -> ArrayType:
        # To add:
        # shift
        # crop
        images = Downsample(pre_processing_options).run(images)
        return images
