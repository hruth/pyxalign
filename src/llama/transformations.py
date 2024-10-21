from abc import ABC, abstractmethod
from array import ArrayType
import numpy as np
import cupy as cp
import scipy
import llama.api.maps as maps
from llama.api.options.device import DeviceOptions, GPUOptions
from llama.api.options.transform import (
    CropOptions,
    DownsampleOptions,
    PreProcessingOptions,
)
from llama.gpu_utils import get_array_module_and_fft_backend


class Transformation(ABC):
    def __init__(self, device_options: DeviceOptions = DeviceOptions):
        pass

    @abstractmethod
    def run(self, images: ArrayType, *args, **kwargs) -> ArrayType:
        pass


class Downsample(Transformation):
    def __init__(
        self,
        options: DownsampleOptions,
    ):
        super().__init__(device_options=options.device_options)
        self.scale = options.scale
        self.function_type = maps.get_downsample_func_by_enum(options.type)
        self.enabled = options.enabled

    def run(self, images: ArrayType) -> ArrayType:
        """Calls one of the image_downsample functions"""
        if self.enabled:
            return self.function_type(images, self.scale)
        else:
            return images


def image_crop(images: ArrayType, crop_options: CropOptions) -> ArrayType:
    pass

def image_crop_pad(images: ArrayType)


def image_shift_fft(images: ArrayType, shift: np.ndarray) -> ArrayType:
    pass


def image_shift_circ(images: ArrayType, shift: np.ndarray) -> ArrayType:
    pass


def image_downsample_fft(images: ArrayType, scale: int) -> ArrayType:
    xp, fft_backend = get_array_module_and_fft_backend(images)

    pad_by = 2
    image_size = np.array(images.shape, dtype=int)[1:]
    image_size_new = np.round(np.ceil(image_size / scale / 2) * 2) + pad_by
    isReal = not xp.issubdtype(images.dtype, xp.complexfloating)

    scale = np.prod(image_size_new - pad_by) / np.prod(image_size)
    downsample = int(np.ceil(np.sqrt(1 / scale)))

    # apply the padding to account for boundary issues
    padWidth = int(downsample * pad_by / 2)
    padShape = cp.pad(images[0], padWidth, "symmetric").shape

    imagesPad = cp.zeros((len(images), padShape[0], padShape[1]), dtype=images.dtype)
    for i in range(len(images)):
        imagesPad[i] = np.pad(images[i], padWidth, "symmetric")
    images = imagesPad
    del imagesPad

    # go to the fourier space
    with scipy.fft.set_backend(fft_backend):
        images = scipy.fft.fft2(images)

        # apply +/-0.5 px shift
        images = image_shift_fft(
            images, np.array([[interpSign * -0.5, interpSign * -0.5]]), applyFFT=False
        )

        # crop in the Fourier space
        images = scipy.fft.ifftshift(
            image_crop_pad(
                scipy.fft.fftshift(images, axes=(1, 2)),
                image_size_new[0],
                image_size_new[1],
            ),
            axes=(1, 2),
        )

        # apply -/+0.5 px shift in the cropped space
        images = image_shift_fft(
            images, np.array([[interpSign * 0.5, interpSign * 0.5]]), applyFFT=False
        )

        images = scipy.fft.ifft2(images)

        # scale to keep the average constant
        images = images * scale

        # remove the padding
        a = int(pad_by / 2)
        images = images[:, a : (image_size_new[0] - a), a : (image_size_new[1] - a)]

        if isReal:
            images = cp.real(images)

    return images


def image_downsample_linear(images: ArrayType, scale: int) -> ArrayType:
    pass


def image_pre_process(
    images: ArrayType, pre_processing_options: PreProcessingOptions
) -> ArrayType:
    # shift
    # crop
    images = Downsample(pre_processing_options).run(images)
    pass
