from typing import Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from llama.alignment.cross_correlation import calculate_alignment_shift
from llama.api.options.alignment import CrossCorrelationOptions
from llama.api.options.device import DeviceOptions
from llama.api.options.plotting import ImageSliderPlotOptions, SliderPlotOptions
from llama.api.options.transform import ShiftOptions
from llama.gpu_utils import pin_memory
from llama.plotting.plotters import make_image_slider_plot, ImagePlotObject
from llama.transformations.helpers import round_to_divisor
from llama.api.constants import divisor
from llama.transformations.classes import Shifter
import copy


class XRFObject:
    def __init__(
        self,
        counts_per_second: dict[int, dict[str, np.ndarray]],
        angles: np.ndarray,
    ):
        self.counts_per_second = counts_per_second  # key: scan number, value: dict with channel key
        self.angles = np.array(angles)
        self.scan_numbers = np.array(list(self.counts_per_second.keys()), dtype=int)
        self.channels = np.array(list(counts_per_second[self.scan_numbers[0]].keys()))

    def create_flourescence_arrays(self):
        self.remove_inconsistent_sizes()
        self.xrf_arrays = {}
        for channel in self.channels:
            self.xrf_arrays[channel] = np.array(
                [array[channel] for array in self.counts_per_second.values()]
            )

    def remove_inconsistent_sizes(self, channel: str = "Total_Fluorescence_Yield"):
        # get all unique shapes
        shapes = [v[channel].shape for v in self.counts_per_second.values()]
        # Get the count per each shape
        n_arrays_per_shape = []
        for shape in set(shapes):
            n_arrays_per_shape += [shape == x for x in shapes]
        idx = np.argmax(n_arrays_per_shape)
        most_common_shape = shapes[idx]
        idx_keep = [x == most_common_shape for x in shapes]
        idx_remove = [not x for x in idx_keep]
        for scan in self.scan_numbers[idx_remove]:
            del self.counts_per_second[scan]
        self.angles = self.angles[idx_keep]
        self.scan_numbers = self.scan_numbers[idx_keep]

    def pad_arrays(self, shape_divisor: int):
        def pad_single_array(array: np.ndarray):
            new_spatial_shape = round_to_divisor(array.shape[1:], "ceil", shape_divisor)
            pad_amount = new_spatial_shape - np.array(array.shape[1:])
            pad_by = np.array(
                (
                    (0, 0),
                    (np.floor(pad_amount[0] / 2), np.ceil(pad_amount[0] / 2)),
                    (np.floor(pad_amount[1] / 2), np.ceil(pad_amount[1] / 2)),
                )
            ).astype(int)

            return np.pad(array, pad_by)

        for channel, array in self.xrf_arrays.items():
            self.xrf_arrays[channel] = pad_single_array(array)

    def pin_arrays(self):
        for channel in self.channels:
            self.xrf_arrays[channel] = pin_memory(self.xrf_arrays[channel])

    def get_cross_correlation_shift(self, options: CrossCorrelationOptions, channel: str):
        dummy_illum_sum = np.ones_like(self.xrf_arrays[channel])[0]
        self.shift = calculate_alignment_shift(
            self.xrf_arrays[channel],
            self.angles,
            dummy_illum_sum,
            options,
        )

    def shift_xrf_objects(self, device_options: DeviceOptions):
        shift_options = ShiftOptions(type="circ", enabled=True, device=device_options)
        for channel in self.channels:
            self.xrf_arrays[channel] = Shifter(shift_options).run(
                self.xrf_arrays[channel], self.shift
            )

    def _get_xrf_plot_object(
        self, channel: str, plot_options: Optional[ImageSliderPlotOptions] = None
    ) -> ImagePlotObject:
        if plot_options is None:
            plot_options = ImageSliderPlotOptions(slider=SliderPlotOptions(title=channel))
        plot_object = ImagePlotObject(self.xrf_arrays[channel], options=plot_options)
        return plot_object

    def plot_xrf_channels(
        self,
        channels: Union[str, list[str]],
        plot_options: Optional[ImageSliderPlotOptions] = None,
        subplot_dims: Optional[Sequence] = None,
    ):
        if isinstance(channels, str):
            channels = [channels]
        plot_objects = []
        for channel in channels:
            plot_objects += [self._get_xrf_plot_object(channel, plot_options)]
        make_image_slider_plot(plot_objects, subplot_dims=subplot_dims)

    def plot_cross_correlation_results(self, device_options: DeviceOptions, channels: list[str]):
        shift_options = ShiftOptions(type="circ", enabled=True, device=device_options)

        # Make the plotting objects
        plot_objects = []
        # n_cols = 2
        for row, channel in enumerate(channels):
            plot_object = self._get_xrf_plot_object(channel)
            plot_objects += [plot_object]

            aligned_plot_object = copy.deepcopy(plot_object)
            aligned_plot_object.array = Shifter(shift_options).run(
                self.xrf_arrays[channel], self.shift
            )
            aligned_plot_object.options.slider.title = "Aligned"
            plot_objects += [aligned_plot_object]

        n_rows = len(channels)
        n_cols = 2
        make_image_slider_plot(plot_objects, (n_rows, n_cols))
        plt.show()
