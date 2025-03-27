from typing import Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from llama.alignment.cross_correlation import CrossCorrelationAligner
from llama.api import enums
from llama.api.options.device import DeviceOptions
from llama.api.options.plotting import ImageSliderPlotOptions, SliderPlotOptions
from llama.api.options.projections import ProjectionOptions
from llama.api.options.task import AlignmentTaskOptions
from llama.data_structures.xrf_projections import XRFProjections
from llama.plotting.plotters import make_image_slider_plot
from llama.timing.timer_utils import clear_timer_globals
from llama.transformations.classes import Shifter
from llama.api.types import r_type
import copy


class XRFTask:
    _primary_channel: str
    projections_dict: dict[str, XRFProjections] = {}

    def __init__(
        self,
        xrf_array_dict: dict[str, np.ndarray],
        angles: np.ndarray,
        scan_numbers: np.ndarray,
        projection_options: ProjectionOptions,
        task_options: AlignmentTaskOptions,
        primary_channel: str,
    ):
        # self.angles = angles
        # self.scan_numbers = scan_numbers
        self.channels = xrf_array_dict.keys()
        self.projection_options = projection_options
        self.task_options = task_options
        self._primary_channel = primary_channel

        self.create_xrf_projections_object(xrf_array_dict, angles, scan_numbers)

    def create_xrf_projections_object(
        self,
        xrf_array_dict: dict[str, np.ndarray],
        angles: np.ndarray,
        scan_numbers: np.ndarray,
    ):
        for channel in self.channels:
            # Make a new projections object
            self.projections_dict[channel] = XRFProjections(
                projections=xrf_array_dict[channel],
                angles=angles,
                options=self.projection_options,
                scan_numbers=scan_numbers,
                # probe_positions=None,
                # center_of_rotation=None,
                # masks=None,
                # probe=None,
                # skip_pre_processing=False,
                # add_center_offset_to_positions
                # shift_manager=
                # transform_tracker=
            )

    @property
    def angles(self):
        return self.projections_dict[self._primary_channel].angles

    @property
    def scan_numbers(self):
        return self.projections_dict[self._primary_channel].scan_numbers

    @property
    def primary_channel(self):
        "The channel used for alignment"
        return self._primary_channel

    @primary_channel.setter
    def primary_channel(self, new_channel: str):
        if new_channel in self.channels:
            self._primary_channel = new_channel
        else:
            print(
                f"{new_channel} not in list of existing channels."
                + "Please choose from the available list of channels:"
            )
            for channel in self.channels:
                print(f"{channel}")

    def pin_all_arrays(self):
        for channel, projections in self.projections_dict.items():
            projections.pin_arrays()

    def get_cross_correlation_shift(self):
        clear_timer_globals()
        self.cross_correlation_aligner = CrossCorrelationAligner(
            projections=self.projections_dict[self._primary_channel],
            options=self.task_options.cross_correlation,
        )
        # Placeholder for actual illum_sum
        self.illum_sum = np.ones_like(self.projections_dict[self._primary_channel].data, dtype=r_type)
        shift = self.cross_correlation_aligner.run(self.illum_sum)
        # Stage shift for all projections
        for channel, projections in self.projections_dict.items():
            projections.shift_manager.stage_shift(
                shift=shift,
                function_type=enums.ShiftType.CIRC,
                alignment_options=self.task_options.cross_correlation,
            )
        projections.plot_staged_shift("Cross-correlation Shift")
        print("Cross-correlation shift stored in shift_manager")

    def apply_staged_shift_to_all_channels(self, device_options: Optional[DeviceOptions] = None):
        for channel, projections in self.projections_dict.items():
            projections.apply_staged_shift(device_options)

    def drop_projections_from_all_channels(self, remove_idx: list[int]):
        for channel, projections in self.projections_dict.items():
            projections.drop_projections(remove_idx=remove_idx)

    def plot_xrf_channels(
        self,
        channels: Union[str, list[str]],
        plot_options: Optional[ImageSliderPlotOptions] = None,
        subplot_dims: Optional[Sequence] = None,
    ):
        if isinstance(channels, str):
            channels = [channels]
        if plot_options is None:
            plot_options = ImageSliderPlotOptions()
        plot_objects = []
        for channel in channels:
            if channel not in channels:
                print(f"The channel {channel} is not a valid selection.")
            plot_options_copy = copy.deepcopy(plot_options)
            plot_options_copy.slider.title = channel
            plot_objects += [self.projections_dict[channel].get_plot_object(plot_options_copy)]
        make_image_slider_plot(plot_objects, subplot_dims=subplot_dims)

    def plot_alignment_results(self, channels: Optional[list[str]] = None, sort_idx: Optional[Sequence] = None):
        shift = self.projections_dict[self._primary_channel].shift_manager.staged_shift
        shift_options = self.projections_dict[
            self._primary_channel
        ].shift_manager.get_staged_shift_options()

        if channels is None:
            channels = [self._primary_channel]

        # Make the plotting objects
        plot_objects = []
        for channel in channels:
            projections = self.projections_dict[channel]
            # Make plot object for original data
            plot_object = projections.get_plot_object(
                ImageSliderPlotOptions(slider=SliderPlotOptions(title=channel))
            )
            plot_objects += [plot_object]
            # Make plot object for aligned data
            aligned_plot_object = copy.deepcopy(plot_object)
            aligned_plot_object.array = Shifter(shift_options).run(projections.data, shift)
            aligned_plot_object.options.slider.title = "Aligned"
            plot_objects += [aligned_plot_object]

        n_rows = len(channels)
        n_cols = 2
        make_image_slider_plot(plot_objects, (n_rows, n_cols), sort_idx=sort_idx)
        plt.show()
