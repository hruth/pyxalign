from typing import Optional
import numpy as np
from pyxalign import LaminographyAlignmentTask
from pyxalign.alignment.cross_correlation import CrossCorrelationAligner
from pyxalign.api import enums
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.data_structures.task import run_projection_matching
from pyxalign.data_structures.xrf_projections import XRFProjections
from pyxalign.timing.timer_utils import clear_timer_globals
from pyxalign.api.types import r_type


class XRFTask:
    def __init__(
        self,
        xrf_array_dict: dict[str, np.ndarray],
        angles: np.ndarray,
        scan_numbers: np.ndarray,
        projection_options: ProjectionOptions,
        alignment_options: AlignmentTaskOptions,
        primary_channel: str,
    ):
        self.pma_object = None
        self.pma_gui_list = []
        self._primary_channel: str
        self.projections_dict: dict[str, XRFProjections] = {}
        # self.angles = angles
        # self.scan_numbers = scan_numbers
        self.channels = xrf_array_dict.keys()
        self.projection_options = projection_options
        self.alignment_options = alignment_options
        self._primary_channel = primary_channel
        self.create_xrf_projections_object(xrf_array_dict, angles, scan_numbers)
        self._center_of_rotation = self.projections_dict[self._primary_channel].center_of_rotation

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

    # @property
    # def phase_projections(self):
    #     return self.projections_dict[self._primary_channel]

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

    @property
    def center_of_rotation(self):
        return self._center_of_rotation

    @center_of_rotation.setter
    def center_of_rotation(self, center_of_rotation: np.ndarray):
        self._center_of_rotation = center_of_rotation
        for _, projections in self.projections_dict.items():
            projections.center_of_rotation = self._center_of_rotation * 1

    def apply_staged_shift_to_all_channels(self, device_options: Optional[DeviceOptions] = None):
        for _, projections in self.projections_dict.items():
            projections.apply_staged_shift(device_options)

    def drop_projections_from_all_channels(self, remove_idx: list[int]):
        for _, projections in self.projections_dict.items():
            projections.drop_projections(remove_idx=remove_idx)

    def pin_all_arrays(self):
        for _, projections in self.projections_dict.items():
            projections.pin_arrays()

    def get_projection_matching_shift(
        self, initial_shift: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # clear existing astra objects
        if self.pma_object is not None:
            if hasattr(self.pma_object, "aligned_projections"):
                self.pma_object.aligned_projections.volume.clear_astra_objects()

        # reset timers
        clear_timer_globals()

        # close old gui windows
        if self.alignment_options.projection_matching.interactive_viewer.close_old_windows:
            self.clear_pma_gui_list()
            if self.pma_object is not None and self.pma_object.gui is not None:
                self.pma_object.gui.close()
        else:
            self.pma_gui_list += [self.pma_object.gui]

        # run the pma algorithm
        self.pma_object, shift = run_projection_matching(
            self.projections_dict[self.primary_channel],
            initial_shift,
            self.alignment_options.projection_matching,
        )

        # Save the resulting alignment shift
        for _, projections in self.projections_dict.items():
            projections.shift_manager.stage_shift(
                shift=shift,
                function_type=enums.ShiftType.FFT,
                alignment_options=self.alignment_options.projection_matching,
            )
        print("Projection-matching shifts stored in shift_manager")

        return shift

    def clear_pma_gui_list(self):
        for gui in self.pma_gui_list:
            gui.close()
        self.pma_gui_list = []

    def get_cross_correlation_shift(self, illum_sum: np.ndarray = None):
        clear_timer_globals()
        self.cross_correlation_aligner = CrossCorrelationAligner(
            projections=self.projections_dict[self._primary_channel],
            options=self.alignment_options.cross_correlation,
        )
        # Placeholder for actual illum_sum
        if illum_sum is None:
            self.illum_sum = np.ones_like(
                self.projections_dict[self._primary_channel].data[0], dtype=r_type
            )
        else:
            self.illum_sum = illum_sum
        shift = self.cross_correlation_aligner.run(self.illum_sum)
        # Stage shift for all projections
        for channel, projections in self.projections_dict.items():
            projections.shift_manager.stage_shift(
                shift=shift,
                function_type=enums.ShiftType.CIRC,
                alignment_options=self.alignment_options.cross_correlation,
            )
        projections.plot_staged_shift("Cross-correlation Shift")
        print("Cross-correlation shift stored in shift_manager")

    # def launch_xrf_projections_viewer(self):
