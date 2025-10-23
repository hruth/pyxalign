from typing import Optional
import numpy as np
import h5py
import copy

# from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.alignment.cross_correlation import CrossCorrelationAligner
from pyxalign.api import enums
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.data_structures.task import run_projection_matching
from pyxalign.data_structures.xrf_projections import XRFProjections
from pyxalign.io.load import load_xrf_projections
from pyxalign.io.save import save_generic_data_structure_to_h5
from pyxalign.io.utils import load_options
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
        center_of_rotation: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        _initialize_from_loaded_data: bool = False,
        _loaded_projections_dict: Optional[dict[str, XRFProjections]] = None,
    ):
        self.projection_options = projection_options
        self.alignment_options = alignment_options
        # force proper typing
        angles = np.array(angles, dtype=r_type)
        scan_numbers = np.array(scan_numbers, dtype=int)

        if not _initialize_from_loaded_data:
            # initialize from dict of arrays
            self.projections_dict: dict[str, XRFProjections] = {}
            self.channels = xrf_array_dict.keys()
            self._primary_channel = primary_channel
            self.create_xrf_projections_object(
                xrf_array_dict, angles, scan_numbers, center_of_rotation, masks
            )
        elif _initialize_from_loaded_data:
            # initialize from dict of projections
            self.projections_dict = _loaded_projections_dict
            self._primary_channel = primary_channel
            self.channels = self.projections_dict.keys()
            # reinforce references
            for channel, proj in self.projections_dict.items():
                proj.options = self.projection_options

        # if center_of_rotation is not None:
        # self.center_of_rotation = center_of_rotation.astype(r_type)
        # else:
        self._center_of_rotation = self.projections_dict[self._primary_channel].center_of_rotation

        # initialize misc variables
        self.pma_object = None
        self.pma_gui_list = []

    def create_xrf_projections_object(
        self,
        xrf_array_dict: dict[str, np.ndarray],
        angles: np.ndarray,
        scan_numbers: np.ndarray,
        center_of_rotation: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ):
        for channel in self.channels:
            # Make a new projections object
            self.projections_dict[channel] = XRFProjections(
                projections=xrf_array_dict[channel],
                angles=copy.copy(angles),
                options=self.projection_options,
                scan_numbers=copy.copy(scan_numbers),
                center_of_rotation=copy.copy(center_of_rotation),
                masks=copy.copy(masks),
                # probe_positions=None,
                # probe=None,
                # skip_pre_processing=False,
                # add_center_offset_to_positions
                # shift_manager=
                # transform_tracker=
            )

    # angles
    @property
    def angles(self) -> np.ndarray:
        # do not allow in-place editing
        angles = self.projections_dict[self._primary_channel].angles
        angles.flags.writeable = False
        return angles

    @angles.setter
    def angles(self, angles: np.ndarray):
        self._set_all_angles(angles)

    def _set_all_angles(self, angles: np.ndarray):
        for _, projections in self.projections_dict.items():
            projections.angles = angles * 1

    @property
    def center_of_rotation(self):
        # do not allow in-place editing
        center_of_rotation = self.projections_dict[self._primary_channel].center_of_rotation
        center_of_rotation.flags.writeable = False
        return center_of_rotation

    @center_of_rotation.setter
    def center_of_rotation(self, center_of_rotation: np.ndarray):
        self._center_of_rotation = center_of_rotation
        for _, projections in self.projections_dict.items():
            projections.center_of_rotation = self._center_of_rotation * 1

    @property
    def scan_numbers(self):
        # do not allow in-place editing
        scan_numbers = self.projections_dict[self._primary_channel].scan_numbers
        scan_numbers.flags.writeable = False
        return scan_numbers

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

    def apply_staged_shift_to_all_channels(self, device_options: Optional[DeviceOptions] = None):
        for _, projections in self.projections_dict.items():
            projections.apply_staged_shift(device_options)

    def undo_last_shift_on_all_channels(self, device_options: Optional[DeviceOptions] = None):
        for _, projections in self.projections_dict.items():
            projections.undo_last_shift(device_options)

    def drop_projections_from_all_channels(self, remove_scans: list[int]):
        for _, projections in self.projections_dict.items():
            projections.drop_projections(remove_scans=remove_scans)

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
        projections.plot_shift(
            shift_type=enums.ShiftManagerMemberType.STAGED_SHIFT,
            title="Cross-correlation Shift",
        )
        print("Cross-correlation shift stored in shift_manager")

    def save_task(self, file_path: str, save_channels: Optional[list[str]] = None):
        if save_channels is None:
            save_channels = self.channels
        else:
            # protect against user error in capitalization
            lower_case_channels = [x.lower() for x in self.channels]
            for i, channel in enumerate(save_channels):
                if channel.lower() in lower_case_channels:
                    idx = np.where([x == channel.lower() for x in lower_case_channels])[0]
                    save_channels[idx] = self.channels[idx]
                else:
                    print(f"Channel '{channel}' not found")
            # primary channel must be included
            if self.primary_channel.lower() not in lower_case_channels:
                save_channels += [self.primary_channel]
            print(save_channels)

        with h5py.File(file_path, "w") as h5_obj:
            proj_channels_group = h5_obj.create_group("projections")
            for channel in save_channels:
                self.projections_dict[channel].save_projections_object(
                    h5_obj=proj_channels_group.create_group(channel)
                )
            save_generic_data_structure_to_h5(
                self.projection_options, h5_obj.create_group("projection_options")
            )
            save_generic_data_structure_to_h5(
                self.alignment_options, h5_obj.create_group("alignment_options")
            )
            h5_obj.create_dataset(name="primary_channel", data=self.primary_channel)
            print(f"XRF task saved to {h5_obj.file.filename}{h5_obj.name}")
            h5_obj["task_file_type"] = "xrf"


def load_xrf_task(file_path: str, exclude_channels: Optional[list[str]] = None) -> XRFTask:
    with h5py.File(file_path, "r") as h5_obj:
        xrf_projections_dict = load_xrf_projections(
            task_h5_obj=h5_obj, exclude_channels=exclude_channels
        )
        primary_channel = h5_obj["primary_channel"][()].decode()
        alignment_options = load_options(h5_obj["alignment_options"], AlignmentTaskOptions)
        projection_options = load_options(h5_obj["projection_options"], ProjectionOptions)
    angles = xrf_projections_dict[primary_channel].angles
    scan_numbers = xrf_projections_dict[primary_channel].scan_numbers
    # masks = xrf_projections_dict[primary_channel].masks
    center_of_rotation = xrf_projections_dict[primary_channel].center_of_rotation
    # xrf_arrays_dict = {channel: proj.data for channel, proj in xrf_projections_dict.items()}
    xrf_task = XRFTask(
        None,
        angles=angles,
        scan_numbers=scan_numbers,
        projection_options=projection_options,
        alignment_options=alignment_options,
        primary_channel=primary_channel,
        center_of_rotation=center_of_rotation,
        # masks=masks,
        _initialize_from_loaded_data=True,
        _loaded_projections_dict=xrf_projections_dict,
    )

    return xrf_task
