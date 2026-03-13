import os
from typing import Optional
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication

from pyxalign.api.options.base import BaseOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.transform import RotationOptions, ShearOptions
from pyxalign.autorunner.abstract import (
    Autorunner,
    save_state_file_wrapper,
    skip_if_loading_from_checkpoint,
    handle_checkpoint,
    _update_all_config_parameters,
    _update_pyxalign_object_settings,
    _get_high_level_config_options,
)
from pyxalign.autorunner.config import AutorunnerConfig
from pyxalign.data_structures.projections import  ComplexProjections
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.interactions.autorunner.initialization_widget import (
    launch_initialization_config_widget,
)
from pyxalign.interactions.autorunner.wrapper import AutorunnerGUIWrapper
from pyxalign.interactions.combined_viewer import launch_combined_alignment_widget
from pyxalign.interactions.io.loader import launch_data_loader
from pyxalign.interactions.mask import launch_mask_builder
from pyxalign.interactions.options.options_editor import (
    launch_basic_options_editor,
)
from pyxalign.interactions.phase_unwrap import launch_phase_unwrap_widget

from pyxalign.interactions.reconstruction_parameter_tuner import (
    launch_reconstruction_parameter_tuner,
)
from pyxalign.io.loaders.base import StandardData

from pyxalign.io.loaders.load_any import load_dataset_from_arbitrary_options
from pyxalign.io.loaders.maps import (
    get_experiment_type_enum_from_options,
    get_loader_options_by_enum,
)
from pyxalign.io.loaders.utils import convert_projection_dict_to_array


class AutorunnerPtycho(Autorunner):
    def __init__(self, state_folder: str):
        self._standardized_data: StandardData
        self.state_folder = state_folder
        file_path = os.path.join(state_folder, "autorunner_state_file.yaml")

        self._state_file_path = None
        if file_path is not None:
            if os.path.exists(file_path):
                self.config: AutorunnerConfig = AutorunnerConfig().load_from_path(file_path)
                # make sure the state_folder path points to the correct location

            else:
                print("Autorunner config not found, using default configuration")
                self.config = AutorunnerConfig()
        else:
            self.config = AutorunnerConfig()

    @property
    def loading_options_path(self) -> str:
        return os.path.join(self.state_folder, "loading_options.yaml")

    def run(self):
        self.app = QApplication.instance() or QApplication([])
        self._edit_autorunner_settings()
        self._create_state_folders_and_files()
        self._get_loading_options()
        self._load_data()
        self._get_initialization_options()
        self._create_projections_object()
        self._get_cross_correlation_alignment()
        self._get_complex_projections_masks()
        self._unwrap_phase()
        self._select_center_of_rotation()
        self._get_phase_projections_masks()
        self._run_projection_matching_sequence()
        self._get_final_reconstruction()
        # save volumes ?

    @save_state_file_wrapper
    def _create_state_folders_and_files(self):
        # Create state folder
        if not os.path.exists(self.state_folder):
            os.mkdir(self.state_folder)
            print(f"Created state folder: {self.state_folder}")
        # create checkpoints folder
        if not os.path.exists(self._checkpoints_folder):
            os.mkdir(self._checkpoints_folder)

        if not self.config.state.use_state_file_settings:
            return
        
        # create the state file
        if not os.path.exists(self._state_file_path):
            self.config.save_to_dict(self._state_file_path)

    # @save_state_file
    def _edit_autorunner_settings(self):
        app = QApplication.instance() or QApplication([])

        valid_checkpoint = False
        while not valid_checkpoint:
            content_gui = launch_basic_options_editor(
                self.config,
                enable_advanced_tab=True,
                basic_options_list=_get_high_level_config_options(),
                open_panels_list=[
                    "checkpoint",
                    "enabled_checkpoints",
                    "interactivity",
                    "state",
                ],
                folder_dialog_fields=["state.state_folder"],
                file_dialog_fields=["loading.initial_options_path", "checkpoint.custom_task_path"],
                label="Update Autorunner Configuration",
                wait_until_closed=False,
            )
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title="Autorunner Configuration",
                task=getattr(self, "task", None),
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
            )
            wrapper.wait_for_user_action()
            # if self.config.state.use_state_file_settings:
            self._state_file_path = os.path.join(
                self.state_folder, "autorunner_state_file.yaml"
            )

            # check that checkpoint exists
            if not self.config.checkpoint.load_from_checkpoint:
                valid_checkpoint = True
            else:
                # checkpoints_folder = os.path.join(self.state_folder, "checkpoints")
                checkpoint_path = os.path.join(
                    self._checkpoints_folder, self.config.checkpoint.which_checkpoint + "_task.h5"
                )
                if (
                    not os.path.exists(checkpoint_path)
                    and not self.config.checkpoint.load_from_custom_task
                ):
                    print(f"There is no {self.config.checkpoint.which_checkpoint} checkpoint file.")
                    print(f"Available checkpoint files:")
                    for file_name in os.listdir(self._checkpoints_folder):
                        print("- " + file_name)
                elif self.config.checkpoint.load_from_custom_task and not os.path.exists(
                    self.config.checkpoint.custom_task_path
                ):
                    print(
                        f"No file found at custom task path: {self.config.checkpoint.custom_task_path}"
                    )
                else:
                    valid_checkpoint = True
        if self.config.state.update_state_file:
            print(f"config.state.update_state_file is True -- the autorunner configuration file will be updated after every step.")
        if self.config.state.use_state_file_settings:
            print(f"config.state.use_state_file_settings is True -- the pyxalign objects' settings will be updated with values from the task file.")


    @skip_if_loading_from_checkpoint
    def _get_loading_options(self):
        # path = self.config.loading.initial_options_path
        options_type = self.config.loading.experiment_type
        self.loading_options: BaseOptions = get_loader_options_by_enum(options_type)
        if self.loading_options_path is not None and os.path.exists(self.loading_options_path):
            self.loading_options.load_from_path(self.loading_options_path)

    @skip_if_loading_from_checkpoint
    @save_state_file_wrapper
    def _load_data(self):
        if self.config.interactivity.loading or self.loading_options is None:
            self._standardized_data, self.loading_options = launch_data_loader(self.loading_options)
        else:
            self._standardized_data = load_dataset_from_arbitrary_options(
                self.loading_options, int(mp.cpu_count() * 0.8)
            )

        if self.config.state.update_state_file:
            # save options
            # initial_options_path = os.path.join(
            #     self.state_folder, "loading_options.yaml"
            # )
            self.loading_options.save_to_dict(self.loading_options_path)
            print(f"Loading options saved to: {self.loading_options_path}")
            # update autorunner config
            self.config.loading.experiment_type = get_experiment_type_enum_from_options(
                self.loading_options
            )
            # self.config.loading.initial_options_path = initial_options_path

    @skip_if_loading_from_checkpoint
    @save_state_file_wrapper
    def _get_initialization_options(self):
        if self.config.interactivity.initialization:
            self.config.initialize = launch_initialization_config_widget(
                self._standardized_data, self.config.initialize
            )

    # @save_state_file  # needs to be done within fnction
    @handle_checkpoint("initialization")
    def _create_projections_object(self):
        # create padded projection array
        new_array_size = self._standardized_data.get_minimum_size_for_projection_array()
        new_array_size += self.config.initialize.pad
        projection_array = convert_projection_dict_to_array(
            self._standardized_data.projections, new_array_size, pad_with_mode=True
        )

        # define projection options
        projection_options = ProjectionOptions()
        # experiment parameters
        projection_options.experiment.laminography_angle = self.config.initialize.laminography_angle
        # projection_options.experiment.sample_thickness = self.config.initialize.sample_thickness
        projection_options.experiment.pixel_size = self._standardized_data.pixel_size
        # input processing
        if self.config.initialize.rotation_angle != 0:
            projection_options.input_processing.rotation = RotationOptions(
                enabled=True, angle=self.config.initialize.rotation_angle
            )
        if self.config.initialize.shear_angle != 0:
            projection_options.input_processing.shear = ShearOptions(
                enabled=True, angle=self.config.initialize.shear_angle
            )

        # create complex_projections object
        complex_projections = ComplexProjections(
            projections=projection_array,
            angles=self._standardized_data.angles,
            scan_numbers=self._standardized_data.scan_numbers,
            options=projection_options,
            probe_positions=list(self._standardized_data.probe_positions.values()),
            probe=self._standardized_data.probe,
            skip_pre_processing=False,
            file_paths=list(self._standardized_data.file_paths.values()),
        )
        self.task = LaminographyAlignmentTask(complex_projections=complex_projections)
        if self.config.initialize.remove_scan_numbers is not None:
            self.task.complex_projections.drop_projections(
                self.config.initialize.remove_scan_numbers
            )

        self._standardized_data = None

        _update_pyxalign_object_settings(self.task, self.config)
        _update_all_config_parameters(self.task, self.config)

    @save_state_file_wrapper
    @handle_checkpoint("cross_correlation")
    def _get_cross_correlation_alignment(self):
        if not self.config.cross_correlation_enabled:
            return

        if self.config.interactivity.cross_correlation:
            content_gui = launch_combined_alignment_widget(
                self.task,
                include_projection_matching=False,
                include_cross_correlation=True,
                wait_until_closed=False,
            )
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title="Cross Correlation Alignment",
                task=self.task,
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
            )
            wrapper.wait_for_user_action()
        else:
            self.task.get_cross_correlation_shift(plot_results=False)
            self.task.complex_projections.apply_staged_shift()

    @save_state_file_wrapper
    @handle_checkpoint("phase_unwrap_masks")
    def _get_complex_projections_masks(self):
        if self.config.interactivity.phase_unwrap_masks:
            content_gui = launch_mask_builder(self.task.complex_projections, wait_until_closed=True)
        else:
            self.task.complex_projections.get_masks_from_probe_positions()

    @save_state_file_wrapper
    @handle_checkpoint("phase_unwrapping")
    def _unwrap_phase(self):
        print("Perform phase unwrapping...")

        if self.config.interactivity.phase_unwrapping:
            content_gui = launch_phase_unwrap_widget(self.task, wait_until_closed=False)
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title="Phase Unwrapping",
                task=self.task,
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
            )
            wrapper.wait_for_user_action()
        else:
            self.task.get_unwrapped_phase()
        self.task.complex_projections = None

    @save_state_file_wrapper
    @handle_checkpoint("pma_masks")
    def _get_phase_projections_masks(self):
        print("Select masks used in projection-matching alignment...")

        if self.config.interactivity.pma_masks:
            content_gui = launch_mask_builder(self.task.phase_projections, wait_until_closed=True)
        else:
            self.task.phase_projections.get_masks_from_probe_positions()

    @save_state_file_wrapper
    @handle_checkpoint("reconstruction_tuning")
    def _select_center_of_rotation(self):
        print("Select reconstruction parameters...")
        app = QApplication.instance() or QApplication([])

        if self.config.interactivity.reconstruction_tuning:
            content_gui = launch_reconstruction_parameter_tuner(
                self.task.phase_projections, wait_until_closed=False
            )
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title="Reconstruction Parameter Tuning",
                task=self.task,
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
            )
            wrapper.wait_for_user_action()

    @save_state_file_wrapper
    @handle_checkpoint("projection_matching")
    def _run_projection_matching_sequence(self):
        if not self.config.projection_matching_enabled:
            return

        if not self.config.interactivity.projection_matching:
            # need to figure out how to specify sequences first
            pass
            #  self.task.phase_projections.apply_staged_shift()
        else:
            content_gui = launch_combined_alignment_widget(
                self.task,
                include_projection_matching=True,
                include_cross_correlation=True,
                # self._options_dict["projection_matching_alignment"]["sequence"],
                wait_until_closed=False,
            )
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title="Projection Matching Sequence",
                task=self.task,
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
            )
            wrapper.wait_for_user_action()

    @save_state_file_wrapper
    @handle_checkpoint("final_reconstruction")
    def _get_final_reconstruction(self):
        print("Select reconstruction parameters...")
        app = QApplication.instance() or QApplication([])

        if self.config.interactivity.reconstruction_tuning:
            content_gui = launch_reconstruction_parameter_tuner(
                self.task.phase_projections, is_already_aligned=True, wait_until_closed=False
            )
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title="Final 3D Reconstruction",
                task=self.task,
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
            )
            wrapper.wait_for_user_action()

    def save_state_file(self):
        if self.config.state.update_state_file:
            if hasattr(self, "task"):
                # config parameters are updated after the event completes
                print("Task updated with state file parameters")
                _update_all_config_parameters(self.task, self.config)
            self.config.save_to_dict(self._state_file_path)
            print(f"Updated state file at {self._state_file_path}")
        else:
            # should always at least update some state file parameters and all checkpoint parameters
            if self._state_file_path is not None and os.path.exists(self._state_file_path):
                current_saved_config: AutorunnerConfig = AutorunnerConfig().load_from_path(self._state_file_path)
                current_saved_config.state.use_state_file_settings = self.config.state.use_state_file_settings
                current_saved_config.state.update_state_file = self.config.state.update_state_file
                current_saved_config.checkpoint = self.config.checkpoint
                current_saved_config.save_to_dict(self._state_file_path)
                print(current_saved_config.state)
                print(self._state_file_path)
