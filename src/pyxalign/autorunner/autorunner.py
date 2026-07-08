import os
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication

from pyxalign.api.options.base import BaseOptions
from pyxalign.autorunner.abstract import (
    Autorunner,
    save_state_file_wrapper,
    skip_if_loading_from_checkpoint,
    handle_checkpoint,
    _update_all_config_parameters,
    _get_high_level_config_options,
)
from pyxalign.autorunner.config import AutorunnerConfig
from pyxalign.autorunner.enums import AutorunnerStep, Checkpoints
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.interactions.autorunner.data_load_and_init_widget import DataLoadAndInitWidget
from pyxalign.autorunner.support import build_complex_projections
from pyxalign.interactions.autorunner.wrapper import AutorunnerGUIWrapper, AutorunnerRestarted
from pyxalign.interactions.combined_viewer import launch_combined_alignment_widget
from pyxalign.interactions.options.options_editor import launch_basic_options_editor
from pyxalign.interactions.dialog_defaults import set_default_dialog_dir
from pyxalign.interactions.phase_unwrap import launch_phase_unwrap_widget
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.load_any import load_dataset_from_arbitrary_options
from pyxalign.io.loaders.maps import (
    get_experiment_type_enum_from_options,
    get_loader_options_by_enum,
)


class AutorunnerPtycho(Autorunner):
    def __init__(self, state_folder: str):
        self._standardized_data: StandardData
        self.state_folder = state_folder
        self._state_file_path = os.path.join(state_folder, "autorunner_state_file.yaml")

        # Initialize various attributes
        self._data_load_init_widget = None
        self.task = None

    @property
    def loading_options_path(self) -> str:
        return os.path.join(self.state_folder, "loading_options.yaml")

    def run(self):
        self.app = QApplication.instance() or QApplication([])
        while True:
            try:
                self._initialize_autorunner_config()
                self._edit_autorunner_settings()
                self._create_state_folders_and_files()
                self._get_loading_options()
                self._load_data_and_create_task()
                # self._create_projections_object()
                self._open_complex_projections_window()
                self._unwrap_phase()
                self._run_projection_matching_sequence()
                break
            except AutorunnerRestarted:
                print("Restarting autorunner...")
                self.task = None

    def _initialize_autorunner_config(self):
        if os.path.exists(self._state_file_path):
            self.config: AutorunnerConfig = AutorunnerConfig().load_from_path(self._state_file_path)
        else:
            print("Autorunner config not found, using default configuration")
            self.config = AutorunnerConfig()

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
        set_default_dialog_dir(self.state_folder)
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
                title=AutorunnerStep.AUTORUNNER_CONFIGURATION_WINDOW,
                task=getattr(self, "task", None),
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
                show_sync_button=False,
                show_restart_button=False,
            )
            wrapper.wait_for_user_action()

            # check that checkpoint exists
            if not self.config.checkpoint.load_from_checkpoint:
                valid_checkpoint = True
            else:
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
            print(
                f"config.state.update_state_file is True -- the autorunner configuration file will be updated after every step."
            )
        if self.config.state.use_state_file_settings:
            print(
                f"config.state.use_state_file_settings is True -- the pyxalign objects' settings will be updated with values from the task file."
            )

    @skip_if_loading_from_checkpoint
    def _get_loading_options(self):
        options_type = self.config.loading.experiment_type
        self.loading_options: BaseOptions = get_loader_options_by_enum(options_type)
        if self.loading_options_path is not None and os.path.exists(self.loading_options_path):
            self.loading_options.load_from_path(self.loading_options_path)

    @save_state_file_wrapper
    @handle_checkpoint(Checkpoints.AFTER_LOADING)
    def _load_data_and_create_task(self):
        if self.config.interactivity.loading or self.loading_options is None:
            content_gui = DataLoadAndInitWidget(
                load_options=self.loading_options,
                initialization_config=self.config.initialize,
                show_finish_button=False,
            )
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title=AutorunnerStep.DATA_LOADER_WINDOW,
                task=getattr(self, "task", None),
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
                show_sync_button=False,
            )
            wrapper.proceed_button.setEnabled(False)
            content_gui.data_loaded.connect(lambda: wrapper.proceed_button.setEnabled(True))
            wrapper.wait_for_user_action()
            self.loading_options = content_gui.loading_options
            complex_projections = content_gui.get_or_build_complex_projections()
        else:
            self._standardized_data = load_dataset_from_arbitrary_options(
                self.loading_options, int(mp.cpu_count() * 0.8)
            )
            build_complex_projections(self._standardized_data, self.config.initialize)
        self.task = LaminographyAlignmentTask(complex_projections=complex_projections)

        if self.config.state.update_state_file:
            self.loading_options.save_to_dict(self.loading_options_path)
            print(f"Loading options saved to: {self.loading_options_path}")
            self.config.loading.experiment_type = get_experiment_type_enum_from_options(
                self.loading_options
            )

    @save_state_file_wrapper
    @handle_checkpoint(Checkpoints.AFTER_COMPLEX_PROJECTIONS_WINDOW)
    def _open_complex_projections_window(self):
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
                title=AutorunnerStep.COMPLEX_PROJECTIONS_WINDOW,
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
    @handle_checkpoint(Checkpoints.AFTER_PHASE_UNWRAPPING_WINDOW)
    def _unwrap_phase(self):
        if self.config.interactivity.phase_unwrapping:
            content_gui = launch_phase_unwrap_widget(self.task, wait_until_closed=False)
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title=AutorunnerStep.PHASE_UNWRAPPING_WINDOW,
                task=self.task,
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
            )
            wrapper.proceed_button.setEnabled(False)
            content_gui.phase_unwrapped.connect(lambda: wrapper.proceed_button.setEnabled(True))
            wrapper.wait_for_user_action()
        else:
            self.task.get_unwrapped_phase()
        self.task.complex_projections = None

    @save_state_file_wrapper
    @handle_checkpoint(Checkpoints.FINAL)
    def _run_projection_matching_sequence(self):
        if not self.config.projection_matching_enabled:
            return

        if not self.config.interactivity.projection_matching:
            # no automation exists
            pass
        else:
            content_gui = launch_combined_alignment_widget(
                self.task,
                include_projection_matching=True,
                include_cross_correlation=True,
                wait_until_closed=False,
            )
            wrapper = AutorunnerGUIWrapper(
                content_gui,
                title=AutorunnerStep.UNWRAPPED_PROJECTIONS_WINDOW,
                task=self.task,
                checkpoints_folder=self._checkpoints_folder,
                config=self.config,
                state_file_path=self._state_file_path,
            )
            wrapper.wait_for_user_action()

    def save_state_file(self):
        if self.config.state.update_state_file:
            if self.task is not None:
                # config parameters are updated after the event completes
                _update_all_config_parameters(self.task, self.config)
            self.config.save_to_dict(self._state_file_path)
            print(f"Updated state file at {self._state_file_path}")
        else:
            # should always at least update some state file parameters and all checkpoint parameters
            if self._state_file_path is not None and os.path.exists(self._state_file_path):
                current_saved_config: AutorunnerConfig = AutorunnerConfig().load_from_path(
                    self._state_file_path
                )
                current_saved_config.state.use_state_file_settings = (
                    self.config.state.use_state_file_settings
                )
                current_saved_config.state.update_state_file = self.config.state.update_state_file
                current_saved_config.checkpoint = self.config.checkpoint
                current_saved_config.save_to_dict(self._state_file_path)
