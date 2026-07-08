import os
from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional
import numpy as np

from pyxalign.api.options_utils import get_all_attribute_names
from pyxalign.api.types import r_type
from pyxalign.autorunner.config import AutorunnerConfig
from pyxalign.autorunner.enums import get_checkpoint_order_value
from pyxalign.data_structures.task import LaminographyAlignmentTask, load_task
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper


class Autorunner(ABC):
    def __init__(self):
        self.task: LaminographyAlignmentTask
        self.config: AutorunnerConfig
        self._standardized_data
        self._state_file_path: str
        self.state_folder: str

    @property
    def _checkpoints_folder(self) -> Optional[str]:
        return os.path.join(self.state_folder, "checkpoints")

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save_state_file(self):
        pass


def save_state_file_wrapper(func):
    """Decorator that saves the config to the state file after method execution."""

    @wraps(func)
    def wrapper(self: Autorunner, *args, **kwargs):
        if self.config.state.use_state_file_settings and self.task is not None:
            # load existing state file settings into task
            _update_pyxalign_object_settings(self.task, self.config)
        result = func(self, *args, **kwargs)
        self.save_state_file()
        return result
    return wrapper

def skip_if_loading_from_checkpoint(func):

    @wraps(func)
    def wrapper(self: Autorunner, *args, **kwargs):
        if not self.config.checkpoint.load_from_checkpoint:
            result = func(self, *args, **kwargs)
        else:
            return
        return result

    return wrapper


def handle_checkpoint(checkpoint: str):
    def checkpoint_inner(func):
        @wraps(func)
        def wrapper(self: Autorunner, *args, **kwargs):
            checkpoint_path = os.path.join(self._checkpoints_folder, checkpoint + "_task.h5")
            if not self.config.checkpoint.load_from_checkpoint:
                result = func(self, *args, **kwargs)
            else:
                # check if past the current checkpoint or not
                current_checkpoint_val = get_checkpoint_order_value(checkpoint)
                loaded_checkpoint_val = get_checkpoint_order_value(
                    self.config.checkpoint.which_checkpoint
                )
                if current_checkpoint_val < loaded_checkpoint_val:
                    # before checkpoint
                    return
                elif current_checkpoint_val == loaded_checkpoint_val:
                    # at checkpoint
                    load_task_wrapped = loading_bar_wrapper(
                        "Loading pyxalign task...", block_all_windows=True
                    )(load_task)
                    if self.config.checkpoint.load_from_custom_task:
                        self.task = load_task_wrapped(self.config.checkpoint.custom_task_path)
                    else:
                        self.task = load_task_wrapped(checkpoint_path)
                    # sync loaded task with settings file
                    if self.config.state.use_state_file_settings:
                        _update_pyxalign_object_settings(self.task, self.config)
                    return
                elif current_checkpoint_val > loaded_checkpoint_val:
                    # after checkpoint
                    result = func(self, *args, **kwargs)

            # save checkpoint if enabled
            if getattr(self.config.checkpoint.enabled_checkpoints, checkpoint):
                if not os.path.exists(self._checkpoints_folder):
                    os.mkdir(self._checkpoints_folder)
                self.task.save_task(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint}")
                if self.config.state.use_state_file_settings:
                    self.config.checkpoint.which_checkpoint = checkpoint
                    self.config.checkpoint.load_from_custom_task = False
                # After proceeding to next checkpoint, custom task loading
                # should be disabled on next run

            return result

        return wrapper

    return checkpoint_inner


def _update_all_config_parameters(task: LaminographyAlignmentTask, config: AutorunnerConfig):
    # - Not all parameters will be updated here, just the ones in the task or projections
    #   objects
    # - It might be better to break into multiple config items for something like reconstruct,
    #   depending on when it is change?

    # Update task level options
    config.cross_correlation = task.options.cross_correlation
    config.projection_matching = task.options.projection_matching  # this should be defaults instead

    # Update projection level options
    if task.phase_projections is not None:
        projections = task.phase_projections
        config.projection_matching_masks_from_position = projections.options.mask_from_positions
        config.projection_matching_masks_from_roi = projections.options.masks_from_roi
    else:
        projections = task.complex_projections
        config.phase_unwrap_masks_from_position = projections.options.mask_from_positions
        config.phase_unwrap_masks_from_roi = projections.options.masks_from_roi
    config.unwrap_phase = projections.options.phase_unwrap
    # reconstruct parameters
    # update sample thickness in config
    config.reconstruct.sample_thickness = projections.options.experiment.sample_thickness
    # update volume width
    config.reconstruct.volume_width = projections.options.volume_width
    # update vertical cor offset in config
    unshifted_center_of_rotation = np.array(projections.data.shape[1:], dtype=r_type) / 2
    config.reconstruct.center_vertical_offset = (
        projections.center_of_rotation[0] - unshifted_center_of_rotation[0]
    )
    config.reconstruct.center_horizontal_offset = (
        projections.center_of_rotation[1] - unshifted_center_of_rotation[1]
    )
    config.reconstruct.reconstruct = projections.options.reconstruct
    config.initialize.laminography_angle = projections.options.experiment.laminography_angle
    print("Updated autorunner config using task object settings")


def _update_pyxalign_object_settings(task: LaminographyAlignmentTask, config: AutorunnerConfig):
    task.options.projection_matching = config.projection_matching
    task.options.cross_correlation = config.cross_correlation

    if task.phase_projections is not None:
        projections = task.phase_projections
        projections.options.mask_from_positions = config.projection_matching_masks_from_position
        projections.options.masks_from_roi = config.projection_matching_masks_from_roi
    else:
        projections = task.complex_projections
        projections.options.mask_from_positions = config.phase_unwrap_masks_from_position
        projections.options.masks_from_roi = config.phase_unwrap_masks_from_roi

    projections.options.phase_unwrap = config.unwrap_phase
    # reconstruct parameters
    # update sample thickness in config
    projections.options.experiment.sample_thickness = config.reconstruct.sample_thickness
    projections.options.volume_width = config.reconstruct.volume_width
    # update vertical cor offset in config
    unshifted_center_of_rotation = np.array(projections.data.shape[1:], dtype=r_type) / 2
    projections.center_of_rotation[0] = (
        config.reconstruct.center_vertical_offset + unshifted_center_of_rotation[0]
    )
    projections.center_of_rotation[1] = (
        config.reconstruct.center_horizontal_offset + unshifted_center_of_rotation[1]
    )
    projections.options.reconstruct = config.reconstruct.reconstruct
    projections.options.experiment.laminography_angle = config.initialize.laminography_angle
    print("Updated task object settings using autorunner config")


def _get_high_level_config_options() -> list[str]:
    high_level_config_options = [
        "state.state_folder",
        "state.use_state_file_settings",
        "state.update_state_file",
        # "state.state_memory_enabled",
        # "state.use_state_file",
        "state.update_state_file",
        # "interactivity",
        # "cross_correlation_enabled",
        # "projection_matching_enabled",
        "enabled_checkpoints",
        "checkpoint",
    ]

    # commenting out to remove interactivity options from basic options panel
    # high_level_config_options += [
    #     "interactivity." + x for x in get_all_attribute_names(AutorunnerConfig().interactivity)
    # ]
    high_level_config_options += [
        "checkpoint." + x for x in get_all_attribute_names(AutorunnerConfig().checkpoint)
    ]
    high_level_config_options += [
        "checkpoint.enabled_checkpoints." + x
        for x in get_all_attribute_names(AutorunnerConfig().checkpoint.enabled_checkpoints)
    ]
    # high_level_config_options += [
    #     "state." + x
    #     for x in get_all_attribute_names(AutorunnerConfig().state)
    # ]

    return high_level_config_options
