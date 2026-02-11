import copy
import os
from abc import ABC, abstractmethod
import yaml
import multiprocessing as mp

from pyxalign.api.options.alignment import CrossCorrelationOptions, ProjectionMatchingOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.transform import RotationOptions, ShearOptions
from pyxalign.autorunner.enums import Checkpoints, get_checkpoint_order_value
from pyxalign.autorunner.io import (
    get_projection_matching_sequence_options,
    get_updated_options,
    load_options_from_yaml,
)
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.data_structures.task import LaminographyAlignmentTask, load_task
from pyxalign.estimate_center import plot_center_of_rotation_estimate_results
from pyxalign.interactions.cross_correlation import launch_cross_correlation_gui
from pyxalign.interactions.io.loader import launch_data_loader
from pyxalign.interactions.mask import launch_mask_builder
from pyxalign.interactions.phase_unwrap import launch_phase_unwrap_widget
from pyxalign.interactions.pma_runner import launch_pma_runner
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.pear.options import BaseLoadOptions, LYNXLoadOptions
from pyxalign.io.loaders.utils import convert_projection_dict_to_array
from pyxalign.io.save import can_fit_in_single_tiff_file


class Autorunner(ABC):
    def __init__(self, file_path: str):
        with open(file_path, "r") as f:
            self._options_dict = yaml.safe_load(f)
        self._setup_results_folders()

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _get_load_options(self):
        pass

    @abstractmethod
    def _load_data(self):
        pass

    def _setup_results_folders(self):
        self.results_folders = {}
        self.results_folders["parent"] = self._options_dict["results"]["results_folder"]
        self.results_folders["final"] = os.path.join(self.results_folders["parent"], "final")
        self.results_folders["projection_matching"] = os.path.join(
            self.results_folders["parent"], "projection_matching"
        )
        self.results_folders["temporary"] = os.path.join(
            self.results_folders["parent"], "temporary"
        )
        for folder in self.results_folders.values():
            if not os.path.exists(folder):
                os.mkdir(folder)


class AutorunnerLYNX(Autorunner):
    def run(self):
        # self._get_load_options()
        self._load_data()
        self._create_projections_object()
        self._get_cross_correlation_alignment()
        self._get_complex_projections_masks()
        self._unwrap_phase()
        self._get_phase_projections_masks()
        self._estimate_center_of_rotation()
        self._run_projection_matching_sequence()
        self._get_volume()
        self._save_volume()

    def _get_load_options(self):
        cfg = self._options_dict["loading"]

        base_load_options = BaseLoadOptions(
            parent_projections_folder=cfg["parent_projections_folder"],
            loader_type=cfg["loader_type"],
            file_pattern=cfg["file_pattern"],
            scan_start=cfg["scan_start"],
            scan_end=cfg["scan_end"],
            select_all_by_default=True,
        )
        self._load_options = LYNXLoadOptions(
            dat_file_path=cfg["tomo_scannumbers_path"],
            base=base_load_options,
            selected_experiment_name=cfg["experiment_name"],
        )

    def _load_data(self):
        if self._options_dict["loading"]["start_from_checkpoint"]["enabled"]:
            self._load_checkpoint_task(
                self._options_dict["loading"]["start_from_checkpoint"]["checkpoint"]
            )
        else:
            cfg = self._options_dict["loading"]
            self._get_load_options()
            if not cfg["interactive"]:
                self._standardized_data = load_data_from_pear_format(
                    n_processes=int(mp.cpu_count() * 0.8),
                    options=self._load_options,
                )
            else:
                gui = launch_data_loader(self._load_options)

    def _create_projections_object(self):
        self._current_checkpoint = Checkpoints.INITIALIZATION
        step_string = "initialization"
        cfg = self._options_dict[step_string]

        if self._skip_to_checkpoint():
            return

        # create padded projection array
        new_array_size = self._standardized_data.get_minimum_size_for_projection_array()
        new_array_size += cfg["pad"]
        projection_array = convert_projection_dict_to_array(
            self._standardized_data.projections, new_array_size, pad_with_mode=True
        )

        # define projection options
        projection_options = ProjectionOptions()
        # experiment parameters
        projection_options.experiment.laminography_angle = cfg["laminography_angle"]
        projection_options.experiment.sample_thickness = cfg["sample_thickness"]
        projection_options.experiment.pixel_size = self._standardized_data.pixel_size
        # input processing
        if cfg["rotation_angle"] != 0:
            projection_options.input_processing.rotation = RotationOptions(
                enabled=True, angle=cfg["rotation_angle"]
            )
        if cfg["shear_angle"] != 0:
            projection_options.input_processing.shear = ShearOptions(
                enabled=True, angle=cfg["shear_angle"]
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
        self.task.complex_projections.drop_projections(cfg["remove_scan_numbers"])

        self._standardized_data = None

        self._save_checkpoint_task(step_string)

    def _get_cross_correlation_alignment(self):
        self._current_checkpoint = Checkpoints.CROSS_CORRELATION_ALIGNMENT
        step_string = "cross_correlation_alignment"
        cfg = self._options_dict[step_string]

        if self._skip_to_checkpoint() or not cfg["enabled"]:
            return

        settings_path = cfg["default_settings_path"]
        if settings_path is not None and os.path.exists(settings_path):
            self.task.options.cross_correlation = load_options_from_yaml(
                settings_path, CrossCorrelationOptions()
            )

        if cfg["interactive"]:
            launch_cross_correlation_gui(
                self.task, projection_type="complex", wait_until_closed=True
            )
        else:
            self.task.get_cross_correlation_shift(plot_results=False)
        self.task.complex_projections.apply_staged_shift()
        self._save_checkpoint_task(step_string)

    def _get_complex_projections_masks(self):
        cfg = self._options_dict["phase_unwrapping"]["masks"]

        if self._skip_to_checkpoint() or not cfg["enabled"]:
            return

        if cfg["threshold"] is not None:
            self.task.complex_projections.options.mask_from_positions.threshold = cfg["threshold"]
        if cfg["interactive"]:
            launch_mask_builder(self.task.complex_projections, wait_until_closed=True)
        else:
            self.task.complex_projections.get_masks_from_probe_positions()

    def _unwrap_phase(self):
        self._current_checkpoint = Checkpoints.PHASE_UNWRAPPING
        step_string = "phase_unwrapping"
        cfg = self._options_dict[step_string]

        if self._skip_to_checkpoint():
            return

        if cfg["interactive"]:
            gui = launch_phase_unwrap_widget(self.task, wait_until_closed=True)
        else:
            self.task.get_unwrapped_phase()
        self.task.complex_projections = None
        self._save_checkpoint_task(step_string)

    def _get_phase_projections_masks(self):
        self._current_checkpoint = Checkpoints.PHASE_PROJECTIONS_MASKS
        step_string = "phase_projections_masks"
        cfg = self._options_dict["phase_projections_masks"]

        if self._skip_to_checkpoint():
            return

        if cfg["threshold"] is not None:
            self.task.phase_projections.options.mask_from_positions.threshold = cfg["threshold"]
        if cfg["interactive"]:
            launch_mask_builder(self.task.phase_projections, wait_until_closed=True)
        else:
            self.task.phase_projections.get_masks_from_probe_positions()
        self._save_checkpoint_task(step_string)

    def _estimate_center_of_rotation(self):
        self._current_checkpoint = Checkpoints.ESTIMATE_CENTER
        step_string = "estimate_center"
        cfg = self._options_dict["estimate_center"]

        if self._skip_to_checkpoint() or not cfg["enabled"]:
            return

        estimate_center_options = copy.deepcopy(self.task.phase_projections.options.estimate_center)

        for i, new_options_dict in enumerate(cfg["sequence"]):
            self.task.phase_projections.options.estimate_center = get_updated_options(
                estimate_center_options, new_options_dict
            )
            # run center estimation code
            center_estimate_results = self.task.phase_projections.estimate_center_of_rotation()
            self.task.phase_projections.center_of_rotation[:] = (
                center_estimate_results.optimal_center_of_rotation
            )
            plot_center_of_rotation_estimate_results(
                center_of_rotation_estimate_results=center_estimate_results,
                projections=self.task.phase_projections.data,
                plot_projection_sum=True,
                save_plot=True,
                save_path=os.path.join(
                    self.results_folders["temporary"], f"estimate_center_{i}.pdf"
                ),
            )
        self._save_checkpoint_task(step_string)

    def _run_projection_matching_sequence(self):
        cfg = self._options_dict["projection_matching_alignment"]
        self.task.options.projection_matching = load_options_from_yaml(
            cfg["default_settings_path"], ProjectionMatchingOptions()
        )
        # update defaults
        self.task.options.projection_matching = get_updated_options(
            self.task.options.projection_matching, cfg["update_defaults"]
        )

        # update the results path
        self.task.options.projection_matching.save.folder = self.results_folders[
            "projection_matching"
        ]

        if not cfg["interactive"]:
            pma_options_list = get_projection_matching_sequence_options(
                self.task.options.projection_matching, cfg["sequence"]
            )
            shift = None
            suffix = self.task.options.projection_matching.save.suffix
            for i, pma_options in enumerate(pma_options_list):
                self.task.options.projection_matching = copy.deepcopy(pma_options)
                # update suffix
                self.task.options.projection_matching.save.suffix = suffix + f"_{i}"
                self.task.get_projection_matching_shift(shift)
        else:
            gui = launch_pma_runner(
                self.task,
                self._options_dict["projection_matching_alignment"]["sequence"],
                wait_until_closed=True,
            )
        self.task.phase_projections.apply_staged_shift()

    def _get_volume(self):
        self.task.phase_projections.get_3D_reconstruction()
        self.task.phase_projections.volume.get_optimal_rotation_of_reconstruction()

    def _save_volume(self):
        self.task.phase_projections.volume.save_as_tiff(
            os.path.join(self.results_folders["final"], "aligned_volume.tiff"),
            crop_to_single_file=True,
        )
        if not can_fit_in_single_tiff_file(self.task.phase_projections.volume.data):
            self.task.phase_projections.volume.save_as_tiff(
                os.path.join(self.results_folders["final"], "aligned_volume_cropped.tiff"),
                crop_to_single_file=True,
            )

    def _save_aligned_task(self):
        self.task.save_task(os.path.join(self.results_folders["final"], "aligned_task.h5"))

    def _save_checkpoint_task(self, step_string: str):
        if self._options_dict["results"]["checkpoints"][step_string]:
            self.task.save_task(self._return_checkpoint_path(step_string))

    def _load_checkpoint_task(self, step_string: str):
        self.task = load_task(self._return_checkpoint_path(step_string))

    def _return_checkpoint_path(self, step_string: str):
        return os.path.join(self.results_folders["temporary"], f"task_after_{step_string}.h5")

    def _skip_to_checkpoint(self):
        if not self._options_dict["loading"]["start_from_checkpoint"]["enabled"]:
            return False
        else:
            current_checkpoint_val = get_checkpoint_order_value(self._current_checkpoint)
            loaded_checkpoint_val = get_checkpoint_order_value(
                self._options_dict["loading"]["start_from_checkpoint"]["checkpoint"]
            )
            return current_checkpoint_val <= loaded_checkpoint_val