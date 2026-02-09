import copy
import os
from abc import ABC, abstractmethod
import yaml
import h5py
import multiprocessing as mp

from pyxalign.api.options.alignment import CrossCorrelationOptions, ProjectionMatchingOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.transform import RotationOptions, ShearOptions
from pyxalign.autorunner.io import (
    get_projection_matching_sequence_options,
    get_updated_options,
    load_options_from_yaml,
)
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.interactions.cross_correlation import launch_cross_correlation_gui
from pyxalign.interactions.io.loader import launch_data_loader
from pyxalign.interactions.mask import launch_mask_builder
from pyxalign.interactions.phase_unwrap import launch_phase_unwrap_widget
from pyxalign.interactions.pma_runner import PMAMasterWidget, launch_pma_runner
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.maps import get_loader_options_by_enum
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.pear.options import BaseLoadOptions, LYNXLoadOptions
from pyxalign.io.loaders.utils import convert_projection_dict_to_array
from pyxalign.io.save import can_fit_in_single_tiff_file
from pyxalign.transformations.functions import shear_positions


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
        self.results_folders["parent"] = self._options_dict["Results"]["ResultsFolder"]
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
        cfg = self._options_dict["Loading"]

        base_load_options = BaseLoadOptions(
            parent_projections_folder=cfg["InputReconstructionsFolder"],
            loader_type=cfg["LoaderType"],
            file_pattern=cfg["FilePattern"],
            scan_start=cfg["ScanStart"],
            scan_end=cfg["ScanEnd"],
            select_all_by_default=True,
        )
        self._load_options = LYNXLoadOptions(
            dat_file_path=cfg["TomographyScannumbersPath"],
            base=base_load_options,
            selected_experiment_name=cfg["ExperimentName"],
        )

    def _load_data(self):
        cfg = self._options_dict["Loading"]
        self._get_load_options()
        if not cfg["Interactive"]:
            self._standardized_data = load_data_from_pear_format(
                n_processes=int(mp.cpu_count() * 0.8),
                options=self._load_options,
            )
        else:
            gui = launch_data_loader(self._load_options)

    def _create_projections_object(self):
        step_string = "initialization"
        cfg = self._options_dict[step_string]

        # creat padded projection array
        new_array_size = self._standardized_data.get_minimum_size_for_projection_array()
        new_array_size += cfg["Pad"]
        projection_array = convert_projection_dict_to_array(
            self._standardized_data.projections, new_array_size, pad_with_mode=True
        )

        # define projection options
        projection_options = ProjectionOptions()
        # experiment parameters
        projection_options.experiment.laminography_angle = cfg["LaminographyAngle"]
        projection_options.experiment.sample_thickness = cfg["SampleThickness"]
        projection_options.experiment.pixel_size = self._standardized_data.pixel_size
        # input processing
        if cfg["RotationAngle"] != 0:
            projection_options.input_processing.rotation = RotationOptions(
                enabled=True, angle=cfg["RotationAngle"]
            )
        if cfg["ShearAngle"] != 0:
            projection_options.input_processing.shear = ShearOptions(
                enabled=True, angle=cfg["ShearAngle"]
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
        step_string = "cross_correlation_alignment"
        cfg = self._options_dict[step_string]
        if not cfg["enabled"]:
            return

        settings_path = cfg["default_settings_path"]
        if settings_path is not None and os.path.exists(settings_path):
            self.task.options.cross_correlation = load_options_from_yaml(
                settings_path, CrossCorrelationOptions()
            )

        if cfg["Interactive"]:
            launch_cross_correlation_gui(
                self.task, projection_type="complex", wait_until_closed=True
            )
        else:
            self.task.get_cross_correlation_shift(plot_results=False)
        self.task.complex_projections.apply_staged_shift()
        self._save_checkpoint_task(step_string)

    def _get_complex_projections_masks(self):
        cfg = self._options_dict["phase_unwrapping"]["Masks"]

        if cfg["Threshold"] is not None:
            self.task.complex_projections.options.mask_from_positions.threshold = cfg["Threshold"]
        if cfg["Interactive"]:
            launch_mask_builder(self.task.complex_projections, wait_until_closed=True)
        else:
            self.task.complex_projections.get_masks_from_probe_positions()

    def _unwrap_phase(self):
        step_string = "phase_unwrapping"
        cfg = self._options_dict[step_string]

        if cfg["Interactive"]:
            gui = launch_phase_unwrap_widget(self.task, wait_until_closed=True)
        else:
            self.task.get_unwrapped_phase()
        self.task.complex_projections = None
        self._save_checkpoint_task(step_string)

    def _get_phase_projections_masks(self):
        step_string = "phase_projections_masks"
        cfg = self._options_dict["phase_projections_masks"]

        if cfg["Threshold"] is not None:
            self.task.phase_projections.options.mask_from_positions.threshold = cfg["Threshold"]
        if cfg["Interactive"]:
            launch_mask_builder(self.task.phase_projections, wait_until_closed=True)
        else:
            self.task.phase_projections.get_masks_from_probe_positions()
        self._save_checkpoint_task(step_string)

    def _estimate_center_of_rotation(self):
        step_string = "estimate_center"
        cfg = self._options_dict["estimate_center"]
        if not cfg["Enabled"]:
            return
        
        estimate_center_options = copy.deepcopy(self.task.phase_projections.options.estimate_center)

        for new_options_dict in cfg["sequence"]:
            self.task.phase_projections.options.estimate_center = get_updated_options(
                estimate_center_options, new_options_dict
            )
            # run center estimation code
            center_estimate_results = self.task.phase_projections.estimate_center_of_rotation()
            self.task.phase_projections.center_of_rotation[:] = (
                center_estimate_results.optimal_center_of_rotation
            )

        # estimate_center_options = self.task.phase_projections.options.estimate_center
        # if cfg["Scale"] is not None:
        #     estimate_center_options.projection_matching.downsample.scale = cfg["Scale"]
        # params = zip(
        #     cfg["HorizontalRanges"],
        #     cfg["HorizontalSpacings"],
        #     cfg["VerticalRanges"],
        #     cfg["VerticalSpacings"],
        # )
        # estimate_center_options.horizontal_coordinate.enabled = True
        # estimate_center_options.vertical_coordinate.enabled = True
        # for h_range, h_spc, v_range, v_spc in params:
        #     estimate_center_options.horizontal_coordinate.range = h_range
        #     estimate_center_options.vertical_coordinate.range = v_range
        #     estimate_center_options.horizontal_coordinate.spacing = h_spc
        #     estimate_center_options.vertical_coordinate.spacing = v_spc
        #     # run center estimation code
        #     center_estimate_results = self.task.phase_projections.estimate_center_of_rotation()
        #     self.task.phase_projections.center_of_rotation[:] = (
        #         center_estimate_results.optimal_center_of_rotation
        #     )
        self._save_checkpoint_task(step_string)

    def _run_projection_matching_sequence(self):
        cfg = self._options_dict["projection_matching_alignment"]
        self.task.options.projection_matching = load_options_from_yaml(
            cfg["default_settings_path"], ProjectionMatchingOptions()
        )
        # update defaults
        self.task.options.projection_matching = get_updated_options(
            self.task.options.projection_matching, cfg["UpdateDefaults"]
        )

        # update the results path
        self.task.options.projection_matching.save.folder = self.results_folders[
            "projection_matching"
        ]

        if not cfg["Interactive"]:
            pma_options_list = get_projection_matching_sequence_options(
                self.task.options.projection_matching, cfg["Sequence"]
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
                self._options_dict["projection_matching_alignment"]["Sequence"],
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
        if self._options_dict["Results"]["checkpoints"][step_string]:
            self.task.save_task(
                os.path.join(self.results_folders["temporary"], f"task_after_{step_string}.h5")
            )
