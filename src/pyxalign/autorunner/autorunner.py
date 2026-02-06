import os
from abc import ABC, abstractmethod
import yaml
import multiprocessing as mp

from pyxalign.api.options.alignment import CrossCorrelationOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.transform import RotationOptions, ShearOptions
from pyxalign.autorunner.io import load_options_from_yaml
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.interactions.cross_correlation import launch_cross_correlation_gui
from pyxalign.interactions.mask import launch_mask_builder
from pyxalign.interactions.phase_unwrap import launch_phase_unwrap_widget
from pyxalign.interactions.pma_runner import PMAMasterWidget
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.maps import get_loader_options_by_enum
from pyxalign.io.loaders.pear.api import load_data_from_pear_format
from pyxalign.io.loaders.pear.options import BaseLoadOptions, LYNXLoadOptions
from pyxalign.io.loaders.utils import convert_projection_dict_to_array
from pyxalign.transformations.functions import shear_positions


class Autorunner(ABC):
    def __init__(self, file_path: str):
        with open(file_path, "r") as f:
            self._options_dict = yaml.safe_load(f)

    def run_reconstruction(self):
        self._get_load_options()
        self._load_data()
        self._create_projections_object()
        self._get_cross_correlation_alignment()
        self._get_complex_projections_masks()
        self._unwrap_phase

    @abstractmethod
    def _get_load_options(self):
        pass

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def _create_projections_object(self):
        pass

    @abstractmethod
    def _get_cross_correlation_alignment(self):
        pass

    @abstractmethod
    def _get_complex_projections_masks(self):
        pass

    @abstractmethod
    def _unwrap_phase(self):
        pass

class AutorunnerLYNX(Autorunner):
    def _get_load_options(self):
        cfg = self._options_dict["Loading"]

        base_load_options = BaseLoadOptions(
            parent_projections_folder=cfg["InputReconstructionsFolder"],
            loader_type=cfg["LoaderType"]["PEAR_V1"],
            file_pattern=cfg["FilePattern"],
            select_all_by_default=True,
        )
        self._load_options = LYNXLoadOptions(
            dat_file_path=os.path.join(
                cfg["InputReconstructionsFolder"], "dat-files/tomography_scannumbers.txt"
            ),
            base=base_load_options,
            selected_experiment_name=cfg["ExperimentName"],
        )

    def _load_data(self):
        cfg = self._options_dict["Loading"]

        self._standardized_data = load_data_from_pear_format(
            n_processes=int(mp.cpu_count() * 0.8),
            options=self._load_options,
        )

    def _create_projections_object(self):
        cfg = self._options_dict["Initialization"]

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

        self._standardized_data = None

    def _get_cross_correlation_alignment(self):
        cfg = self._options_dict["CrossCorrelationAlignment"]
        if not cfg["Enabled"]:
            return

        settings_path = cfg["DefaultSettingsPath"]
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

    def _get_complex_projections_masks(self):
        cfg = self._options_dict["PhaseUnwrapping"]["Masks"]

        if cfg["Threshold"] is not None:
            self.task.complex_projections.options.mask_from_positions.threshold = cfg["Threshold"]
        if cfg["Interactive"]:
            launch_mask_builder(self.task.complex_projections, wait_until_closed=True)
        else:
            self.task.complex_projections.get_masks_from_probe_positions()

    def _get_unwrapped_phase(self):
        cfg = self._options_dict["PhaseUnwrapping"]

        if cfg["Interactive"]:
            gui = launch_phase_unwrap_widget(self.task, wait_until_closed=True)
        else:
            self.task.get_unwrapped_phase()
        self.task.complex_projections = None

    def _get_phase_projections_masks(self):
        cfg = self._options_dict["PhaseProjectionMasks"]

        if cfg["Threshold"] is not None:
            self.task.phase_projections.options.mask_from_positions.threshold = cfg["Threshold"]
        if cfg["Interactive"]:
            launch_mask_builder(self.task.phase_projections, wait_until_closed=True)
        else:
            self.task.phase_projections.get_masks_from_probe_positions()

    def _estimate_center_of_rotation(self):
        cfg = self._options_dict["EstimateCenter"]
        if not cfg["Enabled"]:
            return

        estimate_center_options = self.task.phase_projections.options.estimate_center
        if cfg["Scale"] is not None:
            estimate_center_options.projection_matching.downsample.scale = cfg["Scale"]
        params = zip(
            cfg["HorizontalRanges"],
            cfg["HorizontalSpacings"],
            cfg["VerticalRanges"],
            cfg["VerticalSpacings"],
        )
        estimate_center_options.horizontal_coordinate.enabled = True
        estimate_center_options.vertical_coordinate.enabled = True
        for h_range, h_spc, v_range, v_spc in params:
            estimate_center_options.horizontal_coordinate.range = h_range
            estimate_center_options.vertical_coordinate.range = v_range
            estimate_center_options.horizontal_coordinate.spacing = h_spc
            estimate_center_options.horizontal_coordinate.spacing = v_spc
            # run center estimation code
            center_estimate_results = self.task.phase_projections.estimate_center_of_rotation()
            self.task.phase_projections.center_of_rotation[:] = (
                center_estimate_results.optimal_center_of_rotation
            )

    def _run_projection_matching_sequence(self):
        pass
