"""
Interactive projection matching alignment (PMA) runner with multi-resolution capabilities.

This module provides a comprehensive GUI for running projection matching alignment
algorithms with multi-resolution scanning, real-time visualization, and results
collection. The interface integrates options editing, alignment sequencing management,
and plotting capabilities into a unified tabbed workflow.

Key Components:
- PMAMasterWidget: Main interface for projection matching alignment workflows
- AlignmentResults: Data structure for storing alignment results and parameters
- AlignmentResultsCollection: Widget for visualizing and comparing multiple alignment results
- Multi-resolution alignment sequence support with progress monitoring
- Integration with ProjectionMatchingViewer for real-time visualization
"""

import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Callable, Optional, Union

import cupy as cp
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyxalign.api.options_utils import get_all_attribute_names
import pyxalign.data_structures.task as t
import pyxalign.io.load as load
from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.api.options.transform import DownsampleOptions
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.sequencer import SequencerWidget
from pyxalign.interactions.custom import action_button_style_sheet
from pyxalign.io.utils import OptionsClass
from pyxalign.plotting.interactive.base import MultiThreadedWidget
from pyxalign.plotting.interactive.projection_matching import ProjectionMatchingViewer
from pyxalign.plotting.interactive.utils import OptionsDisplayWidget

basic_pma_settings = [
    "iterations",
    "high_pass_filter",
    "downsample",
    "downsample.scale",
]


class AlignmentResults:
    """
    Data structure for storing projection matching alignment results.

    This class encapsulates the results from a single projection matching
    alignment run, including the computed shifts, initial conditions, and
    the options used for the alignment.

    Parameters
    ----------
    shift : np.ndarray
        Final computed alignment shifts for each projection.
    initial_shift : np.ndarray
        Initial shift values used as starting point for alignment.
    angles : np.ndarray
        Projection angles corresponding to the alignment results.
    pma_options : ProjectionMatchingOptions
        Projection matching options used for this alignment run.
    projection_options : ProjectionOptions
        Projection configuration options used for this alignment run.
    """

    def __init__(
        self,
        shift: np.ndarray,
        initial_shift: np.ndarray,
        angles: np.ndarray,
        options: OptionsClass,
        projection_options: ProjectionOptions,
    ):
        self.shift = shift
        self.initial_shift = initial_shift
        self.angles = angles
        self.pma_options = options
        self.projection_options = projection_options


class AlignmentResultsCollection(QWidget):
    """
    Widget for visualizing and comparing multiple alignment results.

    This widget provides an interface for browsing through multiple alignment
    results, displaying shift plots and alignment options for comparison.
    Users can select different results from a table and view the corresponding
    shift data and configuration parameters.

    Parameters
    ----------
    alignment_results_list : list[AlignmentResults]
        List of alignment results to display and compare.
    parent : QWidget, optional
        Parent widget for this interface.
    """

    def __init__(
        self, alignment_results_list: list[AlignmentResults], parent: Optional[QWidget] = None
    ):
        super().__init__(parent=parent)
        self.alignment_results_list = alignment_results_list

        self.create_shift_plots()
        self.create_options_display()
        self.update_table()

        main_layout = QHBoxLayout(self)

        display_widget = QWidget()
        display_layout = QHBoxLayout()
        display_widget.setLayout(display_layout)
        display_layout.addWidget(self.canvas)

        left_layout = QVBoxLayout()
        table_title = QLabel("Select Alignment Results Index")
        table_title.setStyleSheet("QLabel {font-size: 18px;}")
        left_layout.addWidget(table_title)
        left_layout.addWidget(self.results_table)
        options_title = QLabel("Alignment Options")
        table_title.setStyleSheet("QLabel {font-size: 18px;}")
        left_layout.addWidget(options_title)
        left_layout.addWidget(self.options_display)
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(display_widget, stretch=3)

    def create_shift_plots(self):
        """
        Creates a widget containing:
        1) A QTableWidget on the left listing each AlignmentResults entry by index.
        2) A Matplotlib plot on the right with two stacked axes:
            - The top axis (labeled "horizontal") plots the first column
            of shift and initial_shift.
            - The bottom axis (labeled "vertical") plots the second column
            of shift and initial_shift.
        Clicking on a row in the table updates the plots to show that entry's data.

        Returns:
            QWidget: A QWidget containing the described UI components.
        """
        # Create the table
        self.results_table = QTableWidget(0, 1)
        self.results_table.setHorizontalHeaderLabels(["Index"])
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        # main_layout.addWidget(self.results_table)

        # Create the Matplotlib Figure/Canvas
        self.figure = Figure(figsize=(5, 4), layout="compressed")
        self.canvas = FigureCanvas(self.figure)
        self.ax_horizontal = self.figure.add_subplot(211)
        self.ax_vertical = self.figure.add_subplot(212)

        # Give each subplot a title and axes labels
        axis_directions = ["horizontal", "vertical"]
        for i, ax in enumerate([self.ax_horizontal, self.ax_vertical]):
            ax.set_title(f"{axis_directions[i]} shifts")
            ax.set_ylabel("shift (px)")
            ax.set_xlabel("angle (deg)")

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # main_layout.addWidget(self.canvas)

        # Connect the table's click signal to our plotting function
        self.results_table.currentCellChanged.connect(self.on_table_cell_changed)

    def update_table(self):
        num_results = len(self.alignment_results_list)
        table_length = self.results_table.rowCount()

        # Fill the table with row indices
        for i in range(num_results):
            if i >= table_length:
                self.results_table.insertRow(i)
                item = QTableWidgetItem(str(i))
                self.results_table.setItem(i, 0, item)

    def on_table_cell_changed(self, row: int, column: int):
        self.change_shift_plot_index(row)
        self.change_options_display_index(row)

    def change_shift_plot_index(self, row: int):
        alignment_result = self.alignment_results_list[row]
        sort_idx = np.argsort(alignment_result.angles)
        sorted_angles = alignment_result.angles[sort_idx]

        # Give each subplot a title and axes labels
        axis_directions = ["horizontal", "vertical"]
        for i, ax in enumerate([self.ax_horizontal, self.ax_vertical]):
            ax.clear()
            ax.set_title(f"{axis_directions[i]} shifts")
            ax.set_ylabel("shift (px)")
            ax.set_xlabel("angle (deg)")
            ax.plot(sorted_angles, alignment_result.shift[sort_idx, i], label="final")
            ax.plot(
                sorted_angles,
                alignment_result.initial_shift[sort_idx, i],
                label="initial",
            )
            ax.autoscale(enable=True, axis="x", tight=True)
            ax.legend()
            ax.grid(linestyle=":")

        self.canvas.draw()

    def create_options_display(self):
        # self._options_display_layout = QVBoxLayout()
        self.options_display = OptionsDisplayWidget()

    def change_options_display_index(self, row: int):
        self.options_display.update_options(self.alignment_results_list[row].pma_options)
        self.options_display.update_display()

    # def go_next(self):
    #     """
    #     Move to the next page in the stacked widget.
    #     """
    #     current_index = self.stacked_widget.currentIndex()
    #     next_index = (current_index + 1) % self.stacked_widget.count()
    #     self.stacked_widget.setCurrentIndex(next_index)

    # def go_previous(self):
    #     """
    #     Move to the previous page in the stacked widget.
    #     """
    #     current_index = self.stacked_widget.currentIndex()
    #     prev_index = (current_index - 1) % self.stacked_widget.count()
    #     self.stacked_widget.setCurrentIndex(prev_index)


class PMAMasterWidget(MultiThreadedWidget):
    # edit PMA options
    # Features:
    # - set up multi-resolution alignment scans
    # - launch viewer when scan is started
    # - store: alignment shift results, options used
    # First: set up layout where you can set up a multi-resolution scan.
    # keep it simple by making all resolutions run with the same options.
    def __init__(
        self,
        task: Optional["t.LaminographyAlignmentTask"] = None,
        multi_thread_func: Optional[Callable] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.alignment_results_list: list[AlignmentResults] = []
        self.pma_viewer = None
        self.results_collection_widget = None

        if task is not None:
            self.initialize_page(task)

    def initialize_page(self, task: "t.LaminographyAlignmentTask"):
        self.task = task
        tabs = QTabWidget()
        tabs.setObjectName("main_tabs")
        tabs.setStyleSheet("#main_tabs > QTabBar{font-size: 20px;}")
        layout = QHBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

        self.generate_start_and_stop_buttons()
        self.generate_options_selection_widget()
        self.generate_sequencer()
        self.make_first_tab_layout(tabs)
        self.make_second_tab_layout(tabs)
        self.make_third_tab_layout(tabs)

    def generate_start_and_stop_buttons(self):
        self.button_widget = QWidget(self)
        button_layout = QHBoxLayout()
        self.button_widget.setLayout(button_layout)

        self.start_sequence_button = QPushButton("Start Alignment")
        self.stop_alignment_button = QPushButton("Stop Current Alignment")
        self.stop_sequence_button = QPushButton("Stop Alignment Sequence")

        self.start_sequence_button.pressed.connect(self.start_alignment_sequence)

        self.start_sequence_button.setStyleSheet("QPushButton { background-color: green;}")
        self.stop_alignment_button.setStyleSheet("QPushButton { background-color: red;}")
        self.stop_sequence_button.setStyleSheet("QPushButton { background-color: red;}")

        button_layout.addWidget(self.start_sequence_button)
        button_layout.addWidget(self.stop_alignment_button)
        button_layout.addWidget(self.stop_sequence_button)
        button_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Preferred))

        # "QPushButton { font-weight: bold; font-size: 11pt; color: white; padding: 2px 6px; }"
        self.button_widget.setStyleSheet(action_button_style_sheet)

    def start_alignment_sequence(self):
        options_sequence = self.sequencer.generate_options_sequence(
            self.task.options.projection_matching
        )
        shift = None
        for options in options_sequence:
            shift = self.task.get_projection_matching_shift(initial_shift=shift, options=options)
            self.alignment_results_list += [
                AlignmentResults(
                    shift,
                    self.task.pma_object.initial_shift,
                    self.task.pma_object.aligned_projections.angles,
                    options=options,
                    projection_options=self.task.phase_projections.options,
                )
            ]
            self.update_pma_viewer_tab()
            self.update_results_collection_tab()

    def generate_options_selection_widget(self):
        # create options editor
        self.options_editor = BasicOptionsEditor(
            self.task.options.projection_matching,
            skip_fields=["plot"],
            enable_advanced_tab=True,
            basic_options_list=basic_pma_settings,
            open_panels_list=["downsample"],
            label="Projection Matching Alignment Options"
        )

    def generate_sequencer(self):
        self.sequencer = SequencerWidget(
            self.task.options.projection_matching,
            basic_options_list=basic_pma_settings,
            parent=self,
        )

    def update_pma_viewer_tab(self):
        if self.pma_viewer is not None:
            self.pma_viewer.deleteLater()
            self.pma_viewer.setParent(None)
            self.pma_viewer = None
        self.pma_viewer = ProjectionMatchingViewer(self.task.pma_object)
        self.pma_viewer.initialize_plots(add_stop_button=False)
        self.pma_viewer.update_plots()
        self._pma_viewer_layout.addWidget(self.pma_viewer)

    def update_results_collection_tab(self):
        self.results_collection_widget.update_table()

    def make_first_tab_layout(self, tabs: QTabWidget):
        alignment_setup_widget = QWidget(self)

        layout = QGridLayout()
        layout.addWidget(self.options_editor, 0, 0)
        layout.addWidget(self.sequencer, 0, 1)
        layout.addWidget(self.button_widget, 1, 0, 1, 2)

        alignment_setup_widget.setLayout(layout)
        tabs.addTab(alignment_setup_widget, "Configure && Start")

    def make_second_tab_layout(self, tabs: QTabWidget):
        empty_widget = QWidget()
        self._pma_viewer_layout = QVBoxLayout()
        empty_widget.setLayout(self._pma_viewer_layout)
        tabs.addTab(empty_widget, "Detailed Results")

    def make_third_tab_layout(self, tabs: QTabWidget):
        self.results_collection_widget = AlignmentResultsCollection(self.alignment_results_list)
        empty_widget = QWidget()
        self._results_collection_layout = QVBoxLayout()
        empty_widget.setLayout(self._results_collection_layout)
        tabs.addTab(self.results_collection_widget, "Collected Results")


if __name__ == "__main__":
    import os

    base_folder = os.environ["PYXALIGN_CI_TEST_DATA_DIR"]
    rel_path = "dummy_inputs/cSAXS_e18044_LamNI_201907_16x_downsampled_pre_pma_task.h5"
    task_path = os.path.join(base_folder, rel_path)
    dummy_task = t.load_task(task_path)
    dummy_task.options.projection_matching.iterations = 3
    dummy_task.options.projection_matching.downsample = ProjectionMatchingOptions().downsample
    dummy_task.options.projection_matching.downsample.enabled = True
    dummy_task.options.projection_matching.interactive_viewer.update.enabled = True

    # dummy_task = None

    app = QApplication(sys.argv)
    master_widget = PMAMasterWidget(dummy_task)

    # Use the left half of the screen
    screen_geometry = app.desktop().availableGeometry(master_widget)
    master_widget.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() * 0.75),
        int(screen_geometry.height() * 0.9),
    )

    master_widget.show()
    sys.exit(app.exec_())
