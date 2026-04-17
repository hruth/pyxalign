"""
Interactive projection matching alignment (PMA) runner with multi-resolution capabilities.

This module provides a comprehensive GUI for running projection matching alignment
algorithms with multi-resolution scanning, real-time visualization, and results
collection. The interface integrates options editing, alignment sequencing management,
and plotting capabilities into a unified tabbed workflow.

Key Components:
- PMAMasterWidget: Main interface for projection matching alignment workflows
- PMAResults: Data structure for storing PMA alignment results and parameters (with run_type)
- PMAResultsCollection: Widget for visualizing and comparing multiple PMA alignment results
- Multi-resolution alignment sequence support with progress monitoring
- Integration with ProjectionMatchingViewer for real-time visualization
"""

import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Callable, Optional, Union
import time

import cupy as cp
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QStackedWidget,
    QTabBar,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyxalign.api.options_utils import get_all_attribute_names
import pyxalign.data_structures.task as t
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
import pyxalign.io.load as load
from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.api.options.transform import DownsampleOptions
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.sequencer_v2 import SequencerWidgetV2
from pyxalign.interactions.custom import action_button_style_sheet
from pyxalign.api.types import OptionsClass
from pyxalign.interactions.viewers.base import MultiThreadedWidget
from pyxalign.interactions.viewers.projection_matching import ProjectionMatchingViewer
from pyxalign.interactions.alignment_results import AlignmentResults, AlignmentResultsCollection
from pyxalign.interactions.roi_selector import GetBoxBoundsFromROISelector

basic_pma_settings = [
    "iterations",
    "high_pass_filter",
    "downsample",
    "downsample.scale",
    "save",
    "save.enabled",
    "save.folder",
    "save.suffix",
    "save.save_pma_volume",
    "save.save_pma_projections",
    "regularization",
    "regularization.enabled",
    "regularization.iterations",
    "regularization.local_TV_lambda",
    "regularization.use_gpu",
    # "horizontal_offset",
    # "vertical_offset",
    # "sample_thickness",
    "keep_on_gpu",
    "step_relax",
    "min_step_size",
    "reconstruct",
    "reconstruct.astra",
    "reconstruct.astra.back_project_gpu_indices",
    "reconstruct.astra.forward_project_gpu_indices",
]


class PMAResults(AlignmentResults):
    """
    Data structure for storing projection matching alignment results.

    This class extends AlignmentResults to add PMA-specific attributes,
    including initial_shift_source and run_type to track the alignment
    configuration and execution mode.

    Parameters
    ----------
    shift : np.ndarray
        Final computed alignment shifts for each projection.
    initial_shift : np.ndarray
        Initial shift values used as starting point for alignment.
    angles : np.ndarray
        Projection angles corresponding to the alignment results.
    scan_numbers : np.ndarray
        Scan numbers corresponding to each projection.
    initial_shift_source : str
        Description of the source of the initial shift (e.g., "None", "Previous", "Result 0").
    options : OptionsClass
        Projection matching options used for this alignment run.
    projection_options : ProjectionOptions
        Projection configuration options used for this alignment run.
    run_type : str, optional
        Type of alignment run (e.g., "defaults", "sequence"). Default is "unknown".
    changed_settings : dict, optional
        Dictionary of settings that were changed via the sequencer for this alignment instance.
    total_applied_shift : np.ndarray, optional
        Total applied shift from phase_projections at the time of alignment.
    """

    def __init__(
        self,
        shift: np.ndarray,
        initial_shift: np.ndarray,
        angles: np.ndarray,
        options: OptionsClass,
        projection_options: ProjectionOptions,
        scan_numbers: Optional[np.ndarray] = None,
        initial_shift_source: str = "None",
        run_type: Optional[str] = None,
        changed_settings: Optional[dict] = None,
        total_applied_shift: Optional[np.ndarray] = None,
    ):
        super().__init__(
            shift=shift,
            initial_shift=initial_shift,
            angles=angles,
            options=options,
            projection_options=projection_options,
            scan_numbers=scan_numbers,
        )
        self.initial_shift_source = initial_shift_source
        self.run_type = run_type if run_type is not None else "unknown"
        self.changed_settings = changed_settings if changed_settings is not None else {}
        self.total_applied_shift = total_applied_shift


class PMAResultsCollection(AlignmentResultsCollection):
    """
    Widget for visualizing and comparing multiple PMA alignment results.

    This widget extends AlignmentResultsCollection to add a "Run Type" column
    to the results table, showing whether each alignment was run with defaults
    or as part of a sequence, and displays changed sequencer settings.

    Parameters
    ----------
    alignment_results_list : list[PMAResults]
        List of PMA alignment results to display and compare.
    display_initial_shift : bool, optional
        Whether to display initial shift in plots. Default is True.
    task : t.LaminographyAlignmentTask, optional
        Task object containing phase_projections for accessing applied shifts.
    parent : QWidget, optional
        Parent widget for this interface.
    """

    def __init__(
        self,
        alignment_results_list: list[AlignmentResults],
        display_initial_shift: bool = True,
        task: Optional["t.LaminographyAlignmentTask"] = None,
        projection_viewer: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ):
        # Store parameters for manual layout construction
        self.alignment_results_list = alignment_results_list
        self.display_initial_shift = display_initial_shift
        self.task = task
        self.projection_viewer = projection_viewer
        self.show_with_applied_shifts = False  # Default to current view
        self.current_selected_row = None

        # Call QWidget.__init__ directly, not the parent AlignmentResultsCollection
        QWidget.__init__(self, parent)

        # Create changed settings display widget
        self.changed_settings_display = QLabel()
        self.changed_settings_display.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; }"
        )
        self.changed_settings_display.setWordWrap(True)
        self.changed_settings_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # Create view mode toggle widget
        self.create_view_toggle()

        # Build the layout manually to control widget order
        self.create_shift_plots()
        self.create_options_display()
        self.create_stage_shift_button()
        self.update_table()

        main_layout = QHBoxLayout(self)

        # Right side: plots and shift display mode toggle
        display_widget = QWidget()
        display_layout = QVBoxLayout()
        display_widget.setLayout(display_layout)
        display_layout.addWidget(self.canvas, stretch=1)

        # Add view mode toggle section below plots
        view_toggle_title = QLabel("Shift Display Mode")
        view_toggle_title.setStyleSheet("QLabel {font-size: 18px;}")
        display_layout.addWidget(view_toggle_title)
        display_layout.addWidget(self.view_toggle_widget)

        # Left side: table and settings
        left_layout = QVBoxLayout()

        # Add table section
        table_title = QLabel("Select Alignment Results Index")
        table_title.setStyleSheet("QLabel {font-size: 18px;}")
        left_layout.addWidget(table_title)
        left_layout.addWidget(self.results_table)

        # Add stage shift button
        left_layout.addWidget(self.stage_shift_button)

        # Add changed settings section in a scroll area
        changed_settings_title = QLabel("Sequencer Changed Settings")
        changed_settings_title.setStyleSheet("QLabel {font-size: 18px;}")
        left_layout.addWidget(changed_settings_title)

        # Wrap changed settings display in a scroll area
        settings_scroll = QScrollArea()
        settings_scroll.setWidget(self.changed_settings_display)
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumHeight(100)
        settings_scroll.setMaximumHeight(200)
        left_layout.addWidget(settings_scroll)

        # Add alignment options section
        options_title = QLabel("Alignment Options")
        options_title.setStyleSheet("QLabel {font-size: 18px;}")
        left_layout.addWidget(options_title)
        left_layout.addWidget(self.options_display)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(display_widget, stretch=3)

    def create_view_toggle(self):
        """Create radio button group to toggle between shift display modes."""
        self.view_toggle_widget = QGroupBox()
        view_toggle_layout = QVBoxLayout()
        self.view_toggle_widget.setLayout(view_toggle_layout)

        # Create radio button group
        self.view_button_group = QButtonGroup(self)

        # Default view: show shifts as computed by PMA
        self.default_view_button = QRadioButton("Relative to initial (default)")
        self.default_view_button.setChecked(True)
        self.default_view_button.setStyleSheet("font-size: 12pt;")
        self.default_view_button.toggled.connect(self.on_view_mode_changed)

        # Applied shifts view: add total applied shifts to both initial and final
        self.applied_shifts_view_button = QRadioButton("Include applied shifts from projections")
        self.applied_shifts_view_button.setStyleSheet("font-size: 12pt;")
        self.applied_shifts_view_button.toggled.connect(self.on_view_mode_changed)

        # Add buttons to group and layout
        self.view_button_group.addButton(self.default_view_button)
        self.view_button_group.addButton(self.applied_shifts_view_button)
        view_toggle_layout.addWidget(self.default_view_button)
        view_toggle_layout.addWidget(self.applied_shifts_view_button)

        # Disable applied shifts view if task is not available
        if self.task is None:
            self.applied_shifts_view_button.setEnabled(False)
            self.applied_shifts_view_button.setToolTip("Task not available - cannot access applied shifts")

    def on_view_mode_changed(self):
        """Handle view mode toggle and update the plot."""
        self.show_with_applied_shifts = self.applied_shifts_view_button.isChecked()
        # Refresh the current plot
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            self.change_shift_plot_index(current_row)

    def _get_table_column_count(self) -> int:
        """Return 3 columns: Index, Initial Shift, Run Type."""
        return 3

    def _get_table_headers(self) -> list[str]:
        """Return headers including Run Type."""
        return ["Index", "Initial Shift", "Run Type"]

    def update_table(self):
        """Update table with PMA-specific columns including run type."""
        num_results = len(self.alignment_results_list)
        table_length = self.results_table.rowCount()

        # Fill the table with row indices, initial shift sources, and run types
        for i in range(num_results):
            if i >= table_length:
                self.results_table.insertRow(i)
                # Column 0: Index
                index_item = QTableWidgetItem(str(i))
                self.results_table.setItem(i, 0, index_item)
                # Column 1: Initial Shift Source
                shift_source_item = QTableWidgetItem(self.alignment_results_list[i].initial_shift_source)
                self.results_table.setItem(i, 1, shift_source_item)
                # Column 2: Run Type
                run_type_item = QTableWidgetItem(self.alignment_results_list[i].run_type)
                self.results_table.setItem(i, 2, run_type_item)

    def on_table_cell_changed(self, row: int, column: int):
        """Override to also update changed settings display."""
        self.current_selected_row = row
        if len(self.alignment_results_list) == 0:
            return
        super().on_table_cell_changed(row, column)
        self.update_changed_settings_display(row)

    def update_changed_settings_display(self, row: int):
        """Update the changed settings display for the selected alignment result."""
        alignment_result = self.alignment_results_list[row]
        changed_settings = getattr(alignment_result, 'changed_settings', {})

        if changed_settings:
            # Format the changed settings as a readable string
            settings_text = "<b>Changed Settings:</b><br>"
            for key, value in changed_settings.items():
                settings_text += f"• <b>{key}</b>: {value}<br>"
            self.changed_settings_display.setText(settings_text)
        else:
            self.changed_settings_display.setText("<i>No settings were changed via sequencer for this alignment.</i>")

    def get_total_applied_shift(self, alignment_result: AlignmentResults) -> np.ndarray:
        """
        Get the total applied shift that was stored when this alignment result was created.

        Returns
        -------
        np.ndarray
            Total applied shift array matching the alignment result's scan numbers,
            or zeros if not available.
        """
        # Use the stored total_applied_shift if available
        if hasattr(alignment_result, 'total_applied_shift') and alignment_result.total_applied_shift is not None:
            return alignment_result.total_applied_shift
        else:
            # Return zeros if no stored shift
            return np.zeros_like(alignment_result.shift)

    def change_shift_plot_index(self, row: int):
        """Override to add support for showing shifts with applied offsets."""
        alignment_result = self.alignment_results_list[row]
        sort_idx = np.argsort(alignment_result.angles)
        sorted_angles = alignment_result.angles[sort_idx]

        # Get shifts to plot
        final_shift = alignment_result.shift
        initial_shift = alignment_result.initial_shift

        # If showing with applied shifts, add the total applied shift offset
        if self.show_with_applied_shifts:
            total_applied = self.get_total_applied_shift(alignment_result)
            final_shift = final_shift + total_applied
            initial_shift = initial_shift + total_applied

        # Plot the shifts
        axis_directions = ["horizontal", "vertical"]
        for i, ax in enumerate([self.ax_horizontal, self.ax_vertical]):
            ax.clear()
            ax.set_title(f"{axis_directions[i]} shifts")
            ax.set_ylabel("shift (px)")
            ax.set_xlabel("angle (deg)")
            ax.plot(sorted_angles, final_shift[sort_idx, i], label="final")
            if self.display_initial_shift:
                ax.plot(
                    sorted_angles,
                    initial_shift[sort_idx, i],
                    label="initial",
                )
            ax.autoscale(enable=True, axis="x", tight=True)
            ax.legend()
            ax.grid(linestyle=":")

        self.canvas.draw()

    def create_stage_shift_button(self):
        """Create a button to stage the selected shift."""
        self.stage_shift_button = QPushButton("Stage Selected Shift")
        self.stage_shift_button.setStyleSheet(action_button_style_sheet)
        self.stage_shift_button.clicked.connect(self.on_stage_shift_clicked)

    def on_stage_shift_clicked(self):
        """Handle the stage shift button click."""
        if self.current_selected_row is not None:
            self.stage_shift(self.current_selected_row)

    def stage_shift(self, row: int):
        """
        Stage the shift from the selected alignment result.

        Parameters
        ----------
        row : int
            The index of the alignment result to stage.
        """
        from pyxalign.api import enums

        if self.task is None or self.task.phase_projections is None:
            print("Cannot stage shift: task or phase_projections not available")
            return

        alignment_result = self.alignment_results_list[row]
        shift = alignment_result.shift

        # Stage the shift using the shift_manager
        self.task.phase_projections.shift_manager.stage_shift(
            shift=shift,
            function_type=enums.ShiftType.CIRC,
            alignment_options=self.task.options.projection_matching,
            eliminate_wrapping=True,
        )
        print(f"Shift from alignment result {row} staged successfully")

        # Refresh the Applied Shifts tab in the projection viewer
        if self.projection_viewer is not None:
            self.projection_viewer.refresh_applied_shifts_tab()

    def clear_plots(self):
        """Clear the shift plots when no alignment results are available."""
        for ax in [self.ax_horizontal, self.ax_vertical]:
            ax.clear()
            ax.set_title("")
            ax.text(
                0.5, 0.5, "No alignment results available",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
                color='gray'
            )
        self.canvas.draw()


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
        list_of_updated_settings: Optional[list[dict]] = None,
        multi_thread_func: Optional[Callable] = None,
        parent: Optional[QWidget] = None,
        projection_viewer=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.alignment_results_list: list[PMAResults] = []
        self.pma_viewer = None
        self.results_collection_widget = None
        self.stop_alignment_sequence_flag = False
        self.projection_viewer = projection_viewer
        self.crop_viewer = None

        if task is not None:
            self.initialize_page(task, list_of_updated_settings)

    def initialize_page(
        self,
        task: "t.LaminographyAlignmentTask",
        list_of_updated_settings: Optional[list[dict]] = None,
    ):
        self.task = task
        tabs = QTabWidget()
        tabs.setObjectName("main_tabs")
        tabs.setStyleSheet("#main_tabs > QTabBar{font-size: 20px;}")
        layout = QHBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

        self.generate_start_and_stop_buttons()
        self.generate_options_selection_widget()
        self.generate_sequencer(list_of_updated_settings)
        self.make_first_tab_layout(tabs)
        self.make_second_tab_layout(tabs)
        self.make_third_tab_layout(tabs)

    def generate_start_and_stop_buttons(self):
        # Left button widget (underneath options editor)
        self.left_button_widget = QWidget(self)
        left_button_layout = QHBoxLayout()
        self.left_button_widget.setLayout(left_button_layout)

        self.start_alignment_defaults_button = QPushButton("Start Alignment (Defaults)")
        self.start_sequence_button = QPushButton("Start Alignment Sequence")
        self.stop_alignment_button = QPushButton("Stop Current Alignment")

        # Set fixed width for buttons
        button_width = 250
        self.start_sequence_button.setFixedWidth(button_width)
        self.stop_alignment_button.setFixedWidth(button_width)

        # Create dropdown for initial shift selection for defaults (on the left, aligned left)
        initial_shift_widget = QWidget()
        initial_shift_layout = QVBoxLayout()
        initial_shift_layout.setContentsMargins(0, 0, 0, 0)
        initial_shift_widget.setLayout(initial_shift_layout)
        initial_shift_layout.addWidget(QLabel("Initial shift:"), alignment=Qt.AlignLeft)
        self.initial_shift_combobox = QComboBox()
        self.initial_shift_combobox.addItem("None")
        self.initial_shift_combobox.setFixedWidth(button_width)
        initial_shift_layout.addWidget(self.initial_shift_combobox, alignment=Qt.AlignLeft)

        # Create dropdown for initial shift selection for sequence (on the right)
        sequence_initial_shift_widget = QWidget()
        sequence_initial_shift_layout = QVBoxLayout()
        sequence_initial_shift_layout.setContentsMargins(0, 0, 0, 0)
        sequence_initial_shift_widget.setLayout(sequence_initial_shift_layout)
        sequence_initial_shift_layout.addWidget(QLabel("Initial shift:"), alignment=Qt.AlignLeft)
        self.sequence_initial_shift_combobox = QComboBox()
        self.sequence_initial_shift_combobox.addItem("None")
        self.sequence_initial_shift_combobox.addItem("Previous")
        self.sequence_initial_shift_combobox.addItem("Default")
        self.sequence_initial_shift_combobox.setCurrentText("Previous")
        self.sequence_initial_shift_combobox.setFixedWidth(button_width)
        sequence_initial_shift_layout.addWidget(
            self.sequence_initial_shift_combobox, alignment=Qt.AlignLeft
        )

        # Create vertical layout for stop button (on the right, aligned right)
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_container.setLayout(buttons_layout)

        self.start_sequence_button.pressed.connect(self.start_alignment_sequence)
        self.stop_alignment_button.pressed.connect(self.on_stop_alignment_button_pushed)
        self.start_alignment_defaults_button.pressed.connect(
            self.on_start_alignment_with_defaults_pushed
        )

        self.start_sequence_button.setStyleSheet("QPushButton { background-color: green;}")
        self.start_alignment_defaults_button.setStyleSheet("QPushButton { background-color: green;}")
        self.stop_alignment_button.setStyleSheet("QPushButton { background-color: red;}")

        buttons_layout.addWidget(self.start_alignment_defaults_button, alignment=Qt.AlignRight)
        buttons_layout.addWidget(self.stop_alignment_button, alignment=Qt.AlignRight)

        # Add dropdown on the left, spacer in middle, stop button on the right
        left_button_layout.addWidget(initial_shift_widget, alignment=Qt.AlignLeft)
        left_button_layout.addStretch()
        left_button_layout.addWidget(buttons_container, alignment=Qt.AlignRight)

        # Right button widget (underneath sequencer)
        self.right_button_widget = QWidget(self)
        right_button_layout = QHBoxLayout()
        self.right_button_widget.setLayout(right_button_layout)

        self.stop_sequence_button = QPushButton("Stop Alignment Sequence")
        self.stop_sequence_button.setFixedWidth(button_width)
        self.stop_sequence_button.pressed.connect(self.on_stop_sequence_button_pushed)
        self.stop_sequence_button.setStyleSheet("QPushButton { background-color: red;}")

        # Create vertical layout for sequence buttons (on the right, aligned right)
        sequence_buttons_container = QWidget()
        sequence_buttons_layout = QVBoxLayout()
        sequence_buttons_layout.setContentsMargins(0, 0, 0, 0)
        sequence_buttons_container.setLayout(sequence_buttons_layout)

        sequence_buttons_layout.addWidget(self.start_sequence_button, alignment=Qt.AlignRight)
        sequence_buttons_layout.addWidget(self.stop_sequence_button, alignment=Qt.AlignRight)

        # Add initial shift dropdown on the left, spacer in middle, sequence buttons on the right
        right_button_layout.addWidget(sequence_initial_shift_widget, alignment=Qt.AlignLeft)
        right_button_layout.addStretch()
        right_button_layout.addWidget(sequence_buttons_container, alignment=Qt.AlignRight)

        # Apply button style sheet
        self.left_button_widget.setStyleSheet(action_button_style_sheet)
        self.right_button_widget.setStyleSheet(action_button_style_sheet)

    def set_configure_tab_enabled(self, enabled: bool, run_mode: Optional[str] = None):
        """
        Enable or disable widgets on the Configure & Start tab.

        When disabled, only the appropriate stop button remains enabled to allow
        cancellation of running alignments.

        Parameters
        ----------
        enabled : bool
            If True, enable all widgets. If False, disable all except appropriate stop button.
        run_mode : str, optional
            Type of alignment run: 'defaults' or 'sequence'. Determines which stop button
            remains enabled when disabled. If None, both stop buttons are controlled together.
        """
        # Disable/enable the options editor
        self.options_editor.setEnabled(enabled)

        # Disable/enable the sequencer
        self.sequencer.setEnabled(enabled)

        # Disable/enable the start buttons and initial shift selectors
        self.start_alignment_defaults_button.setEnabled(enabled)
        self.start_sequence_button.setEnabled(enabled)
        self.initial_shift_combobox.setEnabled(enabled)
        self.sequence_initial_shift_combobox.setEnabled(enabled)

        # Update start button appearances based on enabled state
        if enabled:
            # Re-enable with green background
            self.start_alignment_defaults_button.setStyleSheet("QPushButton { background-color: green;}")
            self.start_sequence_button.setStyleSheet("QPushButton { background-color: green;}")
        else:
            # Disabled appearance - gray background
            self.start_alignment_defaults_button.setStyleSheet("QPushButton { background-color: gray; color: darkgray;}")
            self.start_sequence_button.setStyleSheet("QPushButton { background-color: gray; color: darkgray;}")

        # Handle stop buttons based on mode
        if enabled:
            # When re-enabling everything, disable both stop buttons
            self.stop_alignment_button.setEnabled(False)
            self.stop_sequence_button.setEnabled(False)
        else:
            # When disabling, only enable the appropriate stop button based on run mode
            if run_mode == 'defaults':
                # Only enable the single alignment stop button, disable sequence stop button
                self.stop_alignment_button.setEnabled(True)
                self.stop_sequence_button.setEnabled(False)
                self.stop_sequence_button.setStyleSheet("QPushButton { background-color: gray; color: darkgray;}")
            elif run_mode == 'sequence':
                # Enable both stop buttons for sequence mode
                self.stop_alignment_button.setEnabled(True)
                self.stop_sequence_button.setEnabled(True)
            else:
                # Default behavior: enable both stop buttons
                self.stop_alignment_button.setEnabled(True)
                self.stop_sequence_button.setEnabled(True)

        # Reset stop button styles when re-enabling
        if enabled:
            self.stop_alignment_button.setStyleSheet("QPushButton { background-color: red;}")
            self.stop_sequence_button.setStyleSheet("QPushButton { background-color: red;}")

    def calculate_total_applied_shift(self) -> Optional[np.ndarray]:
        """
        Calculate the total applied shift from phase_projections at the current moment.

        Returns
        -------
        np.ndarray or None
            Total applied shift array, or None if no shifts have been applied.
        """
        if not hasattr(self.task, 'phase_projections'):
            return None

        past_shifts = self.task.phase_projections.shift_manager.past_shifts
        if len(past_shifts) == 0:
            return None

        # Calculate total applied shift
        total_applied = np.sum(past_shifts, axis=0)
        return total_applied

    def filter_shift_by_scan_numbers(
        self, shift: np.ndarray, source_scan_numbers: np.ndarray, target_scan_numbers: np.ndarray
    ) -> np.ndarray:
        """
        Filter a shift array to match the current scan numbers.

        When using a previous alignment result as the initial shift, some scans
        may have been removed. This method removes entries from the shift array
        that correspond to removed scans.

        Parameters
        ----------
        shift : np.ndarray
            The shift array from a previous alignment result.
        source_scan_numbers : np.ndarray
            Scan numbers from the previous alignment result.
        target_scan_numbers : np.ndarray
            Current scan numbers in phase_projections.

        Returns
        -------
        np.ndarray
            Filtered shift array matching current scan numbers.
        """
        if source_scan_numbers is None or target_scan_numbers is None:
            # If scan numbers aren't available, return original shift
            return shift

        # Find which scans from the source are still present in the target
        mask = np.isin(source_scan_numbers, target_scan_numbers)

        # Check if all target scans are in source
        if not np.all(np.isin(target_scan_numbers, source_scan_numbers)):
            print(
                "Warning: Some current scans were not present in the selected "
                "initial shift. These will be initialized with zero shift."
            )
            # Create a new shift array initialized to zeros
            filtered_shift = np.zeros((len(target_scan_numbers), shift.shape[1]))
            # Find indices where target scans match source scans
            for i, scan in enumerate(target_scan_numbers):
                source_indices = np.where(source_scan_numbers == scan)[0]
                if len(source_indices) > 0:
                    filtered_shift[i] = shift[source_indices[0]]
            return filtered_shift
        else:
            # All target scans are in source, just filter out removed scans
            filtered_shift = shift[mask]
            # Reorder to match target scan order
            source_filtered = source_scan_numbers[mask]
            reorder_indices = np.array([np.where(source_filtered == scan)[0][0] for scan in target_scan_numbers])
            return filtered_shift[reorder_indices]

    def start_alignment_sequence(self):
        # Disable configure tab widgets during execution, enable only sequence stop button
        self.set_configure_tab_enabled(False, run_mode='sequence')

        try:
            options_sequence = self.sequencer.generate_options_sequence(
                self.task.options.projection_matching
            )
            # Get the changed settings for each sequence item
            changed_settings_sequence = self.sequencer.get_changed_settings_sequence()

            # # shift = None
            # suffix = self.task.options.projection_matching.save.suffix
            for i, options in enumerate(options_sequence):
                # # update suffix
                # options.save.suffix = suffix + f"_{i}"
                # Get initial shift based on combobox selection
                if i == 0:
                    # defaults button determines shift on first iteration
                    initial_shift_source = self.initial_shift_combobox.currentText()
                else:
                    # sequence button determines initial shift on following runs
                    initial_shift_source = self.sequence_initial_shift_combobox.currentText()
                    if initial_shift_source == "Default":
                        initial_shift_source = self.initial_shift_combobox.currentText()

                # Get the changed settings for this particular sequence item
                changed_settings = (
                    changed_settings_sequence[i] if i < len(changed_settings_sequence) else {}
                )

                self.run_projection_matching_instance(
                    options,
                    initial_shift_source,
                    run_type="sequence",
                    changed_settings=changed_settings,
                )
                if self.stop_alignment_sequence_flag:
                    break
        finally:
            # Always re-enable configure tab widgets when done (even if error occurs)
            self.set_configure_tab_enabled(True)
            # reset flags
            self.stop_alignment_sequence_flag = False

    def run_projection_matching_instance(
        self,
        # initial_shift: np.ndarray,
        options: ProjectionMatchingOptions,
        initial_shift_source: str,
        run_type: Optional[str] = None,
        changed_settings: Optional[dict] = None,
    ):
        initial_shift, initial_shift_source = self.get_initial_shift(shift_text=initial_shift_source)
        shift = self.task.get_projection_matching_shift(
            initial_shift=initial_shift, options=options
        )

        # Calculate total applied shift at this moment
        total_applied_shift = self.calculate_total_applied_shift()

        self.alignment_results_list += [
            PMAResults(
                shift,
                self.task.pma_object.initial_shift,
                self.task.pma_object.aligned_projections.angles,
                options=options,
                projection_options=self.task.phase_projections.options,
                scan_numbers=self.task.phase_projections.scan_numbers.copy(),
                initial_shift_source=initial_shift_source,
                run_type=run_type,
                changed_settings=changed_settings,
                total_applied_shift=total_applied_shift,
            )
        ]
        self.update_pma_viewer_tab()
        self.update_results_collection_tab()
        # Refresh the Applied Shifts tab in the projection viewer
        if self.projection_viewer is not None:
            self.projection_viewer.refresh_applied_shifts_tab()
        if self.task.pma_object.gui is not None:
            self.task.pma_object.gui.close()

    def get_initial_shift(self, shift_text: str) -> tuple[np.ndarray, str]:
        if shift_text == "None":
            return None, shift_text
        elif shift_text == "Previous":
            result_index = len(self.alignment_results_list) - 1
        else:
            result_index = int(shift_text.split()[-1])
        selected_result = self.alignment_results_list[result_index]
        # Filter the shift to match current scan numbers
        initial_shift = self.filter_shift_by_scan_numbers(
            shift=selected_result.shift,
            source_scan_numbers=selected_result.scan_numbers,
            target_scan_numbers=self.task.phase_projections.scan_numbers,
        )
        shift_text = f"Result {result_index}"
        return initial_shift, shift_text

    def on_start_alignment_with_defaults_pushed(self):
        # Disable configure tab widgets during execution, enable only defaults stop button
        self.set_configure_tab_enabled(False, run_mode='defaults')

        try:
            initial_shift_source = self.initial_shift_combobox.currentText()
            self.run_projection_matching_instance(
                self.task.options.projection_matching, initial_shift_source, run_type="defaults"
            )
        finally:
            # Always re-enable configure tab widgets when done (even if error occurs)
            self.set_configure_tab_enabled(True)

    def on_stop_sequence_button_pushed(self):
        self.stop_alignment_sequence_flag = True
        self.send_alignment_stop_flag()

    def on_stop_alignment_button_pushed(self):
        self.send_alignment_stop_flag()

    def send_alignment_stop_flag(self):
        loop_started = False
        while not loop_started:
            loop_started = (
                hasattr(self.task, "pma_object") and self.task.pma_object.alignment_loop_started
            )
            time.sleep(0.1)
        self.task.pma_object.external_stop_flag = True

    def generate_options_selection_widget(self):
        # create options editor
        self.options_editor = BasicOptionsEditor(
            self.task.options.projection_matching,
            skip_fields=["plot", "crop"],
            enable_advanced_tab=True,
            basic_options_list=basic_pma_settings,
            open_panels_list=[
                "downsample",
                "save",
                "regularization",
                "reconstruct",
                "astra",
            ],
            label="Projection Matching Alignment Options",
        )
        # add button for editing crop region
        self.open_crop_viewer_button = QPushButton("Edit Crop Region/Alignment ROI")
        self.open_crop_viewer_button.clicked.connect(self.show_cropped_projections_viewer)
        self.open_crop_viewer_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.options_editor.form_layout.addRow("", self.open_crop_viewer_button)

    def generate_sequencer(self, list_of_updated_settings: Optional[list[dict]] = None):
        self.sequencer = SequencerWidgetV2(
            self.task.options.projection_matching,
            list_of_updated_settings,
            basic_options_list=basic_pma_settings,
            parent=self,
        )

    def show_cropped_projections_viewer(self):
        """Open the ROI selector window for setting crop region."""
        self.crop_viewer = GetBoxBoundsFromROISelector(
            self.task.phase_projections,
            self.options_editor._data.crop
        )
        self.crop_viewer.rectangular_roi_selected.connect(self.update_crop_options)
        self.crop_viewer.show()

    def update_crop_options(self):
        """Update crop options from the ROI selector."""
        self.task.options.projection_matching.crop = self.crop_viewer.options
        self.crop_viewer.close()

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
        self.update_initial_shift_combobox()

    def update_initial_shift_combobox(self):
        """Update the initial shift combobox with current results."""
        # Store current selection
        current_text = self.initial_shift_combobox.currentText()

        # Clear and rebuild
        self.initial_shift_combobox.clear()
        self.initial_shift_combobox.addItem("None")
        self.initial_shift_combobox.addItem("Previous")

        # Add all results from the collection
        for i in range(len(self.alignment_results_list)):
            self.initial_shift_combobox.addItem(f"Result {i}")

        # Try to restore previous selection
        index = self.initial_shift_combobox.findText(current_text)
        if index >= 0:
            self.initial_shift_combobox.setCurrentIndex(index)

    def clear_alignment_results(self):
        """
        Clear all alignment results from the collection.

        This method is called when shift operations (apply or undo) are performed
        on the ProjectionViewer, as those operations invalidate previously computed
        alignment results.
        """
        # Clear the alignment results list
        self.alignment_results_list.clear()

        # Clear the PMA viewer tab
        if self.pma_viewer is not None:
            self.pma_viewer.deleteLater()
            self.pma_viewer.setParent(None)
            self.pma_viewer = None

        # Update the results collection widget to reflect empty results
        self.results_collection_widget.alignment_results_list = self.alignment_results_list
        # Clear all rows from the results table
        self.results_collection_widget.results_table.setRowCount(0)
        # Clear the changed settings display
        self.results_collection_widget.changed_settings_display.setText(
            "<i>No alignment results available.</i>"
        )
        # Clear the plots on the collected results tab
        self.results_collection_widget.clear_plots()

        # Reset the initial shift combobox to only show "None" and "Previous"
        self.update_initial_shift_combobox()

    def make_first_tab_layout(self, tabs: QTabWidget):
        alignment_setup_widget = QWidget(self)

        # Create left side container
        left_container = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.options_editor, stretch=1)  # Expand to fill vertical space
        left_layout.addWidget(self.left_button_widget, stretch=0)  # Keep at minimum size
        left_container.setLayout(left_layout)

        # Create vertical separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(2)

        # Create right side container
        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.sequencer, stretch=1)  # Expand to fill vertical space
        right_layout.addWidget(self.right_button_widget, stretch=0)  # Keep at minimum size
        right_container.setLayout(right_layout)

        # Main layout with both containers and separator
        layout = QHBoxLayout()
        layout.addWidget(left_container)
        layout.addWidget(separator)
        layout.addWidget(right_container)

        alignment_setup_widget.setLayout(layout)
        tabs.addTab(alignment_setup_widget, "Configure && Start")

    def make_second_tab_layout(self, tabs: QTabWidget):
        empty_widget = QWidget()
        self._pma_viewer_layout = QVBoxLayout()
        empty_widget.setLayout(self._pma_viewer_layout)
        tabs.addTab(empty_widget, "Detailed Results")

    def make_third_tab_layout(self, tabs: QTabWidget):
        self.results_collection_widget = PMAResultsCollection(
            self.alignment_results_list,
            task=self.task,
            projection_viewer=self.projection_viewer,
        )
        empty_widget = QWidget()
        self._results_collection_layout = QVBoxLayout()
        empty_widget.setLayout(self._results_collection_layout)
        tabs.addTab(self.results_collection_widget, "Collected Results")


@switch_to_matplotlib_qt_backend
def launch_pma_runner(
    task: t.LaminographyAlignmentTask,
    list_of_updated_settings: Optional[list[dict]] = None,
    wait_until_closed: bool = False,
):
    # may want to move this to the PMA runner tab
    app = QApplication.instance() or QApplication([])
    gui = PMAMasterWidget(task, list_of_updated_settings)
    gui.show()
    gui.setAttribute(Qt.WA_DeleteOnClose)
    if wait_until_closed:
        app.exec_()
    return gui


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
