"""
Data structures for storing and visualizing alignment results.

This module provides base classes for storing alignment results from various
alignment algorithms (projection matching, cross-correlation, etc.) and
widgets for visualizing and comparing multiple alignment results.

Key Components:
- AlignmentResults: Base data structure for storing alignment results and parameters
- AlignmentResultsCollection: Base widget for visualizing and comparing multiple alignment results
"""

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.types import OptionsClass
from pyxalign.interactions.viewers.utils import OptionsDisplayWidget


class AlignmentResults:
    """
    Base data structure for storing alignment results.

    This class encapsulates the results from a single alignment run,
    including the computed shifts, initial conditions, and the options
    used for the alignment.

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
    options : OptionsClass
        Alignment options used for this alignment run.
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
        scan_numbers: Optional[np.ndarray] = None,
    ):
        self.shift = shift
        self.initial_shift = initial_shift
        self.angles = angles
        self.scan_numbers = scan_numbers
        self.pma_options = options
        self.projection_options = projection_options


class AlignmentResultsCollection(QWidget):
    """
    Base widget for visualizing and comparing multiple alignment results.

    This widget provides an interface for browsing through multiple alignment
    results, displaying shift plots and alignment options for comparison.
    Users can select different results from a table and view the corresponding
    shift data and configuration parameters.

    Parameters
    ----------
    alignment_results_list : list[AlignmentResults]
        List of alignment results to display and compare.
    display_initial_shift : bool, optional
        Whether to display initial shift in plots. Default is True.
    parent : QWidget, optional
        Parent widget for this interface.
    """

    def __init__(
        self,
        alignment_results_list: list[AlignmentResults],
        display_initial_shift: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent=parent)
        self.alignment_results_list = alignment_results_list
        self.display_initial_shift = display_initial_shift

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
        # Create the table - default 2 columns, subclasses can override
        num_columns = self._get_table_column_count()
        self.results_table = QTableWidget(0, num_columns)
        self.results_table.setHorizontalHeaderLabels(self._get_table_headers())
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

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

        # Connect the table's click signal to our plotting function
        self.results_table.currentCellChanged.connect(self.on_table_cell_changed)

    def _get_table_column_count(self) -> int:
        """Return the number of columns for the results table. Override in subclasses."""
        return 1

    def _get_table_headers(self) -> list[str]:
        """Return the header labels for the results table. Override in subclasses."""
        return ["Index"]

    def update_table(self):
        """Update the table with alignment results. Override in subclasses for custom columns."""
        num_results = len(self.alignment_results_list)
        table_length = self.results_table.rowCount()

        # Fill the table with row indices
        for i in range(num_results):
            if i >= table_length:
                self.results_table.insertRow(i)
                # Column 0: Index
                index_item = QTableWidgetItem(str(i))
                self.results_table.setItem(i, 0, index_item)

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
            if self.display_initial_shift:
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
        self.options_display = OptionsDisplayWidget()

    def change_options_display_index(self, row: int):
        self.options_display.update_options(self.alignment_results_list[row].pma_options)
        self.options_display.update_display()
