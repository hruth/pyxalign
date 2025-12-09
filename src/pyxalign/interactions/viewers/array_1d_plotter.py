"""
Interactive 1D array plotter widget with slider and checkbox selection.

This module provides a widget for plotting 1D arrays with both single array
selection via slider and multi-array selection via checkboxes.
"""

from typing import List, Optional, Union
import numpy as np
import cupy as cp

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QScrollArea,
    QPushButton,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QSpacerItem,
    QApplication,
)

from pyxalign.interactions.viewers.base import IndexSelectorWidget
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend

# Color palette for multi-array plotting
PLOT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class Array1DPlotterWidget(QWidget):
    """
    Interactive widget for plotting 1D arrays with slider and checkbox selection.

    Features:
    - Slider for array navigation and single array viewing
    - Single checkbox to add/remove current array to/from multi-array plot
    - Arrays remain plotted when navigating to different indices
    - Clear All functionality to return to single array view
    - Real-time plot updates

    Parameters
    ----------
    array_data : Union[List[np.ndarray], np.ndarray]
        Either a list of 1D numpy arrays to plot, or a 2D numpy array where
        the first dimension is indexed along (each row becomes a 1D array).
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self, array_data: Union[List[np.ndarray], np.ndarray], parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        # Validate and store input arrays
        self.array_list = self._validate_and_convert_arrays(array_data)
        self.num_arrays = len(self.array_list)

        if self.num_arrays == 0:
            raise ValueError("array_data must contain at least one array")

        # Initialize UI components
        self.plot_widget: Optional[pg.PlotWidget] = None
        self.index_selector: Optional[IndexSelectorWidget] = None
        self.add_to_plot_checkbox: Optional[QCheckBox] = None

        # Track which array indices should be plotted
        self.plotted_indices: set = set()

        # Setup the user interface
        self.setup_ui()

        # Initialize with first array displayed
        self.update_single_plot(0)

    def _validate_and_convert_arrays(
        self, array_data: Union[List[np.ndarray], np.ndarray]
    ) -> List[np.ndarray]:
        """
        Validate input data and convert to list of 1D numpy arrays.

        Parameters
        ----------
        array_data : Union[List[np.ndarray], np.ndarray]
            Either a list of 1D arrays or a 2D array to be sliced along first dimension.

        Returns
        -------
        List[np.ndarray]
            List of validated 1D numpy arrays.

        Raises
        ------
        ValueError
            If input is invalid or empty.
        """
        if isinstance(array_data, (list, tuple)):
            return self._process_array_list(array_data)
        elif isinstance(array_data, np.ndarray) or cp.get_array_module(array_data) == cp:
            return self._process_2d_array(array_data)
        else:
            raise ValueError(
                "array_data must be either a list of 1D arrays or a 2D array. "
                f"Got {type(array_data)}"
            )

    def _process_array_list(self, array_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a list of arrays, validating each is 1D.

        Parameters
        ----------
        array_list : List[np.ndarray]
            List of arrays to validate and convert.

        Returns
        -------
        List[np.ndarray]
            List of validated 1D numpy arrays.

        Raises
        ------
        ValueError
            If any array is not 1D or if the list is empty.
        """
        if not array_list:
            raise ValueError("array_list cannot be empty")

        converted_arrays = []
        for i, array in enumerate(array_list):
            # Convert cupy arrays to numpy
            if cp.get_array_module(array) == cp:
                array = array.get()

            # Validate that array is 1D
            if array.ndim != 1:
                raise ValueError(f"Array at index {i} is not 1D (shape: {array.shape})")

            converted_arrays.append(array)

        return converted_arrays

    def _process_2d_array(self, array_2d: np.ndarray) -> List[np.ndarray]:
        """
        Process a 2D array by slicing along the first dimension.

        Parameters
        ----------
        array_2d : np.ndarray
            2D array to slice into 1D arrays.

        Returns
        -------
        List[np.ndarray]
            List of 1D arrays, one for each row of the input.

        Raises
        ------
        ValueError
            If array is not 2D or is empty.
        """
        # Convert cupy arrays to numpy
        if cp.get_array_module(array_2d) == cp:
            array_2d = array_2d.get()

        # Validate that array is 2D
        if array_2d.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {array_2d.ndim}D array (shape: {array_2d.shape})"
            )

        if array_2d.shape[0] == 0:
            raise ValueError("2D array cannot have zero rows")

        # Slice along first dimension to create list of 1D arrays
        converted_arrays = []
        for i in range(array_2d.shape[0]):
            converted_arrays.append(array_2d[i, :])

        return converted_arrays

    def setup_ui(self):
        """Setup the main user interface layout."""
        # Main horizontal layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Setup control panel and plot area
        control_panel = self.setup_control_panel()
        plot_area = self.setup_plot_area()

        # Add to main layout with appropriate sizing
        main_layout.addWidget(control_panel, 1)  # Control panel takes 1/4 of space
        main_layout.addWidget(plot_area, 3)  # Plot area takes 3/4 of space

        # Apply styling
        self.setStyleSheet("""
            QGroupBox {
                font-size: 13pt;
                font-weight: bold;
            }
            QCheckBox {
                font-size: 12pt;
            }
            QLabel {
                font-size: 11pt;
            }
            QPushButton {
                font-size: 12pt;
                padding: 5px 10px;
                min-height: 25px;
            }
        """)

    def setup_control_panel(self) -> QWidget:
        """
        Setup the left control panel with slider and single checkbox.

        Returns
        -------
        QWidget
            The control panel widget.
        """
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)

        # Array navigation section
        navigation_group = QGroupBox("Array Navigation")
        navigation_layout = QVBoxLayout()
        navigation_group.setLayout(navigation_layout)

        # Index selector widget (slider)
        self.index_selector = IndexSelectorWidget(
            num_frames=self.num_arrays,
            start_index=0,
            include_play_button=True,
            hide_controls=False,
            parent=self,
        )

        # Connect slider to update functions
        self.index_selector.slider.valueChanged.connect(self.update_single_plot)
        self.index_selector.spinbox.valueChanged.connect(self.update_single_plot)
        self.index_selector.slider.valueChanged.connect(self.update_add_to_plot_checkbox)
        self.index_selector.spinbox.valueChanged.connect(self.update_add_to_plot_checkbox)

        navigation_layout.addWidget(self.index_selector)

        # Multi-array plotting section
        plotting_group = QGroupBox("Multi-Array Plotting")
        plotting_layout = QVBoxLayout()
        plotting_group.setLayout(plotting_layout)

        # Single checkbox for adding current array to plot
        self.add_to_plot_checkbox = QCheckBox("Add current array to plot")
        self.add_to_plot_checkbox.clicked.connect(self.update_plotted_arrays)
        plotting_layout.addWidget(self.add_to_plot_checkbox)

        # Control buttons
        button_layout = QHBoxLayout()
        clear_all_button = QPushButton("Clear All")
        clear_all_button.clicked.connect(self.clear_all_plotted_arrays)
        button_layout.addWidget(clear_all_button)
        plotting_layout.addLayout(button_layout)

        # Add sections to control panel
        control_layout.addWidget(navigation_group)
        control_layout.addWidget(plotting_group)

        # Add spacer to push everything to top
        control_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        return control_panel

    def setup_plot_area(self) -> QWidget:
        """
        Setup the right plot area with pyqtgraph widget.

        Returns
        -------
        QWidget
            The plot area widget.
        """
        plot_area = QWidget()
        plot_layout = QVBoxLayout()
        plot_area.setLayout(plot_layout)

        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.setLabel("bottom", "Index")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        plot_layout.addWidget(self.plot_widget)

        return plot_area

    def update_single_plot(self, index: int):
        """
        Update the plot to show the current array and any plotted arrays.

        Parameters
        ----------
        index : int
            Index of the array to plot.
        """
        if not (0 <= index < self.num_arrays):
            return

        # Update the plot with current array and any plotted arrays
        self.update_plot()

    def update_plot(self):
        """Update the plot to show current array and all plotted arrays."""
        # Clear the plot
        self.plot_widget.clear()

        # Get current index
        current_index = self.index_selector.slider.value()

        # Create a set of all arrays to plot (current + plotted)
        arrays_to_plot = set(self.plotted_indices)
        arrays_to_plot.add(current_index)  # Always include current array

        # Convert to sorted list for consistent plotting order
        arrays_to_plot_list = sorted(list(arrays_to_plot))

        # Plot all arrays
        for plot_idx, array_idx in enumerate(arrays_to_plot_list):
            array = self.array_list[array_idx]
            x_data = np.arange(len(array))
            color = PLOT_COLORS[plot_idx % len(PLOT_COLORS)]

            # Use different line style for current array if it's not in plotted set
            if array_idx == current_index and array_idx not in self.plotted_indices:
                # Current array not in plotted set - use thicker line or different style
                pen = pg.mkPen(color=color, width=3, style=Qt.SolidLine)
                name = f"Array {array_idx} (current)"
            else:
                # Regular plotted array
                pen = pg.mkPen(color=color, width=2)
                name = f"Array {array_idx}"

            self.plot_widget.plot(x_data, array, pen=pen, name=name)

        # Update title based on mode
        if not self.plotted_indices:
            # Only current array is shown
            array = self.array_list[current_index]
            self.plot_widget.setTitle(
                f"Single Array View - Array {current_index} (length: {len(array)})"
            )
        else:
            # Multi-array view with current array always visible
            num_plotted = len(self.plotted_indices)
            if current_index in self.plotted_indices:
                self.plot_widget.setTitle(
                    f"Multi-Array View - {num_plotted} arrays plotted (current: {current_index})"
                )
            else:
                self.plot_widget.setTitle(
                    f"Multi-Array View - {num_plotted} arrays plotted + current ({current_index})"
                )

    def update_add_to_plot_checkbox(self):
        """Update the checkbox state based on current index."""
        current_index = self.index_selector.slider.value()
        is_plotted = current_index in self.plotted_indices

        # Block signals to prevent recursive calls
        self.add_to_plot_checkbox.blockSignals(True)
        self.add_to_plot_checkbox.setChecked(is_plotted)
        self.add_to_plot_checkbox.blockSignals(False)

    def update_plotted_arrays(self):
        """Add or remove current array from plotted set based on checkbox state."""
        current_index = self.index_selector.slider.value()

        if self.add_to_plot_checkbox.isChecked():
            # Add current index to plotted set
            self.plotted_indices.add(current_index)
        else:
            # Remove current index from plotted set
            self.plotted_indices.discard(current_index)

        # Update the plot
        self.update_plot()

    def clear_all_plotted_arrays(self):
        """Clear all plotted arrays and return to single array view."""
        self.plotted_indices.clear()

        # Update checkbox state
        self.update_add_to_plot_checkbox()

        # Update the plot
        self.update_plot()

    def get_plotted_array_indices(self) -> List[int]:
        """
        Get the indices of currently plotted arrays.

        Returns
        -------
        List[int]
            List of indices of plotted arrays.
        """
        return sorted(list(self.plotted_indices))

    def set_plotted_arrays(self, indices: List[int]):
        """
        Set which arrays should be plotted.

        Parameters
        ----------
        indices : List[int]
            List of array indices to plot.
        """
        # Clear current set and add new indices
        self.plotted_indices.clear()
        for index in indices:
            if 0 <= index < self.num_arrays:
                self.plotted_indices.add(index)

        # Update checkbox state and plot
        self.update_add_to_plot_checkbox()
        self.update_plot()


@switch_to_matplotlib_qt_backend
def launch_array_1d_plotter(
    array_data: Union[List[np.ndarray], np.ndarray], wait_until_closed: bool = False
) -> Array1DPlotterWidget:
    """
    Launch the 1D array plotter GUI.

    Parameters
    ----------
    array_data : Union[List[np.ndarray], np.ndarray]
        Either a list of 1D arrays to plot, or a 2D array where the first
        dimension is indexed along (each row becomes a 1D array).
    wait_until_closed : bool, optional
        If True, the application starts a blocking call until the GUI window
        is closed. Default is False.

    Returns
    -------
    Array1DPlotterWidget
        The created widget instance.

    Example
    -------
    Create and launch a plotter for multiple 1D arrays::

        # Using list of 1D arrays
        arrays = [np.sin(np.linspace(0, 2*np.pi, 100)),
                  np.cos(np.linspace(0, 2*np.pi, 100)),
                  np.tan(np.linspace(0, np.pi/4, 100))]
        gui = launch_array_1d_plotter(arrays)

        # Using 2D array
        data_2d = np.random.randn(5, 100)  # 5 arrays of length 100
        gui = launch_array_1d_plotter(data_2d)
    """
    app = QApplication.instance() or QApplication([])
    gui = Array1DPlotterWidget(array_data)
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()

    if wait_until_closed:
        app.exec_()

    return gui