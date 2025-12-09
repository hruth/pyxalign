"""
Interactive 1D array plotter widgets with slider and checkbox selection.

This module provides widgets for plotting 1D arrays with both single array
selection via slider and multi-array selection via checkboxes. It includes:

- Array1DPlotterWidget: Single plotter for one set of 1D arrays
- LinkedArray1DPlotterWidget: Multiple linked plotters for comparing datasets
"""

import array
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
    QGridLayout,
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
    - Current array is always visible, regardless of checkbox state
    - Auto-scale Y axis control to prevent/allow rescaling when changing indices
    - Clear All functionality to return to single array view
    - Real-time plot updates

    Parameters
    ----------
    array_data : Union[List[np.ndarray], np.ndarray]
        Either a list of 1D numpy arrays to plot, or a 2D numpy array where
        the first dimension is indexed along (each row becomes a 1D array).
    extra_title_strings_list : Optional[List[str]], optional
        List of strings to append to the title for each array; must have the
        same length as the number of arrays.
    hide_multi_array_controls : bool, optional
        If True, hide the multi-array plotting controls (checkbox and Clear All button).
        Default is False. This is useful when the widget is part of a LinkedArray1DPlotterWidget
        where only the first widget should show these controls.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        array_data: Union[List[np.ndarray], np.ndarray],
        extra_title_strings_list: Optional[List[str]] = None,
        hide_multi_array_controls: bool = False,
        hide_navigation_title: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        # Validate and store input arrays
        self.array_list = self._validate_and_convert_arrays(array_data)
        self.num_arrays = len(self.array_list)

        if self.num_arrays == 0:
            raise ValueError("array_data must contain at least one array")

        # Store extra title strings
        self.extra_title_strings_list = extra_title_strings_list
        if self.extra_title_strings_list is not None:
            if len(self.extra_title_strings_list) != self.num_arrays:
                raise ValueError(
                    f"extra_title_strings_list length ({len(self.extra_title_strings_list)}) "
                    f"must match number of arrays ({self.num_arrays})"
                )

        # Store control visibility settings
        self.hide_multi_array_controls = hide_multi_array_controls
        self.hide_navigation_title = hide_navigation_title

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
        # Main horizontal layout with minimal spacing
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        main_layout.setSpacing(5)  # Minimal spacing between control panel and plot area
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
        control_layout.setContentsMargins(0, 0, 0, 0)  # No extra margins
        control_layout.setSpacing(5)  # Minimal spacing between sections
        control_panel.setLayout(control_layout)

        # Array navigation section (conditionally with or without title)
        if self.hide_navigation_title:
            # Create a simple widget without group box title
            navigation_container = QWidget()
            navigation_layout = QVBoxLayout()
            navigation_layout.setContentsMargins(0, 0, 0, 0)
            navigation_layout.setSpacing(3)
            navigation_container.setLayout(navigation_layout)
        else:
            # Create a group box with title
            navigation_container = QGroupBox("Array Navigation")
            navigation_layout = QVBoxLayout()
            navigation_layout.setContentsMargins(5, 5, 5, 5)
            navigation_layout.setSpacing(3)
            navigation_container.setLayout(navigation_layout)

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

        # Connect play button and timer functionality
        self.index_selector.play_button.clicked.connect(self.toggle_play)
        self.index_selector.play_timer.timeout.connect(self.next_frame)

        # Initialize play state
        self.playing = False

        # Set up playback speed if available
        if hasattr(self.index_selector, "playback_speed_spin"):
            # Set initial timer interval based on playback speed
            speed_hz = self.index_selector.playback_speed_spin.value()
            interval_ms = int(1000 / speed_hz)
            self.index_selector.play_timer.setInterval(interval_ms)

        navigation_layout.addWidget(self.index_selector)

        # Multi-array plotting section (conditionally visible)
        if not self.hide_multi_array_controls:
            plotting_group = QGroupBox("Multi-Array Plotting")
            plotting_layout = QVBoxLayout()
            plotting_layout.setContentsMargins(5, 5, 5, 5)
            plotting_layout.setSpacing(3)
            plotting_group.setLayout(plotting_layout)

            # Single checkbox for adding current array to plot
            self.add_to_plot_checkbox = QCheckBox("Add current array to plot")
            self.add_to_plot_checkbox.clicked.connect(self.update_plotted_arrays)
            plotting_layout.addWidget(self.add_to_plot_checkbox)

            # Control buttons
            button_layout = QHBoxLayout()
            self.clear_all_button = QPushButton("Clear All")
            self.clear_all_button.clicked.connect(self.clear_all_plotted_arrays)
            button_layout.addWidget(self.clear_all_button)
            plotting_layout.addLayout(button_layout)

            # Add sections to control panel
            control_layout.addWidget(navigation_container)
            control_layout.addWidget(plotting_group)
        else:
            # Only add navigation container if multi-array controls are hidden
            control_layout.addWidget(navigation_container)

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
        plot_layout.setContentsMargins(0, 0, 0, 0)  # No extra margins
        plot_layout.setSpacing(5)  # Minimal spacing between checkbox and plot
        plot_area.setLayout(plot_layout)

        # Auto-scale checkbox above the plot
        self.auto_scale_checkbox = QCheckBox("Auto-scale Y axis")
        self.auto_scale_checkbox.setChecked(True)  # Default to auto-scaling enabled
        plot_layout.addWidget(self.auto_scale_checkbox)

        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.setLabel("bottom", "Index")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # Disable x-axis mouse interaction by default
        self.plot_widget.setMouseEnabled(x=False, y=True)

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
        # Store current y-axis range if auto-scaling is disabled
        y_range = None
        if not self.auto_scale_checkbox.isChecked():
            view_box = self.plot_widget.getViewBox()
            y_range = view_box.viewRange()[1]  # Get current Y range [min, max]

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

        # Restore y-axis range if auto-scaling is disabled
        if y_range is not None:
            view_box = self.plot_widget.getViewBox()
            view_box.setYRange(y_range[0], y_range[1], padding=0)

        # Update title based on mode
        base_title = f"Array {current_index}"
        if self.extra_title_strings_list is not None:
            base_title += self.extra_title_strings_list[current_index]

        if not self.plotted_indices:
            # Only current array is shown
            array = self.array_list[current_index]
            self.plot_widget.setTitle(f"Single Array View - {base_title} (length: {len(array)})")
        else:
            # Multi-array view with current array always visible
            num_plotted = len(self.plotted_indices)
            if current_index in self.plotted_indices:
                self.plot_widget.setTitle(
                    f"Multi-Array View - {num_plotted} arrays plotted (current: {base_title})"
                )
            else:
                self.plot_widget.setTitle(
                    f"Multi-Array View - {num_plotted} arrays plotted + current ({base_title})"
                )

    def update_add_to_plot_checkbox(self):
        """Update the checkbox state based on current index."""
        if self.add_to_plot_checkbox is None:
            return

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

    def toggle_play(self):
        """Toggle play/pause functionality for the array navigation."""
        if self.playing:
            self.index_selector.play_timer.stop()
            self.index_selector.play_button.setText("Play")
        else:
            self.index_selector.play_timer.start()
            self.index_selector.play_button.setText("Pause")
        self.playing = not self.playing

    def next_frame(self):
        """Advance to the next frame in the sequence, wrapping around at the end."""
        current = self.index_selector.slider.value()
        next_idx = (current + 1) % self.num_arrays
        self.index_selector.slider.setValue(next_idx)


class LinkedArray1DPlotterWidget(QWidget):
    """
    Widget that displays multiple Array1DPlotterWidget instances with linked index selectors.

    This widget creates multiple Array1DPlotterWidget instances arranged in a grid layout,
    with their index selectors linked so that changing the index in one widget updates
    all others. This is useful for comparing multiple sets of 1D arrays simultaneously.

    Parameters
    ----------
    array_data_list : List[Union[List[np.ndarray], np.ndarray]]
        List where each entry is either a list of 1D numpy arrays or a 2D numpy array.
        Each entry will be used to create one Array1DPlotterWidget.
    plot_titles : Optional[List[str]], optional
        List of titles to display above each plot. Must have the same length as
        array_data_list if provided.
    extra_title_strings_list : Optional[List[List[str]]], optional
        List of lists of strings to append to titles. The outer list corresponds to
        each Array1DPlotterWidget, and the inner lists are the extra title strings
        for each array within that widget.
    n_rows : Optional[int], optional
        Number of rows in the grid layout. If None, defaults to 1 row.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        array_data_list: List[Union[List[np.ndarray], np.ndarray]],
        plot_titles: Optional[List[str]] = None,
        extra_title_strings_list: Optional[List[List[str]]] = None,
        n_rows: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if n_rows is None:
            n_rows = len(array_data_list)

        if not array_data_list:
            raise ValueError("array_data_list must contain at least one entry")

        # Validate plot_titles if provided
        if plot_titles is not None:
            if len(plot_titles) != len(array_data_list):
                raise ValueError(
                    f"plot_titles length ({len(plot_titles)}) "
                    f"must match array_data_list length ({len(array_data_list)})"
                )

        # Validate extra_title_strings_list if provided
        if extra_title_strings_list is not None:
            if len(extra_title_strings_list) != len(array_data_list):
                raise ValueError(
                    f"extra_title_strings_list length ({len(extra_title_strings_list)}) "
                    f"must match array_data_list length ({len(array_data_list)})"
                )

        # Setup grid layout
        self.setup_grid_layout(array_data_list, plot_titles, extra_title_strings_list, n_rows)

        # Link all index selectors
        self.link_index_selectors()

        # Setup synchronized multi-array controls
        self.setup_synchronized_multi_array_controls()

    def setup_grid_layout(
        self,
        array_data_list: List[Union[List[np.ndarray], np.ndarray]],
        plot_titles: Optional[List[str]],
        extra_title_strings_list: Optional[List[List[str]]],
        n_rows: Optional[int],
    ):
        """Setup the grid layout with Array1DPlotterWidget instances and optional titles."""
        # Calculate grid dimensions
        n = len(array_data_list)
        if n_rows is None:
            n_rows = 1
        n_cols = int(np.ceil(n / n_rows))

        # Create main grid layout with minimal spacing
        layout = QGridLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        layout.setSpacing(5)  # Minimal spacing between widgets
        self.setLayout(layout)

        # Create Array1DPlotterWidget instances
        self.plotter_widget_list: List[Array1DPlotterWidget] = []

        for i, array_data in enumerate(array_data_list):
            # Get extra title strings for this widget if provided
            extra_titles = None
            if extra_title_strings_list is not None:
                extra_titles = extra_title_strings_list[i]

            # Create the Array1DPlotterWidget
            # Hide multi-array controls and navigation title for all but the first widget
            hide_controls = i != 0
            hide_nav_title = i != 0
            plotter_widget = Array1DPlotterWidget(
                array_data=array_data,
                extra_title_strings_list=extra_titles,
                hide_multi_array_controls=hide_controls,
                hide_navigation_title=hide_nav_title,
                parent=self,
            )

            # Hide index selector controls for all but the first widget
            if i != 0:
                plotter_widget.index_selector.hide()

            self.plotter_widget_list.append(plotter_widget)

            # Calculate grid position
            row, col = np.unravel_index(i, (n_rows, n_cols))

            # If plot titles are provided, modify the plotter widget to include title
            if plot_titles is not None:
                # Add title to the plot area of this specific widget
                self._add_title_to_plotter(plotter_widget, plot_titles[i])

            # Add plotter widget to grid layout
            layout.addWidget(plotter_widget, row, col)

    def _add_title_to_plotter(self, plotter_widget: Array1DPlotterWidget, title: str):
        """Add a title label above the pyqtgraph plot area within the plotter widget."""
        # Get the plot area widget from the plotter
        plot_area = None
        for child in plotter_widget.children():
            if isinstance(child, QWidget) and hasattr(child, "layout"):
                layout = child.layout()
                if layout and layout.count() > 0:
                    # Check if this layout contains a PlotWidget
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        if item and item.widget() and hasattr(item.widget(), "plot"):
                            plot_area = child
                            break
                if plot_area:
                    break

        if plot_area is None:
            # Fallback: find the plot area by looking for the widget with the plot_widget
            for child in plotter_widget.children():
                if isinstance(child, QWidget):
                    for grandchild in child.children():
                        if grandchild == plotter_widget.plot_widget:
                            plot_area = child
                            break
                    if plot_area:
                        break

        if plot_area and plot_area.layout():
            plot_layout = plot_area.layout()

            # Create title label
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("""
                QLabel {
                    font-size: 14pt;
                    font-weight: bold;
                    color: #333333;
                    padding: 5px;
                    background-color: #f0f0f0;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    margin-bottom: 2px;
                }
            """)

            # Insert title at the beginning of the plot area layout
            # Find the position of the auto-scale checkbox (should be first)
            auto_scale_pos = -1
            for i in range(plot_layout.count()):
                item = plot_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QCheckBox):
                    if hasattr(item.widget(), "text") and "Auto-scale" in item.widget().text():
                        auto_scale_pos = i
                        break

            if auto_scale_pos >= 0:
                # Insert title after the auto-scale checkbox
                plot_layout.insertWidget(auto_scale_pos + 1, title_label)
            else:
                # Fallback: insert at the beginning
                plot_layout.insertWidget(0, title_label)

    def link_index_selectors(self):
        """Link all index selectors so they stay synchronized."""
        if len(self.plotter_widget_list) <= 1:
            return

        # Get the primary (first) widget's index selector
        primary_widget = self.plotter_widget_list[0]
        primary_selector = primary_widget.index_selector

        # Link all other widgets to the primary widget
        for i, widget in enumerate(self.plotter_widget_list[1:], 1):
            secondary_selector = widget.index_selector

            # Connect primary to secondary
            primary_selector.slider.valueChanged.connect(secondary_selector.slider.setValue)
            primary_selector.spinbox.valueChanged.connect(secondary_selector.spinbox.setValue)

            # Connect secondary to primary
            secondary_selector.slider.valueChanged.connect(primary_selector.slider.setValue)
            secondary_selector.spinbox.valueChanged.connect(primary_selector.spinbox.setValue)

            # Also link play functionality
            primary_selector.play_button.clicked.connect(
                lambda checked, w=widget: w.toggle_play()
                if w.playing != primary_widget.playing
                else None
            )

    def get_current_index(self) -> int:
        """
        Get the current index from the primary (first) widget.

        Returns
        -------
        int
            Current index value.
        """
        if self.plotter_widget_list:
            return self.plotter_widget_list[0].index_selector.slider.value()
        return 0

    def set_current_index(self, index: int):
        """
        Set the current index for all linked widgets.

        Parameters
        ----------
        index : int
            Index to set.
        """
        if self.plotter_widget_list:
            primary_selector = self.plotter_widget_list[0].index_selector
            primary_selector.slider.setValue(index)

    def get_plotted_arrays_for_widget(self, widget_index: int) -> List[int]:
        """
        Get the plotted array indices for a specific widget.

        Parameters
        ----------
        widget_index : int
            Index of the widget to query.

        Returns
        -------
        List[int]
            List of plotted array indices for the specified widget.
        """
        if 0 <= widget_index < len(self.plotter_widget_list):
            return self.plotter_widget_list[widget_index].get_plotted_array_indices()
        return []

    def set_plotted_arrays_for_widget(self, widget_index: int, indices: List[int]):
        """
        Set the plotted arrays for a specific widget.

        Parameters
        ----------
        widget_index : int
            Index of the widget to modify.
        indices : List[int]
            List of array indices to plot for the specified widget.
        """
        if 0 <= widget_index < len(self.plotter_widget_list):
            self.plotter_widget_list[widget_index].set_plotted_arrays(indices)

    def setup_synchronized_multi_array_controls(self):
        """Setup synchronized multi-array controls for the primary widget."""
        if not self.plotter_widget_list:
            return

        # Get the primary widget (first one with visible controls)
        primary_widget = self.plotter_widget_list[0]

        # Only proceed if the primary widget has multi-array controls
        if primary_widget.add_to_plot_checkbox is None:
            return

        # Disconnect the original signals from the primary widget
        primary_widget.add_to_plot_checkbox.clicked.disconnect()
        primary_widget.clear_all_button.clicked.disconnect()

        # Connect to synchronized methods
        primary_widget.add_to_plot_checkbox.clicked.connect(self.sync_update_plotted_arrays)
        primary_widget.clear_all_button.clicked.connect(self.sync_clear_all_plotted_arrays)

        # Also connect index changes to update the checkbox state
        primary_widget.index_selector.slider.valueChanged.connect(self.sync_update_checkbox_state)
        primary_widget.index_selector.spinbox.valueChanged.connect(self.sync_update_checkbox_state)

    def sync_update_plotted_arrays(self):
        """Synchronized method to add/remove current array from all widgets."""
        if not self.plotter_widget_list:
            return

        primary_widget = self.plotter_widget_list[0]
        if primary_widget.add_to_plot_checkbox is None:
            return

        current_index = primary_widget.index_selector.slider.value()
        is_checked = primary_widget.add_to_plot_checkbox.isChecked()

        # Apply the same operation to all widgets
        for widget in self.plotter_widget_list:
            if is_checked:
                widget.plotted_indices.add(current_index)
            else:
                widget.plotted_indices.discard(current_index)

            # Update the plot for each widget
            widget.update_plot()

    def sync_clear_all_plotted_arrays(self):
        """Synchronized method to clear all plotted arrays from all widgets."""
        # Clear plotted arrays from all widgets
        for widget in self.plotter_widget_list:
            widget.plotted_indices.clear()
            widget.update_plot()

        # Update the checkbox state
        self.sync_update_checkbox_state()

    def sync_update_checkbox_state(self):
        """Update the checkbox state based on whether current index is plotted in any widget."""
        if not self.plotter_widget_list:
            return

        primary_widget = self.plotter_widget_list[0]
        if primary_widget.add_to_plot_checkbox is None:
            return

        current_index = primary_widget.index_selector.slider.value()

        # Check if current index is plotted in any widget
        is_plotted_anywhere = any(
            current_index in widget.plotted_indices for widget in self.plotter_widget_list
        )

        # Update checkbox state
        primary_widget.add_to_plot_checkbox.blockSignals(True)
        primary_widget.add_to_plot_checkbox.setChecked(is_plotted_anywhere)
        primary_widget.add_to_plot_checkbox.blockSignals(False)

    def clear_all_plotted_arrays(self):
        """Clear all plotted arrays for all widgets."""
        for widget in self.plotter_widget_list:
            widget.clear_all_plotted_arrays()


@switch_to_matplotlib_qt_backend
def launch_array_1d_plotter(
    array_data: Union[List[np.ndarray], np.ndarray],
    extra_title_strings_list: Optional[List[str]] = None,
    wait_until_closed: bool = False,
) -> Array1DPlotterWidget:
    """
    Launch the 1D array plotter GUI.

    Parameters
    ----------
    array_data : Union[List[np.ndarray], np.ndarray]
        Either a list of 1D arrays to plot, or a 2D array where the first
        dimension is indexed along (each row becomes a 1D array).
    extra_title_strings_list : Optional[List[str]], optional
        List of strings to append to the title for each array; must have the
        same length as the number of arrays.
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

        # Using 2D array with extra titles
        data_2d = np.random.randn(5, 100)  # 5 arrays of length 100
        titles = [" - Sine Wave", " - Cosine Wave", " - Random 1", " - Random 2", " - Random 3"]
        gui = launch_array_1d_plotter(data_2d, extra_title_strings_list=titles)
    """
    app = QApplication.instance() or QApplication([])
    gui = Array1DPlotterWidget(array_data, extra_title_strings_list=extra_title_strings_list)
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()

    if wait_until_closed:
        app.exec_()

    return gui


@switch_to_matplotlib_qt_backend
def launch_linked_array_1d_plotter(
    array_data_list: List[Union[List[np.ndarray], np.ndarray]],
    plot_titles: Optional[List[str]] = None,
    extra_title_strings_list: Optional[List[List[str]]] = None,
    n_rows: Optional[int] = None,
    wait_until_closed: bool = False,
) -> LinkedArray1DPlotterWidget:
    """
    Launch multiple linked 1D array plotter GUIs.

    This function creates multiple Array1DPlotterWidget instances with linked
    index selectors, allowing for easy comparison of multiple sets of 1D arrays.

    Parameters
    ----------
    array_data_list : List[Union[List[np.ndarray], np.ndarray]]
        List where each entry is either a list of 1D numpy arrays or a 2D numpy array.
        Each entry will be used to create one Array1DPlotterWidget.
    plot_titles : Optional[List[str]], optional
        List of titles to display above each plot. Must have the same length as
        array_data_list if provided.
    extra_title_strings_list : Optional[List[List[str]]], optional
        List of lists of strings to append to titles. The outer list corresponds to
        each Array1DPlotterWidget, and the inner lists are the extra title strings
        for each array within that widget.
    n_rows : Optional[int], optional
        Number of rows in the grid layout. If None, defaults to 1 row.
    wait_until_closed : bool, optional
        If True, the application starts a blocking call until the GUI window
        is closed. Default is False.

    Returns
    -------
    LinkedArray1DPlotterWidget
        The created linked widget instance.

    Example
    -------
    Create and launch linked plotters for comparing different datasets::

        # Create sample data
        x = np.linspace(0, 2*np.pi, 100)
        dataset1 = [np.sin(x), np.cos(x), np.tan(x/2)]
        dataset2 = [np.sin(2*x), np.cos(2*x), np.tan(x)]

        # Launch linked plotters
        gui = launch_linked_array_1d_plotter([dataset1, dataset2])

        # With plot titles and custom array titles
        plot_titles = ["Dataset 1", "Dataset 2"]
        titles1 = [" - Sin", " - Cos", " - Tan/2"]
        titles2 = [" - Sin(2x)", " - Cos(2x)", " - Tan"]
        gui = launch_linked_array_1d_plotter(
            [dataset1, dataset2],
            plot_titles=plot_titles,
            extra_title_strings_list=[titles1, titles2],
            n_rows=2
        )
    """
    app = QApplication.instance() or QApplication([])
    gui = LinkedArray1DPlotterWidget(
        array_data_list=array_data_list,
        plot_titles=plot_titles,
        extra_title_strings_list=extra_title_strings_list,
        n_rows=n_rows,
    )
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()

    if wait_until_closed:
        app.exec_()

    return gui
