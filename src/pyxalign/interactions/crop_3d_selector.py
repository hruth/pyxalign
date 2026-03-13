"""
Interactive 3D cropping tool with bounding box visualization.

This module provides a tool for interactively cropping 3D arrays with
per-axis control and interactive bounding box visualization that updates
as the viewing axis changes.
"""

from typing import Optional, Sequence
import numpy as np
import copy
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSpinBox,
    QGroupBox,
    QGridLayout,
    QPushButton,
    QApplication,
)

from pyxalign.api.options.transform import Crop3DOptions
from pyxalign.api.options_utils import print_options
from pyxalign.interactions.viewers.base import ArrayViewer
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend


class Crop3DSelector(QWidget):
    """
    Interactive widget for 3D cropping with per-axis controls.

    This widget allows users to specify crop regions in all three dimensions
    using both spinboxes and an interactive bounding box. The bounding box
    updates to show the correct projection for the currently selected axis.

    Signals
    -------
    crop_changed : Crop3DOptions
        Emitted when crop parameters change, containing updated Crop3DOptions object.
    """

    crop_changed = pyqtSignal(object)  # Will emit Crop3DOptions object

    def __init__(
        self,
        array3d: np.ndarray,
        crop_options: Optional[Crop3DOptions] = None,
        sort_idx: Optional[Sequence] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the 3D crop selector widget.

        Args:
            array3d: 3D numpy array to display and crop
            crop_options: Optional crop configuration. If None, uses default values.
            sort_idx: Optional sorting indices for array display
            parent: Optional parent widget
        """
        super().__init__(parent)

        # Store array and validate
        if array3d is None:
            raise ValueError("array3d cannot be None")
        if array3d.ndim != 3:
            raise ValueError(f"array3d must be 3-dimensional, got {array3d.ndim}D")

        self.array3d = array3d

        # Use provided options or create default
        if crop_options is None:
            self.crop_options = Crop3DOptions()
        else:
            self.crop_options = copy.deepcopy(crop_options)

        # Convert defaults to length of array
        if self.crop_options.horizontal_range is None:
            self.crop_options.horizontal_range = array3d.shape[2]
        if self.crop_options.vertical_range is None:
            self.crop_options.vertical_range = array3d.shape[1]
        if self.crop_options.depth_range is None:
            self.crop_options.depth_range = array3d.shape[0]

        # Initialize ArrayViewer
        self.array_viewer = ArrayViewer(array3d, sort_idx=sort_idx, hide_axis_controls=False)

        # Initialize ROI graphics item and spinboxes
        self.roi_item = None
        self.spinboxes = {}
        self._updating_from_graphics = False  # Flag to prevent recursive updates

        # Setup UI and ROI graphics
        self.setup_ui()
        self.setup_roi_graphics()

        # Connect to axis change events
        self.array_viewer.prev_axis_button.clicked.connect(self.on_axis_changed)
        self.array_viewer.next_axis_button.clicked.connect(self.on_axis_changed)

        # Set window properties
        self.setWindowTitle("3D Crop Selector")
        self.resize(900, 800)

    def setup_ui(self):
        """Setup the widget layout."""
        layout = QVBoxLayout()

        # Main array viewer with ROI overlay
        layout.addWidget(self.array_viewer)

        # Crop parameter display
        crop_info = self.create_crop_info_display()
        layout.addWidget(crop_info)

        self.setLayout(layout)

    def create_crop_info_display(self):
        """Create crop parameter controls with spinboxes for each axis."""
        # Create group box for crop controls
        crop_group = QGroupBox("3D Crop Parameters")
        crop_group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")

        # Create grid layout for organized parameter display
        grid_layout = QGridLayout()

        # Set reasonable ranges based on array dimensions
        spinbox_max_val = int(1e7)

        # Create spinboxes for each crop parameter
        self.spinboxes = {}

        # Horizontal (X-axis) parameters
        grid_layout.addWidget(QLabel("Horizontal Center:"), 0, 0)
        self.spinboxes["horizontal_offset"] = QSpinBox()
        self.spinboxes["horizontal_offset"].setRange(-spinbox_max_val, spinbox_max_val)
        self.spinboxes["horizontal_offset"].setValue(self.crop_options.horizontal_offset)
        self.spinboxes["horizontal_offset"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["horizontal_offset"], 0, 1)

        grid_layout.addWidget(QLabel("Horizontal Width:"), 0, 2)
        self.spinboxes["horizontal_range"] = QSpinBox()
        self.spinboxes["horizontal_range"].setRange(1, spinbox_max_val)
        self.spinboxes["horizontal_range"].setValue(self.crop_options.horizontal_range)
        self.spinboxes["horizontal_range"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["horizontal_range"], 0, 3)

        # Vertical (Y-axis) parameters
        grid_layout.addWidget(QLabel("Vertical Center:"), 1, 0)
        self.spinboxes["vertical_offset"] = QSpinBox()
        self.spinboxes["vertical_offset"].setRange(-spinbox_max_val, spinbox_max_val)
        self.spinboxes["vertical_offset"].setValue(self.crop_options.vertical_offset)
        self.spinboxes["vertical_offset"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["vertical_offset"], 1, 1)

        grid_layout.addWidget(QLabel("Vertical Width:"), 1, 2)
        self.spinboxes["vertical_range"] = QSpinBox()
        self.spinboxes["vertical_range"].setRange(1, spinbox_max_val)
        self.spinboxes["vertical_range"].setValue(self.crop_options.vertical_range)
        self.spinboxes["vertical_range"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["vertical_range"], 1, 3)

        # Depth (Z-axis) parameters
        grid_layout.addWidget(QLabel("Depth Center:"), 2, 0)
        self.spinboxes["depth_offset"] = QSpinBox()
        self.spinboxes["depth_offset"].setRange(-spinbox_max_val, spinbox_max_val)
        self.spinboxes["depth_offset"].setValue(self.crop_options.depth_offset)
        self.spinboxes["depth_offset"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["depth_offset"], 2, 1)

        grid_layout.addWidget(QLabel("Depth Width:"), 2, 2)
        self.spinboxes["depth_range"] = QSpinBox()
        self.spinboxes["depth_range"].setRange(1, spinbox_max_val)
        self.spinboxes["depth_range"].setValue(self.crop_options.depth_range)
        self.spinboxes["depth_range"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["depth_range"], 2, 3)

        # Style the spinboxes
        for spinbox in self.spinboxes.values():
            spinbox.setStyleSheet("QSpinBox { font-size: 12pt; }")
            spinbox.setMinimumWidth(100)

        # Style the labels
        for i in range(grid_layout.count()):
            item = grid_layout.itemAt(i)
            if item and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("QLabel { font-size: 12pt; }")

        crop_group.setLayout(grid_layout)
        return crop_group

    def setup_roi_graphics(self):
        """Initialize the pyqtgraph ROI item based on current viewing axis."""
        # Create the ROI item
        self.roi_item = pg.RectROI(
            pos=[0, 0],
            size=[100, 100],
            pen=pg.mkPen(color="r", width=2),  # Red outline, 2px width
            rotatable=False,
            scaleSnap=True,
            translateSnap=True,
        )

        # Add scale handles on all four corners
        self.roi_item.addScaleHandle([0, 0], [1, 1])  # Top-left
        self.roi_item.addScaleHandle([1, 0], [0, 1])  # Top-right
        self.roi_item.addScaleHandle([0, 1], [1, 0])  # Bottom-left
        self.roi_item.addScaleHandle([1, 1], [0, 0])  # Bottom-right

        # Add to the ArrayViewer's plot
        self.array_viewer.plot_item.addItem(self.roi_item)

        # Connect ROI change signal
        self.roi_item.sigRegionChanged.connect(self.on_roi_changed)

        # Update ROI to match current axis and crop options
        self.update_roi_for_current_axis()

    def get_axis_mapping(self):
        """
        Get the mapping between 3D crop parameters and 2D display coordinates
        for the current viewing axis.

        Returns:
            tuple: (horizontal_param, vertical_param, horizontal_range_param, vertical_range_param,
                   horizontal_shape, vertical_shape)
                   where params are strings like 'horizontal_offset', 'depth_offset', etc.
        """
        axis = self.array_viewer.options.slider_axis

        # axis 0: viewing down depth axis, showing vertical (Y) vs horizontal (X)
        # axis 1: viewing down vertical axis, showing depth (Z) vs horizontal (X)
        # axis 2: viewing down horizontal axis, showing depth (Z) vs vertical (Y)

        if axis == 0:
            # Viewing axis 0: X (horizontal) is left-right, Y (vertical) is up-down
            return (
                "horizontal_offset",
                "vertical_offset",
                "horizontal_range",
                "vertical_range",
                self.array3d.shape[2],
                self.array3d.shape[1],
            )
        elif axis == 1:
            # Viewing axis 1: X (horizontal) is left-right, Z (depth) is up-down
            return (
                "horizontal_offset",
                "depth_offset",
                "horizontal_range",
                "depth_range",
                self.array3d.shape[2],
                self.array3d.shape[0],
            )
        else:  # axis == 2
            # Viewing axis 2: Y (vertical) is left-right, Z (depth) is up-down
            return (
                "vertical_offset",
                "depth_offset",
                "vertical_range",
                "depth_range",
                self.array3d.shape[1],
                self.array3d.shape[0],
            )

    def update_roi_for_current_axis(self):
        """Update the ROI graphics to match the current axis view."""
        if self.roi_item is None:
            return

        # Get the axis mapping
        (
            h_offset_param,
            v_offset_param,
            h_range_param,
            v_range_param,
            h_shape,
            v_shape,
        ) = self.get_axis_mapping()

        # Get the crop parameters for this axis view
        h_offset = getattr(self.crop_options, h_offset_param)
        v_offset = getattr(self.crop_options, v_offset_param)
        h_range = getattr(self.crop_options, h_range_param)
        v_range = getattr(self.crop_options, v_range_param)

        # Calculate image centers for this view
        image_center_x = h_shape // 2
        image_center_y = v_shape // 2

        # Convert from relative offset to absolute center position
        absolute_center_x = image_center_x + h_offset
        absolute_center_y = image_center_y + v_offset

        # Convert to top-left position for pg.RectROI
        pos_x = absolute_center_x - h_range // 2
        pos_y = absolute_center_y - v_range // 2

        # Temporarily disconnect signal to avoid recursive updates
        self.roi_item.sigRegionChanged.disconnect(self.on_roi_changed)

        # Update ROI item
        self.roi_item.setPos([pos_x, pos_y])
        self.roi_item.setSize([h_range, v_range])

        # Reconnect signal
        self.roi_item.sigRegionChanged.connect(self.on_roi_changed)

    def on_roi_changed(self):
        """Update Crop3DOptions when user modifies ROI graphics item."""
        if self._updating_from_graphics:
            return

        self._updating_from_graphics = True

        pos = self.roi_item.pos()
        size = self.roi_item.size()

        # Get the axis mapping
        (
            h_offset_param,
            v_offset_param,
            h_range_param,
            v_range_param,
            h_shape,
            v_shape,
        ) = self.get_axis_mapping()

        # Convert from position/size back to absolute center coordinates
        absolute_center_x = int(pos[0] + size[0] / 2)
        absolute_center_y = int(pos[1] + size[1] / 2)
        extent_x = int(size[0])
        extent_y = int(size[1])

        # Calculate image centers
        image_center_x = h_shape // 2
        image_center_y = v_shape // 2

        # Convert absolute center to relative offset from image center
        relative_offset_x = absolute_center_x - image_center_x
        relative_offset_y = absolute_center_y - image_center_y

        # Update the Crop3DOptions object with relative offsets
        setattr(self.crop_options, h_offset_param, relative_offset_x)
        setattr(self.crop_options, v_offset_param, relative_offset_y)
        setattr(self.crop_options, h_range_param, extent_x)
        setattr(self.crop_options, v_range_param, extent_y)

        # Update the spinboxes without triggering their signals
        self.update_spinboxes_from_options()

        # Emit the updated options
        self.crop_changed.emit(self.crop_options)

        self._updating_from_graphics = False

    def on_spinbox_changed(self):
        """Update ROI graphics when user modifies spinbox values."""
        if self._updating_from_graphics:
            return

        # Update crop options from spinbox values
        self.crop_options.horizontal_offset = self.spinboxes["horizontal_offset"].value()
        self.crop_options.vertical_offset = self.spinboxes["vertical_offset"].value()
        self.crop_options.depth_offset = self.spinboxes["depth_offset"].value()
        self.crop_options.horizontal_range = self.spinboxes["horizontal_range"].value()
        self.crop_options.vertical_range = self.spinboxes["vertical_range"].value()
        self.crop_options.depth_range = self.spinboxes["depth_range"].value()

        # Update the graphics item
        self.update_roi_for_current_axis()

        # Emit the updated options
        self.crop_changed.emit(self.crop_options)

    def on_axis_changed(self):
        """Handle axis change events from the ArrayViewer."""
        # Update the ROI to show the correct bounding box for the new axis
        self.update_roi_for_current_axis()

    def update_spinboxes_from_options(self):
        """Update spinbox values from current crop options without triggering signals."""
        # Temporarily block signals to prevent recursive updates
        for spinbox in self.spinboxes.values():
            spinbox.blockSignals(True)

        self.spinboxes["horizontal_offset"].setValue(self.crop_options.horizontal_offset)
        self.spinboxes["vertical_offset"].setValue(self.crop_options.vertical_offset)
        self.spinboxes["depth_offset"].setValue(self.crop_options.depth_offset)
        self.spinboxes["horizontal_range"].setValue(self.crop_options.horizontal_range)
        self.spinboxes["vertical_range"].setValue(self.crop_options.vertical_range)
        self.spinboxes["depth_range"].setValue(self.crop_options.depth_range)

        # Re-enable signals
        for spinbox in self.spinboxes.values():
            spinbox.blockSignals(False)

    def get_crop_options(self) -> Crop3DOptions:
        """
        Get the current crop options.

        Returns:
            Current Crop3DOptions object with updated parameters
        """
        return self.crop_options

    def set_crop_options(self, crop_options: Crop3DOptions):
        """
        Set new crop options and update the display.

        Args:
            crop_options: New crop configuration to apply
        """
        self.crop_options = crop_options

        # Update the graphics item
        self.update_roi_for_current_axis()

        # Update spinboxes
        self.update_spinboxes_from_options()

        # Emit change signal
        self.crop_changed.emit(self.crop_options)

    def start(self):
        """Show the widget."""
        self.show()


class GetCrop3DOptionsFromSelector(QWidget):
    """
    Wrapper widget for returning Crop3DOptions after interactive selection.

    This widget provides a "Select and Finish" button that closes the
    selection window and returns the configured Crop3DOptions.
    """

    crop_3d_selected = pyqtSignal()

    def __init__(
        self,
        array3d: np.ndarray,
        crop_options: Optional[Crop3DOptions] = None,
        sort_idx: Optional[Sequence] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the wrapper widget.

        Args:
            array3d: 3D numpy array to crop
            crop_options: Optional initial crop configuration
            sort_idx: Optional sorting indices for array display
            parent: Optional parent widget
        """
        super().__init__(parent)

        if crop_options is None:
            crop_options = Crop3DOptions()

        self.crop_selector = Crop3DSelector(
            array3d=array3d,
            crop_options=crop_options,
            sort_idx=sort_idx,
        )

        self.finish_button = QPushButton(text="Select and Finish")
        self.finish_button.clicked.connect(self.finish)
        self.setup_ui()

        # Set window properties
        self.setWindowTitle("3D Crop Selection")
        self.resize(900, 850)

    def finish(self):
        """Handle the finish button click."""
        self.options = self.crop_selector.get_crop_options()
        self.options.enabled = True
        self.crop_3d_selected.emit()

    def setup_ui(self):
        """Setup the widget layout."""
        layout = QVBoxLayout()

        # Crop selector
        layout.addWidget(self.crop_selector)

        # Finish button
        layout.addWidget(self.finish_button, alignment=Qt.AlignRight)
        self.finish_button.setStyleSheet("background-color: blue; color: white;")

        self.setLayout(layout)


@switch_to_matplotlib_qt_backend
def launch_crop_3d_selector(
    array3d: np.ndarray,
    crop_options: Optional[Crop3DOptions] = None,
    sort_idx: Optional[Sequence] = None,
) -> Crop3DOptions:
    """
    Launch a GUI for interactively selecting a 3D crop region.

    This function displays an interactive widget where users can specify
    crop boundaries for all three dimensions using both spinboxes and
    an interactive bounding box that updates as the viewing axis changes.

    Args:
        array3d: 3D numpy array to crop
        crop_options: Optional initial crop configuration. If None, uses defaults.
        sort_idx: Optional sorting indices for array display

    Returns:
        Crop3DOptions instance with the selected crop parameters and enabled=True

    Example:
        Select crop options for a 3D reconstruction::

            crop_options = launch_crop_3d_selector(reconstruction_volume)
            cropped_volume = apply_crop_3d(reconstruction_volume, crop_options)
    """
    app = QApplication.instance() or QApplication([])

    if crop_options is None:
        crop_options = Crop3DOptions()

    gui = GetCrop3DOptionsFromSelector(
        array3d=array3d,
        crop_options=crop_options,
        sort_idx=sort_idx,
    )

    # Define a slot to handle the signal containing the selected options
    result = {}

    def on_crop_3d_selected():
        result["data"] = gui.options
        app.quit()

    gui.crop_3d_selected.connect(on_crop_3d_selected)

    gui.show()
    app.exec()
    gui.close()

    if result != {}:
        # Return the result after the app closes
        crop_options = result["data"]
        print_options(crop_options)
    else:
        crop_options = None

    return crop_options
