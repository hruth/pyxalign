"""
Interactive ROI selection widget for 3D arrays.

This module provides tools for interactively selecting rectangular regions
of interest in 3D array data using click-and-drag mouse interaction.
"""

from typing import Optional, Sequence, Union

import numpy as np
import copy
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QApplication,
    QSpinBox,
    QGroupBox,
    QGridLayout,
    QPushButton,
)

from pyxalign.api.options.roi import ROIOptions, ROIType, RectangularROIOptions
from pyxalign.api.options.transform import CropOptions
from pyxalign.api.options_utils import print_options
import pyxalign.data_structures.projections as p
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper
from pyxalign.interactions.viewers.base import ArrayViewer
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.transformations.helpers import force_rectangular_roi_in_bounds


class ROISelector(QWidget):
    """
    Interactive widget for selecting rectangular regions of interest in 3D arrays.

    The ROI is shared across all frames of the 3D array and parameterized by
    center coordinates and extents in pixel units. Horizontal and vertical offsets
    are relative to the center of the image.

    Signals
    -------
    roi_changed : ROIOptions
        Emitted when ROI parameters change, containing updated ROIOptions object.
    """

    roi_changed = pyqtSignal(object)  # Will emit ROIOptions object

    def __init__(
        self,
        array3d: np.ndarray,
        roi_options: Optional[ROIOptions] = None,
        sort_idx: Optional[Sequence] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the ROI selector widget.

        Args:
            array3d: 3D numpy array to display and select ROI from
            roi_options: Optional ROI configuration. If None, uses default values.
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
        if roi_options is None:
            self.roi_options = ROIOptions()
        else:
            self.roi_options = copy.deepcopy(roi_options)

        # convert defaults to length of array
        if self.roi_options.rectangle.horizontal_range is None:
            # update None to array size
            self.roi_options.rectangle.horizontal_range = array3d.shape[2]
        if self.roi_options.rectangle.vertical_range is None:
            self.roi_options.rectangle.vertical_range = array3d.shape[1]

        # Initialize ArrayViewer
        self.array_viewer = ArrayViewer(array3d, sort_idx=sort_idx)

        # Initialize ROI graphics item and spinboxes
        self.roi_item = None
        self.spinboxes = {}
        self._updating_from_graphics = False  # Flag to prevent recursive updates

        # Setup UI and ROI graphics
        self.setup_ui()
        self.setup_roi_graphics()

        # Set window properties
        self.setWindowTitle("ROI Selector")
        self.resize(900, 750)

    def setup_ui(self):
        """Setup the widget layout."""
        layout = QVBoxLayout()

        # Main array viewer with ROI overlay
        layout.addWidget(self.array_viewer)

        # ROI parameter display
        roi_info = self.create_roi_info_display()
        layout.addWidget(roi_info)

        self.setLayout(layout)

    def create_roi_info_display(self):
        """Create ROI parameter controls with spinboxes for each attribute."""
        # Create group box for ROI controls
        roi_group = QGroupBox("ROI Parameters")
        roi_group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")

        # Create grid layout for organized parameter display
        grid_layout = QGridLayout()

        # Set reasonable ranges based on array dimensions
        max_x = self.array3d.shape[2]
        max_y = self.array3d.shape[1]
        center_x = max_x // 2
        center_y = max_y // 2

        # Create spinboxes for each ROI parameter
        self.spinboxes = {}

        spinbox_max_val = int(1e7)
        # Horizontal offset (relative to center)
        grid_layout.addWidget(QLabel("Horizontal Offset (from center):"), 0, 0)
        self.spinboxes["horizontal_offset"] = QSpinBox()
        self.spinboxes["horizontal_offset"].setRange(-spinbox_max_val, spinbox_max_val)
        self.spinboxes["horizontal_offset"].setValue(self.roi_options.rectangle.horizontal_offset)
        self.spinboxes["horizontal_offset"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["horizontal_offset"], 0, 1)

        # Vertical offset (relative to center)
        grid_layout.addWidget(QLabel("Vertical Offset (from center):"), 1, 0)
        self.spinboxes["vertical_offset"] = QSpinBox()
        self.spinboxes["vertical_offset"].setRange(-spinbox_max_val, spinbox_max_val)
        self.spinboxes["vertical_offset"].setValue(self.roi_options.rectangle.vertical_offset)
        self.spinboxes["vertical_offset"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["vertical_offset"], 1, 1)

        # Horizontal range (width)
        grid_layout.addWidget(QLabel("Horizontal Range (Width):"), 0, 2)
        self.spinboxes["horizontal_range"] = QSpinBox()
        self.spinboxes["horizontal_range"].setRange(1, spinbox_max_val)
        self.spinboxes["horizontal_range"].setValue(self.roi_options.rectangle.horizontal_range)
        self.spinboxes["horizontal_range"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["horizontal_range"], 0, 3)

        # Vertical range (height)
        grid_layout.addWidget(QLabel("Vertical Range (Height):"), 1, 2)
        self.spinboxes["vertical_range"] = QSpinBox()
        self.spinboxes["vertical_range"].setMinimum(1)
        self.spinboxes["vertical_range"].setRange(1, spinbox_max_val)
        self.spinboxes["vertical_range"].setValue(self.roi_options.rectangle.vertical_range)
        self.spinboxes["vertical_range"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["vertical_range"], 1, 3)

        # Style the spinboxes
        for spinbox in self.spinboxes.values():
            spinbox.setStyleSheet("QSpinBox { font-size: 12pt; }")
            spinbox.setMinimumWidth(80)

        # Style the labels
        for i in range(grid_layout.count()):
            item = grid_layout.itemAt(i)
            if item and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("QLabel { font-size: 12pt; }")

        roi_group.setLayout(grid_layout)
        return roi_group

    def setup_roi_graphics(self):
        """Initialize the pyqtgraph ROI item with parameters from roi_options."""
        rect_opts = self.roi_options.rectangle

        # # Handle default values (0) by setting reasonable defaults based on array size
        # if rect_opts.horizontal_range == 0 or rect_opts.vertical_range == 0:
        #     # Set default size to 1/4 of the image dimensions
        #     default_width = max(50, self.array3d.shape[2] // 4)
        #     default_height = max(50, self.array3d.shape[1] // 4)

        #     if rect_opts.horizontal_range == 0:
        #         rect_opts.horizontal_range = default_width
        #     if rect_opts.vertical_range == 0:
        #         rect_opts.vertical_range = default_height
                # Handle default values (0) by setting reasonable defaults based on array size
        # Set default size to 1/4 of the image dimensions
        if rect_opts.horizontal_range is None:
            default_width = max(50, self.array3d.shape[2] // 4)
            if rect_opts.horizontal_range == 0:
                rect_opts.horizontal_range = default_width
        if rect_opts.vertical_range is None:
            default_height = max(50, self.array3d.shape[1] // 4)
            if rect_opts.vertical_range == 0:
                rect_opts.vertical_range = default_height

        # Calculate image center
        image_center_x = self.array3d.shape[2] // 2
        image_center_y = self.array3d.shape[1] // 2

        # Convert from relative offset to absolute center position
        # Offsets are relative to image center, so add them to image center
        absolute_center_x = image_center_x + rect_opts.horizontal_offset
        absolute_center_y = image_center_y + rect_opts.vertical_offset

        # Convert from center/extent to position/size for pg.RectROI
        # pg.RectROI expects [x, y] position (top-left) and [width, height] size
        pos_x = absolute_center_x - rect_opts.horizontal_range // 2
        pos_y = absolute_center_y - rect_opts.vertical_range // 2

        self.roi_item = pg.RectROI(
            pos=[pos_x, pos_y],
            size=[rect_opts.horizontal_range, rect_opts.vertical_range],
            pen=pg.mkPen(color="r", width=2),  # Red outline, 2px width
            rotatable=False,
            scaleSnap=True,
            translateSnap=True,
        )
        # add scale handle in all corners
        # Top-left corner (resize from top-left, bottom-right stays fixed)
        self.roi_item.addScaleHandle([0, 0], [1, 1])
        
        # Top-right corner (resize from top-right, bottom-left stays fixed)
        self.roi_item.addScaleHandle([1, 0], [0, 1])
        
        # Bottom-left corner (resize from bottom-left, top-right stays fixed)
        self.roi_item.addScaleHandle([0, 1], [1, 0])
        
        # Bottom-right corner (resize from bottom-right, top-left stays fixed)
        self.roi_item.addScaleHandle([1, 1], [0, 0])

        # Add to the ArrayViewer's plot
        self.array_viewer.plot_item.addItem(self.roi_item)

        # Connect ROI change signal
        self.roi_item.sigRegionChanged.connect(self.on_roi_changed)

        # Initial update of spinboxes to match the ROI
        self.update_spinboxes_from_roi()

    def on_roi_changed(self):
        """Update ROIOptions when user modifies ROI graphics item."""
        if self._updating_from_graphics:
            return

        self._updating_from_graphics = True

        pos = self.roi_item.pos()
        size = self.roi_item.size()

        # Convert from position/size back to absolute center coordinates
        absolute_center_x = int(pos[0] + size[0] / 2)
        absolute_center_y = int(pos[1] + size[1] / 2)
        extent_x = int(size[0])
        extent_y = int(size[1])

        # Calculate image center
        image_center_x = self.array3d.shape[2] // 2
        image_center_y = self.array3d.shape[1] // 2

        # Convert absolute center to relative offset from image center
        relative_offset_x = absolute_center_x - image_center_x
        relative_offset_y = absolute_center_y - image_center_y

        # Update the ROIOptions object with relative offsets
        self.roi_options.rectangle.horizontal_offset = relative_offset_x
        self.roi_options.rectangle.vertical_offset = relative_offset_y
        self.roi_options.rectangle.horizontal_range = extent_x
        self.roi_options.rectangle.vertical_range = extent_y

        # Update the spinboxes without triggering their signals
        self.update_spinboxes_from_roi()

        # Emit the updated options
        self.roi_changed.emit(self.roi_options)

        self._updating_from_graphics = False

    def on_spinbox_changed(self):
        """Update ROI graphics when user modifies spinbox values."""
        if self._updating_from_graphics:
            return

        # Update ROI options from spinbox values
        self.roi_options.rectangle.horizontal_offset = self.spinboxes["horizontal_offset"].value()
        self.roi_options.rectangle.vertical_offset = self.spinboxes["vertical_offset"].value()
        self.roi_options.rectangle.horizontal_range = self.spinboxes["horizontal_range"].value()
        self.roi_options.rectangle.vertical_range = self.spinboxes["vertical_range"].value()

        # Update the graphics item
        self.update_roi_graphics_from_options()

        # Emit the updated options
        self.roi_changed.emit(self.roi_options)

    def update_spinboxes_from_roi(self):
        """Update spinbox values from current ROI options without triggering signals."""
        rect = self.roi_options.rectangle

        # Temporarily block signals to prevent recursive updates
        for spinbox in self.spinboxes.values():
            spinbox.blockSignals(True)

        self.spinboxes["horizontal_offset"].setValue(rect.horizontal_offset)
        self.spinboxes["vertical_offset"].setValue(rect.vertical_offset)
        self.spinboxes["horizontal_range"].setValue(rect.horizontal_range)
        self.spinboxes["vertical_range"].setValue(rect.vertical_range)

        # Re-enable signals
        for spinbox in self.spinboxes.values():
            spinbox.blockSignals(False)

    def update_roi_graphics_from_options(self):
        """Update the ROI graphics item from current ROI options."""
        rect_opts = self.roi_options.rectangle

        # Calculate image center
        image_center_x = self.array3d.shape[2] // 2
        image_center_y = self.array3d.shape[1] // 2

        # Convert from relative offset to absolute center position
        absolute_center_x = image_center_x + rect_opts.horizontal_offset
        absolute_center_y = image_center_y + rect_opts.vertical_offset

        # Convert to top-left position for pg.RectROI
        pos_x = absolute_center_x - rect_opts.horizontal_range // 2
        pos_y = absolute_center_y - rect_opts.vertical_range // 2

        # Temporarily disconnect signal to avoid recursive updates
        self.roi_item.sigRegionChanged.disconnect(self.on_roi_changed)

        # Update ROI item
        self.roi_item.setPos([pos_x, pos_y])
        self.roi_item.setSize([rect_opts.horizontal_range, rect_opts.vertical_range])

        # Reconnect signal
        self.roi_item.sigRegionChanged.connect(self.on_roi_changed)

    def get_roi_options(self) -> ROIOptions:
        """
        Get the current ROI options.

        Returns:
            Current ROIOptions object with updated parameters
        """
        return self.roi_options

    def set_roi_options(self, roi_options: ROIOptions):
        """
        Set new ROI options and update the display.

        Args:
            roi_options: New ROI configuration to apply
        """
        self.roi_options = roi_options

        # Update the graphics item using the same logic as update_roi_graphics_from_options
        self.update_roi_graphics_from_options()

        # Update spinboxes
        self.update_spinboxes_from_roi()

        # Emit change signal
        self.roi_changed.emit(self.roi_options)


class MaskFromROISelector(QWidget):
    masks_created = pyqtSignal()

    def __init__(self, projections: "p.Projections", parent: Optional[QWidget] = None):
        """
        Widget for choosing masks by selecting the ROI interactly.

        Args:
            projections (Projections): Projections object to create masks for.
            parent: Optional parent widget
        """
        super().__init__(parent)

        self.projections = projections
        self.roi_selector = ROISelector(
            array3d=projections.data,
            roi_options=projections.options.masks_from_roi,
            sort_idx=np.argsort(projections.angles),
        )
        self.finish_button = QPushButton(text="Select and Finish")
        self.finish_button.clicked.connect(self.finish)
        self.setup_ui()

    def finish(self):
        self.projections.options.masks_from_roi = self.roi_selector.roi_options
        wrapped_func = loading_bar_wrapper("Creating masks from selected area...", True)(
            self.projections.get_masks_from_roi_selection
        )
        wrapped_func()
        self.projections.get_masks_from_roi_selection()
        self.masks_created.emit()
        self.close()

    def setup_ui(self):
        """Setup the widget layout."""
        layout = QVBoxLayout()

        # roi selection
        layout.addWidget(self.roi_selector)
        # finish button
        layout.addWidget(self.finish_button, alignment=Qt.AlignRight)
        self.finish_button.setStyleSheet("background-color: blue; color: white;")

        self.setLayout(layout)


class GetBoxBoundsFromROISelector(QWidget):
    rectangular_roi_selected = pyqtSignal()

    def __init__(
        self,
        projections: "p.Projections",
        options: Union[CropOptions, RectangularROIOptions],
        parent: Optional[QWidget] = None,
    ):
        """
        Widget for returning an instance of `CropOptions` or 
        `RectangularROIOptions` by interactively selecting its values.

        Args:
            projections (Projections): Projections object to create the 
                options instance for.
            options: Options to initialize the ROISelector with. The 
                type of `options` dictates the type of options that will
                be created.
        """
        super().__init__(parent)

        self.return_type = options.__class__

        roi_options = ROIOptions(shape=ROIType.RECTANGULAR)
        if options.__class__.__qualname__ == "CropOptions":
            # convert CropOptions to ROIOptions
            roi_options.rectangle.horizontal_range = options.horizontal_range
            roi_options.rectangle.vertical_range = options.vertical_range
            roi_options.rectangle.horizontal_offset = options.horizontal_offset
            roi_options.rectangle.vertical_offset = options.vertical_offset
        elif options.__class__.__qualname__ == "RectangularROIOptions":
            roi_options.rectangle = options
        else:
            raise ValueError

        self.projections = projections
        self.roi_selector = ROISelector(
            array3d=projections.data,
            roi_options=roi_options,
            sort_idx=np.argsort(projections.angles),
        )
        self.finish_button = QPushButton(text="Select and Finish")
        self.finish_button.clicked.connect(self.finish)
        self.setup_ui()

    def finish(self):
        self.roi_selector.roi_options.rectangle = force_rectangular_roi_in_bounds(
            self.roi_selector.roi_options.rectangle,
            array_2d_size=self.projections.data.shape[1:],
        )

        self.options = self.return_type(
            horizontal_range=self.roi_selector.roi_options.rectangle.horizontal_range,
            vertical_range=self.roi_selector.roi_options.rectangle.vertical_range,
            horizontal_offset=self.roi_selector.roi_options.rectangle.horizontal_offset,
            vertical_offset=self.roi_selector.roi_options.rectangle.vertical_offset,
        )
        if self.return_type.__qualname__ == "CropOptions":
            self.options.enabled = True
        self.rectangular_roi_selected.emit()

    def setup_ui(self):
        """Setup the widget layout."""
        layout = QVBoxLayout()

        # roi selection
        layout.addWidget(self.roi_selector)
        # finish button
        layout.addWidget(self.finish_button, alignment=Qt.AlignRight)
        self.finish_button.setStyleSheet("background-color: blue; color: white;")

        self.setLayout(layout)


@switch_to_matplotlib_qt_backend
def launch_mask_selection_from_roi(
    projections: "p.Projections",
    wait_until_closed: bool = False,
) -> MaskFromROISelector:
    """
    Launch the ROI selector GUI for interactively selecting the mask
    from setting a region of interest. The masks will be generated and
    stored in the `masks` attribute of the `projections` input argument.

    Args:
        projections (Projections): Projections object to create masks for.
        wait_until_closed (bool): If True, blocks until the GUI window is closed

    Returns:
        MaskFromROISelector widget instance

    Example:
        Launch GUI for a phase projections in a 
        task::

            mask_gui = pyxalign.gui.launch_mask_selection_from_roi(task.phase_projections)

        Next, view the masks using the `ProjectionViewer` gui::

            projections_gui = pyxalign.gui.launch_projection_viewer(task.phase_projections)
    """
    app = QApplication.instance() or QApplication([])
    gui = MaskFromROISelector(projections)
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui


@switch_to_matplotlib_qt_backend
def launch_crop_window_selection(
    projections: "p.Projections",
    crop_options: Optional[CropOptions] = None,
) -> CropOptions:
    """
    Launch a GUI that lets you interactively select a region of 
    interest. This function returns an instance of `CropOptions` with 
    the values selected in the GUI.

    Args:
        projections (Projections): Projections object that will be 
            cropped.

    Returns:
        CropOptions instance with selected values.

    Example:
        Set the crop options for projection-matching 
        alignment::

            crop_options = pyxalign.gui.launch_crop_window_selection(task.phase_projections)
            task.options.projection_matching.crop = crop_options
    """
    app = QApplication.instance() or QApplication([])
    if crop_options is None:
        crop_options = CropOptions()
    gui = GetBoxBoundsFromROISelector(projections, crop_options)

    # Define a slot to handle the signal containing the
    # loaded data
    result = {}

    def on_crop_region_selected():
        result["data"] = gui.options
        app.quit()

    gui.rectangular_roi_selected.connect(on_crop_region_selected)

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
