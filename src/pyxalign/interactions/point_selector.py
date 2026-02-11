"""
Interactive point selection widget for 2D images.

This module provides tools for interactively clicking points on a 2D image
and displaying the selected coordinates in spinboxes.
"""

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QGroupBox,
    QGridLayout,
    QPushButton,
)


class PointSelector(QWidget):
    """
    Interactive widget for selecting a point on a 2D image.

    The user can click anywhere on the image to select a point. The coordinates
    are displayed in spinboxes at the bottom of the window and can also be
    manually edited.

    Signals
    -------
    point_changed : tuple
        Emitted when point coordinates change, containing (x, y) tuple.
    point_selected : tuple
        Emitted when user clicks "Select and Finish", containing (x, y) tuple.
    """

    point_changed = pyqtSignal(object)  # Will emit (x, y) tuple
    point_selected = pyqtSignal(object)  # Will emit (x, y) tuple on finish

    def __init__(
        self,
        image: np.ndarray,
        initial_point: Optional[tuple] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the point selector widget.

        Args:
            image: 2D numpy array to display
            initial_point: Optional (x, y) tuple for initial point position.
                If None, defaults to center of image.
            parent: Optional parent widget
        """
        super().__init__(parent)

        # Store image and validate
        if image is None:
            raise ValueError("image cannot be None")
        if image.ndim != 2:
            raise ValueError(f"image must be 2-dimensional, got {image.ndim}D")

        self.image = image

        # Set initial point
        if initial_point is None:
            self.point_x = image.shape[1] // 2
            self.point_y = image.shape[0] // 2
        else:
            self.point_x = int(initial_point[0])
            self.point_y = int(initial_point[1])

        # Initialize graphics items and spinboxes
        self.plot_widget = None
        self.image_item = None
        self.point_item = None
        self.spinboxes = {}
        self.finish_button = None
        self._updating_from_click = False  # Flag to prevent recursive updates

        # Setup UI
        self.setup_ui()
        self.setup_graphics()

        # Set window properties
        self.setWindowTitle("Point Selector")
        self.resize(800, 700)

    def setup_ui(self):
        """Setup the widget layout."""
        layout = QVBoxLayout()

        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        layout.addWidget(self.plot_widget)

        # Point coordinate display
        point_info = self.create_point_info_display()
        layout.addWidget(point_info)

        # Finish button
        self.finish_button = QPushButton(text="Select and Finish")
        self.finish_button.clicked.connect(self.finish)
        self.finish_button.setStyleSheet("background-color: blue; color: white;")
        layout.addWidget(self.finish_button, alignment=Qt.AlignRight)

        self.setLayout(layout)

    def create_point_info_display(self):
        """Create point coordinate controls with spinboxes."""
        # Create group box for point coordinates
        point_group = QGroupBox("Point Coordinates")
        point_group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")

        # Create grid layout for organized parameter display
        grid_layout = QGridLayout()

        # Set ranges based on image dimensions
        max_x = self.image.shape[1] - 1
        max_y = self.image.shape[0] - 1

        # Create spinboxes for x and y coordinates
        self.spinboxes = {}

        # X coordinate
        grid_layout.addWidget(QLabel("X Position:"), 0, 0)
        self.spinboxes["x"] = QSpinBox()
        self.spinboxes["x"].setRange(0, max_x)
        self.spinboxes["x"].setValue(self.point_x)
        self.spinboxes["x"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["x"], 0, 1)

        # Y coordinate
        grid_layout.addWidget(QLabel("Y Position:"), 0, 2)
        self.spinboxes["y"] = QSpinBox()
        self.spinboxes["y"].setRange(0, max_y)
        self.spinboxes["y"].setValue(self.point_y)
        self.spinboxes["y"].valueChanged.connect(self.on_spinbox_changed)
        grid_layout.addWidget(self.spinboxes["y"], 0, 3)

        # Style the spinboxes
        for spinbox in self.spinboxes.values():
            spinbox.setStyleSheet("QSpinBox { font-size: 12pt; }")
            spinbox.setMinimumWidth(100)

        # Style the labels
        for i in range(grid_layout.count()):
            item = grid_layout.itemAt(i)
            if item and isinstance(item.widget(), QLabel):
                item.widget().setStyleSheet("QLabel { font-size: 12pt; }")

        point_group.setLayout(grid_layout)
        return point_group

    def setup_graphics(self):
        """Initialize the pyqtgraph image and point marker."""
        # Display the image
        self.image_item = pg.ImageItem()
        self.image_item.setImage(self.image.T)
        self.plot_widget.addItem(self.image_item)

        # Create a scatter plot item for the point marker
        self.point_item = pg.ScatterPlotItem(
            size=15,
            pen=pg.mkPen(color="r", width=2),
            brush=pg.mkBrush(255, 0, 0, 120),
        )
        self.plot_widget.addItem(self.point_item)

        # Set initial point position
        self.update_point_graphics()

        # Connect mouse click signal
        self.image_item.mouseClickEvent = self.on_image_clicked

    def on_image_clicked(self, event):
        """Handle mouse click on the image."""
        if event.button() == Qt.LeftButton:
            self._updating_from_click = True

            # Get click position in image coordinates
            pos = event.pos()
            x = int(pos.x())
            y = int(pos.y())

            # Clamp to image bounds
            x = max(0, min(x, self.image.shape[1] - 1))
            y = max(0, min(y, self.image.shape[0] - 1))

            # Update point coordinates
            self.point_x = x
            self.point_y = y

            # Update graphics and spinboxes
            self.update_point_graphics()
            self.update_spinboxes_from_point()

            # Emit signal
            self.point_changed.emit((self.point_x, self.point_y))

            self._updating_from_click = False

            event.accept()

    def on_spinbox_changed(self):
        """Update point position when user modifies spinbox values."""
        if self._updating_from_click:
            return

        # Update point coordinates from spinbox values
        self.point_x = self.spinboxes["x"].value()
        self.point_y = self.spinboxes["y"].value()

        # Update the graphics item
        self.update_point_graphics()

        # Emit signal
        self.point_changed.emit((self.point_x, self.point_y))

    def update_spinboxes_from_point(self):
        """Update spinbox values from current point coordinates without triggering signals."""
        # Temporarily block signals to prevent recursive updates
        for spinbox in self.spinboxes.values():
            spinbox.blockSignals(True)

        self.spinboxes["x"].setValue(self.point_x)
        self.spinboxes["y"].setValue(self.point_y)

        # Re-enable signals
        for spinbox in self.spinboxes.values():
            spinbox.blockSignals(False)

    def update_point_graphics(self):
        """Update the point marker position on the image."""
        self.point_item.setData(
            x=[self.point_x],
            y=[self.point_y],
        )

    def get_point(self) -> tuple:
        """
        Get the current point coordinates.

        Returns:
            Tuple of (x, y) coordinates
        """
        return (self.point_x, self.point_y)

    def set_point(self, x: int, y: int):
        """
        Set new point coordinates and update the display.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        # Clamp to image bounds
        x = max(0, min(x, self.image.shape[1] - 1))
        y = max(0, min(y, self.image.shape[0] - 1))

        self.point_x = x
        self.point_y = y

        # Update graphics and spinboxes
        self.update_point_graphics()
        self.update_spinboxes_from_point()

        # Emit signal
        self.point_changed.emit((self.point_x, self.point_y))

    def finish(self):
        """Emit point_selected signal and close the widget."""
        self.point_selected.emit((self.point_x, self.point_y))
        self.close()


def launch_point_selector(
    image: np.ndarray,
    initial_point: Optional[tuple] = None,
) -> tuple:
    """
    Launch the point selector GUI for interactively selecting a point on a 2D image.

    The GUI will remain open until the user clicks "Select and Finish", at which
    point it closes and returns the selected coordinates.

    Args:
        image: 2D numpy array to display
        initial_point: Optional (x, y) tuple for initial point position

    Returns:
        Tuple of (x, y) coordinates of the selected point

    Example:
        Launch GUI and get the selected point::

            x, y = launch_point_selector(my_image)
            print(f"Selected point: ({x}, {y})")
    """
    app = QApplication.instance() or QApplication([])
    gui = PointSelector(image, initial_point=initial_point)

    # Define a slot to handle the signal containing the selected point
    result = {}

    def on_point_selected(point):
        result["point"] = point
        app.quit()

    gui.point_selected.connect(on_point_selected)

    gui.show()
    app.exec()
    gui.close()

    if result:
        return result["point"]
    else:
        return None
