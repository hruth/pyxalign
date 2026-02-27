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

from pyxalign.interactions.viewers.base import ArrayViewer


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
        projections: Optional[np.ndarray] = None,
    ):
        """
        Initialize the point selector widget.

        Args:
            image: 2D numpy array to display (used as fallback or sum of projections)
            initial_point: Optional (x, y) tuple for initial point position.
                If None, defaults to center of image.
            parent: Optional parent widget
            projections: Optional 3D numpy array of projections. If provided,
                defaults to showing ArrayViewer with single projections.
        """
        super().__init__(parent)

        # Store image and validate
        if image is None:
            raise ValueError("image cannot be None")
        if image.ndim != 2:
            raise ValueError(f"image must be 2-dimensional, got {image.ndim}D")

        self.image = image
        self.projections = projections
        self.show_sum = False  # Default to showing single projections if available

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
        self.array_viewer = None
        self.spinboxes = {}
        self.finish_button = None
        self.toggle_button = None
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

        # Create display area (either ArrayViewer or pyqtgraph plot widget)
        if self.projections is not None and not self.show_sum:
            # Use ArrayViewer for single projections
            self.array_viewer = ArrayViewer(
                array3d=self.projections,
                hide_climit_controls=True,
                parent=self
            )
            # Hide play button, spinbox, and playback speed controls - keep only slider
            if hasattr(self.array_viewer, 'play_button'):
                self.array_viewer.play_button.hide()
            if hasattr(self.array_viewer, 'spinbox'):
                self.array_viewer.spinbox.hide()
                # Also hide the "index" label under the spinbox
                if self.array_viewer.spinbox.parent():
                    self.array_viewer.spinbox.parent().hide()
            if hasattr(self.array_viewer.indexing_widget, 'playback_speed_spin'):
                self.array_viewer.indexing_widget.playback_speed_spin.hide()
                # Also hide the label for playback speed if it exists
                if self.array_viewer.indexing_widget.playback_speed_spin.parent():
                    parent = self.array_viewer.indexing_widget.playback_speed_spin.parent()
                    if parent != self.array_viewer.indexing_widget:
                        parent.hide()
            layout.addWidget(self.array_viewer)
        else:
            # Create pyqtgraph plot widget for sum of projections
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setAspectLocked(True)
            layout.addWidget(self.plot_widget)

        # Point coordinate display
        point_info = self.create_point_info_display()
        layout.addWidget(point_info)

        # Create button layout
        button_layout = QHBoxLayout()

        # Toggle button (only if projections are available)
        if self.projections is not None:
            button_text = "Show Single Projections" if self.show_sum else "Show Sum of Projections"
            self.toggle_button = QPushButton(text=button_text)
            self.toggle_button.clicked.connect(self.toggle_display_mode)
            button_layout.addWidget(self.toggle_button)

        # Finish button
        self.finish_button = QPushButton(text="Select and Finish")
        self.finish_button.clicked.connect(self.finish)
        self.finish_button.setStyleSheet("background-color: blue; color: white;")
        button_layout.addWidget(self.finish_button, alignment=Qt.AlignRight)

        layout.addLayout(button_layout)

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
        if self.projections is not None and not self.show_sum:
            # For ArrayViewer, we need to add point marker to the image display
            if hasattr(self.array_viewer, 'image_item'):
                # Create a scatter plot item for the point marker
                self.point_item = pg.ScatterPlotItem(
                    size=15,
                    pen=pg.mkPen(color="r", width=2),
                    brush=pg.mkBrush(255, 0, 0, 120),
                )
                self.array_viewer.plot_item.addItem(self.point_item)

                # Set initial point position
                self.update_point_graphics()

                # Connect mouse click signal
                self.array_viewer.image_item.mouseClickEvent = self.on_image_clicked
        else:
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

    def toggle_display_mode(self):
        """Toggle between single projections (ArrayViewer) and sum of projections."""
        if self.projections is None:
            return

        # Toggle the mode
        self.show_sum = not self.show_sum

        # Get the existing layout
        main_layout = self.layout()

        # Remove all widgets from layout but keep references to preserve what we need
        # We need to remove: display widget (index 0), point info (index 1), button layout (index 2)

        # Remove display widget (ArrayViewer or PlotWidget)
        display_item = main_layout.itemAt(0)
        if display_item and display_item.widget():
            widget = display_item.widget()
            main_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

        # Reset display references
        self.plot_widget = None
        self.image_item = None
        self.array_viewer = None
        self.point_item = None

        # Create new display widget based on mode
        if self.projections is not None and not self.show_sum:
            # Use ArrayViewer for single projections
            self.array_viewer = ArrayViewer(
                array3d=self.projections,
                hide_climit_controls=True,
                parent=self
            )
            # Hide play button, spinbox, and playback speed controls - keep only slider
            if hasattr(self.array_viewer, 'play_button'):
                self.array_viewer.play_button.hide()
            if hasattr(self.array_viewer, 'spinbox'):
                self.array_viewer.spinbox.hide()
                # Also hide the "index" label under the spinbox
                if self.array_viewer.spinbox.parent():
                    self.array_viewer.spinbox.parent().hide()
            if hasattr(self.array_viewer.indexing_widget, 'playback_speed_spin'):
                self.array_viewer.indexing_widget.playback_speed_spin.hide()
                # Also hide the label for playback speed if it exists
                if self.array_viewer.indexing_widget.playback_speed_spin.parent():
                    parent = self.array_viewer.indexing_widget.playback_speed_spin.parent()
                    if parent != self.array_viewer.indexing_widget:
                        parent.hide()
            main_layout.insertWidget(0, self.array_viewer)
        else:
            # Create pyqtgraph plot widget for sum of projections
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setAspectLocked(True)
            main_layout.insertWidget(0, self.plot_widget)

        # Setup graphics for the new display
        self.setup_graphics()

        # Update toggle button text
        if self.show_sum:
            self.toggle_button.setText("Show Single Projections")
        else:
            self.toggle_button.setText("Show Sum of Projections")

    def finish(self):
        """Emit point_selected signal and close the widget."""
        self.point_selected.emit((self.point_x, self.point_y))
        self.close()


def launch_point_selector(
    image: np.ndarray,
    initial_point: Optional[tuple] = None,
    projections: Optional[np.ndarray] = None,
) -> tuple:
    """
    Launch the point selector GUI for interactively selecting a point on a 2D image.

    The GUI will remain open until the user clicks "Select and Finish", at which
    point it closes and returns the selected coordinates.

    Args:
        image: 2D numpy array to display (used as fallback or sum of projections)
        initial_point: Optional (x, y) tuple for initial point position
        projections: Optional 3D numpy array of projections. If provided,
            defaults to showing ArrayViewer with single projections.

    Returns:
        Tuple of (x, y) coordinates of the selected point

    Example:
        Launch GUI and get the selected point::

            x, y = launch_point_selector(my_image)
            print(f"Selected point: ({x}, {y})")
    """
    app = QApplication.instance() or QApplication([])
    gui = PointSelector(image, initial_point=initial_point, projections=projections)

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
