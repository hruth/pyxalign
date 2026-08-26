"""
Interactive GUI for visualizing a 3D volume and computing histograms of crop regions.

This module provides a widget that combines an ArrayViewer (with axis controls visible)
with an interactive rectangular crop region selector.  Clicking "Calculate Histogram"
computes and displays the histogram for the selected crop of the currently displayed
frame.
"""

from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.interactions.viewers.base import ArrayViewer


class VolumeHistogramViewer(QWidget):
    """Widget for viewing a 3D volume and computing histograms of crop regions.

    Displays a 3D volume with an ArrayViewer (axis-cycling controls visible), overlays
    an interactive rectangular ROI, and can compute and display a histogram of pixel
    values within the selected region of the currently displayed frame.
    """

    def __init__(
        self,
        volume: np.ndarray,
        sort_idx: Optional[Sequence] = None,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the volume histogram viewer.

        Args:
            volume: 3D numpy array to display and analyze.
            sort_idx: Optional index sorting passed to the ArrayViewer.
            parent: Optional parent widget.
        """
        super().__init__(parent)

        if volume.ndim != 3:
            raise ValueError(f"volume must be 3-dimensional, got {volume.ndim}D")

        self.volume = volume

        self._roi_item: Optional[pg.RectROI] = None
        self._spinboxes: dict = {}
        self._updating_from_graphics: bool = False

        self._setup_array_viewer(sort_idx)
        self._setup_ui()
        self._setup_roi()
        self._connect_signals()

        self.setWindowTitle("Volume Histogram Viewer")
        self.resize(1000, 950)

    # ------------------------------------------------------------------
    # Setup methods
    # ------------------------------------------------------------------

    def _setup_array_viewer(self, sort_idx: Optional[Sequence]) -> None:
        """Create the ArrayViewer with axis controls visible."""
        self.array_viewer = ArrayViewer(
            self.volume,
            sort_idx=sort_idx,
            hide_axis_controls=False,
        )

    def _setup_ui(self) -> None:
        """Build the main widget layout."""
        # Right panel: crop controls, button, histogram stacked vertically
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)
        right_panel.addWidget(self._build_roi_controls_group())
        right_panel.addWidget(self._build_calculate_button(), alignment=Qt.AlignRight)
        right_panel.addWidget(self._build_histogram_plot(), stretch=1)

        # ArrayViewer on the left, right panel on the right
        main_layout = QHBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(self.array_viewer, stretch=2)
        main_layout.addLayout(right_panel, stretch=1)

        self.setLayout(main_layout)

    def _build_roi_controls_group(self) -> QGroupBox:
        """Create the grouped ROI parameter spinbox controls."""
        group = QGroupBox("Crop Region")
        group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        pg_width, pg_height = self._current_image_dimensions()
        large_max = int(1e7)

        spinbox_specs = [
            ("horizontal_offset", "Horizontal Offset (from center):", 0, 0, -large_max, large_max, 0),
            ("vertical_offset",   "Vertical Offset (from center):",   1, 0, -large_max, large_max, 0),
            ("horizontal_range",  "Horizontal Range (Width):",         0, 2, 1, large_max, pg_width),
            ("vertical_range",    "Vertical Range (Height):",          1, 2, 1, large_max, pg_height),
        ]

        self._spinboxes = {}
        for key, label_text, row, col, min_val, max_val, default in spinbox_specs:
            label = QLabel(label_text)
            label.setStyleSheet("QLabel { font-size: 12pt; }")
            grid.addWidget(label, row, col)

            spinbox = QSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(default)
            spinbox.setStyleSheet("QSpinBox { font-size: 12pt; }")
            spinbox.setMinimumWidth(90)
            grid.addWidget(spinbox, row, col + 1)

            self._spinboxes[key] = spinbox

        group.setLayout(grid)
        return group

    def _build_calculate_button(self) -> QPushButton:
        """Create the Calculate Histogram action button."""
        self.calculate_button = QPushButton("Calculate Histogram")
        self.calculate_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #2196F3; color: white;"
            "  font-size: 13pt; font-weight: bold;"
            "  padding: 6px 24px; border-radius: 4px;"
            "}"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:pressed { background-color: #0D47A1; }"
        )
        return self.calculate_button

    def _build_histogram_plot(self) -> pg.PlotWidget:
        """Create the pyqtgraph PlotWidget used to display histograms."""
        self.histogram_plot = pg.PlotWidget()
        self.histogram_plot.setLabel("left", "Count")
        self.histogram_plot.setLabel("bottom", "Pixel Value")
        self.histogram_plot.setTitle("Histogram — press 'Calculate Histogram' to compute")
        self.histogram_plot.setMinimumHeight(200)
        return self.histogram_plot

    def _setup_roi(self) -> None:
        """Overlay a RectROI on the ArrayViewer covering the full current frame."""
        pg_width, pg_height = self._current_image_dimensions()

        self._roi_item = pg.RectROI(
            pos=[0, 0],
            size=[pg_width, pg_height],
            pen=pg.mkPen(color="r", width=2),
            rotatable=False,
            scaleSnap=True,
            translateSnap=True,
        )
        self._roi_item.addScaleHandle([0, 0], [1, 1])
        self._roi_item.addScaleHandle([1, 0], [0, 1])
        self._roi_item.addScaleHandle([0, 1], [1, 0])
        self._roi_item.addScaleHandle([1, 1], [0, 0])

        self.array_viewer.plot_item.addItem(self._roi_item)
        self._sync_spinboxes_from_roi()

    def _connect_signals(self) -> None:
        """Connect all widget signals to their slots."""
        self._roi_item.sigRegionChanged.connect(self._on_roi_changed)
        for spinbox in self._spinboxes.values():
            spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.calculate_button.clicked.connect(self._calculate_histogram)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _current_image_dimensions(self) -> tuple:
        """Return (pg_width, pg_height) of the currently displayed pyqtgraph image.

        Returns:
            Tuple of (width, height) in pyqtgraph image coordinates.
        """
        if (
            self._roi_item is None
            and hasattr(self, "array_viewer")
            and self.array_viewer.image_item.image is not None
        ):
            img = self.array_viewer.image_item.image
            return (img.shape[0], img.shape[1])

        # Derive from volume shape and current slider axis before image is rendered
        axis = self.array_viewer.options.slider_axis
        other_dims = [self.volume.shape[i] for i in range(3) if i != axis]
        # display_frame applies np.transpose, so pyqtgraph x=other_dims[1], y=other_dims[0]
        return (other_dims[1], other_dims[0])

    def _image_center(self) -> tuple:
        """Return (center_x, center_y) in pyqtgraph image coordinates."""
        pg_width, pg_height = self._current_image_dimensions()
        return (pg_width // 2, pg_height // 2)

    # ------------------------------------------------------------------
    # ROI / spinbox synchronisation
    # ------------------------------------------------------------------

    def _on_roi_changed(self) -> None:
        """Update spinboxes when the user drags or resizes the ROI."""
        if self._updating_from_graphics:
            return
        self._updating_from_graphics = True
        self._sync_spinboxes_from_roi()
        self._updating_from_graphics = False

    def _on_spinbox_changed(self) -> None:
        """Update the ROI graphics when the user edits spinbox values."""
        if self._updating_from_graphics:
            return
        self._sync_roi_from_spinboxes()

    def _sync_spinboxes_from_roi(self) -> None:
        """Read the current ROI position/size and update spinboxes accordingly."""
        pos = self._roi_item.pos()
        size = self._roi_item.size()

        center_x, center_y = self._image_center()
        roi_center_x = int(pos[0] + size[0] / 2)
        roi_center_y = int(pos[1] + size[1] / 2)

        for spinbox in self._spinboxes.values():
            spinbox.blockSignals(True)

        self._spinboxes["horizontal_offset"].setValue(roi_center_x - center_x)
        self._spinboxes["vertical_offset"].setValue(roi_center_y - center_y)
        self._spinboxes["horizontal_range"].setValue(int(size[0]))
        self._spinboxes["vertical_range"].setValue(int(size[1]))

        for spinbox in self._spinboxes.values():
            spinbox.blockSignals(False)

    def _sync_roi_from_spinboxes(self) -> None:
        """Move/resize the ROI to match the current spinbox values."""
        center_x, center_y = self._image_center()

        h_offset = self._spinboxes["horizontal_offset"].value()
        v_offset = self._spinboxes["vertical_offset"].value()
        h_range = self._spinboxes["horizontal_range"].value()
        v_range = self._spinboxes["vertical_range"].value()

        roi_center_x = center_x + h_offset
        roi_center_y = center_y + v_offset

        pos_x = roi_center_x - h_range // 2
        pos_y = roi_center_y - v_range // 2

        self._roi_item.sigRegionChanged.disconnect(self._on_roi_changed)
        self._roi_item.setPos([pos_x, pos_y])
        self._roi_item.setSize([h_range, v_range])
        self._roi_item.sigRegionChanged.connect(self._on_roi_changed)

    # ------------------------------------------------------------------
    # Histogram calculation
    # ------------------------------------------------------------------

    def _calculate_histogram(self) -> None:
        """Extract the ROI region from the current frame and plot its histogram."""
        roi_data = self._roi_item.getArrayRegion(
            self.array_viewer.image_item.image,
            self.array_viewer.image_item,
        )

        if roi_data is None or roi_data.size == 0:
            self.histogram_plot.setTitle("No data in selected region.")
            return

        pixel_values = roi_data.ravel()
        pixel_values = pixel_values[np.isfinite(pixel_values)]

        if pixel_values.size == 0:
            self.histogram_plot.setTitle("Selected region contains no finite values.")
            return

        num_bins = min(256, max(10, int(np.sqrt(pixel_values.size))))
        counts, bin_edges = np.histogram(pixel_values, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        self.histogram_plot.clear()
        bar_item = pg.BarGraphItem(
            x=bin_centers,
            height=counts,
            width=bin_width * 0.9,
            brush=pg.mkBrush(color=(100, 150, 255, 180)),
            pen=pg.mkPen(color=(60, 100, 200)),
        )
        self.histogram_plot.addItem(bar_item)
        self.histogram_plot.setTitle(
            f"Histogram — {pixel_values.size:,} pixels | "
            f"min={pixel_values.min():.4g}, max={pixel_values.max():.4g}, "
            f"mean={pixel_values.mean():.4g}"
        )


@switch_to_matplotlib_qt_backend
def launch_volume_histogram_viewer(
    volume: np.ndarray,
    sort_idx: Optional[Sequence] = None,
    wait_until_closed: bool = False,
) -> VolumeHistogramViewer:
    """Launch the volume histogram viewer GUI.

    Displays the volume in an ArrayViewer with axis controls, allows selection
    of a 2D crop region via an interactive overlay, and computes/displays the
    histogram of pixel values in the selected region for the current frame.

    Args:
        volume: 3D numpy array to view and analyze.
        sort_idx: Optional index sorting passed to the ArrayViewer.
        wait_until_closed: If True, blocks until the GUI window is closed.

    Returns:
        VolumeHistogramViewer widget instance.

    Example:
        Inspect pixel value distributions in a reconstruction volume::

            gui = pyxalign.gui.launch_volume_histogram_viewer(task.volume)
    """
    app = QApplication.instance() or QApplication([])
    gui = VolumeHistogramViewer(volume, sort_idx=sort_idx)
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui
