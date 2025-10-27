from turtle import title
import matplotlib
from pyxalign.io.loaders.base import StandardData

import sys
import numpy as np
from typing import Optional, Dict
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
)
import pyqtgraph as pg

from pyxalign.interactions.viewers.base import IndexSelectorWidget


class StandardDataViewer(QWidget):
    """
    A PyQt5 Widget for displaying the data stored in StandardData using pyqtgraph.

    Key Changes from the basic version:
      - If projections are complex, display np.angle(projection).
      - Histogram/ROI panels on the right of ImageView are hidden.
      - A slider, spinbox, and play button select which projection is displayed.
      - Added a spinbox for setting the playback speed (ms per frame).
      - A second tab shows probe positions for the selected scan.
      - Additional data is displayed below in a horizontal layout:
         * On the left: pixel size and probe shape (stacked vertically).
         * On the right: a 2-col table with scan numbers and angles.
    """

    def __init__(self, data: Optional[StandardData] = None, parent=None):
        super().__init__(parent)

        self.data = data  # May be None at init
        self.setWindowTitle("StandardData Viewer")
        self.resize(1200, 700)

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)
        title_label = QLabel("Loaded Data")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14pt; 
            }
            """)
        self.main_layout.addWidget(title_label, alignment=Qt.AlignHCenter)

        # # Timer and play control
        # self.play_timer = QTimer()
        # self.play_timer.timeout.connect(self._advance_frame)
        self.is_playing = False

        # Build UI
        self._create_tabbed_display()
        self._create_slider_controls()
        self._create_additional_data_panel()

        # If data is provided, populate now
        if self.data is not None:
            self._populate_from_data()

    def setStandardData(self, data: StandardData):
        """Assign a StandardData object to the viewer and populate the UI."""
        self.data = data
        self._populate_from_data()

    # ---------------------------
    # UI CREATION
    # ---------------------------
    def _create_slider_controls(self):
        """
        Create controls for selecting which projection to view:
          - A horizontal slider
          - A spin box
          - A Play/Pause button
          - A Playback Speed spin box (in ms).
        """
        # controls_layout = QVBoxLayout()

        self.indexing_widget = IndexSelectorWidget(0, 0)
        self.slider, self.spinbox, self.play_button, self.play_timer = (
            self.indexing_widget.slider,
            self.indexing_widget.spinbox,
            self.indexing_widget.play_button,
            self.indexing_widget.play_timer,
        )
        self.slider.valueChanged.connect(self._on_index_changed)
        self.spinbox.valueChanged.connect(self._on_index_changed)
        self.play_button.clicked.connect(self._on_play_clicked)
        self.play_timer.timeout.connect(self._advance_frame)

        self.main_layout.addWidget(self.indexing_widget)

    def _create_tabbed_display(self):
        """
        Create a QTabWidget with two tabs:
          1) Projections (pyqtgraph ImageView)
          2) Probe positions (pyqtgraph PlotWidget)
        """
        self.tab_widget = QTabWidget()

        # -- Tab 1: Projections --
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        # Show axes for x/y ticks
        view = self.image_view.getView()
        view.invertY(False)
        view.showAxis("left", True)
        view.showAxis("bottom", True)
        view.setLabel("left", "Y")
        view.setLabel("bottom", "X")

        proj_widget = QWidget()
        proj_layout = QVBoxLayout(proj_widget)
        proj_layout.addWidget(self.image_view)
        self.tab_widget.addTab(proj_widget, "Projections")

        # -- Tab 2: Probe positions --
        self.probe_pos_plot = pg.PlotWidget()
        self.probe_pos_plot.showGrid(x=True, y=True)
        self.probe_pos_plot.setLabel("left", "Probe Y")
        self.probe_pos_plot.setLabel("bottom", "Probe X")
        self.probe_pos_plot.setAspectLocked(True)

        probe_pos_widget = QWidget()
        probe_pos_layout = QVBoxLayout(probe_pos_widget)
        probe_pos_layout.addWidget(self.probe_pos_plot)
        self.tab_widget.addTab(probe_pos_widget, "Probe Positions")

        # Connect tab change signal to handle synchronization
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        self.main_layout.addWidget(self.tab_widget, stretch=1)

    def _create_additional_data_panel(self):
        """
        Create a horizontal layout beneath the main tabs to display:
          - Pixel size (top) and probe shape (bottom) on the left (stacked).
          - A table of [Scan Number, Angles] on the right.
        """
        self.additional_layout = QHBoxLayout()

        # Left: pixel size & probe shape in a vertical layout
        self.left_vbox = QVBoxLayout()
        self.pixel_size_label = QLabel("Pixel Size:")
        self.probe_shape_label = QLabel("Probe Shape:")
        self.left_vbox.addWidget(self.pixel_size_label)
        self.left_vbox.addWidget(self.probe_shape_label)

        # Right: table of scan numbers and angles
        self.scan_angles_table = QTableWidget()
        self.scan_angles_table.setColumnCount(2)
        self.scan_angles_table.setHorizontalHeaderLabels(["Scan #", "Angle"])
        self.scan_angles_table.verticalHeader().setVisible(False)

        # Put the left vbox and the table in the horizontal layout
        self.additional_layout.addLayout(self.left_vbox)
        self.additional_layout.addWidget(self.scan_angles_table)

        # Add to main layout
        self.main_layout.addLayout(self.additional_layout)

    # ---------------------------
    # POPULATING & UPDATING UI
    # ---------------------------
    def _populate_from_data(self):
        """
        Populate the UI elements based on assigned StandardData:
          - Set slider range
          - Fill table of scan numbers/angles
          - Display pixel size & probe shape
          - Show first projection by default
        """
        if not self.data or len(self.data.scan_numbers) == 0:
            # No data to show
            self.slider.setMaximum(0)
            self.spinbox.setMaximum(0)
            self.scan_angles_table.setRowCount(0)
            return

        num_scans = len(self.data.scan_numbers)
        self.slider.setMaximum(num_scans - 1)
        self.spinbox.setMaximum(num_scans - 1)

        # Fill table with [Scan #, Angle]
        self.scan_angles_table.setRowCount(num_scans)
        for i, scan_num in enumerate(self.data.scan_numbers):
            angle_val = self.data.angles[i] if i < len(self.data.angles) else 0.0

            # Column 0: Scan #
            sn_item = QTableWidgetItem(str(scan_num))
            self.scan_angles_table.setItem(i, 0, sn_item)

            # Column 1: Angle
            angle_item = QTableWidgetItem(str(angle_val))
            self.scan_angles_table.setItem(i, 1, angle_item)

        # Pixel size & probe shape
        px_text = "Pixel Size: " + (str(self.data.pixel_size) if self.data.pixel_size else "None")
        self.pixel_size_label.setText(px_text)

        probe_text = "Probe Shape: "
        if self.data.probe is not None:
            probe_text += str(self.data.probe.shape)
        else:
            probe_text += "None"
        self.probe_shape_label.setText(probe_text)

        # Show the first projection
        self.slider.setValue(0)
        self.spinbox.setValue(0)
        self._update_display(0)

    def _on_index_changed(self, value: int):
        """
        Callback for both the slider and spinbox. Keeps them synchronized and updates the display.
        """
        # Prevent signals from looping
        if self.sender() == self.slider:
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(value)
            self.spinbox.blockSignals(False)
        elif self.sender() == self.spinbox:
            self.slider.blockSignals(True)
            self.slider.setValue(value)
            self.slider.blockSignals(False)

        self._update_display(value)

    def _update_display(self, idx: int):
        """
        Update the projection view and conditionally update probe positions based on current tab.
        """
        if not self.data or idx >= len(self.data.scan_numbers):
            return

        scan_num = self.data.scan_numbers[idx]
        self._set_projection_in_viewer(scan_num)
        
        # Only update probe positions if that tab is currently visible
        if self.tab_widget.currentIndex() == 1:  # Probe positions tab index
            self._update_probe_positions(scan_num)

    def _set_projection_in_viewer(self, scan_num: int):
        """
        Retrieve the projection for the given scan number and display it.
        If the projection is complex, display np.angle(projection).
        """
        if not self.data or not self.data.projections:
            return

        proj = self.data.projections.get(scan_num)
        if proj is None:
            empty_data = np.zeros((10, 10))
            self.image_view.setImage(empty_data, autoLevels=True)
            return

        if np.iscomplexobj(proj):
            data_to_show = np.angle(proj)
        else:
            data_to_show = proj

        self.image_view.setImage(np.transpose(data_to_show), autoLevels=True)

    def _update_probe_positions(self, scan_num: int):
        """
        Clear the probe_pos_plot and draw a scatter of probe positions for the given scan, if available.
        """
        self.probe_pos_plot.clear()

        if self.data and self.data.probe_positions and scan_num in self.data.probe_positions:
            pos_array = self.data.probe_positions[scan_num]
            self.probe_pos_plot.plot(
                pos_array[:, 1],
                pos_array[:, 0],
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush="r",
            )

    def _on_tab_changed(self, index: int):
        """
        Handle tab changes to synchronize probe positions when switching to that tab.
        """
        if index == 1:  # Probe positions tab
            # Update probe positions to match current projection index
            current_idx = self.slider.value()
            if self.data and current_idx < len(self.data.scan_numbers):
                scan_num = self.data.scan_numbers[current_idx]
                self._update_probe_positions(scan_num)

    # ---------------------------
    # PLAYBACK LOGIC
    # ---------------------------
    def _on_play_clicked(self):
        """
        Toggle play/pause of stepping through frames.
        """
        if not self.data or len(self.data.scan_numbers) == 0:
            return
        if not self.is_playing:
            # Start playing
            self.is_playing = True
            self.play_button.setText("Pause")
            # Use the current playback speed
            self.play_timer.start()
        else:
            # Pause
            self.is_playing = False
            self.play_button.setText("Play")
            self.play_timer.stop()

    def _on_playback_speed_changed(self, value: int):
        """
        Update the interval (ms) at which the frames advance,
        if playback is active this immediately changes the timer rate.
        """
        if self.is_playing:
            self.play_timer.setInterval(value)

    def _advance_frame(self):
        """
        Timer callback: step the slider forward by 1, looping back at the end.
        """
        current_val = self.slider.value()
        max_val = self.slider.maximum()
        if current_val < max_val:
            self.slider.setValue(current_val + 1)
        else:
            # Loop to the beginning
            self.slider.setValue(0)


def launch_standard_data_viewer(
    standard_data: StandardData,
    wait_until_closed: bool = False,
):
    app = QApplication.instance() or QApplication([])
    gui = StandardDataViewer(standard_data)
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create viewer (initially no data)
    viewer = StandardDataViewer()
    viewer.show()

    # Build some dummy data
    dummy_scans = np.array([100, 200, 300])
    # Example: 256x256 arrays, some real, some complex
    dummy_projections = {
        100: np.random.rand(256, 256),
        200: np.random.rand(256, 256) + 1j * np.random.rand(256, 256),
        300: np.random.rand(256, 256),
    }
    dummy_angles = np.linspace(0, 180, 3)
    dummy_probe_positions = {
        100: np.random.rand(10, 2),
        200: np.random.rand(8, 2),
        300: np.random.rand(12, 2),
    }
    dummy_probe = np.random.rand(64, 64)

    standard_data = StandardData(
        projections=dummy_projections,
        angles=dummy_angles,
        scan_numbers=dummy_scans,
        probe_positions=dummy_probe_positions,
        probe=dummy_probe,
        pixel_size=1.234,
    )

    # Assign data to viewer after initialization
    viewer.setStandardData(standard_data)

    sys.exit(app.exec_())
