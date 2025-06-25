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
    QTabWidget
)
import pyqtgraph as pg


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

        # Timer and play control
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._advance_frame)
        self.is_playing = False

        # Build UI
        self._create_slider_controls()
        self._create_tabbed_display()
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
        controls_layout = QHBoxLayout()

        # Slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_index_changed)
        controls_layout.addWidget(self.frame_slider)

        # SpinBox
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.valueChanged.connect(self._on_index_changed)
        controls_layout.addWidget(self.frame_spin)

        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._on_play_clicked)
        controls_layout.addWidget(self.play_button)

        # Playback Speed label
        self.playback_speed_label = QLabel("Playback Speed (ms):")
        controls_layout.addWidget(self.playback_speed_label)

        # Playback Speed spin box
        self.playback_speed_spin = QSpinBox()
        self.playback_speed_spin.setRange(10, 2000)  # Set a reasonable range
        self.playback_speed_spin.setValue(500)       # Default interval = 500ms
        self.playback_speed_spin.valueChanged.connect(self._on_playback_speed_changed)
        controls_layout.addWidget(self.playback_speed_spin)

        # Add this controls layout to the main layout
        self.main_layout.addLayout(controls_layout)

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
        view.showAxis('left', True)
        view.showAxis('bottom', True)
        view.setLabel('left', 'Y')
        view.setLabel('bottom', 'X')

        proj_widget = QWidget()
        proj_layout = QVBoxLayout(proj_widget)
        proj_layout.addWidget(self.image_view)
        self.tab_widget.addTab(proj_widget, "Projections")

        # -- Tab 2: Probe positions --
        self.probe_pos_plot = pg.PlotWidget()
        self.probe_pos_plot.showGrid(x=True, y=True)
        self.probe_pos_plot.setLabel('left', 'Probe Y')
        self.probe_pos_plot.setLabel('bottom', 'Probe X')
        self.probe_pos_plot.setAspectLocked(True)

        probe_pos_widget = QWidget()
        probe_pos_layout = QVBoxLayout(probe_pos_widget)
        probe_pos_layout.addWidget(self.probe_pos_plot)
        self.tab_widget.addTab(probe_pos_widget, "Probe Positions")

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
            self.frame_slider.setMaximum(0)
            self.frame_spin.setMaximum(0)
            self.scan_angles_table.setRowCount(0)
            return

        num_scans = len(self.data.scan_numbers)
        self.frame_slider.setMaximum(num_scans - 1)
        self.frame_spin.setMaximum(num_scans - 1)

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
        self.frame_slider.setValue(0)
        self.frame_spin.setValue(0)
        self._update_display(0)

    def _on_index_changed(self, value: int):
        """
        Callback for both the slider and spinbox. Keeps them synchronized and updates the display.
        """
        # Prevent signals from looping
        if self.sender() == self.frame_slider:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(value)
            self.frame_spin.blockSignals(False)
        elif self.sender() == self.frame_spin:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(value)
            self.frame_slider.blockSignals(False)

        self._update_display(value)

    def _update_display(self, idx: int):
        """
        Update both the projection view and the probe positions tab based on the selected index.
        """
        if not self.data or idx >= len(self.data.scan_numbers):
            return

        scan_num = self.data.scan_numbers[idx]
        self._set_projection_in_viewer(scan_num)
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

        self.image_view.setImage(data_to_show, autoLevels=True)

    def _update_probe_positions(self, scan_num: int):
        """
        Clear the probe_pos_plot and draw a scatter of probe positions for the given scan, if available.
        """
        self.probe_pos_plot.clear()

        if (
            self.data
            and self.data.probe_positions
            and scan_num in self.data.probe_positions
        ):
            pos_array = self.data.probe_positions[scan_num]
            self.probe_pos_plot.plot(
                pos_array[:, 0],
                pos_array[:, 1],
                pen=None,
                symbol='o',
                symbolSize=5,
                symbolBrush='r'
            )

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
            self.play_timer.start(self.playback_speed_spin.value())
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
        current_val = self.frame_slider.value()
        max_val = self.frame_slider.maximum()
        if current_val < max_val:
            self.frame_slider.setValue(current_val + 1)
        else:
            # Loop to the beginning
            self.frame_slider.setValue(0)


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
