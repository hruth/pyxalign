"""
Interactive GUI for tuning 3D reconstruction parameters.

This module provides a PyQt5 GUI for interactively adjusting reconstruction
parameters (sample thickness, center of rotation) and viewing the resulting
3D reconstruction.
"""
from typing import Optional
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDoubleSpinBox,
    QLabel,
    QGroupBox,
    QSizePolicy,
    QSpacerItem,
)
from PyQt5.QtCore import Qt
from pyxalign.interactions.viewers.base import ArrayViewer
from pyxalign.api.options.plotting import ArrayViewerOptions
from pyxalign.interactions.point_selector import PointSelector
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
import pyxalign.data_structures.projections as p


class ReconstructionParameterTuner(QWidget):
    """Interactive GUI for tuning reconstruction parameters.

    This widget allows users to adjust sample thickness and center of rotation
    coordinates, then regenerate and view the 3D reconstruction with updated
    parameters.

    Args:
        phase_projections: PhaseProjections object containing the data to reconstruct.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        phase_projections: "p.PhaseProjections",
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.phase_projections = phase_projections
        self.array_viewer = None
        self.point_selector = None

        self.setWindowTitle("3D Reconstruction Parameter Tuner")
        self.resize(1600, 900)

        # Create the UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Main layout: horizontal split
        main_layout = QHBoxLayout()

        # ===== LEFT PANEL: Input Controls =====
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Create parameter controls
        param_group = QGroupBox("Reconstruction Parameters")
        param_group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        param_layout = QVBoxLayout()

        # Sample thickness spinbox
        thickness_layout = QHBoxLayout()
        thickness_label = QLabel("Sample Thickness (m):")
        thickness_label.setStyleSheet("font-size: 11pt;")
        self.thickness_spinbox = QDoubleSpinBox()
        self.thickness_spinbox.setDecimals(9)
        self.thickness_spinbox.setMinimum(0.0)
        self.thickness_spinbox.setMaximum(1.0)
        self.thickness_spinbox.setSingleStep(1e-6)
        self.thickness_spinbox.setValue(
            self.phase_projections.options.experiment.sample_thickness
        )
        self.thickness_spinbox.setStyleSheet("font-size: 11pt;")
        self.thickness_spinbox.valueChanged.connect(self.on_thickness_changed)
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_spinbox)
        thickness_layout.addStretch()

        # Add parameter controls to layout
        param_layout.addLayout(thickness_layout)
        param_group.setLayout(param_layout)

        # Create point selector for center of rotation
        # Use sum of projections as the image for point selection
        projection_sum = np.sum(self.phase_projections.data, axis=0)
        initial_center = (
            int(self.phase_projections.center_of_rotation[1]),
            int(self.phase_projections.center_of_rotation[0])
        )
        self.point_selector = PointSelector(
            image=projection_sum,
            initial_point=initial_center,
            parent=self
        )
        # Remove the finish button from point selector since we're embedding it
        self.point_selector.finish_button.hide()
        # Connect the point_changed signal to update center of rotation
        self.point_selector.point_changed.connect(self.on_center_of_rotation_changed)

        # Create center of rotation group
        cor_group = QGroupBox("Center of Rotation Selection")
        cor_group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        cor_layout = QVBoxLayout()
        cor_layout.addWidget(self.point_selector)
        cor_group.setLayout(cor_layout)

        # Create reconstruct button
        self.reconstruct_button = QPushButton("Run 3D Reconstruction")
        self.reconstruct_button.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px;")
        self.reconstruct_button.clicked.connect(self.on_reconstruct_clicked)

        # Add widgets to left panel
        left_layout.addWidget(param_group)
        left_layout.addWidget(cor_group)
        left_layout.addWidget(self.reconstruct_button)
        left_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # ===== RIGHT PANEL: Volume Display =====
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Create label for volume viewer
        volume_label = QLabel("3D Reconstruction Volume")
        volume_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        volume_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(volume_label)

        # Create placeholder for array viewer
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout()
        self.viewer_container.setLayout(self.viewer_layout)
        right_layout.addWidget(self.viewer_container)

        # Add left and right panels to main layout
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=2)

        self.setLayout(main_layout)

    def on_thickness_changed(self, value: float):
        """Update sample thickness when spinbox value changes."""
        self.phase_projections.options.experiment.sample_thickness = value

    def on_center_of_rotation_changed(self, point: tuple):
        """Update center of rotation when point selector changes.

        Args:
            point: Tuple of (x, y) coordinates from the point selector.
        """
        x, y = point
        # PointSelector returns (x, y), but center_of_rotation is stored as [y, x]
        self.phase_projections.center_of_rotation[1] = x
        self.phase_projections.center_of_rotation[0] = y

    def on_reconstruct_clicked(self):
        """Generate 3D reconstruction and display it."""
        # Disable button during reconstruction
        self.reconstruct_button.setEnabled(False)
        self.reconstruct_button.setText("Reconstructing...")

        try:
            # Run reconstruction
            self.phase_projections.get_3D_reconstruction()

            # Update or create array viewer
            if self.array_viewer is None:
                # Create new array viewer
                self.array_viewer = ArrayViewer(
                    array3d=self.phase_projections.volume.data,
                    options=ArrayViewerOptions(
                        slider_axis=0,
                        start_index=int(self.phase_projections.volume.data.shape[0] / 2),
                    ),
                )
                self.viewer_layout.addWidget(self.array_viewer)
            else:
                # Update existing array viewer with new volume data
                self.array_viewer.array3d = self.phase_projections.volume.data
                self.array_viewer.refresh_frame()

        finally:
            # Re-enable button
            self.reconstruct_button.setEnabled(True)
            self.reconstruct_button.setText("Run 3D Reconstruction")

    def start(self):
        """Show the widget."""
        self.show()


@switch_to_matplotlib_qt_backend
def launch_reconstruction_parameter_tuner(
    phase_projections: "p.PhaseProjections",
    wait_until_closed: bool = False,
) -> ReconstructionParameterTuner:
    """Launch the reconstruction parameter tuner GUI.

    This GUI allows interactive adjustment of reconstruction parameters
    (sample thickness and center of rotation) and displays the resulting
    3D reconstruction using the ArrayViewer.

    Args:
        phase_projections: PhaseProjections object containing the data.
        wait_until_closed: If True, the application starts a blocking call
            until the GUI window is closed.

    Returns:
        The ReconstructionParameterTuner widget instance.

    Example:
        Launch the parameter tuning GUI::

            gui = pyxalign.gui.launch_reconstruction_parameter_tuner(
                task.phase_projections
            )
    """
    app = QApplication.instance() or QApplication([])
    gui = ReconstructionParameterTuner(phase_projections=phase_projections)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui
