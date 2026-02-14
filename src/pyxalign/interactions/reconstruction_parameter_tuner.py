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
    QCheckBox,
    QComboBox,
    QScrollArea,
    QSpinBox,
    QLineEdit,
)
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QRegExpValidator
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper
from pyxalign.interactions.viewers.base import ArrayViewer
from pyxalign.api.options.plotting import ArrayViewerOptions
from pyxalign.interactions.point_selector import PointSelector
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
import pyxalign.data_structures.projections as p
from pyxalign.api import enums


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that displays values in scientific notation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDecimals(10)  # Set high precision for internal calculations
        # Set a validator that accepts scientific notation
        validator = QRegExpValidator(QRegExp(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?"))
        self.lineEdit().setValidator(validator)

    def textFromValue(self, value: float) -> str:
        """Convert value to scientific notation string."""
        return f"{value:.2e}"

    def valueFromText(self, text: str) -> float:
        """Convert text to float value."""
        try:
            return float(text)
        except ValueError:
            return 0.0

    def validate(self, text: str, pos: int):
        """Override validate to accept scientific notation."""
        # Allow scientific notation during input
        validator = QRegExpValidator(QRegExp(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?"))
        return validator.validate(text, pos)


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

        # Reconstruction Method subsection
        method_group = QGroupBox("Reconstruction Method")
        method_group.setStyleSheet("QGroupBox { font-size: 12pt; font-weight: bold; }")
        method_group_layout = QVBoxLayout()

        # Reconstruction method dropdown
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        method_label.setStyleSheet("font-size: 11pt;")
        self.method_combobox = QComboBox()
        for method in enums.ReconstructionMethods:
            self.method_combobox.addItem(method.value, method)
        # Set current value
        current_method = self.phase_projections.options.reconstruct.method
        index = self.method_combobox.findData(current_method)
        if index >= 0:
            self.method_combobox.setCurrentIndex(index)
        self.method_combobox.setStyleSheet("font-size: 11pt;")
        self.method_combobox.currentIndexChanged.connect(self.on_method_changed)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combobox)
        method_layout.addStretch()
        method_group_layout.addLayout(method_layout)

        # ASTRA Options
        self.astra_controls = []

        # Algorithm Type
        astra_algorithm_layout = QHBoxLayout()
        astra_algorithm_label = QLabel("Algorithm Type:")
        astra_algorithm_label.setStyleSheet("font-size: 11pt;")
        self.astra_algorithm_lineedit = QLineEdit()
        self.astra_algorithm_lineedit.setText(
            self.phase_projections.options.reconstruct.astra.algorithm_type
        )
        self.astra_algorithm_lineedit.setStyleSheet("font-size: 11pt;")
        self.astra_algorithm_lineedit.setEnabled(False)
        self.astra_algorithm_lineedit.textChanged.connect(self.on_astra_algorithm_changed)
        astra_algorithm_layout.addWidget(astra_algorithm_label)
        astra_algorithm_layout.addWidget(self.astra_algorithm_lineedit)
        astra_algorithm_layout.addStretch()
        method_group_layout.addLayout(astra_algorithm_layout)
        self.astra_controls.extend([astra_algorithm_label, self.astra_algorithm_lineedit])

        # SART Options
        self.sart_controls = []

        # Iterations
        sart_iterations_layout = QHBoxLayout()
        sart_iterations_label = QLabel("Iterations:")
        sart_iterations_label.setStyleSheet("font-size: 11pt;")
        self.sart_iterations_spinbox = QSpinBox()
        self.sart_iterations_spinbox.setMinimum(1)
        self.sart_iterations_spinbox.setMaximum(1000)
        self.sart_iterations_spinbox.setValue(
            self.phase_projections.options.reconstruct.sart.iterations
        )
        self.sart_iterations_spinbox.setStyleSheet("font-size: 11pt;")
        self.sart_iterations_spinbox.valueChanged.connect(self.on_sart_iterations_changed)
        sart_iterations_layout.addWidget(sart_iterations_label)
        sart_iterations_layout.addWidget(self.sart_iterations_spinbox)
        sart_iterations_layout.addStretch()
        method_group_layout.addLayout(sart_iterations_layout)
        self.sart_controls.extend([sart_iterations_label, self.sart_iterations_spinbox])

        # Use Circular Constraint
        sart_circular_layout = QHBoxLayout()
        sart_circular_label = QLabel("Use Circular Constraint:")
        sart_circular_label.setStyleSheet("font-size: 11pt;")
        self.sart_circular_checkbox = QCheckBox()
        self.sart_circular_checkbox.setChecked(
            self.phase_projections.options.reconstruct.sart.use_circular_constraint
        )
        self.sart_circular_checkbox.setStyleSheet("font-size: 11pt;")
        self.sart_circular_checkbox.stateChanged.connect(self.on_sart_circular_changed)
        sart_circular_layout.addWidget(sart_circular_label)
        sart_circular_layout.addWidget(self.sart_circular_checkbox)
        sart_circular_layout.addStretch()
        method_group_layout.addLayout(sart_circular_layout)
        self.sart_controls.extend([sart_circular_label, self.sart_circular_checkbox])

        # Relaxation
        sart_relaxation_layout = QHBoxLayout()
        sart_relaxation_label = QLabel("Relaxation:")
        sart_relaxation_label.setStyleSheet("font-size: 11pt;")
        self.sart_relaxation_spinbox = QDoubleSpinBox()
        self.sart_relaxation_spinbox.setDecimals(6)
        self.sart_relaxation_spinbox.setMinimum(0.0)
        self.sart_relaxation_spinbox.setMaximum(1.0)
        self.sart_relaxation_spinbox.setSingleStep(0.01)
        self.sart_relaxation_spinbox.setValue(
            self.phase_projections.options.reconstruct.sart.relaxation
        )
        self.sart_relaxation_spinbox.setStyleSheet("font-size: 11pt;")
        self.sart_relaxation_spinbox.valueChanged.connect(self.on_sart_relaxation_changed)
        sart_relaxation_layout.addWidget(sart_relaxation_label)
        sart_relaxation_layout.addWidget(self.sart_relaxation_spinbox)
        sart_relaxation_layout.addStretch()
        method_group_layout.addLayout(sart_relaxation_layout)
        self.sart_controls.extend([sart_relaxation_label, self.sart_relaxation_spinbox])

        # N Subtomograms
        sart_subtomograms_layout = QHBoxLayout()
        sart_subtomograms_label = QLabel("N Subtomograms:")
        sart_subtomograms_label.setStyleSheet("font-size: 11pt;")
        self.sart_subtomograms_spinbox = QSpinBox()
        self.sart_subtomograms_spinbox.setMinimum(1)
        self.sart_subtomograms_spinbox.setMaximum(100)
        self.sart_subtomograms_spinbox.setValue(
            self.phase_projections.options.reconstruct.sart.n_subtomograms
        )
        self.sart_subtomograms_spinbox.setStyleSheet("font-size: 11pt;")
        self.sart_subtomograms_spinbox.valueChanged.connect(self.on_sart_subtomograms_changed)
        sart_subtomograms_layout.addWidget(sart_subtomograms_label)
        sart_subtomograms_layout.addWidget(self.sart_subtomograms_spinbox)
        sart_subtomograms_layout.addStretch()
        method_group_layout.addLayout(sart_subtomograms_layout)
        self.sart_controls.extend([sart_subtomograms_label, self.sart_subtomograms_spinbox])

        method_group.setLayout(method_group_layout)

        # Add method group to main param layout
        param_layout.addWidget(method_group)

        # Reconstruction Size subsection
        size_group = QGroupBox("Reconstruction Size")
        size_group.setStyleSheet("QGroupBox { font-size: 12pt; font-weight: bold; }")
        size_layout = QVBoxLayout()

        # Sample thickness spinbox
        thickness_layout = QHBoxLayout()
        thickness_label = QLabel("Sample Thickness (m):")
        thickness_label.setStyleSheet("font-size: 11pt;")
        self.thickness_spinbox = ScientificDoubleSpinBox()
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

        # Use custom width checkbox
        use_custom_width_layout = QHBoxLayout()
        use_custom_width_label = QLabel("Use Custom Width:")
        use_custom_width_label.setStyleSheet("font-size: 11pt;")
        self.use_custom_width_checkbox = QCheckBox()
        self.use_custom_width_checkbox.setChecked(
            self.phase_projections.options.volume_width.use_custom_width
        )
        self.use_custom_width_checkbox.setStyleSheet("font-size: 11pt;")
        self.use_custom_width_checkbox.stateChanged.connect(self.on_use_custom_width_changed)
        use_custom_width_layout.addWidget(use_custom_width_label)
        use_custom_width_layout.addWidget(self.use_custom_width_checkbox)
        use_custom_width_layout.addStretch()

        # Width type dropdown
        width_type_layout = QHBoxLayout()
        self.width_type_label = QLabel("Width Type:")
        self.width_type_label.setStyleSheet("font-size: 11pt;")
        self.width_type_combobox = QComboBox()
        for width_type in enums.VolumeWidthTypes:
            self.width_type_combobox.addItem(width_type.value, width_type)
        # Set current value
        current_width_type = self.phase_projections.options.volume_width.width_type
        index = self.width_type_combobox.findData(current_width_type)
        if index >= 0:
            self.width_type_combobox.setCurrentIndex(index)
        self.width_type_combobox.setStyleSheet("font-size: 11pt;")
        self.width_type_combobox.currentIndexChanged.connect(self.on_width_type_changed)
        width_type_layout.addWidget(self.width_type_label)
        width_type_layout.addWidget(self.width_type_combobox)
        width_type_layout.addStretch()

        # Multiplier spinbox
        self.multiplier_layout = QHBoxLayout()
        self.multiplier_label = QLabel("Multiplier:")
        self.multiplier_label.setStyleSheet("font-size: 11pt;")
        self.multiplier_spinbox = QDoubleSpinBox()
        self.multiplier_spinbox.setDecimals(6)
        self.multiplier_spinbox.setMinimum(0.0)
        self.multiplier_spinbox.setMaximum(100.0)
        self.multiplier_spinbox.setSingleStep(0.1)
        self.multiplier_spinbox.setValue(
            self.phase_projections.options.volume_width.multiplier
        )
        self.multiplier_spinbox.setStyleSheet("font-size: 11pt;")
        self.multiplier_spinbox.valueChanged.connect(self.on_multiplier_changed)
        self.multiplier_layout.addWidget(self.multiplier_label)
        self.multiplier_layout.addWidget(self.multiplier_spinbox)
        self.multiplier_layout.addStretch()

        # Width meters spinbox
        self.width_meters_layout = QHBoxLayout()
        self.width_meters_label = QLabel("Width (meters):")
        self.width_meters_label.setStyleSheet("font-size: 11pt;")
        self.width_meters_spinbox = ScientificDoubleSpinBox()
        self.width_meters_spinbox.setMinimum(0.0)
        self.width_meters_spinbox.setMaximum(1.0)
        self.width_meters_spinbox.setSingleStep(1e-6)
        # Handle None value
        width_meters_value = self.phase_projections.options.volume_width.width_meters
        if width_meters_value is not None:
            self.width_meters_spinbox.setValue(width_meters_value)
        else:
            self.width_meters_spinbox.setValue(0.0)
        self.width_meters_spinbox.setStyleSheet("font-size: 11pt;")
        self.width_meters_spinbox.valueChanged.connect(self.on_width_meters_changed)
        self.width_meters_layout.addWidget(self.width_meters_label)
        self.width_meters_layout.addWidget(self.width_meters_spinbox)
        self.width_meters_layout.addStretch()

        # Add size controls to size group layout
        size_layout.addLayout(thickness_layout)
        size_layout.addLayout(use_custom_width_layout)
        size_layout.addLayout(width_type_layout)
        size_layout.addLayout(self.multiplier_layout)
        size_layout.addLayout(self.width_meters_layout)
        size_group.setLayout(size_layout)

        # Add size group to main param layout
        param_layout.addWidget(size_group)
        param_group.setLayout(param_layout)

        # Update visibility and enabled state based on initial values
        self.update_method_controls_visibility()
        self.update_width_controls_visibility()
        self.update_width_controls_enabled_state()

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

        # Wrap left panel in a scroll area
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidget(left_panel)
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # ===== RIGHT PANEL: Volume Display =====
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Create group box for volume display
        volume_group = QGroupBox("3D Reconstruction Volume")
        volume_group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        volume_group_layout = QVBoxLayout()

        # Create placeholder for array viewer
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout()
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_container.setLayout(self.viewer_layout)
        volume_group_layout.addWidget(self.viewer_container)
        volume_group.setLayout(volume_group_layout)

        # Add volume group to right panel
        right_layout.addWidget(volume_group)

        # Add left and right panels to main layout
        main_layout.addWidget(left_scroll_area, stretch=1)
        main_layout.addWidget(right_panel, stretch=2)

        self.setLayout(main_layout)

    def on_method_changed(self, index: int):
        """Update reconstruction method when combobox selection changes."""
        method = self.method_combobox.itemData(index)
        self.phase_projections.options.reconstruct.method = method
        self.update_method_controls_visibility()

    def on_astra_algorithm_changed(self, text: str):
        """Update ASTRA algorithm type when text changes."""
        self.phase_projections.options.reconstruct.astra.algorithm_type = text

    def on_sart_iterations_changed(self, value: int):
        """Update SART iterations when spinbox value changes."""
        self.phase_projections.options.reconstruct.sart.iterations = value

    def on_sart_circular_changed(self, state: int):
        """Update SART use_circular_constraint when checkbox state changes."""
        self.phase_projections.options.reconstruct.sart.use_circular_constraint = bool(state)

    def on_sart_relaxation_changed(self, value: float):
        """Update SART relaxation when spinbox value changes."""
        self.phase_projections.options.reconstruct.sart.relaxation = value

    def on_sart_subtomograms_changed(self, value: int):
        """Update SART n_subtomograms when spinbox value changes."""
        self.phase_projections.options.reconstruct.sart.n_subtomograms = value

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

    def on_use_custom_width_changed(self, state: int):
        """Update use_custom_width when checkbox state changes."""
        self.phase_projections.options.volume_width.use_custom_width = bool(state)
        self.update_width_controls_enabled_state()

    def on_width_type_changed(self, index: int):
        """Update width_type when combobox selection changes."""
        width_type = self.width_type_combobox.itemData(index)
        self.phase_projections.options.volume_width.width_type = width_type
        self.update_width_controls_visibility()

    def on_multiplier_changed(self, value: float):
        """Update multiplier when spinbox value changes."""
        self.phase_projections.options.volume_width.multiplier = value

    def on_width_meters_changed(self, value: float):
        """Update width_meters when spinbox value changes."""
        self.phase_projections.options.volume_width.width_meters = value

    def update_method_controls_visibility(self):
        """Show/hide method-specific controls based on selected reconstruction method."""
        method = self.method_combobox.currentData()

        # Show/hide ASTRA controls
        is_astra = method == enums.ReconstructionMethods.ASTRA
        for control in self.astra_controls:
            control.setVisible(is_astra)

        # Show/hide SART controls
        is_sart = method == enums.ReconstructionMethods.SART
        for control in self.sart_controls:
            control.setVisible(is_sart)

    def update_width_controls_visibility(self):
        """Show/hide width controls based on width_type selection."""
        width_type = self.width_type_combobox.currentData()

        # Show/hide multiplier controls
        is_multiplier = width_type == enums.VolumeWidthTypes.MULTIPLIER
        self.multiplier_label.setVisible(is_multiplier)
        self.multiplier_spinbox.setVisible(is_multiplier)

        # Show/hide meters controls
        is_meters = width_type == enums.VolumeWidthTypes.METERS
        self.width_meters_label.setVisible(is_meters)
        self.width_meters_spinbox.setVisible(is_meters)

    def update_width_controls_enabled_state(self):
        """Enable/disable width controls based on use_custom_width checkbox."""
        enabled = self.use_custom_width_checkbox.isChecked()

        self.width_type_label.setEnabled(enabled)
        self.width_type_combobox.setEnabled(enabled)
        self.multiplier_label.setEnabled(enabled)
        self.multiplier_spinbox.setEnabled(enabled)
        self.width_meters_label.setEnabled(enabled)
        self.width_meters_spinbox.setEnabled(enabled)

    def on_reconstruct_clicked(self):
        """Generate 3D reconstruction and display it."""
        # Disable button during reconstruction
        self.reconstruct_button.setEnabled(False)
        self.reconstruct_button.setText("Reconstructing...")

        try:
            # Run reconstruction
            load_bar_func_wrapper = loading_bar_wrapper(
                "Getting 3D reconstruction...", block_all_windows=True
            )(self.phase_projections.get_3D_reconstruction)
            load_bar_func_wrapper()

            # Update or create array viewer
            if self.array_viewer is None:
                # Create new array viewer
                self.array_viewer = ArrayViewer(
                    array3d=self.phase_projections.volume.data,
                    options=ArrayViewerOptions(
                        slider_axis=0,
                        start_index=int(self.phase_projections.volume.data.shape[0] / 2),
                    ),
                    hide_axis_controls=False,
                    include_array_saving_widget=True,
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
