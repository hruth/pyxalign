"""
Interactive GUI for tuning 3D reconstruction parameters.

This module provides a PyQt5 GUI for interactively adjusting reconstruction
parameters (sample thickness, center of rotation) and viewing the resulting
3D reconstruction.
"""
from typing import Optional
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
    QStackedWidget,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QRegExpValidator
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper
from pyxalign.interactions.viewers.base import ArrayViewer
from pyxalign.api.options.plotting import ArrayViewerOptions
from pyxalign.interactions.point_selector import PointSelector
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
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
        is_already_aligned: bool = False,
    ):
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.phase_projections = phase_projections
        self.array_viewer = None
        self.point_selector = None
        self.current_page = 0  # Track current page (0 = reconstruction params, 1 = post-processing)
        self.is_already_aligned = is_already_aligned

        self.setWindowTitle("3D Reconstruction Parameter Tuner")
        self.resize(1600, 900)

        # Create the UI
        self.init_ui()

        # Auto-display volume if one is already present
        self._display_existing_volume()

    def init_ui(self):
        """Initialize the user interface."""
        # Main layout: horizontal split
        main_layout = QHBoxLayout()

        # ===== LEFT PANEL: Input Controls =====
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()

        # Create page 0: Reconstruction Parameters
        recon_params_page = self.create_reconstruction_parameters_page()
        self.stacked_widget.addWidget(recon_params_page)

        # Create page 1: Post-processing (placeholder for now)
        postproc_page = self.create_postprocessing_page()
        self.stacked_widget.addWidget(postproc_page)

        # Add stacked widget to left layout
        left_layout.addWidget(self.stacked_widget)

        # Create reconstruct button (moved here to be at the bottom)
        self.reconstruct_button = QPushButton("Run 3D Reconstruction")
        self.reconstruct_button.setStyleSheet("background-color: blue; color: white; font-size: 12pt; font-weight: bold; padding: 10px;")
        self.reconstruct_button.clicked.connect(self.on_reconstruct_clicked)
        left_layout.addWidget(self.reconstruct_button)

        # Create navigation buttons
        nav_layout = QHBoxLayout()

        # Left arrow button
        self.left_nav_button = QPushButton("← Reconstruction Parameters")
        self.left_nav_button.setStyleSheet("font-size: 11pt; padding: 8px;")
        self.left_nav_button.clicked.connect(self.on_left_nav_clicked)
        nav_layout.addWidget(self.left_nav_button)

        # Add stretch to push buttons to edges
        nav_layout.addStretch()

        # Right arrow button
        self.right_nav_button = QPushButton("Post-processing →")
        self.right_nav_button.setStyleSheet("font-size: 11pt; padding: 8px;")
        self.right_nav_button.clicked.connect(self.on_right_nav_clicked)
        nav_layout.addWidget(self.right_nav_button)

        # Add navigation layout to left panel
        left_layout.addLayout(nav_layout)

        # Update button states based on current page
        self.update_navigation_buttons()

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

        # Delete button row: right-aligned, above the volume group box
        button_row = QHBoxLayout()
        button_row.addStretch()
        self.delete_volume_button = QPushButton("delete volume from memory")
        self.delete_volume_button.setStyleSheet(
            "background-color: #C0392B; color: white; font-size: 11pt; padding: 4px 14px;"
        )
        self.delete_volume_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.delete_volume_button.clicked.connect(self.on_delete_volume_clicked)
        self.delete_volume_button.setVisible(False)
        button_row.addWidget(self.delete_volume_button)
        right_layout.addLayout(button_row)

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

        right_layout.addWidget(volume_group)

        # Add left and right panels to main layout
        main_layout.addWidget(left_scroll_area, stretch=1)
        main_layout.addWidget(right_panel, stretch=2)

        self.setLayout(main_layout)

    def create_reconstruction_parameters_page(self):
        """Create the reconstruction parameters page."""
        page = QWidget()
        page_layout = QVBoxLayout()
        page.setLayout(page_layout)

        # Create parameter controls
        param_group = QGroupBox("Reconstruction Parameters")
        param_group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        param_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        param_layout = QVBoxLayout()

        # Create tab widget for Basic and Advanced options
        param_tab_widget = QTabWidget()
        param_tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create Basic tab
        basic_tab = QWidget()
        basic_tab_layout = QVBoxLayout()
        basic_tab.setLayout(basic_tab_layout)

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
            bool(self.phase_projections.options.reconstruct.sart.use_circular_constraint)
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

        # Initial Volume
        sart_initial_volume_layout = QHBoxLayout()
        sart_initial_volume_label = QLabel("Initial Volume:")
        sart_initial_volume_label.setStyleSheet("font-size: 11pt;")
        self.sart_initial_volume_combobox = QComboBox()
        for initial_volume in enums.SARTInitialVolumes:
            self.sart_initial_volume_combobox.addItem(initial_volume.value, initial_volume)
        # Set current value
        current_initial_volume = self.phase_projections.options.reconstruct.sart.initial_volume
        index = self.sart_initial_volume_combobox.findData(current_initial_volume)
        if index >= 0:
            self.sart_initial_volume_combobox.setCurrentIndex(index)
        self.sart_initial_volume_combobox.setStyleSheet("font-size: 11pt;")
        self.sart_initial_volume_combobox.currentIndexChanged.connect(self.on_sart_initial_volume_changed)
        sart_initial_volume_layout.addWidget(sart_initial_volume_label)
        sart_initial_volume_layout.addWidget(self.sart_initial_volume_combobox)
        sart_initial_volume_layout.addStretch()
        method_group_layout.addLayout(sart_initial_volume_layout)
        self.sart_controls.extend([sart_initial_volume_label, self.sart_initial_volume_combobox])

        method_group.setLayout(method_group_layout)

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
        self.thickness_spinbox.setMaximum(100_000.0)
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

        # Create center of rotation group for basic tab
        cor_group = QGroupBox("Center of Rotation Selection")
        cor_group.setStyleSheet("QGroupBox { font-size: 12pt; font-weight: bold; }")
        cor_layout = QVBoxLayout()

        # Create button to open point selector
        self.select_cor_button = QPushButton("Select Center of Rotation")
        self.select_cor_button.clicked.connect(self.open_point_selector)
        self.select_cor_button.setStyleSheet("QPushButton { font-size: 12pt; padding: 10px; }")
        cor_layout.addWidget(self.select_cor_button)

        cor_group.setLayout(cor_layout)

        # Create reconstruction geometry group
        geom_group = QGroupBox("Reconstruction Geometry")
        geom_group.setStyleSheet("QGroupBox { font-size: 12pt; font-weight: bold; }")
        geom_layout = QVBoxLayout()

        # Laminography angle spinbox
        lamino_angle_layout = QHBoxLayout()
        lamino_angle_label = QLabel("Laminography Angle (degrees):")
        lamino_angle_label.setStyleSheet("font-size: 11pt;")
        self.lamino_angle_spinbox = QDoubleSpinBox()
        self.lamino_angle_spinbox.setDecimals(6)
        self.lamino_angle_spinbox.setMinimum(-360.0)
        self.lamino_angle_spinbox.setMaximum(360.0)
        self.lamino_angle_spinbox.setSingleStep(0.1)
        self.lamino_angle_spinbox.setValue(
            self.phase_projections.options.experiment.laminography_angle
        )
        self.lamino_angle_spinbox.setStyleSheet("font-size: 11pt;")
        self.lamino_angle_spinbox.valueChanged.connect(self.on_lamino_angle_changed)
        lamino_angle_layout.addWidget(lamino_angle_label)
        lamino_angle_layout.addWidget(self.lamino_angle_spinbox)
        lamino_angle_layout.addStretch()

        # Tilt angle spinbox
        tilt_angle_layout = QHBoxLayout()
        tilt_angle_label = QLabel("Tilt Angle (degrees):")
        tilt_angle_label.setStyleSheet("font-size: 11pt;")
        self.tilt_angle_spinbox = QDoubleSpinBox()
        self.tilt_angle_spinbox.setDecimals(6)
        self.tilt_angle_spinbox.setMinimum(-360.0)
        self.tilt_angle_spinbox.setMaximum(360.0)
        self.tilt_angle_spinbox.setSingleStep(0.1)
        self.tilt_angle_spinbox.setValue(
            self.phase_projections.options.reconstruct.geometry.tilt_angle
        )
        self.tilt_angle_spinbox.setStyleSheet("font-size: 11pt;")
        self.tilt_angle_spinbox.valueChanged.connect(self.on_tilt_angle_changed)
        tilt_angle_layout.addWidget(tilt_angle_label)
        tilt_angle_layout.addWidget(self.tilt_angle_spinbox)
        tilt_angle_layout.addStretch()

        # Skew angle spinbox
        skew_angle_layout = QHBoxLayout()
        skew_angle_label = QLabel("Skew Angle (degrees):")
        skew_angle_label.setStyleSheet("font-size: 11pt;")
        self.skew_angle_spinbox = QDoubleSpinBox()
        self.skew_angle_spinbox.setDecimals(6)
        self.skew_angle_spinbox.setMinimum(-360.0)
        self.skew_angle_spinbox.setMaximum(360.0)
        self.skew_angle_spinbox.setSingleStep(0.1)
        self.skew_angle_spinbox.setValue(
            self.phase_projections.options.reconstruct.geometry.skew_angle
        )
        self.skew_angle_spinbox.setStyleSheet("font-size: 11pt;")
        self.skew_angle_spinbox.valueChanged.connect(self.on_skew_angle_changed)
        skew_angle_layout.addWidget(skew_angle_label)
        skew_angle_layout.addWidget(self.skew_angle_spinbox)
        skew_angle_layout.addStretch()

        # Add angle controls to geometry group layout
        geom_layout.addLayout(lamino_angle_layout)
        geom_layout.addLayout(tilt_angle_layout)
        geom_layout.addLayout(skew_angle_layout)
        geom_group.setLayout(geom_layout)

        # Add method, size, cor, and geometry groups to basic tab
        basic_tab_layout.addWidget(method_group)
        basic_tab_layout.addWidget(size_group)
        basic_tab_layout.addWidget(cor_group)
        basic_tab_layout.addWidget(geom_group)
        basic_tab_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Create Advanced tab
        advanced_tab = QWidget()
        advanced_tab_layout = QVBoxLayout()
        advanced_tab.setLayout(advanced_tab_layout)

        # Add BasicOptionsEditor for phase_projections.options.reconstruct
        self.reconstruct_options_editor = BasicOptionsEditor(
            data=self.phase_projections.options.reconstruct,
            # label="Edit Reconstruction Settings",
            skip_fields=[
                "method",
                "sart",
                "regularization",
                "astra.algorithm_type",
                "geometry.tilt_angle",
                "geometry.skew_angle",
                "geometry",
            ],
            open_panels_list=["astra", "geometry"],
            label="",
        )
        advanced_tab_layout.addWidget(self.reconstruct_options_editor)

        # Add tabs to tab widget
        param_tab_widget.addTab(basic_tab, "Basic")
        param_tab_widget.addTab(advanced_tab, "Advanced")

        # Add tab widget to param layout
        param_layout.addWidget(param_tab_widget)
        param_group.setLayout(param_layout)

        # Update visibility and enabled state based on initial values
        self.update_method_controls_visibility()
        self.update_width_controls_visibility()
        self.update_width_controls_enabled_state()

        # Point selector will be created when button is clicked
        self.point_selector = None

        # Add widgets to page
        page_layout.addWidget(param_group)

        return page

    def create_postprocessing_page(self):
        """Create the post-processing page."""
        page = QWidget()
        page_layout = QVBoxLayout()
        page.setLayout(page_layout)

        # Post-processing controls
        postproc_group = QGroupBox("Post-processing")
        postproc_group.setStyleSheet("QGroupBox { font-size: 13pt; font-weight: bold; }")
        postproc_layout = QVBoxLayout()

        # Rotation angles subsection
        rotation_group = QGroupBox("Rotation Angles")
        rotation_group.setStyleSheet("QGroupBox { font-size: 12pt; font-weight: bold; }")
        rotation_layout = QVBoxLayout()

        # Estimate rotation angles button
        self.estimate_rotation_button = QPushButton("Estimate Rotation Angles")
        self.estimate_rotation_button.setStyleSheet("font-size: 11pt; padding: 8px;")
        self.estimate_rotation_button.clicked.connect(self.on_estimate_rotation_clicked)
        rotation_layout.addWidget(self.estimate_rotation_button)

        # Rotation angle spinboxes
        # X rotation
        x_rotation_layout = QHBoxLayout()
        x_rotation_label = QLabel("X Rotation (degrees):")
        x_rotation_label.setStyleSheet("font-size: 11pt;")
        self.x_rotation_spinbox = QDoubleSpinBox()
        self.x_rotation_spinbox.setDecimals(6)
        self.x_rotation_spinbox.setMinimum(-360.0)
        self.x_rotation_spinbox.setMaximum(360.0)
        self.x_rotation_spinbox.setSingleStep(0.1)
        self.x_rotation_spinbox.setValue(0.0)
        self.x_rotation_spinbox.setStyleSheet("font-size: 11pt;")
        x_rotation_layout.addWidget(x_rotation_label)
        x_rotation_layout.addWidget(self.x_rotation_spinbox)
        x_rotation_layout.addStretch()
        rotation_layout.addLayout(x_rotation_layout)

        # Y rotation
        y_rotation_layout = QHBoxLayout()
        y_rotation_label = QLabel("Y Rotation (degrees):")
        y_rotation_label.setStyleSheet("font-size: 11pt;")
        self.y_rotation_spinbox = QDoubleSpinBox()
        self.y_rotation_spinbox.setDecimals(6)
        self.y_rotation_spinbox.setMinimum(-360.0)
        self.y_rotation_spinbox.setMaximum(360.0)
        self.y_rotation_spinbox.setSingleStep(0.1)
        self.y_rotation_spinbox.setValue(0.0)
        self.y_rotation_spinbox.setStyleSheet("font-size: 11pt;")
        y_rotation_layout.addWidget(y_rotation_label)
        y_rotation_layout.addWidget(self.y_rotation_spinbox)
        y_rotation_layout.addStretch()
        rotation_layout.addLayout(y_rotation_layout)

        # Z rotation
        z_rotation_layout = QHBoxLayout()
        z_rotation_label = QLabel("Z Rotation (degrees):")
        z_rotation_label.setStyleSheet("font-size: 11pt;")
        self.z_rotation_spinbox = QDoubleSpinBox()
        self.z_rotation_spinbox.setDecimals(6)
        self.z_rotation_spinbox.setMinimum(-360.0)
        self.z_rotation_spinbox.setMaximum(360.0)
        self.z_rotation_spinbox.setSingleStep(0.1)
        self.z_rotation_spinbox.setValue(0.0)
        self.z_rotation_spinbox.setStyleSheet("font-size: 11pt;")
        z_rotation_layout.addWidget(z_rotation_label)
        z_rotation_layout.addWidget(self.z_rotation_spinbox)
        z_rotation_layout.addStretch()
        rotation_layout.addLayout(z_rotation_layout)

        # Apply rotation button
        self.apply_rotation_button = QPushButton("Apply Rotation to Volume")
        self.apply_rotation_button.setStyleSheet("font-size: 11pt; padding: 8px;")
        self.apply_rotation_button.clicked.connect(self.on_apply_rotation_clicked)
        rotation_layout.addWidget(self.apply_rotation_button)

        rotation_group.setLayout(rotation_layout)
        postproc_layout.addWidget(rotation_group)

        postproc_group.setLayout(postproc_layout)
        page_layout.addWidget(postproc_group)
        page_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Initially disable all controls until reconstruction is available
        self.update_postprocessing_controls_state(enabled=False)

        return page

    def on_left_nav_clicked(self):
        """Navigate to the previous page (Reconstruction Parameters)."""
        if self.current_page > 0:
            self.current_page -= 1
            self.stacked_widget.setCurrentIndex(self.current_page)
            self.update_navigation_buttons()

    def on_right_nav_clicked(self):
        """Navigate to the next page (Post-processing)."""
        if self.current_page < self.stacked_widget.count() - 1:
            self.current_page += 1
            self.stacked_widget.setCurrentIndex(self.current_page)
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update navigation button states based on current page."""
        # Disable/enable left button
        if self.current_page == 0:
            self.left_nav_button.setEnabled(False)
            self.left_nav_button.setStyleSheet(
                "font-size: 11pt; padding: 8px; color: gray;"
            )
        else:
            self.left_nav_button.setEnabled(True)
            self.left_nav_button.setStyleSheet("font-size: 11pt; padding: 8px;")

        # Disable/enable right button
        if self.current_page >= self.stacked_widget.count() - 1:
            self.right_nav_button.setEnabled(False)
            self.right_nav_button.setStyleSheet(
                "font-size: 11pt; padding: 8px; color: gray;"
            )
        else:
            self.right_nav_button.setEnabled(True)
            self.right_nav_button.setStyleSheet("font-size: 11pt; padding: 8px;")

    def update_postprocessing_controls_state(self, enabled: bool):
        """Enable or disable post-processing controls based on reconstruction availability.

        Args:
            enabled: True to enable controls, False to disable and gray them out.
        """
        self.estimate_rotation_button.setEnabled(enabled)
        self.x_rotation_spinbox.setEnabled(enabled)
        self.y_rotation_spinbox.setEnabled(enabled)
        self.z_rotation_spinbox.setEnabled(enabled)
        self.apply_rotation_button.setEnabled(enabled)

        # Update styling to show grayed out state
        if not enabled:
            self.estimate_rotation_button.setStyleSheet("font-size: 11pt; padding: 8px; color: gray;")
            self.apply_rotation_button.setStyleSheet("font-size: 11pt; padding: 8px; color: gray;")
        else:
            self.estimate_rotation_button.setStyleSheet("font-size: 11pt; padding: 8px;")
            self.apply_rotation_button.setStyleSheet("font-size: 11pt; padding: 8px;")

    def reset_postprocessing_values(self):
        """Reset all post-processing values to defaults."""
        # Reset rotation spinboxes to 0
        self.x_rotation_spinbox.setValue(0.0)
        self.y_rotation_spinbox.setValue(0.0)
        self.z_rotation_spinbox.setValue(0.0)

    def on_estimate_rotation_clicked(self):
        """Estimate optimal rotation angles for the volume."""
        # Check if volume exists
        if self.phase_projections.volume is None or self.phase_projections.volume.data is None:
            # Show error or warning
            print("Error: No volume available. Please run 3D reconstruction first.")
            return

        # Disable button during estimation
        self.estimate_rotation_button.setEnabled(False)
        self.estimate_rotation_button.setText("Estimating...")

        try:
            # Run estimation
            load_bar_func_wrapper = loading_bar_wrapper(
                "Estimating optimal rotation angles...", block_all_windows=True
            )(self.phase_projections.volume.get_optimal_rotation_of_reconstruction)
            load_bar_func_wrapper()

            # Populate spinboxes with estimated values
            if hasattr(self.phase_projections.volume, 'optimal_rotation_angles'):
                self.x_rotation_spinbox.setValue(self.phase_projections.volume.optimal_rotation_angles[0])
                self.y_rotation_spinbox.setValue(self.phase_projections.volume.optimal_rotation_angles[1])
                self.z_rotation_spinbox.setValue(self.phase_projections.volume.optimal_rotation_angles[2])

        finally:
            # Re-enable button
            self.estimate_rotation_button.setEnabled(True)
            self.estimate_rotation_button.setText("Estimate Rotation Angles")

    def on_apply_rotation_clicked(self):
        """Apply rotation to the volume using the specified angles."""
        # Check if volume exists
        if self.phase_projections.volume is None or self.phase_projections.volume.data is None:
            # Show error or warning
            print("Error: No volume available. Please run 3D reconstruction first.")
            return

        # Disable button during rotation
        self.apply_rotation_button.setEnabled(False)
        self.apply_rotation_button.setText("Rotating...")

        try:
            # Get values from spinboxes and set optimal_rotation_angles
            self.phase_projections.volume.optimal_rotation_angles = [
                self.x_rotation_spinbox.value(),
                self.y_rotation_spinbox.value(),
                self.z_rotation_spinbox.value()
            ]

            # Apply rotation
            load_bar_func_wrapper = loading_bar_wrapper(
                "Rotating volume...", block_all_windows=True
            )(self.phase_projections.volume.rotate_reconstruction)
            load_bar_func_wrapper()

            # Refresh array viewer
            if self.array_viewer is not None:
                self.array_viewer.array3d = self.phase_projections.volume.data
                self.array_viewer.refresh_frame()

            # Reset rotation values to 0 after applying
            self.reset_postprocessing_values()

        finally:
            # Re-enable button
            self.apply_rotation_button.setEnabled(True)
            self.apply_rotation_button.setText("Apply Rotation to Volume")

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

    def on_sart_initial_volume_changed(self, index: int):
        """Update SART initial_volume when combobox selection changes."""
        initial_volume = self.sart_initial_volume_combobox.itemData(index)
        self.phase_projections.options.reconstruct.sart.initial_volume = initial_volume

    def on_thickness_changed(self, value: float):
        """Update sample thickness when spinbox value changes."""
        self.phase_projections.options.experiment.sample_thickness = value

    def on_lamino_angle_changed(self, value: float):
        """Update laminography angle when spinbox value changes."""
        self.phase_projections.options.experiment.laminography_angle = value

    def on_tilt_angle_changed(self, value: float):
        """Update tilt angle when spinbox value changes."""
        self.phase_projections.options.reconstruct.geometry.tilt_angle = value

    def on_skew_angle_changed(self, value: float):
        """Update skew angle when spinbox value changes."""
        self.phase_projections.options.reconstruct.geometry.skew_angle = value

    def open_point_selector(self):
        """Open the point selector window for selecting center of rotation."""
        initial_center = (
            int(self.phase_projections.center_of_rotation[1]),
            int(self.phase_projections.center_of_rotation[0]),
        )

        # Create and show point selector
        # Pass None for image - PointSelector will calculate projection sum lazily when needed
        self.point_selector = PointSelector(
            image=None,
            initial_point=initial_center,
            projections=self.phase_projections.data,
        )

        # Make the window modal to block interaction with other windows
        self.point_selector.setWindowModality(Qt.ApplicationModal)

        # Connect the point_selected signal to update center of rotation
        self.point_selector.point_selected.connect(self.on_point_selected)

        # If already aligned, disable x-position control
        if self.is_already_aligned:
            self.point_selector.spinboxes["x"].setEnabled(False)
            self.point_selector.spinboxes["x"].setStyleSheet(
                "QSpinBox { font-size: 12pt; color: gray; }"
            )

        self.point_selector.show()

    def on_point_selected(self, point: tuple):
        """Update center of rotation when point is selected and close window.

        Args:
            point: Tuple of (x, y) coordinates from the point selector.
        """
        x, y = point
        # PointSelector returns (x, y), but center_of_rotation is stored as [y, x]
        # Only update x if not already aligned
        if not self.is_already_aligned:
            self.phase_projections.center_of_rotation[1] = x
        self.phase_projections.center_of_rotation[0] = y

    def on_center_of_rotation_changed(self, point: tuple):
        """Update center of rotation when point selector changes.

        Args:
            point: Tuple of (x, y) coordinates from the point selector.
        """
        x, y = point
        # PointSelector returns (x, y), but center_of_rotation is stored as [y, x]
        # Only update x if not already aligned
        if not self.is_already_aligned:
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

            # Enable post-processing controls now that reconstruction is available
            self.update_postprocessing_controls_state(enabled=True)

            # Reset post-processing values to defaults
            self.reset_postprocessing_values()

            self.delete_volume_button.setVisible(True)

        finally:
            # Re-enable button
            self.reconstruct_button.setEnabled(True)
            self.reconstruct_button.setText("Run 3D Reconstruction")

    def on_delete_volume_clicked(self):
        """Delete the volume array from memory, clear the display, and release ASTRA objects."""
        self.phase_projections.volume.data = None
        if self.array_viewer is not None:
            self.viewer_layout.removeWidget(self.array_viewer)
            self.array_viewer.deleteLater()
            self.array_viewer = None
        self.phase_projections.volume.clear_astra_objects()
        self.delete_volume_button.setVisible(False)
        self.update_postprocessing_controls_state(enabled=False)

    def _display_existing_volume(self):
        """Display volume data already present at phase_projections.volume.data."""
        if (
            self.phase_projections.volume is not None
            and self.phase_projections.volume.data is not None
        ):
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
            self.delete_volume_button.setVisible(True)
            self.update_postprocessing_controls_state(enabled=True)

    def start(self):
        """Show the widget."""
        self.show()


@switch_to_matplotlib_qt_backend
def launch_reconstruction_parameter_tuner(
    phase_projections: "p.PhaseProjections",
    wait_until_closed: bool = False,
    is_already_aligned: bool = False,
) -> ReconstructionParameterTuner:
    """Launch the reconstruction parameter tuner GUI.

    This GUI allows interactive adjustment of reconstruction parameters
    (sample thickness and center of rotation) and displays the resulting
    3D reconstruction using the ArrayViewer.

    Args:
        phase_projections: PhaseProjections object containing the data.
        wait_until_closed: If True, the application starts a blocking call
            until the GUI window is closed.
        is_already_aligned: If True, prevents modification of the x-position
            in the center of rotation selection. Defaults to False.

    Returns:
        The ReconstructionParameterTuner widget instance.

    Example:
        Launch the parameter tuning GUI::

            gui = pyxalign.gui.launch_reconstruction_parameter_tuner(
                task.phase_projections
            )
    """
    app = QApplication.instance() or QApplication([])
    gui = ReconstructionParameterTuner(
        phase_projections=phase_projections,
        is_already_aligned=is_already_aligned
    )
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui
