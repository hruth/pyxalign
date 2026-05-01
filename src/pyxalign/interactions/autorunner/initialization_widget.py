"""
Widget for editing InitializationConfig with data preview.

This module provides an interactive GUI for configuring initialization parameters
while viewing the loaded data.
"""

import copy
from typing import Optional
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QApplication,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from pyxalign.interactions.io.input_data_viewer import StandardDataViewer
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.api.options.projections import ProjectionOptions
from pyxalign.api.options.transform import RotationOptions, ShearOptions
from pyxalign.data_structures.projections import ComplexProjections
from pyxalign.interactions.viewers.base import ArrayViewer
from pyxalign.autorunner.config import InitializationConfig
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend, center_window_on_screen
from pyxalign.io.loaders.base import StandardData
from pyxalign.io.loaders.utils import convert_projection_dict_to_array


class InitializationConfigWidget(QWidget):
    """
    Widget for editing InitializationConfig with StandardDataViewer.

    The widget displays:
    - Left side: StandardDataViewer showing loaded projection data
    - Right side: BasicOptionsEditor for editing InitializationConfig
    - Bottom right: Green button to confirm and close

    Signals:
        config_confirmed: Emitted when the initialize button is clicked
    """

    config_confirmed = pyqtSignal()

    def __init__(
        self,
        standard_data: Optional[StandardData] = None,
        initialization_config: Optional[InitializationConfig] = None,
        parent=None,
    ):
        """
        Initialize the widget.

        Args:
            standard_data: StandardData object containing projection data
            initialization_config: InitializationConfig instance to edit.
                If None, creates a new instance with defaults.
            parent: Parent widget
        """
        super().__init__(parent)

        self._standard_data = standard_data
        self._array_viewer: Optional[ArrayViewer] = None
        self.complex_projections: Optional[ComplexProjections] = None
        self._config_at_last_initialize: Optional[InitializationConfig] = None

        # Store or create config
        if initialization_config is None:
            initialization_config = InitializationConfig()
        self.config = initialization_config

        # Setup UI
        self.setWindowTitle("Initialize Projections Configuration")
        self.resize(1600, 800)

        # Create main horizontal layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left side: StandardDataViewer
        self.data_viewer = StandardDataViewer(data=standard_data, parent=self)
        main_layout.addWidget(self.data_viewer, stretch=2)

        # Right side: vertical layout for options editor and button
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Options editor
        self.options_editor = BasicOptionsEditor(
            data=self.config,
            skip_fields=[],
            label="Initialization Configuration",
            parent=self,
        )
        right_layout.addWidget(self.options_editor)

        # Green button
        self.initialize_button = QPushButton("Preview Projections Object")
        self.initialize_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 14px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            """
        )

        # Make text bold
        button_font = QFont()
        button_font.setBold(True)
        self.initialize_button.setFont(button_font)

        # Connect button signal
        self.initialize_button.clicked.connect(self._on_initialize_clicked)

        right_layout.addWidget(self.initialize_button)

        # Add right panel to main layout
        main_layout.addWidget(right_panel, stretch=1)

    def _on_initialize_clicked(self):
        """Build a ComplexProjections object from current config and display its data in an ArrayViewer."""
        if self._array_viewer is not None:
            self._array_viewer.close()
            self._array_viewer = None

        new_array_size = self._standard_data.get_minimum_size_for_projection_array()
        new_array_size += self.config.pad
        projection_array = convert_projection_dict_to_array(
            self._standard_data.projections, new_array_size, pad_with_mode=True
        )

        projection_options = ProjectionOptions()
        projection_options.experiment.laminography_angle = self.config.laminography_angle
        projection_options.experiment.pixel_size = self._standard_data.pixel_size
        if self.config.rotation_angle != 0:
            projection_options.input_processing.rotation = RotationOptions(
                enabled=True, angle=self.config.rotation_angle
            )
        if self.config.shear_angle != 0:
            projection_options.input_processing.shear = ShearOptions(
                enabled=True, angle=self.config.shear_angle
            )

        complex_projections = ComplexProjections(
            projections=projection_array,
            angles=self._standard_data.angles,
            scan_numbers=self._standard_data.scan_numbers,
            options=projection_options,
            probe_positions=list(self._standard_data.probe_positions.values()),
            probe=self._standard_data.probe,
            skip_pre_processing=False,
            file_paths=list(self._standard_data.file_paths.values()),
        )
        if self.config.remove_scan_numbers is not None:
            complex_projections.drop_projections(self.config.remove_scan_numbers)

        self.complex_projections = complex_projections
        self._config_at_last_initialize = copy.deepcopy(self.config)
        self._array_viewer = ArrayViewer(array3d=complex_projections.data)
        self._array_viewer.setWindowTitle("Projection Array Preview")
        self._array_viewer.show()

    def closeEvent(self, event):
        if self._array_viewer is not None:
            self._array_viewer.close()
            self._array_viewer = None
        super().closeEvent(event)

    def setStandardData(self, data: StandardData):
        """
        Set the StandardData to display in the viewer.

        Args:
            data: StandardData object to display
        """
        self._standard_data = data
        self.data_viewer.setStandardData(data)
        if self._array_viewer is not None:
            self._array_viewer.close()
            self._array_viewer = None


@switch_to_matplotlib_qt_backend
def launch_initialization_config_widget(
    standard_data: StandardData,
    initialization_config: Optional[InitializationConfig] = None,
    wait_until_closed: bool = True,
):
    """
    Launch the initialization config editor GUI.

    This function creates and shows a blocking modal dialog that allows the user
    to edit InitializationConfig while viewing the loaded data. The function
    blocks until the user clicks "Initialize Projections Object" or closes the window.

    Args:
        standard_data: StandardData object containing projection data to display
        initialization_config: InitializationConfig instance to edit.
            If None, creates a new instance with defaults.

    Returns:
        InitializationConfig: The edited configuration if the user clicked the
            initialize button, or None if the window was closed without confirming.

    Example:
        >>> from pyxalign.io.loaders import load_data_from_pear_format
        >>> from pyxalign.gui import launch_initialization_config_widget
        >>>
        >>> # Load data
        >>> standard_data = load_data_from_pear_format(...)
        >>>
        >>> # Launch editor
        >>> config = launch_initialization_config_widget(standard_data)
        >>>
        >>> if config is not None:
        >>>     print(f"Initializing with pad={config.pad}")
        >>>     # Use config to initialize projections...
    """
    app = QApplication.instance() or QApplication([])

    gui = InitializationConfigWidget(
        standard_data=standard_data,
        initialization_config=initialization_config,
    )

    if not wait_until_closed:
        return gui

    def on_config_confirmed():
        app.quit()

    gui.config_confirmed.connect(on_config_confirmed)

    center_window_on_screen(gui, width_fraction=0.75, height_fraction=0.75)
    gui.show()
    app.exec()
    gui.close()

    return gui.config

    # if result:
    #     return result["config"]
    # else:
    #     return None
