"""
Widget for editing InitializationConfig with data preview.

This module provides an interactive GUI for configuring initialization parameters
while viewing the loaded data.
"""

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
from pyxalign.autorunner.config import InitializationConfig
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.io.loaders.base import StandardData


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
        self.initialize_button = QPushButton("Initialize Projections Object")
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
        """Handle initialize button click."""
        self.config_confirmed.emit()
        self.close()

    def setStandardData(self, data: StandardData):
        """
        Set the StandardData to display in the viewer.

        Args:
            data: StandardData object to display
        """
        self.data_viewer.setStandardData(data)


@switch_to_matplotlib_qt_backend
def launch_initialization_config_widget(
    standard_data: StandardData,
    initialization_config: Optional[InitializationConfig] = None,
) -> Optional[InitializationConfig]:
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

    # Store result
    result = {}

    def on_config_confirmed():
        result["config"] = gui.config
        app.quit()

    gui.config_confirmed.connect(on_config_confirmed)

    gui.show()
    app.exec()
    gui.close()

    if result:
        return result["config"]
    else:
        return None
