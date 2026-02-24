"""
GUI wrapper widget for autorunner steps.

Provides a wrapper that adds 'Proceed' and 'End Process' buttons to autorunner GUIs.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QApplication


class AutorunnerProcessEnded(Exception):
    """Exception raised when user ends the autorunner process."""

    pass


class AutorunnerGUIWrapper(QWidget):
    """
    Wrapper widget that contains the original GUI and adds control buttons.

    Provides 'Proceed' and 'End Process' buttons to control the autorunner workflow.
    """

    def __init__(self, content_widget: QWidget, title: str = "Autorunner Step"):
        super().__init__()
        self.content_widget = content_widget
        self.should_proceed = False
        self.should_end_process = False

        self.setWindowTitle(title)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Add the original content widget
        main_layout.addWidget(self.content_widget)

        # Add button layout
        button_layout = QHBoxLayout()

        # Add spacer to push buttons to the right
        button_layout.addStretch()

        # End Process button (red, on the left)
        self.end_button = QPushButton("End Process")
        self.end_button.setStyleSheet(
            "QPushButton { background-color: #ff6b6b; color: white; font-weight: bold; }"
        )
        self.end_button.clicked.connect(self._on_end_process)
        button_layout.addWidget(self.end_button)

        # Proceed button (green, on the right)
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.setStyleSheet(
            "QPushButton { background-color: #51cf66; color: white; font-weight: bold; }"
        )
        self.proceed_button.clicked.connect(self._on_proceed)
        button_layout.addWidget(self.proceed_button)

        main_layout.addLayout(button_layout)

    def _on_proceed(self):
        """Handle proceed button click."""
        self.should_proceed = True
        self.close()

    def _on_end_process(self):
        """Handle end process button click."""
        self.should_end_process = True
        self.close()

    def wait_for_user_action(self):
        """Wait for the user to click either proceed or end process."""
        app = QApplication.instance() or QApplication([])
        self.show()
        app.exec_()

        if self.should_end_process:
            raise AutorunnerProcessEnded("User requested to end the autorunner process")

        return self.should_proceed
