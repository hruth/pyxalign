"""
GUI wrapper widget for autorunner steps.

Provides a wrapper that adds 'Proceed' and 'End Process' buttons to autorunner GUIs.
"""

import os
from typing import Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QApplication, QFileDialog


class AutorunnerProcessEnded(Exception):
    """Exception raised when user ends the autorunner process."""

    pass


class AutorunnerGUIWrapper(QWidget):
    """
    Wrapper widget that contains the original GUI and adds control buttons.

    Provides 'Proceed' and 'End Process' buttons to control the autorunner workflow.
    """

    def __init__(
        self,
        content_widget: QWidget,
        title: str = "Autorunner Step",
        task=None,
        checkpoints_folder: Optional[str] = None,
    ):
        super().__init__()
        self.content_widget = content_widget
        self.should_proceed = False
        self.should_end_process = False
        self.task = task
        self.checkpoints_folder = checkpoints_folder

        self.setWindowTitle(title)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Add the original content widget
        main_layout.addWidget(self.content_widget)

        # Add button layout
        button_layout = QHBoxLayout()

        # Save Current Task button (on the left)
        if self.task is not None:
            self.save_task_button = QPushButton("Save Current Task")
            self.save_task_button.setStyleSheet(
                "QPushButton { background-color: #4dabf7; color: white; font-weight: bold; }"
            )
            self.save_task_button.clicked.connect(self._on_save_task)
            button_layout.addWidget(self.save_task_button)

        # Add spacer to push end/proceed buttons to the right
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

    def _on_save_task(self):
        """Handle save current task button click."""
        if self.task is None:
            return

        # Determine the default directory
        default_dir = self.checkpoints_folder if self.checkpoints_folder is not None else ""

        # Ensure the checkpoints folder exists if specified
        if default_dir and not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)

        # Open file dialog for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Task",
            default_dir,
            "HDF5 Files (*.h5);;All Files (*)",
        )

        if file_path:
            # Ensure the file has .h5 extension
            if not file_path.endswith(".h5"):
                file_path += ".h5"

            # Save the task
            self.task.save_task(file_path)
            print(f"Task saved to: {file_path}")

    def wait_for_user_action(self):
        """Wait for the user to click either proceed or end process."""
        app = QApplication.instance() or QApplication([])
        self.show()
        app.exec_()

        if self.should_end_process:
            raise AutorunnerProcessEnded("User requested to end the autorunner process")

        return self.should_proceed
