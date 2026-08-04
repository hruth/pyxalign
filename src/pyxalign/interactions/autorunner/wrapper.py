"""
GUI wrapper widget for autorunner steps.

Provides a wrapper that adds 'Proceed' and 'End Process' buttons to autorunner GUIs.
"""

import os
from typing import Optional
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QApplication,
    QFileDialog,
    QDialog,
    QListWidget,
    QStackedWidget,
    QTextBrowser,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QTextCharFormat, QTextCursor

from pyxalign.autorunner.abstract import _update_all_config_parameters
from pyxalign.autorunner.enums import AutorunnerStep
from pyxalign.interactions.utils.misc import center_window_on_screen


# _HEADING_COLORS = {1: QColor("#4dabf7"), 2: QColor("#74c0fc")}
_HEADING_COLORS = {1: QColor("#060C78"), 2: QColor("#143dad")}



def _style_headings(browser):
    """Apply colors to h1/h2 blocks after markdown is loaded."""
    block = browser.document().begin()
    while block.isValid():
        level = block.blockFormat().headingLevel()
        if level in _HEADING_COLORS:
            fmt = QTextCharFormat()
            fmt.setForeground(_HEADING_COLORS[level])
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(fmt)
        block = block.next()


_STEP_HELP_TEXTS = {
    AutorunnerStep.AUTORUNNER_CONFIGURATION_WINDOW: """\
# Autorunner Configuration

This step lets you configure the overall autorunner settings before processing begins.

You can typically click **Proceed** without changing any of these parameters for most analyses.

## Tips for "state" settings

- The "state" parameters dictate how the program interacts with the state file. The state
  file (`autorunner_state_file.yaml`) can be found in the state folder.
- Enabling `use_state_file_settings` means that the settings recorded in the state
  file will be used to update the settings shown in the GUI.
- Enabling `update_state_file` means that any changes to the settings in the GUI will
  be written into the state file anytime **Proceed** is clicked.

## Tips for "checkpoint" settings

- You can skip steps in the pyxalign-autorunner program by loading from a checkpoint. The
  "checkpoint" settings configure this behavior.
- **Example:** say you had to close the window after the loading screen and want to pick up
  where you left off. Skip the loading screen by:
    1. Enable `load_from_checkpoint`
    2. Change `which_checkpoint` to `after_loading`
    3. Click **Proceed**
- You can open any step with any existing pyxalign task file. Load a custom task and skip
  to any step by:
    1. Enable `load_from_checkpoint`
    2. Change `which_checkpoint` to the appropriate starting point
    3. Enable `load_from_custom_task`
    4. Specify the path to the task using the `custom_task_path` dialog
    5. Click **Proceed**
""",
    AutorunnerStep.DATA_LOADER_WINDOW: """\
# Initialization

*TODO: Add help text for this step.*

This step configures the initialization parameters for the projections object.
""",
    AutorunnerStep.COMPLEX_PROJECTIONS_WINDOW: """\
# Cross Correlation Alignment

*TODO: Add help text for this step.*

This step performs cross-correlation alignment on the projections.
""",
    AutorunnerStep.PHASE_UNWRAPPING_WINDOW: """\
# Phase Unwrapping

*TODO: Add help text for this step.*

This step unwraps the phase of the complex projections.
""",
    AutorunnerStep.UNWRAPPED_PROJECTIONS_WINDOW: """\
# Projection Matching Sequence

*TODO: Add help text for this step.*

This step runs the projection-matching alignment sequence.
""",
}

_GENERAL_HELP_TEXT = """\
# General Help

## Tips

- Hover over settings and buttons to see a tooltip.
"""


class HelpDialog(QDialog):
    """Sidebar-based help/tips dialog with step-specific and general content."""

    def __init__(self, step_title: AutorunnerStep, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help / Tips")
        self.resize(640, 420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        pages = [
            ("Step Tips", _STEP_HELP_TEXTS.get(step_title, f"No tips available for: {step_title}")),
            ("General Help", _GENERAL_HELP_TEXT),
        ]

        outer_layout = QHBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.setLayout(outer_layout)

        # Left sidebar
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(130)
        self.sidebar.setStyleSheet(
            "QListWidget { background-color: #2e2e2e; border: none; }"
            "QListWidget::item { color: #cccccc; padding: 10px 12px; border: none; }"
            "QListWidget::item:selected { background-color: #444; color: white; }"
            "QListWidget::item:hover:!selected { background-color: #3a3a3a; }"
        )
        for name, _ in pages:
            self.sidebar.addItem(name)
        outer_layout.addWidget(self.sidebar)

        # Right side: content area + font size controls
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        outer_layout.addLayout(right_layout)

        self._font_size = 10
        self._browsers = []
        self._texts = []
        self.stacked = QStackedWidget()
        for _, text in pages:
            browser = QTextBrowser()
            browser.setReadOnly(True)
            browser.setStyleSheet("QTextBrowser { padding: 12px; }")
            self.stacked.addWidget(browser)
            self._browsers.append(browser)
            self._texts.append(text)
        self._refresh_browsers()
        right_layout.addWidget(self.stacked)

        # Font size controls
        font_bar = QHBoxLayout()
        font_bar.setContentsMargins(8, 4, 8, 4)
        font_bar.addStretch()
        for label, slot in (("−", self._zoom_out), ("+", self._zoom_in)):
            btn = QPushButton(label)
            btn.setFixedWidth(36)
            btn.setStyleSheet("QPushButton { font-weight: bold; }")
            btn.clicked.connect(slot)
            font_bar.addWidget(btn)
        right_layout.addLayout(font_bar)

        self.sidebar.currentRowChanged.connect(self.stacked.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

    def _refresh_browsers(self):
        font = QFont()
        font.setPointSize(self._font_size)
        for browser, text in zip(self._browsers, self._texts):
            browser.setFont(font)
            browser.setMarkdown(text)
            _style_headings(browser)

    def _zoom_in(self):
        self._font_size += 1
        self._refresh_browsers()

    def _zoom_out(self):
        self._font_size = max(6, self._font_size - 1)
        self._refresh_browsers()


class AutorunnerProcessEnded(Exception):
    """Exception raised when user ends the autorunner process."""

    pass


class AutorunnerRestarted(Exception):
    """Exception raised when user requests to restart the autorunner."""

    pass


class AutorunnerGUIWrapper(QWidget):
    """
    Wrapper widget that contains the original GUI and adds control buttons.

    Provides 'Proceed' and 'End Process' buttons to control the autorunner workflow.
    """

    def __init__(
        self,
        content_widget: QWidget,
        title: AutorunnerStep = AutorunnerStep.AUTORUNNER_CONFIGURATION_WINDOW,
        task=None,
        checkpoints_folder: Optional[str] = None,
        config=None,
        state_file_path: Optional[str] = None,
        show_sync_button: bool = True,
        show_restart_button: bool = True,
    ):
        super().__init__()
        self.content_widget = content_widget
        self.should_proceed = False
        self.should_end_process = False
        self.task = task
        self.checkpoints_folder = checkpoints_folder
        self.config = config
        self.state_file_path = state_file_path

        self.should_restart = False
        self.setWindowTitle(title)
        self._step_title = title

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top bar: Help/Tips button on the right
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.help_button = QPushButton("Help / Tips")
        self.help_button.setStyleSheet(
            "QPushButton { background-color: #868e96; color: white; font-weight: bold; }"
        )
        self.help_button.setToolTip("Open help and tips for this step.")
        self.help_button.clicked.connect(self._on_help)
        top_bar.addWidget(self.help_button)
        main_layout.addLayout(top_bar)

        # Add the original content widget
        main_layout.addWidget(self.content_widget)

        # Add button layout
        button_layout = QHBoxLayout()

        # Save Current Task button (on the left)
        if self.task is not None:
            self.save_task_button = QPushButton("Save Current Task")
            self.save_task_button.setStyleSheet(
                "QPushButton { background-color: #1971c2; color: white; font-weight: bold; }"
            )
            self.save_task_button.setToolTip(
                "Open a file dialog window and save the current task object to an HDF5 file."
            )
            self.save_task_button.clicked.connect(self._on_save_task)
            button_layout.addWidget(self.save_task_button)

        # Sync to State File button (on the left)
        if show_sync_button and self.config is not None and self.state_file_path is not None:
            self.sync_state_button = QPushButton("Sync to State File")
            self.sync_state_button.setStyleSheet(
                "QPushButton { background-color: #6741d9; color: white; font-weight: bold; }"
            )
            self.sync_state_button.setToolTip(
                "Update the state file with the current task's parameters."
            )
            self.sync_state_button.clicked.connect(self._on_sync_to_state_file)
            button_layout.addWidget(self.sync_state_button)

        # Add spacer to push end/proceed buttons to the right
        button_layout.addStretch()

        # Restart Autorunner button (orange)
        if show_restart_button:
            self.restart_button = QPushButton("Restart Autorunner")
            self.restart_button.setStyleSheet(
                "QPushButton { background-color: #d9480f; color: white; font-weight: bold; }"
            )
            self.restart_button.setToolTip(
                "Restart the autorunner; this will return you to the autorunner configuration window."
            )
            self.restart_button.clicked.connect(self._on_restart)
            button_layout.addWidget(self.restart_button)

        # End Process button (red)
        self.end_button = QPushButton("End Process")
        self.end_button.setStyleSheet(
            "QPushButton { background-color: #c92a2a; color: white; font-weight: bold; }"
        )
        self.end_button.setToolTip("Stop the autorunner and close all windows.")
        self.end_button.clicked.connect(self._on_end_process)
        button_layout.addWidget(self.end_button)

        # Proceed button (green, on the right)
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.setStyleSheet(
            "QPushButton { background-color: #2f9e44; color: white; font-weight: bold; }"
            "QPushButton:disabled { background-color: #adb5bd; color: #6c757d; }"
        )
        self.proceed_button.setToolTip("Continue to the next step.")
        self.proceed_button.clicked.connect(self._on_proceed)
        button_layout.addWidget(self.proceed_button)

        main_layout.addLayout(button_layout)

    def _on_help(self):
        """Open the Help/Tips dialog (non-modal)."""
        if not hasattr(self, "_help_dialog") or not self._help_dialog.isVisible():
            self._help_dialog = HelpDialog(self._step_title, parent=self)
            self._help_dialog.show()
        else:
            self._help_dialog.raise_()
            self._help_dialog.activateWindow()

    def _on_proceed(self):
        self.should_proceed = True
        QApplication.closeAllWindows()
        QApplication.instance().quit()

    def _on_end_process(self):
        reply = QMessageBox.question(
            self,
            "End Process",
            "Are you sure you want to end the process?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.should_end_process = True
            QApplication.closeAllWindows()
            QApplication.instance().quit()

    def _on_restart(self):
        reply = QMessageBox.question(
            self,
            "Restart Autorunner",
            "Are you sure you want to restart?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.should_restart = True
            QApplication.closeAllWindows()
            QApplication.instance().quit()

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

            # Save the task (include pma sequence volumes so they are preserved)
            self.task.save_task(file_path, save_pma_sequence_volumes=True)
            print(f"Task saved to: {file_path}")

    def _on_sync_to_state_file(self):
        """Handle sync to state file button click."""
        if self.config is None or self.state_file_path is None:
            return

        # Update all config parameters from the task
        _update_all_config_parameters(self.task, self.config)

        # Save the config to the state file
        self.config.save_to_dict(self.state_file_path)
        print(f"Configuration synced to state file: {self.state_file_path}")

    def wait_for_user_action(self):
        """Wait for the user to click either proceed or end process."""
        app = QApplication.instance() or QApplication([])
        center_window_on_screen(self, width_fraction=0.75, height_fraction=0.75)
        self.show()
        app.exec_()

        if self.should_end_process:
            raise AutorunnerProcessEnded("User requested to end the autorunner process")

        if self.should_restart:
            raise AutorunnerRestarted("User requested to restart the autorunner")

        return self.should_proceed
