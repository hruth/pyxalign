"""
Interactive phase unwrapping widget for complex projections.

This module provides a GUI interface for unwrapping the phase of complex projections
using configurable options and real-time visualization of results.
"""

import sys
import traceback
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
)

from pyxalign.api.options.options import PhaseUnwrapOptions
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.interactions.custom import action_button_style_sheet
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.plotting.interactive.base import ArrayViewer


class PhaseUnwrapWidget(QWidget):
    """
    Interactive widget for unwrapping the phase of complex projections.

    This widget provides an options editor for PhaseUnwrapOptions, an ArrayViewer
    for displaying the unwrapped phase results, and a button to execute the
    phase unwrapping process.

    Signals
    -------
    phase_unwrapped : np.ndarray
        Emitted when phase unwrapping is completed, containing the unwrapped phase data.
    """

    phase_unwrapped = pyqtSignal()#np.ndarray)

    def __init__(
        self, task: LaminographyAlignmentTask, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.task = task

        self.setWindowTitle("Phase Unwrapping")
        self.resize(1200, 800)

        # Create main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Create left panel for options and controls
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout()
        left_panel.setLayout(left_panel_layout)
        left_panel.setMaximumWidth(400)

        # Create options editor for PhaseUnwrapOptions
        self.options_editor = BasicOptionsEditor(self.task.complex_projections.options.phase_unwrap)
        left_panel_layout.addWidget(self.options_editor)

        # Create unwrap phase button
        self.unwrap_button = QPushButton("Unwrap Phase")
        self.unwrap_button.setStyleSheet(action_button_style_sheet)
        self.unwrap_button.clicked.connect(self.unwrap_phase)
        self.unwrap_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Initially disable button if no task is provided
        if self.task is None or self.task.complex_projections is None:
            self.unwrap_button.setEnabled(False)

        left_panel_layout.addWidget(self.unwrap_button)

        # Add spacer to push everything to the top
        left_panel_layout.addStretch()

        # Create ArrayViewer for displaying unwrapped phase
        self.array_viewer = ArrayViewer(process_func=lambda x: x)
        self.array_viewer.setEnabled(False)  # Initially disabled

        # Add widgets to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.array_viewer)

    def set_task(self, task: LaminographyAlignmentTask):
        """
        Set the LaminographyAlignmentTask for this widget.

        Parameters
        ----------
        task : LaminographyAlignmentTask
            The task containing complex projections to unwrap.
        """
        self.task = task

        # Enable the unwrap button if we have complex projections
        if self.task is not None and self.task.complex_projections is not None:
            self.unwrap_button.setEnabled(True)
        else:
            self.unwrap_button.setEnabled(False)

    def unwrap_phase(self):
        """
        Execute phase unwrapping using the current options and display results.
        """
        if self.task is None or self.task.complex_projections is None:
            print("No complex projections available for phase unwrapping")
            return

        try:
            # Disable button during processing
            self.unwrap_button.setEnabled(False)
            self.unwrap_button.setText("Unwrapping Phase...")

            # # Update the task's phase unwrap options with current editor values
            # self.task.complex_projections.options.phase_unwrap = self.options_editor._data

            # Perform phase unwrapping
            print("Starting phase unwrapping...")
            self.task.get_unwrapped_phase()

            # # Get the unwrapped phase data
            # unwrapped_phase = self.task.phase_projections.data

            # Update ArrayViewer with unwrapped phase
            sort_idx = np.argsort(self.task.phase_projections.angles)
            title_strings = [f"scan {x}" for x in self.task.phase_projections.scan_numbers]

            self.array_viewer.reinitialize_all(
                self.task.phase_projections.data,
                sort_idx=sort_idx,
                extra_title_strings_list=title_strings,
            )

            # Enable the ArrayViewer
            self.array_viewer.setEnabled(True)

            # Emit signal with unwrapped phase data
            self.phase_unwrapped.emit()#unwrapped_phase)

            print("Phase unwrapping completed successfully")

        except Exception as e:
            print(f"An error occurred: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
        finally:
            # Re-enable button and reset text
            self.unwrap_button.setEnabled(True)
            self.unwrap_button.setText("Unwrap Phase")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = PhaseUnwrapWidget()
    widget.show()
    sys.exit(app.exec_())
