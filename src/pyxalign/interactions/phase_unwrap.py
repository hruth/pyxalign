"""
Interactive phase unwrapping widget for complex projections.

This module provides a GUI interface for unwrapping the phase of complex projections
using configurable options and real-time visualization of results.
"""

import sys
import traceback
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QMessageBox,
    QErrorMessage,
    QDialog,
    QSpacerItem,
)

from pyxalign.api.options.options import (
    AirGapRampRemovalOptions,
    GradientIntegrationUnwrapOptions,
    IterativeResidualUnwrapOptions,
    PhaseUnwrapOptions,
)
from pyxalign.api.options_utils import get_all_attribute_names
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.interactions.custom import action_button_style_sheet
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.roi_selector import GetBoxBoundsFromROISelector
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.interactions.viewers.base import ArrayViewer


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

    phase_unwrapped = pyqtSignal()  # np.ndarray)

    def __init__(self, task: LaminographyAlignmentTask, parent: Optional[QWidget] = None):
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
        basic_options_list = (
            ["method", "gradient_integration", "iterative_residual", "remove_ramp_using_air_gap"]
            + get_all_attribute_names(
                GradientIntegrationUnwrapOptions(), parent_prefix="gradient_integration"
            )
            + get_all_attribute_names(
                IterativeResidualUnwrapOptions(), parent_prefix="iterative_residual"
            )
            + get_all_attribute_names(
                AirGapRampRemovalOptions(), parent_prefix="remove_ramp_using_air_gap"
            )
        )
        self.options_editor = BasicOptionsEditor(
            self.task.complex_projections.options.phase_unwrap,
            basic_options_list=basic_options_list,
            enable_advanced_tab=True,
            skip_fields=[
                "remove_ramp_using_air_gap.air_region",
                "remove_ramp_using_air_gap.air_region.horizontal_range",
                "remove_ramp_using_air_gap.air_region.vertical_range",
                "remove_ramp_using_air_gap.air_region.horizontal_offset",
                "remove_ramp_using_air_gap.air_region.vertical_offset",
                "remove_ramp_using_air_gap.air_region.return_view",
            ],
            open_panels_list=["remove_ramp_using_air_gap"],
        )
        # add button for showing cropped projections
        self.open_crop_viewer_button = QPushButton(" Edit Air Gap ROI ")
        self.open_crop_viewer_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.open_crop_viewer_button.clicked.connect(self.show_cropped_projections_viewer)
        self.options_editor.form_layout.addRow("", self.open_crop_viewer_button)
        left_panel_layout.addWidget(self.options_editor, stretch=2)

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

    def show_cropped_projections_viewer(self):
        self.crop_viewer = GetBoxBoundsFromROISelector(
            self.task.complex_projections,
            self.task.complex_projections.options.phase_unwrap.remove_ramp_using_air_gap.air_region,
            # self.options_editor._data.remove_ramp_using_air_gap.air_region
        )
        self.crop_viewer.rectangular_roi_selected.connect(self.update_crop_options)
        self.crop_viewer.show()

    def update_crop_options(self):
        self.task.complex_projections.options.phase_unwrap.remove_ramp_using_air_gap.air_region = (
            self.crop_viewer.options
        )
        self.crop_viewer.close()

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
        try:
            # Disable button during processing
            self.unwrap_button.setEnabled(False)
            self.unwrap_button.setText("Unwrapping Phase...")

            # # Update the task's phase unwrap options with current editor values
            # self.task.complex_projections.options.phase_unwrap = self.options_editor._data

            # Perform phase unwrapping
            # print("Starting phase unwrapping...")
            wrapped_func = loading_bar_wrapper("Unwrapping phase...", block_all_windows=True)(
                func=self.task.get_unwrapped_phase
            )
            wrapped_func()

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

            # Emit signal indicating unwrapping completion
            self.phase_unwrapped.emit()

            print("Phase unwrapping completed successfully")

        except Exception as ex:
            traceback.print_exc()
            # error_dialog = QErrorMessage()
            # error_dialog.showMessage(str(ex))
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setInformativeText(str(ex))
            msg.setWindowTitle("Error")
            spacer = QSpacerItem(500, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
            layout = msg.layout()
            layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())
            msg.exec_()
        finally:
            # Re-enable button and reset text
            self.unwrap_button.setEnabled(True)
            self.unwrap_button.setText("Unwrap Phase")


@switch_to_matplotlib_qt_backend
def launch_phase_unwrap_widget(
    task: LaminographyAlignmentTask,
    wait_until_closed: bool = False,
) -> PhaseUnwrapWidget:
    """Launch the GUI for obtaining and displaying the unwrapped phase.

    Args:
        task (LaminographyAlignmentTask): the task, which must have a
            complex_projections attribute.
        wait_until_closed (bool): if `True`, the application starts a
            blocking call until the GUI window is closed.

    Returns: a PhaseUnwrapWidget instance

    Example:
        Open the phase unwrapping
        GUI::

            gui = pyxalign.gui.launch_phase_unwrap_widget(task)
    """
    app = QApplication.instance() or QApplication([])
    gui = PhaseUnwrapWidget(task)
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = PhaseUnwrapWidget()
    widget.show()
    sys.exit(app.exec_())
