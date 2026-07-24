import sys
from typing import Callable, Optional

import numpy as np
import copy
import pyqtgraph as pg
import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyxalign.api import enums
from pyxalign.api.options.plotting import ArrayViewerOptions
from pyxalign.api.options_utils import get_all_attribute_names
from pyxalign.api.types import r_type
import pyxalign.data_structures.task as t
import pyxalign.data_structures.projections as p
from pyxalign.gpu_utils import create_empty_pinned_array_like
from pyxalign.interactions.alignment_results import AlignmentResults, AlignmentResultsCollection
from pyxalign.interactions.custom import action_button_style_sheet
from pyxalign.api.options.alignment import CrossCorrelationOptions
from pyxalign.api.options.transform import CropOptions, ShiftOptions
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.roi_selector import GetBoxBoundsFromROISelector
from pyxalign.interactions.viewers.arrays import get_projection_title_strings
from pyxalign.interactions.viewers.base import ArrayViewer, MultiThreadedWidget
from pyxalign.transformations.classes import Cropper, Shifter
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper


class CCResultsCollection(AlignmentResultsCollection):
    """
    Widget for visualizing and comparing multiple cross-correlation alignment results.

    This widget extends AlignmentResultsCollection to add a stage shift button
    for cross-correlation alignment results.

    Parameters
    ----------
    alignment_results_list : list[AlignmentResults]
        List of alignment results to display and compare.
    display_initial_shift : bool, optional
        Whether to display initial shift in plots. Default is False.
    task : t.LaminographyAlignmentTask, optional
        Task object containing projections for staging shifts.
    projection_type : enums.ProjectionType, optional
        Type of projections being aligned (PHASE or COMPLEX).
    projection_viewer : QWidget, optional
        ProjectionViewer widget for refreshing after staging shifts.
    parent : QWidget, optional
        Parent widget for this interface.
    """

    def __init__(
        self,
        alignment_results_list: list[AlignmentResults],
        display_initial_shift: bool = False,
        task: Optional["t.LaminographyAlignmentTask"] = None,
        projection_type: Optional[enums.ProjectionType] = None,
        projection_viewer: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ):
        self.task = task
        self.projection_type = projection_type
        self.projection_viewer = projection_viewer

        # Call parent __init__ with stage_shift_callback
        super().__init__(
            alignment_results_list=alignment_results_list,
            display_initial_shift=display_initial_shift,
            stage_shift_callback=self.stage_shift,
            parent=parent,
        )

    def stage_shift(self, row: int):
        """
        Stage the shift from the selected alignment result.

        Parameters
        ----------
        row : int
            The index of the alignment result to stage.
        """
        if self.task is None:
            print("Cannot stage shift: task not available")
            return

        # Determine which projections to use based on projection_type
        if self.projection_type == enums.ProjectionType.PHASE:
            projections = self.task.phase_projections
        elif self.projection_type == enums.ProjectionType.COMPLEX:
            projections = self.task.complex_projections
        else:
            # Auto-detect based on what's available
            if self.task.phase_projections is not None:
                projections = self.task.phase_projections
            elif self.task.complex_projections is not None:
                projections = self.task.complex_projections
            else:
                print("Cannot stage shift: no projections available")
                return

        if projections is None:
            print("Cannot stage shift: projections not available")
            return

        alignment_result = self.alignment_results_list[row]
        shift = alignment_result.shift

        # Stage the shift using the shift_manager
        projections.shift_manager.stage_shift(
            shift=shift,
            function_type=enums.ShiftType.CIRC,
            alignment_options=self.task.options.cross_correlation,
            eliminate_wrapping=True,
        )
        print(f"Shift from cross-correlation alignment result {row} staged successfully")

        # Refresh the Applied Shifts tab in the projection viewer
        if self.projection_viewer is not None:
            self.projection_viewer.refresh_applied_shifts_tab()


class _ProjectionComparisonWindow(QWidget):
    """Top-level window showing pre- and post-alignment projections side by side.

    Shifting is performed when the window is opened. The shifted array is freed
    when the window is closed.
    """

    def __init__(
        self,
        projections: "p.Projections",
        shift: np.ndarray,
        sort_idx: np.ndarray,
        title_strings: list,
        on_closed: Optional[Callable] = None,
    ):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Pre- and Post-Alignment Projections")
        self._on_closed = on_closed
        self._pinned_array = None

        # Perform the shift when the window opens
        shifter = Shifter(ShiftOptions(type=enums.ShiftType.FFT, enabled=True, eliminate_wrapping=True))
        wrapped_shift_func = loading_bar_wrapper(
            load_message="Shifting projections for display...",
            block_all_windows=True,
        )(func=shifter.run)
        self._pinned_array = create_empty_pinned_array_like(projections.data)
        self._pinned_array = wrapped_shift_func(
            images=projections.data,
            shift=shift.astype(r_type),
            pinned_results=self._pinned_array,
        )

        # Create viewers (must be instance attrs to avoid GC)
        self._pre_viewer = ArrayViewer(
            array3d=projections.data,
            sort_idx=sort_idx,
            return_index_selector_seperately=True,
            extra_title_strings_list=title_strings,
            options=ArrayViewerOptions(
                additional_spinbox_indexing=[projections.scan_numbers],
                additional_spinbox_titles=["scan number"],
            ),
        )
        self._post_viewer = ArrayViewer(
            array3d=self._pinned_array,
            sort_idx=sort_idx,
            return_index_selector_seperately=True,
            extra_title_strings_list=title_strings,
        )

        # Link sliders
        self._pre_viewer.slider.valueChanged.connect(self._post_viewer.slider.setValue)
        self._post_viewer.slider.valueChanged.connect(self._pre_viewer.slider.setValue)

        pre_label = QLabel("Pre Alignment")
        pre_label.setStyleSheet("QLabel { font-size: 14pt;}")
        post_label = QLabel("Post Alignment")
        post_label.setStyleSheet("QLabel { font-size: 14pt;}")

        pre_layout = QVBoxLayout()
        pre_layout.addWidget(pre_label)
        pre_layout.addWidget(self._pre_viewer)
        post_layout = QVBoxLayout()
        post_layout.addWidget(post_label)
        post_layout.addWidget(self._post_viewer)

        viewers_layout = QHBoxLayout()
        viewers_layout.addLayout(pre_layout)
        viewers_layout.addLayout(post_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(viewers_layout)
        main_layout.addWidget(self._pre_viewer.indexing_widget)
        self.setLayout(main_layout)
        self.resize(1400, 700)

    def closeEvent(self, event):
        self._pinned_array = None
        if self._on_closed is not None:
            self._on_closed()
        super().closeEvent(event)


class CrossCorrelationMasterWidget(MultiThreadedWidget):
    def __init__(
        self,
        task: Optional["t.LaminographyAlignmentTask"] = None,
        projection_type: Optional[enums.ProjectionType] = None,
        projection_viewer: Optional[QWidget] = None,
        multi_thread_func: Optional[Callable] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.task = task
        self.projection_viewer = projection_viewer
        # If only one type of projection exists, use that type
        if self.task.phase_projections is None:  # only has complex projections
            self.projection_type = enums.ProjectionType.COMPLEX
        elif self.task.complex_projections is None:  # only has phase projections
            self.projection_type = enums.ProjectionType.PHASE
        else:  # has both types of projections
            if projection_type is not None:
                self.projection_type = projection_type
            else:
                projection_type = enums.ProjectionType.PHASE

        self.crop_viewer = None
        self.alignment_results_list: list[AlignmentResults] = []
        self.results_collection_widget = None
        self.last_shift = None
        self._comparison_window = None

        if task is not None:
            self.initialize_page(task)

    @property
    def projections(self) -> "p.Projections":
        if self.projection_type == enums.ProjectionType.PHASE:
            return self.task.phase_projections
        elif self.projection_type == enums.ProjectionType.COMPLEX:
            return self.task.complex_projections

    def initialize_page(self, task: "t.LaminographyAlignmentTask"):
        tabs = QTabWidget()
        tabs.setObjectName("main_tabs")
        tabs.setStyleSheet("#main_tabs > QTabBar{font-size: 20px;}")
        layout = QHBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

        # Make tab for setup
        self.make_options_setup_and_results_tab_layout(tabs)
        # Make display for resulting shift
        self.make_results_tab_layout(tabs)

    def start_alignment(self):
        wrapped_func = loading_bar_wrapper(
            load_message="Getting cross-correlation alignment...",
            block_all_windows=True,
        )(func=self.task.get_cross_correlation_shift)
        shift = wrapped_func(
            projection_type=self.projection_type,
            plot_results=False,
        )
        # Update the main plot
        self.update_shift_results_plot(shift)
        # Add to the collected results
        self.alignment_results_list += [
            AlignmentResults(
                shift,
                shift * 0,
                self.projections.angles,
                options=copy.deepcopy(self.task.options.cross_correlation),
                projection_options=self.projections.options,
            )
        ]
        self.results_collection_widget.update_table()

        # Store the shift and enable the comparison window button
        self.last_shift = shift
        self.view_projections_button.setEnabled(True)

        # Refresh the Applied Shifts tab in the projection viewer
        if self.projection_viewer is not None:
            self.projection_viewer.refresh_applied_shifts_tab()

    def make_options_setup_and_results_tab_layout(self, tabs: QTabWidget):
        alignment_setup_widget = QWidget(self)

        if self.projection_type == enums.ProjectionType.PHASE:
            proj = self.task.phase_projections
        else:
            proj = self.task.complex_projections

        # Make options editor
        basic_options_list = [
            "binning",
            "remove_slow_variation",
            "filter_position",
            "filter_data",
            "crop",
        ]
        basic_options_list += get_all_attribute_names(CropOptions(), parent_prefix="crop")
        self.options_editor = BasicOptionsEditor(
            self.task.options.cross_correlation,
            skip_fields=[
                "precision",
                "crop.horizontal_range",
                "crop.vertical_range",
                "crop.horizontal_offset",
                "crop.vertical_offset",
                "crop.return_view"
            ],
            enable_advanced_tab=True,
            basic_options_list=basic_options_list,
            open_panels_list=["crop"],
            label="Cross Correlation Alignment Options",
        )
        # Make start button
        self.start_button = QPushButton("Start Alignment")
        self.start_button.setStyleSheet(action_button_style_sheet)
        self.start_button.clicked.connect(self.start_alignment)
        self.start_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        # Add button for showing cropped projections
        self.open_crop_viewer_button = QPushButton("Edit Crop Region/Alignment ROI")
        self.open_crop_viewer_button.clicked.connect(self.show_cropped_projections_viewer)
        self.open_crop_viewer_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.options_editor.form_layout.addRow("", self.open_crop_viewer_button)

        # Button for opening the pre/post alignment comparison window
        self.view_projections_button = QPushButton("View pre- and post-alignment projections")
        self.view_projections_button.clicked.connect(self.open_projection_comparison_window)
        self.view_projections_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.view_projections_button.setEnabled(False)

        # Create button layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignLeft)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.view_projections_button)

        # Add shift results viewer
        self.create_shift_results_plot()

        # Store sort index and title strings for use by the comparison window
        self.title_strings = get_projection_title_strings(
            self.projections.scan_numbers, self.projections.angles
        )
        self.sort_idx = np.argsort(proj.angles)

        # Add editor and start button to sub-layout
        inputs_layout = QVBoxLayout()
        inputs_layout.addWidget(self.options_editor, stretch=2)
        inputs_layout.addLayout(buttons_layout)

        outputs_layout = QVBoxLayout()
        outputs_layout.addWidget(self.canvas)

        # Finalize layout
        layout = QHBoxLayout()
        layout.addLayout(inputs_layout)
        layout.addLayout(outputs_layout)
        alignment_setup_widget.setLayout(layout)
        tabs.addTab(alignment_setup_widget, "Configure && Start")

    def open_projection_comparison_window(self):
        """Open the pre/post-alignment comparison window, or raise it if already open."""
        if self._comparison_window is not None and self._comparison_window.isVisible():
            self._comparison_window.raise_()
            self._comparison_window.activateWindow()
            return
        self._comparison_window = _ProjectionComparisonWindow(
            projections=self.projections,
            shift=self.last_shift,
            sort_idx=self.sort_idx,
            title_strings=self.title_strings,
            on_closed=self._on_comparison_window_closed,
        )
        self._comparison_window.show()

    def _on_comparison_window_closed(self):
        self._comparison_window = None

    def make_results_tab_layout(self, tabs: QTabWidget):
        self.results_collection_widget = CCResultsCollection(
            alignment_results_list=self.alignment_results_list,
            display_initial_shift=False,
            task=self.task,
            projection_type=self.projection_type,
            projection_viewer=self.projection_viewer,
        )
        empty_widget = QWidget()
        self._results_collection_layout = QVBoxLayout()
        empty_widget.setLayout(self._results_collection_layout)
        tabs.addTab(self.results_collection_widget, "Collected Results")

    def create_shift_results_plot(self):
        # Create the pyqtgraph GraphicsLayoutWidget
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.canvas = self.graphics_layout  # Keep canvas reference for layout compatibility
        self.plot_item = self.graphics_layout.addPlot()
        self.plot_item.setTitle("Cross Correlation Shift")
        self.plot_item.setLabel("left", "shift (px)")
        self.plot_item.setLabel("bottom", "angle (deg)")
        self.plot_item.showGrid(x=True, y=True)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def update_shift_results_plot(self, shift: np.ndarray):
        self.plot_item.clear()
        sort_idx = np.argsort(self.projections.angles)

        # Plot horizontal and vertical shifts
        angles_sorted = self.projections.angles[sort_idx]
        horizontal_shift = shift[sort_idx, 0]
        vertical_shift = shift[sort_idx, 1]

        # Plot with different colors and labels
        self.plot_item.plot(angles_sorted, horizontal_shift, pen="b", name="Horizontal")
        self.plot_item.plot(angles_sorted, vertical_shift, pen="r", name="Vertical")

        # Add legend
        self.plot_item.addLegend()

    def show_cropped_projections_viewer(self):
        self.crop_viewer = GetBoxBoundsFromROISelector(self.projections, self.options_editor._data.crop)
        self.crop_viewer.rectangular_roi_selected.connect(self.update_crop_options)
        self.crop_viewer.show()

    def update_crop_options(self):
        self.task.options.cross_correlation.crop = self.crop_viewer.options
        self.crop_viewer.close()

    def reinitialize_widget(self, task: "t.LaminographyAlignmentTask"):
        """
        Reinitialize the widget with updated projections from the task.

        This method should be called when the task's phase_projections or
        complex_projections have been updated. When projections are updated,
        their .data, .scan_numbers, and .angles attributes will be different.

        Parameters
        ----------
        task : LaminographyAlignmentTask
            Task with updated projections (phase_projections or complex_projections).

        Notes
        -----
        This method will:
        - Update the task reference
        - Close the comparison window if open
        - Update sort_idx and title_strings
        - Clear the alignment results list
        """
        print("reinitializing widget")
        # Update the task reference
        self.task = task

        projections = self.projections

        # Update title strings and sort index
        self.title_strings = get_projection_title_strings(
            projections.scan_numbers, projections.angles
        )
        print("scan numbers length:", len(projections.scan_numbers))
        self.sort_idx = np.argsort(projections.angles)

        # Close the comparison window and reset state
        if self._comparison_window is not None:
            self._comparison_window.close()
            self._comparison_window = None
        self.last_shift = None
        self.view_projections_button.setEnabled(False)

        # Clear the alignment results
        self.clear_alignment_results()

    def clear_alignment_results(self):
        """
        Clear all alignment results and reset viewers.

        This method is called when shift operations (apply or undo) are performed
        on the ProjectionViewer, as those operations invalidate previously computed
        alignment results. It clears:
        - Alignment results list
        - Results collection table
        - Comparison window (if open)
        - Cross-correlation shift plot
        """
        # Clear the alignment results list
        self.alignment_results_list.clear()

        # Update the results collection widget to reflect empty results
        self.results_collection_widget.alignment_results_list = self.alignment_results_list
        # Clear all rows from the results table
        self.results_collection_widget.results_table.setRowCount(0)
        # Clear the plots
        if hasattr(self.results_collection_widget, 'clear_plots'):
            self.results_collection_widget.clear_plots()

        # Close the comparison window if open
        if self._comparison_window is not None:
            self._comparison_window.close()
            self._comparison_window = None

        # Disable the view projections button
        self.view_projections_button.setEnabled(False)
        self.last_shift = None

        # Clear the cross-correlation shift plot
        if hasattr(self, 'plot_item') and self.plot_item is not None:
            self.plot_item.clear()
            # Remove the legend if it exists
            if hasattr(self.plot_item, 'legend') and self.plot_item.legend is not None and self.plot_item.legend.scene() is not None:
                self.plot_item.legend.scene().removeItem(self.plot_item.legend)


@switch_to_matplotlib_qt_backend
def launch_cross_correlation_gui(
    task: "t.LaminographyAlignmentTask",
    projection_type: Optional[enums.ProjectionType] = None,
    wait_until_closed: bool = False,
) -> CrossCorrelationMasterWidget:
    """Launch the cross-correlation alignment GUI. This GUI lets you
    interactively change the cross-correlation alignment options, view the
    projections before and after alignment, and track alignment results for
    different combinations of alignment options.

    Args:
        task (LaminographyAlignmentTask): The task with projections to
            align.
        projection_type (Optional[ProjectionType]): If the `task` has
            both `phase_projections` and `complex_projections`,
            `projection_type` specifies the projection to align.
        wait_until_closed (bool): if `True`, the application starts a
            blocking call until the GUI window is closed.

    Example:
        **Align the projections in a task object**

        First, launch the cross-correlation gui::

            gui = pyxalign.gui.launch_cross_correlation_gui(task, "phase")

        Clicking the "start alignment" button will run the cross-
        correlation alignment algorithm with the selected parameters.
        Each time you run the alignment, the shift manager tool in the
        `Projections` object will store the shift. Once you are happy
        with the alignment, close the window and shift the projections::

            task.phase_projections.apply_staged_shift()

        To see the previously applied shifts, you can launch the
        projection viewer and click on the tab titled "applied shifts"::

            gui = pyxalign.gui.launch_projection_viewer(task.phase_projections)

        If you want to undo a previously applied shift, you can do::

            task.phase_projections.undo_last_shift()
    """
    app = QApplication.instance() or QApplication([])
    if projection_type is None:
        if task.complex_projections is not None and task.phase_projections is not None:
            print("projection_type was not specified, defaulting to projection_type='phase'")
            projection_type = enums.ProjectionType.PHASE
        if task.complex_projections is None:
            projection_type = enums.ProjectionType.PHASE
        elif task.phase_projections is None:
            projection_type = enums.ProjectionType.COMPLEX
    else:
        projection_type = projection_type.lower()
    gui = CrossCorrelationMasterWidget(task=task, projection_type=projection_type)
    gui.setAttribute(Qt.WA_DeleteOnClose)
    gui.show()

    if wait_until_closed:
        app.exec_()

    return gui


if __name__ == "__main__":
    import sys
    import argparse

    # must enter a path to the task file
    parser = argparse.ArgumentParser()
    parser.add_argument("task_path", help="Path to a path to a task file")
    args = parser.parse_args()
    task_path = args.task_path

    dummy_task = t.load_task(task_path)
    dummy_task.options.cross_correlation = CrossCorrelationOptions()

    app = QApplication(sys.argv)
    master_widget = CrossCorrelationMasterWidget(task=dummy_task, projection_type="PHASE")

    # Use the left half of the screen
    screen_geometry = app.desktop().availableGeometry(master_widget)
    master_widget.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() * 0.75),
        int(screen_geometry.height() * 0.9),
    )

    master_widget.show()
    sys.exit(app.exec_())
