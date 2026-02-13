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
from pyxalign.api.options.alignment import CrossCorrelationOptions
from pyxalign.api.options.transform import CropOptions, ShiftOptions
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.custom import action_button_style_sheet
from pyxalign.interactions.roi_selector import GetBoxBoundsFromROISelector
from pyxalign.interactions.viewers.arrays import get_projection_title_strings
from pyxalign.interactions.viewers.base import ArrayViewer, MultiThreadedWidget
from pyxalign.transformations.classes import Cropper, Shifter
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper


class CrossCorrelationMasterWidget(MultiThreadedWidget):
    def __init__(
        self,
        task: Optional["t.LaminographyAlignmentTask"] = None,
        projection_type: Optional[enums.ProjectionType] = None,
        multi_thread_func: Optional[Callable] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.task = task
        # If only one type of projection exists, use that type
        if self.task.phase_projections is None:  # only has complex projections
            self.projection_type = enums.ProjectionType.COMPLEX
        elif self.task.complex_projections is None:  # only has phase projections
            self.projection_type = enums.ProjectionType.PHASE
        else:  # has both types of projections
            self.projection_type = projection_type

        self.crop_viewer = None
        self.alignment_results_list: list[AlignmentResults] = []
        self.results_collection_widget = None

        if task is not None:
            self.initialize_page(task)

    @property
    def projections(self) -> "p.Projections":
        if self.projection_type == enums.ProjectionType.PHASE:
            return self.task.phase_projections
        elif self.projection_type == enums.ProjectionType.COMPLEX:
            return self.task.complex_projections

    def initialize_page(self, task: "t.LaminographyAlignmentTask"):
        self.pinned_array = create_empty_pinned_array_like(self.projections.data)
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
            projection_type=self.projection_type,  # should perhaps move the type into "options"
            plot_results=False,
        )
        # update the main plot
        self.update_shift_results_plot(shift)
        # update the collections plot
        # this should probably be absorbed into a method of alignment results list
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

        shifter = Shifter(
            ShiftOptions(type=enums.ShiftType.FFT, enabled=True, eliminate_wrapping=True)
        )
        wrapped_shift_func = loading_bar_wrapper(
            load_message="Shifting projections for display...",
            block_all_windows=True,
        )(func=shifter.run)
        self.pinned_array = wrapped_shift_func(
            images=self.projections.data,
            shift=shift.astype(r_type),
            pinned_results=self.pinned_array,
        )
        # self.pinned_array = shift_func.run(
        #     images=self.projections.data,
        #     shift=shift.astype(r_type),
        #     pinned_results=self.pinned_array,
        # )

        self.post_alignment_viewer.reinitialize_all(
            self.pinned_array,
            sort_idx=self.sort_idx,
            extra_title_strings_list=self.title_strings,
        )
        self.post_alignment_viewer.indexing_widget.spinbox.setValue(
            self.pre_alignment_viewer.indexing_widget.spinbox.value()
        )
        # Enable the ArrayViewer
        self.post_alignment_viewer.setEnabled(True)

    def make_options_setup_and_results_tab_layout(self, tabs: QTabWidget):
        alignment_setup_widget = QWidget(self)

        if self.projection_type == enums.ProjectionType.PHASE:
            proj = self.task.phase_projections
        else:
            proj = self.task.complex_projections  # fixed

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
        # self.start_button.setStyleSheet("QPushButton { background-color: green;}")
        self.start_button.setStyleSheet(action_button_style_sheet)
        self.start_button.clicked.connect(self.start_alignment)
        self.start_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        # add button for showing cropped projections
        self.open_crop_viewer_button = QPushButton("Edit Crop Region/Alignment ROI")
        self.open_crop_viewer_button.clicked.connect(self.show_cropped_projections_viewer)
        self.open_crop_viewer_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.options_editor.form_layout.addRow("", self.open_crop_viewer_button)
        # create button layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignLeft)
        buttons_layout.addWidget(self.start_button)
        # buttons_layout.addWidget(self.open_crop_viewer_button)
        # add shift results viewer
        self.create_shift_results_plot()
        # add editor and start button to sub-layout
        inputs_layout = QVBoxLayout()
        inputs_layout.addWidget(self.options_editor, stretch=2)
        inputs_layout.addLayout(buttons_layout)
        # inputs_layout.addWidget(self.canvas, stretch=1)
        # inputs_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Preferred, QSizePolicy.Expanding))
        # inputs_layout.addWidget(self.start_button)

        # Make results display for showing before and after
        self.title_strings = get_projection_title_strings(
                self.projections.scan_numbers, self.projections.angles
            )
        self.sort_idx = np.argsort(proj.angles)
        self.pre_alignment_viewer = ArrayViewer(
            array3d=proj.data,
            sort_idx=self.sort_idx,
            return_index_selector_seperately=True,
            extra_title_strings_list=self.title_strings,
            options=ArrayViewerOptions(
                additional_spinbox_indexing=[self.projections.scan_numbers],
                additional_spinbox_titles=["scan number"],
            )
        )

        pre_align_label = QLabel("Pre Alignment")
        pre_align_label.setStyleSheet("QLabel { font-size: 14pt;}")
        # viewer for showing aligned data
        self.post_alignment_viewer = ArrayViewer(return_index_selector_seperately=True)
        self.post_alignment_viewer.setEnabled(False)  # Initially disabled
        post_align_label = QLabel("Post Alignment")
        post_align_label.setStyleSheet("QLabel { font-size: 14pt;}")
        # link sliders (link the rest at some point later)
        self.pre_alignment_viewer.slider.valueChanged.connect(
            self.post_alignment_viewer.slider.setValue
        )
        self.post_alignment_viewer.slider.valueChanged.connect(
            self.pre_alignment_viewer.slider.setValue
        )
        # add results to sub-layout
        pre_align_layout = QVBoxLayout()
        pre_align_layout.addWidget(pre_align_label)
        pre_align_layout.addWidget(self.pre_alignment_viewer)
        post_align_layout = QVBoxLayout()
        post_align_layout.addWidget(post_align_label)
        post_align_layout.addWidget(self.post_alignment_viewer)
        viewers_layout = QHBoxLayout()
        viewers_layout.addLayout(pre_align_layout)
        viewers_layout.addLayout(post_align_layout)
        outputs_layout = QVBoxLayout()
        outputs_layout.addLayout(viewers_layout)
        outputs_layout.addWidget(self.pre_alignment_viewer.indexing_widget)
        outputs_layout.addWidget(self.canvas)

        # Finalize layout
        layout = QHBoxLayout()
        layout.addLayout(inputs_layout)
        layout.addLayout(outputs_layout)
        alignment_setup_widget.setLayout(layout)
        tabs.addTab(alignment_setup_widget, "Configure && Start")

    def make_results_tab_layout(self, tabs: QTabWidget):
        self.results_collection_widget = AlignmentResultsCollection(
            self.alignment_results_list, display_initial_shift=False
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
        # self.options_editor._data.crop.horizontal_range = self.crop_viewer.crop_options


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
    parser.add_argument("task_path", help="Path to a task file")
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
