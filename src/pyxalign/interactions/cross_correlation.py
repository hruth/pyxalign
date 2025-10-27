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
from pyxalign.api.options_utils import get_all_attribute_names
from pyxalign.api.types import r_type
import pyxalign.data_structures.task as t
import pyxalign.data_structures.projections as p
from pyxalign.gpu_utils import create_empty_pinned_array_like
from pyxalign.interactions.pma_runner import AlignmentResults, AlignmentResultsCollection
from pyxalign.api.options.alignment import CrossCorrelationOptions
from pyxalign.api.options.transform import CropOptions, ShiftOptions
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.custom import action_button_style_sheet
from pyxalign.interactions.viewers.base import ArrayViewer, MultiThreadedWidget
from pyxalign.transformations.classes import Cropper, Shifter


class CrossCorrelationMasterWidget(MultiThreadedWidget):
    def __init__(
        self,
        task: Optional["t.LaminographyAlignmentTask"] = None,
        projection_type: enums.ProjectionType = enums.ProjectionType.COMPLEX,
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
        else:
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
        # Make display comparing shifted and unshifted projections
        # to do

    def start_alignment(self):
        shift = self.task.get_cross_correlation_shift(
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
        shift_func = Shifter(ShiftOptions(type=enums.ShiftType.FFT, enabled=True))
        self.pinned_array = shift_func.run(
            images=self.projections.data,
            shift=shift.astype(r_type),
            pinned_results=self.pinned_array,
        )

        sort_idx = np.argsort(self.projections.angles)
        title_strings = [
            f" scan {scan}, angle {angle:0.2f}"
            for scan, angle in zip(self.projections.scan_numbers, self.projections.angles)
        ]
        self.post_alignment_viewer.reinitialize_all(
            self.pinned_array,
            sort_idx=sort_idx,
            extra_title_strings_list=title_strings,
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
        basic_options_list = ["binning", "filter_position", "filter_data", "crop"]
        basic_options_list += get_all_attribute_names(CropOptions(), parent_prefix="crop")
        print(basic_options_list)
        self.options_editor = BasicOptionsEditor(
            self.task.options.cross_correlation,
            skip_fields=["precision"],
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
        self.open_crop_viewer_button = QPushButton("View Cropped Projections")
        self.open_crop_viewer_button.clicked.connect(self.show_cropped_projections_viewer)
        self.open_crop_viewer_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        # create button layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignLeft)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.open_crop_viewer_button)
        # add shift results viewer
        self.create_shift_results_plot()
        # add editor and start button to sub-layout
        inputs_layout = QVBoxLayout()
        inputs_layout.addWidget(self.options_editor)
        inputs_layout.addWidget(self.canvas)
        inputs_layout.addLayout(buttons_layout)
        # inputs_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Preferred, QSizePolicy.Expanding))
        # inputs_layout.addWidget(self.start_button)

        # Make results display for showing before and after
        self.pre_alignment_viewer = ArrayViewer(array3d=proj.data, sort_idx=np.argsort(proj.angles))
        pre_align_label = QLabel("Pre Alignment")
        pre_align_label.setStyleSheet("QLabel { font-size: 14pt;}")
        # viewer for showing aligned data
        self.post_alignment_viewer = ArrayViewer()
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
        viewers_layout = QVBoxLayout()
        viewers_layout.addWidget(pre_align_label)
        viewers_layout.addWidget(self.pre_alignment_viewer)
        viewers_layout.addWidget(post_align_label)
        viewers_layout.addWidget(self.post_alignment_viewer)

        # Finalize layout
        layout = QHBoxLayout()
        layout.addLayout(inputs_layout)
        layout.addLayout(viewers_layout)
        alignment_setup_widget.setLayout(layout)
        tabs.addTab(alignment_setup_widget, "Configure && Start")

    def make_results_tab_layout(self, tabs: QTabWidget):
        self.results_collection_widget = AlignmentResultsCollection(self.alignment_results_list)
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
        self.crop_viewer = ArrayViewer(
            array3d=Cropper(self.options_editor._data.crop).run(self.projections.data),
            sort_idx=np.argsort(self.projections.angles),
        )
        # self.crop_viewer.setAttribute(Qt.WA_DeleteOnClose)
        self.crop_viewer.show()


def launch_cross_correlation_gui(
    task: "t.LaminographyAlignmentTask",
    projection_type: enums.ProjectionType.COMPLEX,
    wait_until_closed: bool = False,
):
    app = QApplication.instance() or QApplication([])
    gui = CrossCorrelationMasterWidget(task=task, projection_type=projection_type)
    gui.setAttribute(Qt.WA_DeleteOnClose)

    gui.show()
    if wait_until_closed:
        app.exec_()


if __name__ == "__main__":
    import sys
    import argparse

    # must enter a path to the task file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_path",
        help="Path to a task file"
    )
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
