from ctypes import alignment
from pyxalign.api.options.options import ExperimentOptions
from pyxalign.api.options.projections import ProjectionOptions, ProjectionTransformOptions
from pyxalign.api.options.task import AlignmentTaskOptions
from pyxalign.api.options_utils import get_all_attribute_names
from pyxalign.data_structures.projections import ComplexProjections, Projections
from pyxalign.data_structures.task import LaminographyAlignmentTask
from pyxalign.gpu_utils import pin_memory
from pyxalign.interactions.custom import MultipleOfDivisorSpinBox, action_button_style_sheet
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.phase_unwrap import PhaseUnwrapWidget
from pyxalign.io.loaders.base import StandardData
import matplotlib

import sys
import numpy as np
from typing import Optional
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QFrame,
)
from pyxalign.plotting.interactive.arrays import ProjectionViewer
import sip
import pyqtgraph as pg

from pyxalign.io.loaders.utils import convert_projection_dict_to_array
from pyxalign.plotting.interactive.base import ArrayViewer


class CreateProjectionArrayWidget(QWidget):
    array_created_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.standard_data = None
        self.projection_array = None
        self.array_viewer = None

        # Main layout (vertical), top-left aligned.
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setLayout(layout)

        self.add_array_viewer()
        self.add_dict_to_array_converter()

    def set_standard_data(self, standard_data: StandardData):
        self.standard_data = standard_data
        print("CreateProjectionArrayWidget received StandardData")
        # Update the spin boxes
        array_sizes = [array.shape for array in self.standard_data.projections.values()]
        max_width = np.max([size[0] for size in array_sizes])
        max_height = np.max([size[1] for size in array_sizes])
        self.new_shape_x.setValue(self.new_shape_x._round_to_nearest_divisor(max_width))
        self.new_shape_y.setValue(self.new_shape_y._round_to_nearest_divisor(max_height))

    def add_array_viewer(self):
        self.array_viewer = ArrayViewer(process_func=np.angle)
        self.array_viewer.setDisabled(True)
        self.layout().addWidget(self.array_viewer)

    def add_dict_to_array_converter(self):
        # Container layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # Spinboxes layout
        spin_boxes_layout = QHBoxLayout()
        spin_boxes_layout.setContentsMargins(0, 0, 0, 0)
        spin_boxes_layout.setSpacing(5)
        spin_boxes_layout.setAlignment(Qt.AlignLeft)

        new_shape_label = QLabel("Set array shape:")
        self.new_shape_x = MultipleOfDivisorSpinBox()
        self.new_shape_y = MultipleOfDivisorSpinBox()

        self.new_shape_x.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.new_shape_y.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        spin_boxes_layout.addWidget(new_shape_label, 0, Qt.AlignLeft)
        spin_boxes_layout.addWidget(self.new_shape_x, 0, Qt.AlignLeft)
        spin_boxes_layout.addWidget(self.new_shape_y, 0, Qt.AlignLeft)

        main_layout.addLayout(spin_boxes_layout)

        # Button to create the projection array
        create_projection_array_button = QPushButton("Create Projection Array")
        create_projection_array_button.clicked.connect(self.on_create_array_button_clicked)
        create_projection_array_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        main_layout.addWidget(create_projection_array_button)

        self.layout().addLayout(main_layout)

    def on_create_array_button_clicked(self):
        if not self.standard_data:
            print("No StandardData loaded in CreateProjectionArrayWidget.")
            return

        new_shape = (self.new_shape_x.value(), self.new_shape_y.value())
        self.projection_array = convert_projection_dict_to_array(
            self.standard_data.projections,
            delete_projection_dict=False,
            pad_with_mode=True,
            new_shape=new_shape,
        )
        self.array_created_signal.emit(self.projection_array)

        # Show result in array_viewer
        sort_idx = np.argsort(self.standard_data.angles)
        title_strings = [f"scan {x}" for x in self.standard_data.scan_numbers]
        if np.iscomplexobj(self.projection_array):
            self.array_viewer.process_func = np.angle
        else:
            self.array_viewer.process_func = lambda x: x
        self.array_viewer.reinitialize_all(
            self.projection_array,
            sort_idx=sort_idx,
            extra_title_strings_list=title_strings,
        )
        self.array_viewer.setDisabled(False)


class InitializeProjectionsObjectWidget(QWidget):
    object_created_signal = pyqtSignal(LaminographyAlignmentTask)

    def __init__(
        self,
        standard_data: Optional[StandardData] = None,
        default_projection_options: Optional[ProjectionOptions] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.standard_data = standard_data

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.add_options_editor(default_projection_options)
        self.create_object_button = QPushButton("Create Projections Object")
        self.create_object_button.clicked.connect(self.initialize_projections)
        layout.addWidget(self.create_object_button)

    def initialize_projections(self):
        self.projection_array[:] = pin_memory(self.projection_array)
        complex_projections = ComplexProjections(
            projections=self.projection_array,
            angles=self.standard_data.angles,
            scan_numbers=self.standard_data.scan_numbers,
            options=self.options_editor._data,
            probe_positions=list(self.standard_data.probe_positions.values()),
            probe=self.standard_data.probe,
            skip_pre_processing=False,
        )

        task = LaminographyAlignmentTask(
            options=AlignmentTaskOptions(),
            complex_projections=complex_projections,
        )
        self.object_created_signal.emit(task)

    def add_options_editor(self, default_projection_options: Optional[ProjectionOptions] = None):
        if default_projection_options is None:
            default_projection_options = ProjectionOptions()

        keep_fields = ["input_processing", "experiment"]
        skip_fields = list(
            np.setdiff1d(
                get_all_attribute_names(ProjectionOptions(), max_level=0),
                keep_fields,
            )
        )
        skip_fields += [
            "input_processing.mask_downsample_type",
            "mask_downsample_use_gaussian_filter",
        ]
        open_panels_list = [
            "experiment",
            "input_processing",
            "input_processing.rotation",
            "input_processing.shear",
        ]

        self.options_editor = BasicOptionsEditor(
            default_projection_options, skip_fields=skip_fields, open_panels_list=open_panels_list
        )
        self.layout().addWidget(self.options_editor)

    def set_standard_data(self, standard_data: StandardData):
        self.standard_data = standard_data
        print("InitializeProjectionsObjectWidget received StandardData")

    def set_projection_array(self, projection_array: np.ndarray):
        self.projection_array = projection_array
        print("InitializeProjectionsObjectWidget received projection_array")


class ProjectionPagesWidget(QWidget):
    """
    Multi-page widget that displays:
      - First page: CreateProjectionArrayWidget
      - Second page: InitializeProjectionsObjectWidget

    Navigation between pages is done with Back and Forward buttons at the bottom.
    """

    def __init__(self, standard_data: StandardData, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.standard_data = standard_data

        # Create the two pages and pass in any data they need
        self.create_array_widget = CreateProjectionArrayWidget()
        self.create_array_widget.set_standard_data(standard_data)

        self.create_object_widget = InitializeProjectionsObjectWidget(standard_data=standard_data)

        # Create a QStackedWidget to hold both pages
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.create_array_widget)  # Index 0
        self.stacked_widget.addWidget(self.create_object_widget)  # Index 1

        # Navigation buttons
        self.back_button = QPushButton("← Back")
        self.forward_button = QPushButton("Forward →")
        self.back_button.clicked.connect(self.go_to_previous_page)
        self.forward_button.clicked.connect(self.go_to_next_page)

        # Set up main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)

        # Create a layout for navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.back_button, alignment=Qt.AlignBottom | Qt.AlignLeft)
        nav_layout.addWidget(self.forward_button, alignment=Qt.AlignBottom | Qt.AlignRight)

        main_layout.addLayout(nav_layout)
        self.setLayout(main_layout)

        self.update_navigation_buttons()

    def go_to_previous_page(self):
        self.stacked_widget.setCurrentIndex(0)
        self.update_navigation_buttons()

    def go_to_next_page(self):
        self.stacked_widget.setCurrentIndex(1)
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        # Enable/disable buttons based on current page
        if self.stacked_widget.currentIndex() == 0:
            self.back_button.setDisabled(True)
            self.forward_button.setDisabled(False)
        else:
            self.back_button.setDisabled(False)
            self.forward_button.setDisabled(True)


class MainProjectionTab(QWidget):
    """
    Main widget that just shows a button. When that button is clicked,
    it opens a separate window (or dialog) with the two-page widget.
    """

    object_created_signal = pyqtSignal(LaminographyAlignmentTask)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.standard_data = None
        self.task = None
        self.init_window = None  # the top-level or separate window for the pages
        self.complex_projections_viewer = None
        self.phase_projections_viewer = None
        self.phase_unwrap_window = None  # window for phase unwrapping

        # Main layout for this widget
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create buttons layout
        buttons_layout = QVBoxLayout()

        # Create initialize projections object button
        self.open_initializer_button = QPushButton("Initialize Projections Object")
        self.open_initializer_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.open_initializer_button.clicked.connect(self.open_interactive_window)
        self.open_initializer_button.setDisabled(True)

        # Create phase unwrap button
        self.open_phase_unwrap_button = QPushButton("Unwrap Phase")
        self.open_phase_unwrap_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.open_phase_unwrap_button.clicked.connect(self.open_phase_unwrap_window)
        self.open_phase_unwrap_button.setDisabled(True)

        # add buttons to layout
        buttons_layout.addWidget(self.open_initializer_button, alignment=Qt.AlignTop | Qt.AlignLeft)
        buttons_layout.addWidget(
            self.open_phase_unwrap_button, alignment=Qt.AlignTop | Qt.AlignLeft
        )
        main_layout.addLayout(buttons_layout)

        # update button style
        style_sheet = """
            QPushButton {
                font-size: 12pt; 
                padding: 4px 6px;
            }
            """
        self.open_initializer_button.setStyleSheet(style_sheet)
        self.open_phase_unwrap_button.setStyleSheet(style_sheet)

    def on_standard_data_loaded(self, standard_data: StandardData):
        """
        This method can be called externally once the StandardData is ready/loaded.
        """
        self.standard_data = standard_data
        self.open_initializer_button.setDisabled(False)
        self.create_interactive_window()

    def create_interactive_window(self):
        # Create a new top-level widget containing our stacked pages
        self.init_window = QWidget()
        self.init_window.setWindowTitle("Projection Initialization")

        layout = QVBoxLayout()
        self.init_window.setLayout(layout)

        # Instantiate the two-page widget with the latest standard_data
        pages_widget = ProjectionPagesWidget(self.standard_data)
        layout.addWidget(pages_widget)
        self.create_array_widget = pages_widget.create_array_widget
        self.create_object_widget = pages_widget.create_object_widget
        # connect signal that passes the projection array to the object initializer widget
        self.create_array_widget.array_created_signal.connect(
            self.create_object_widget.set_projection_array
        )
        # # connect signal so that projections viewer is created after the object is loaded
        # self.create_object_widget.object_created_signal.connect(
        #     self.insert_complex_projections_viewer
        # )
        # connect the signal to the main projection initializer widget
        self.create_object_widget.object_created_signal.connect(self.on_task_loaded)

    def insert_complex_projections_viewer(self):
        if self.complex_projections_viewer is not None:
            sip.delete(self.complex_projections_viewer)
        self.complex_projections_viewer = ProjectionViewer(
            self.task.complex_projections, enable_dropping=True
        )
        self.layout().addWidget(self.complex_projections_viewer)
        # connect masks created signal to disabling of phase unwrap button
        self.complex_projections_viewer.masks_created.connect(
            self.set_phase_unwrapping_button_state
        )

    def insert_phase_projections_viewer(self):
        if self.phase_projections_viewer is not None:
            sip.delete(self.phase_projections_viewer)
        self.phase_projections_viewer = ProjectionViewer(
            self.task.phase_projections, enable_dropping=True
        )
        self.layout().addWidget(self.phase_projections_viewer)

    @pyqtSlot(LaminographyAlignmentTask)
    def on_task_loaded(self, task: LaminographyAlignmentTask):
        self.task = task
        self.insert_complex_projections_viewer()
        self.insert_phase_projections_viewer()
        self.set_phase_unwrapping_button_state()
        # emit signal meant for external widgets
        self.object_created_signal.emit(task)

    def set_phase_unwrapping_button_state(self):
        if (
            self.task.complex_projections is not None
            and self.task.complex_projections.masks is not None
        ):
            self.open_phase_unwrap_button.setEnabled(True)

    def open_interactive_window(self):
        self.init_window.show()

    def open_phase_unwrap_window(self):
        """
        Creates and shows the phase unwrapping widget in a separate window.
        """
        if self.task is None:
            print("No task available for phase unwrapping.")
            return

        if self.phase_unwrap_window is None or sip.isdeleted(self.phase_unwrap_window):
            self.phase_unwrap_window = PhaseUnwrapWidget(task=self.task)
        self.phase_unwrap_window.phase_unwrapped.connect(self.insert_phase_projections_viewer)
        self.phase_unwrap_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainProjectionTab()
    widget.show()
    sys.exit(app.exec_())
