from typing import Callable, Optional
from pyxalign.api.maps import get_process_func_by_enum
from pyxalign.api.options.plotting import ArrayViewerOptions, ProjectionViewerOptions
import pyxalign.data_structures.projections as p
from pyxalign.plotting.interactive.base import ArrayViewer, MultiThreadedWidget
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QTabWidget,
    QSizePolicy,
    QSpacerItem,
    QGroupBox,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import numpy as np
from matplotlib.figure import Figure
import matplotlib
from pyxalign.plotting.interactive.utils import OptionsDisplayWidget

from pyxalign.timing.timer_utils import timer

color_list = list(matplotlib.colors.TABLEAU_COLORS.values())


class VolumeViewer(MultiThreadedWidget):
    """Widget for frames of a 3D reconstruction"""

    def __init__(
        self,
        volume: np.ndarray,
        multi_thread_func: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.volume = volume

        self.depth_viewer = ArrayViewer(
            array3d=volume,
            options=ArrayViewerOptions(
                slider_axis=0,
                start_index=int(volume.shape[0] / 2),
            ),
        )
        self.side_viewer_1 = ArrayViewer(
            array3d=volume,
            options=ArrayViewerOptions(
                slider_axis=1,
                start_index=int(volume.shape[1] / 2),
            ),
        )
        self.side_viewer_2 = ArrayViewer(
            array3d=volume,
            options=ArrayViewerOptions(
                slider_axis=2,
                start_index=int(volume.shape[2] / 2),
            ),
        )

        # Layout
        layout = QHBoxLayout()
        side_view_layout = QVBoxLayout()
        layout.addWidget(self.depth_viewer)
        side_view_layout.addWidget(self.side_viewer_1)
        side_view_layout.addWidget(self.side_viewer_2)
        layout.addLayout(side_view_layout)
        self.setLayout(layout)

    @timer()
    def update_arrays(self, volume: np.ndarray):
        self.depth_viewer.array3d = volume
        self.side_viewer_1.array3d = volume
        self.side_viewer_2.array3d = volume
        # update the viewer display
        self.depth_viewer.refresh_frame()
        self.side_viewer_1.refresh_frame()
        self.side_viewer_2.refresh_frame()

    def start(self):
        self.show()


class ProjectionViewer(MultiThreadedWidget):
    """Widget for viewing projections"""

    def __init__(
        self,
        projections: "p.Projections",
        options: Optional[ProjectionViewerOptions] = None,
        multi_thread_func: Optional[Callable] = None,
        include_options: bool = True,
        include_shifts: bool = True,
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.projections = projections
        if options is None:
            options = ProjectionViewerOptions()
        self.options = options
        self.resize(1300, 900)

        if np.iscomplex(projections.data[0, 0, 0]) and options.process_func is None:
            self.process_func = np.angle
        else:
            self.process_func = get_process_func_by_enum(options.process_func)

        if self.options.sort:
            sort_idx = np.argsort(projections.angles)
        else:
            sort_idx = None
        self.array_viewer = ArrayViewer(array3d=projections.data, sort_idx=sort_idx)

        button_group_box = self.build_array_selector()

        # setup tabs and layout
        tabs = QTabWidget()
        layout = QHBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)
        # setup array view layout
        array_view_layout = QHBoxLayout()
        array_view_layout.addWidget(button_group_box)
        array_view_layout.addWidget(self.array_viewer)
        array_view_widget = QWidget()
        array_view_widget.setLayout(array_view_layout)
        # add tabs
        tabs.addTab(array_view_widget, "Array Viewer")
        # add tab showing past shifts
        if include_shifts:
            tabs.addTab(AllShiftsViewer(projections), "Applied Shifts")
        if include_options:
            # create options viewer
            self.options_display = OptionsDisplayWidget(projections.options)
            tabs.addTab(self.options_display, "Projection Options")

    def build_array_selector(self) -> QWidget:
        self.projections_name = "projections"
        self.masks_name = "masks"
        self.projections_plus_masks_name = "projections + masks"
        self.forward_projections_name = "forward projections"
        self.residuals_name = "projections - forward projections"
        self.array_names = [self.projections_name]
        if self.projections.masks is not None:
            self.array_names += [self.masks_name]
            self.array_names += [self.projections_plus_masks_name]
        has_forward_projection = (
            hasattr(self.projections, "volume")
            and self.projections.volume.forward_projections is not None
        )
        if has_forward_projection:
            self.array_names += [self.forward_projections_name]
            self.array_names += [self.residuals_name]

        # Build button group
        self.radio_button_group = QButtonGroup(parent=self)
        button_layout = QVBoxLayout()

        # Add each button
        for array_name in self.array_names:
            rb = QRadioButton(array_name, self)
            rb.setChecked(array_name == self.projections_name)
            self.radio_button_group.addButton(rb)
            button_layout.addWidget(rb)
            rb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            rb.setStyleSheet("font-size: 12pt;")
        self.radio_button_group.buttonClicked.connect(self.update_arrays)

        # Format button layout
        button_layout.setSpacing(10)  # Reduce space between widgets
        button_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Wrap the button layout in a QGroupBox
        button_group_box = QGroupBox("Array Selection")
        button_group_box.setStyleSheet("QGroupBox { font-size: 13pt; }")
        button_group_box.setLayout(button_layout)

        return button_group_box

    def update_arrays(self):
        # Update the data in the array viewer
        checked_button_name = self.radio_button_group.checkedButton().text()
        if checked_button_name == self.projections_name:
            self.array_viewer.array3d = self.process_func(self.projections.data)
        elif checked_button_name == self.masks_name:
            self.array_viewer.array3d = self.projections.masks
        elif checked_button_name == self.projections_plus_masks_name:
            self.array_viewer.array3d = self.projections.masks + self.projections.data
        elif checked_button_name == self.forward_projections_name:
            self.array_viewer.array3d = self.projections.volume.forward_projections.data
        elif checked_button_name == self.residuals_name:
            self.array_viewer.array3d = (
                self.projections.data - self.projections.volume.forward_projections.data
            )
        # update the viewer display
        self.array_viewer.refresh_frame()

    def start(self):
        self.show()


class AllShiftsViewer(MultiThreadedWidget):
    def __init__(
        self,
        projections: "p.Projections",
        multi_thread_func: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )

        self.shifts_list = projections.shift_manager.past_shifts
        self.staged_shift = projections.shift_manager.staged_shift
        self.sort_idx = np.argsort(projections.angles)
        self.angles = projections.angles
        self.pixel_size = projections.pixel_size
        self.init_ui()
        self.update_plot()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # === Left panel: axis mode and checkboxes ===
        control_layout = QVBoxLayout()

        # Checkboxes for array selection
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_widget)
        self.checkboxes = []
        
        if len(self.shifts_list) > 0:
            # Add checkbox for total shift
            self.shifts_list = [np.sum(self.shifts_list, 0)] + self.shifts_list
            cb = QCheckBox("Total of applied shifts")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_plot)
            self.checkboxes.append(cb)
            self.checkbox_layout.addWidget(cb)
            # Add checkboxes for the rest of the shifts
            for i in range(1, len(self.shifts_list)):
                cb = QCheckBox(f"Applied shift {i}")
                cb.setChecked(True)
                cb.stateChanged.connect(self.update_plot)
                self.checkboxes.append(cb)
                self.checkbox_layout.addWidget(cb)

        if np.any(self.staged_shift != 0):
            # Add checkbox for the staged shift
            self.shifts_list = self.shifts_list + [self.staged_shift]
            cb = QCheckBox("Staged shift")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_plot)
            self.checkboxes.append(cb)
            self.checkbox_layout.addWidget(cb)

        # format checkboxes
        for cb in self.checkboxes:
            cb.setStyleSheet("font-size: 12pt;")
        # format layout
        self.checkbox_layout.setSpacing(10)  # Reduce space between widgets
        self.checkbox_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        # wrap the button layout in a QGroupBox
        button_group_box = QGroupBox("Select shifts to display")
        button_group_box.setStyleSheet("QGroupBox { font-size: 13pt; }")
        button_group_box.setLayout(self.checkbox_layout)

        # scroll = QScrollArea()
        # scroll.setWidgetResizable(True)
        # scroll.setWidget(self.checkbox_widget)
        # control_layout.addWidget(scroll)
        # control_layout.addStretch()
        # control_layout.addWidget(self.checkbox_widget)
        control_layout.addWidget(button_group_box)

        # === Right panel: matplotlib plot ===
        self.figure = Figure(layout="compressed")
        self.canvas = FigureCanvas(self.figure)
        self.ax = [self.figure.add_subplot(211), self.figure.add_subplot(212)]
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(plot_layout, 4)

    def update_plot(self):
        for j in range(2):
            self.ax[j].clear()
        for i, cb in enumerate(self.checkboxes):
            if cb.isChecked():
                array = self.shifts_list[i]
                for j in range(2):
                    self.ax[j].plot(
                        self.angles[self.sort_idx],
                        array[self.sort_idx, j],
                        label=cb.text(),
                        color=color_list[i],
                    )
                    # self.ax[j].legend()
                    self.ax[j].grid(linestyle=":")
                    self.ax[j].autoscale(enable=True, axis="x", tight=True)
                    self.ax[j].set_ylabel(f"Shift ({self.pixel_size * 1e9:.0f} nm px)")
                    self.ax[j].set_xlabel("Angle (deg)")
        self.ax[0].set_title("Horizontal Shifts")
        self.ax[1].set_title("Vertical Shifts")
        self.ax[0].legend(bbox_to_anchor=(1.1, 1.05))
        self.canvas.draw()

    def start(self):
        self.show()
