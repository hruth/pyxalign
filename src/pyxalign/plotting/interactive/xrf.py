from typing import Callable, Optional
from pyxalign.api.maps import get_process_func_by_enum
from pyxalign.api.options.plotting import ArrayViewerOptions, ProjectionViewerOptions
import pyxalign.data_structures.projections as p
from pyxalign.plotting.interactive.arrays import ProjectionViewer, VolumeViewer
from pyxalign.plotting.interactive.base import ArrayViewer, MultiThreadedWidget
import pyxalign.data_structures.xrf_task as x
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
    QLabel,
    QScrollArea,
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


class XRFChannelSelector:
    def __init__(
        self, channels: list[str], primary_channel: str, connected_function: callable, parent=None
    ):
        self.scroll_area = QScrollArea(parent)
        self.scroll_area.setWidgetResizable(True)

        self.radio_group = QButtonGroup(parent)
        self.radio_group.setExclusive(True)

        widget = QWidget(self.scroll_area)
        layout = QVBoxLayout(widget)

        for channel in channels:
            radio_button = QRadioButton(channel, widget)
            layout.addWidget(radio_button)
            self.radio_group.addButton(radio_button)

            # Set the initially selected channel
            if channel == primary_channel:
                radio_button.setChecked(True)

            # Set the initially selected channel
            if channel == primary_channel:
                radio_button.setChecked(True)

        self.radio_group.buttonClicked.connect(connected_function)
        self.scroll_area.setWidget(widget)

        title_label = QLabel("Channel Selection:")
        title_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        self.layout = QVBoxLayout()
        self.layout.addWidget(title_label)
        self.layout.addWidget(self.scroll_area)

        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMaximumWidth(200)


class XRFProjectionsViewer(MultiThreadedWidget):
    """Widget for viewing all XRF projections"""

    def __init__(
        self,
        xrf_task: "x.XRFTask",
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
        self.xrf_task = xrf_task
        self.resize(1460, 850)

        self.channel_selector = XRFChannelSelector(
            self.xrf_task.channels,
            self.xrf_task.primary_channel,
            connected_function=self.change_projection_viewer_channel,
            parent=self,
        )

        # Projection viewer
        self.projection_viewer = ProjectionViewer(
            self.xrf_task.projections_dict[self.xrf_task.primary_channel],
            include_options=include_options,
            include_shifts=include_shifts,
        )

        # Add to layout
        layout = QHBoxLayout()
        self.setLayout(layout)
        # layout.addWidget(self.channel_selector.scroll_area)
        layout.addLayout(self.channel_selector.layout)
        layout.addWidget(self.projection_viewer)

    def change_projection_viewer_channel(self):
        current_channel = self.channel_selector.radio_group.checkedButton().text()
        self.projection_viewer.projections = self.xrf_task.projections_dict[current_channel]
        self.update_radio_button_group()
        self.projection_viewer.update_arrays()

    def update_radio_button_group(self):
        hide_buttons_list = []
        if self.projection_viewer.projections.masks is None:
            hide_buttons_list += [
                self.projection_viewer.masks_name,
                self.projection_viewer.projections_plus_masks_name,
            ]
        if self.projection_viewer.has_forward_projection():
            hide_buttons_list += [self.projection_viewer.forward_projections_name]
        self.hide_selected_buttons(hide_buttons_list)
        # reset checked button to measured projections if the current selection was hidden
        if self.projection_viewer.radio_button_group.checkedButton().text() in hide_buttons_list:
            self.projection_viewer.radio_button_dict[
                self.projection_viewer.projections_name
            ].setChecked(True)

    def hide_selected_buttons(self, button_names: list[str]):
        for rb in self.projection_viewer.radio_button_group.buttons():
            if rb.text() in button_names:
                rb.hide()
            else:
                rb.show()


class XRFVolumeViewer(MultiThreadedWidget):
    """Widget for viewing all XRF volumes"""

    def __init__(
        self,
        xrf_task: "x.XRFTask",
        multi_thread_func: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1460, 850)
        self.xrf_task = xrf_task

        self.volume_viewer = VolumeViewer(
            self.xrf_task.projections_dict[self.xrf_task.primary_channel].volume.data
        )

        self.channel_selector = XRFChannelSelector(
            self.xrf_task.channels,
            self.xrf_task.primary_channel,
            connected_function=self.change_projection_viewer_channel,
            parent=self,
        )

        # Add to layout
        layout = QHBoxLayout()
        self.setLayout(layout)
        # layout.addWidget(self.channel_selector.scroll_area)
        layout.addLayout(self.channel_selector.layout)
        layout.addWidget(self.volume_viewer)

    def has_volume(self, channel: str):
        return self.xrf_task.projections_dict[channel].volume is not None

    def change_projection_viewer_channel(self):
        current_channel = self.channel_selector.radio_group.checkedButton().text()
        if self.has_volume(current_channel):
            self.volume_viewer.update_arrays(
                self.xrf_task.projections_dict[current_channel].volume.data
            )


class XRFTaskViewer(MultiThreadedWidget):
    def __init__(
        self,
        xrf_task: "x.XRFTask",
        multi_thread_func: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.xrf_task = xrf_task
        self.setWindowTitle("XRF Task Overview")
        self.resize(1400, 900)

        tabs = QTabWidget()
        # Phase projections tab
        tabs.addTab(
            XRFProjectionsViewer(xrf_task),
            "Projections",
        )
        # # 3D volume tab
        # if task.phase_projections.volume.data is not None:
        #     tabs.addTab(
        #         VolumeViewer(task.phase_projections.volume.data),
        #         "3D Reconstruction",
        #     )

        # if task.pma_object is not None:
        #     pma_gui = ProjectionMatchingViewer(task.pma_object)
        #     pma_gui.initialize_plots(add_stop_button=False)
        #     pma_gui.update_plots()
        #     tabs.addTab(pma_gui, "Projection Matching")

        # layout = QVBoxLayout()
        # layout.addWidget(tabs)
        # self.setLayout(layout)
