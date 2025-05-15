from typing import Callable, Optional
from pyxalign.api.maps import get_process_func_by_enum
from pyxalign.api.options.plotting import ArrayViewerOptions, ProjectionViewerOptions
import pyxalign.data_structures.projections as p
from pyxalign.plotting.interactive.arrays import ProjectionViewer
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
    QComboBox,
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

        # Replace your old combo box code with something like this:
        self.channel_selector_scroll_area = QScrollArea(self)
        self.channel_selector_scroll_area.setWidgetResizable(True)

        self.channel_radio_group = QButtonGroup(self)
        self.channel_radio_group.setExclusive(True)

        channels_widget = QWidget(self.channel_selector_scroll_area)
        channels_layout = QVBoxLayout(channels_widget)

        for channel in self.xrf_task.channels:
            radio_button = QRadioButton(channel, channels_widget)
            channels_layout.addWidget(radio_button)
            self.channel_radio_group.addButton(radio_button)

            # Set the initially selected channel
            if channel == self.xrf_task.primary_channel:
                radio_button.setChecked(True)

            # Connect the toggled signal so your slot is called when a button is selected
            radio_button.toggled.connect(
                lambda checked, ch=channel: self.change_projection_viewer_channel(ch)
                if checked
                else None
            )
        self.channel_selector_scroll_area.setWidget(channels_widget)

        # Projection viewer
        self.projection_viewer = ProjectionViewer(
            self.xrf_task.projections_dict[self.xrf_task.primary_channel],
            include_options=include_options,
            include_shifts=include_shifts,
        )

        # Add to layout
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.channel_selector_scroll_area)
        layout.addWidget(self.projection_viewer)

    def change_projection_viewer_channel(self, current_channel: str):
        print(current_channel)
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
