from typing import Callable, Optional
import cupy as cp
from pyxalign.api.maps import get_process_func_by_enum
from pyxalign.api.options import ProjectionViewerOptions
from pyxalign.api.options.plotting import ArrayViewerOptions, ProjectionViewerOptions
from pyxalign.api.options_utils import get_all_attribute_names
import pyxalign.data_structures.projections as p
from pyxalign.gpu_utils import return_cpu_array
from pyxalign.interactions.mask import launch_mask_builder
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.utils.loading_decorator import loading_bar_wrapper
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.interactions.viewers.base import ArrayViewer, IndexSelectorWidget, MultiThreadedWidget
from PyQt5.QtWidgets import (
    QApplication,
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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import numpy as np
from matplotlib.figure import Figure
import matplotlib
from pyxalign.interactions.viewers.utils import (
    OptionsDisplayWidget,
    get_strings_from_table_widget,
    sync_checkboxes,
)

from pyxalign.timing.timer_utils import timer

color_list = list(matplotlib.colors.TABLEAU_COLORS.values())


class VolumeViewer(MultiThreadedWidget):
    """Widget for frames of a 3D reconstruction."""

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

        # Remove clim auto-scale selector from all but one array
        self.side_viewer_1.auto_clim_check_box.hide()
        self.side_viewer_2.auto_clim_check_box.hide()
        sync_checkboxes(
            self.depth_viewer.auto_clim_check_box,
            self.side_viewer_1.auto_clim_check_box,
            self.side_viewer_2.auto_clim_check_box,
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
    """Widget for viewing projections."""

    masks_created = pyqtSignal(np.ndarray)

    def __init__(
        self,
        projections: "p.Projections",
        options: Optional[ProjectionViewerOptions] = None,
        multi_thread_func: Optional[Callable] = None,
        include_options: bool = True,
        include_shifts: bool = True,
        display_only: bool = True,
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
        self.projection_dropping_widget = None
        self.options_editor = None
        self.resize(1300, 900)

        if np.iscomplexobj(projections.data) and options.process_func is None:
            self.process_func = np.angle
        else:
            self.process_func = get_process_func_by_enum(options.process_func)

        if self.options.sort:
            sort_idx = np.argsort(projections.angles)
        else:
            sort_idx = None
        self.array_viewer = ArrayViewer(
            array3d=projections.data,
            sort_idx=sort_idx,
            extra_title_strings_list=get_projection_title_strings(
                self.projections.scan_numbers, self.projections.angles
            ),
            process_func=self.process_func,
            options=ArrayViewerOptions(
                additional_spinbox_indexing=[self.projections.scan_numbers],
                additional_spinbox_titles=["scan number"],
            ),
        )

        # build the array selection widget
        self.build_array_selector()
        # create button for launch the scan removal tool
        if not display_only:
            # create button for scan removal tool
            open_scan_removal_button = QPushButton("Open Scan Removal Window")
            open_scan_removal_button.clicked.connect(self.open_scan_removal_window)
            # create button for the mask creation tol
            open_mask_creation_button = QPushButton("Open Mask Creation Window")
            open_mask_creation_button.clicked.connect(self.open_mask_creation_window)
            # create button for editing properties
            open_options_editor_button = QPushButton("Edit Projection Parameters")
            open_options_editor_button.clicked.connect(self.open_options_editor)

        # setup tabs and layout
        tabs = QTabWidget()
        layout = QHBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)
        # setup array view layout
        array_view_layout = QHBoxLayout()

        # setup control panel on the left
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout()
        left_panel.setLayout(left_panel_layout)
        array_view_layout.addWidget(left_panel)
        left_panel_layout.addWidget(self.button_group_box)
        if not display_only:
            left_panel_layout.addWidget(open_scan_removal_button)
            left_panel_layout.addWidget(open_mask_creation_button)
            left_panel_layout.addWidget(open_options_editor_button)
        left_panel_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

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

    def open_options_editor(self):
        if self.options_editor is None:
            all_attributes = get_all_attribute_names(self.projections.options)
            # include only experiment attributes
            basic_options_list = [x for x in all_attributes if "experiment" in x]
            # skip all other attributes
            skip_fields = [x for x in all_attributes if "experiment" not in x]
            # create options editor widget
            self.options_editor = BasicOptionsEditor(
                self.projections.options,
                basic_options_list=basic_options_list,
                skip_fields=skip_fields,
                open_panels_list=["experiment"],
                label="Projections Options Editor",
            )
        self.options_editor.show()

    def open_scan_removal_window(self):
        if self.projection_dropping_widget is None:
            self.projection_dropping_widget = ScanRemovalTool(self.projections, self.array_viewer)
        self.projection_dropping_widget.show()

    def open_mask_creation_window(self):
        # build masks from probe positions using the mask builder gui
        self.mask_gui = launch_mask_builder(
            self.projections,
            wait_until_closed=False,
        )
        self.mask_gui.masks_created.connect(self.on_masks_created)

    def on_masks_created(self):
        # update viewer so that new masks are shown
        self.update_array_selector()
        self.array_viewer.refresh_frame()

    def update_array_selector(self):
        add_masks = self.projections.masks is not None and (self.masks_name not in self.array_names)
        add_forward_projection = self.has_forward_projection() and (
            self.forward_projections_name not in self.array_names
        )
        add_buttons = []
        if add_masks:
            add_buttons += [self.masks_name, self.projections_plus_masks_name]
        if add_forward_projection:
            add_buttons += [self.forward_projections_name, self.residuals_name]
        self.array_names += add_buttons
        for array_name in add_buttons:
            self.add_button_to_group(array_name)

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
        if self.has_forward_projection():
            self.array_names += [self.forward_projections_name]
            self.array_names += [self.residuals_name]

        # Build button group
        self.radio_button_dict: dict[str, QRadioButton] = {}
        self.radio_button_group = QButtonGroup(parent=self)
        self.button_group_box = QGroupBox("Array Selection")
        button_layout = QVBoxLayout()
        self.button_group_box.setLayout(button_layout)
        self.button_group_box.setStyleSheet("QGroupBox { font-size: 13pt; }")

        # Add each button
        for array_name in self.array_names:
            self.add_button_to_group(array_name)
        self.radio_button_group.buttonClicked.connect(self.update_arrays)

        # Format button layout
        button_layout.setSpacing(10)  # Reduce space between widgets

    def add_button_to_group(self, array_name: str):
        rb = QRadioButton(array_name, self)
        self.radio_button_dict[array_name] = rb
        rb.setChecked(array_name == self.projections_name)
        self.radio_button_group.addButton(rb)
        self.button_group_box.layout().addWidget(rb)
        rb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        rb.setStyleSheet("font-size: 12pt;")

    def has_forward_projection(self):
        return (
            hasattr(self.projections, "volume")
            and self.projections.volume.forward_projections is not None
        )

    def update_arrays(self):
        # Update the data in the array viewer
        checked_button_name = self.radio_button_group.checkedButton().text()
        if checked_button_name == self.projections_name:
            self.array_viewer.process_func = self.process_func
            self.array_viewer.array3d = self.projections.data
        elif checked_button_name == self.masks_name:
            self.array_viewer.process_func = lambda x: x
            self.array_viewer.array3d = self.projections.masks
        elif checked_button_name == self.projections_plus_masks_name:
            # multiplying with the mask might be faster, and display might be
            # more intuitive for the user
            self.array_viewer.process_func = lambda x: x
            self.array_viewer.array3d = self.projections.masks + self.process_func(
                self.projections.data
            )
        elif checked_button_name == self.forward_projections_name:
            self.array_viewer.array3d = self.projections.volume.forward_projections.data
        elif checked_button_name == self.residuals_name:
            projections = return_cpu_array(self.projections.data)
            self.array_viewer.array3d = (
                projections - self.projections.volume.forward_projections.data
            )
        # update the viewer display
        self.array_viewer.refresh_frame()

    def start(self):
        self.show()


class ScanRemovalTool(QWidget):
    scan_column = 0
    angle_column = 1
    file_path_column = 2

    def __init__(
        self,
        projections: "p.Projections",
        array_viewer: ArrayViewer,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setWindowTitle("Scan Removal Tool")
        self.projections = projections

        self.array_viewer = array_viewer
        projection_dropping_widget = self.build_projection_dropper()

        # build layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.addWidget(projection_dropping_widget)

    def build_projection_dropper(self) -> QWidget:
        widget_layout = QVBoxLayout()
        # create the checkbox widget
        self.mark_for_removal_check_box = QCheckBox("Mark for removal", self)
        self.mark_for_removal_check_box.clicked.connect(self.update_staged_for_removal_list)
        self.array_viewer.slider.valueChanged.connect(self.update_mark_for_removal_check_box)
        # create table widget for show scans staged for removal
        self.staged_for_removal_table = QTableWidget(self)
        self.staged_for_removal_table.setColumnCount(4)
        self.staged_for_removal_table.setHorizontalHeaderLabels(
            ["Index", "Scan Number", "Angle (deg)", "File Path"]
        )
        self.staged_for_removal_table.currentCellChanged.connect(self.table_item_selected)
        # create table widget for previously removed scans
        self.removed_scans_table = QTableWidget(self)
        self.removed_scans_table.setColumnCount(3)
        self.removed_scans_table.setHorizontalHeaderLabels(
            ["Scan Number", "Angle (deg)", "File Path"]
        )
        for row_index, scan in enumerate(np.sort(self.projections.dropped_scan_numbers)):
            self.removed_scans_table.insertRow(row_index)
            # insert scan num
            self.removed_scans_table.setItem(
                row_index, self.scan_column, QTableWidgetItem(str(scan))
            )
            # insert angle
            if scan in self.projections.dropped_angles.keys():
                angle = self.projections.dropped_angles[scan]
                self.removed_scans_table.setItem(
                    row_index, self.angle_column, QTableWidgetItem(str(angle))
                )
            # insert file path
            if scan in self.projections.dropped_file_paths.keys():
                file_path = self.projections.dropped_file_paths[scan]
                self.removed_scans_table.setItem(
                    row_index, self.file_path_column, QTableWidgetItem(file_path)
                )
        # create the button for permanently dropping projections
        drop_projections_button = QPushButton("Permanently Remove Scans", self)
        drop_projections_button.pressed.connect(self.remove_staged_projections)
        # Create new index selector and attach it to the array_viewer's index selection widget
        index_selector_widget = IndexSelectorWidget(
            self.array_viewer.num_frames,
            self.array_viewer.slider.value(),
            include_play_button=False,
            parent=self,
        )
        # index_selector_widget.spin_play_layout.insertWidget(0, QLabel("index", self))
        index_selector_widget.slider.setMinimum(0)
        index_selector_widget.slider.setMaximum(self.array_viewer.slider.maximum())
        index_selector_widget.slider.setValue(self.array_viewer.slider.value())
        index_selector_widget.slider.valueChanged.connect(self.array_viewer.slider.setValue)
        self.array_viewer.slider.valueChanged.connect(index_selector_widget.slider.setValue)

        # insert widgets into layout
        widget_layout.addWidget(QLabel("Scans staged for removal", self))
        widget_layout.addWidget(self.staged_for_removal_table)
        widget_layout.addWidget(QLabel("Previously removed scans", self))
        widget_layout.addWidget(self.removed_scans_table)
        widget_layout.addWidget(drop_projections_button)

        # widget_layout.addLayout(index_selection_layout)
        widget_layout.addWidget(index_selector_widget)
        widget_layout.addWidget(self.mark_for_removal_check_box)  # temp location
        # format list widget style
        widget_group_box = QGroupBox()
        widget_group_box.setStyleSheet("QGroupBox { font-size: 13pt; }")
        widget_group_box.setLayout(widget_layout)

        self.setStyleSheet("QLabel { font-size: 11pt;}")

        return widget_group_box

    def remove_staged_projections(self):
        # remove scans from projection object
        remove_scan_numbers = []
        for row in range(self.staged_for_removal_table.rowCount()):
            remove_scan_numbers += [
                int(self.staged_for_removal_table.item(row, self.scan_column + 1).text())
            ]
        # drop projections
        drop_projections_wrapped = loading_bar_wrapper("Removing projections...")(
            self.projections.drop_projections
        )
        drop_projections_wrapped(remove_scan_numbers)
        # clear rows
        self.staged_for_removal_table.blockSignals(True)
        self.staged_for_removal_table.setRowCount(0)
        self.staged_for_removal_table.blockSignals(False)
        # update table of dropped scans
        new_rows_count = len(remove_scan_numbers)
        for i in range(new_rows_count):
            row_index = self.removed_scans_table.rowCount()
            self.removed_scans_table.insertRow(row_index)
            scan = remove_scan_numbers[i]
            # insert scan
            self.removed_scans_table.setItem(
                row_index, self.scan_column, QTableWidgetItem(str(scan))
            )
            # insert angle
            if scan in self.projections.dropped_angles.keys():
                angle = self.projections.dropped_angles[scan]
                self.removed_scans_table.setItem(
                    row_index, self.angle_column, QTableWidgetItem(str(angle))
                )
            # insert file path
            if scan in self.projections.dropped_file_paths.keys():
                file_path = self.projections.dropped_file_paths[scan]
                self.removed_scans_table.setItem(
                    row_index, self.file_path_column, QTableWidgetItem(file_path)
                )
        # un-check scan removal checkbox
        self.mark_for_removal_check_box.blockSignals(True)
        self.mark_for_removal_check_box.setChecked(False)
        self.mark_for_removal_check_box.blockSignals(False)
        sort_idx = np.argsort(self.projections.angles)
        # re-initialize array viewer
        self.array_viewer.reinitialize_all(
            array3d=self.projections.data,
            sort_idx=sort_idx,
            extra_title_strings_list=get_projection_title_strings(
                self.projections.scan_numbers, self.projections.angles
            ),
            new_additional_spinbox_indexing=[self.projections.scan_numbers],
        )

    def table_item_selected(self, row: int):
        index = int(self.staged_for_removal_table.item(row, 0).text())
        self.array_viewer.update_index_externally(index)

    def update_mark_for_removal_check_box(self):
        "Update the scan removal checkbox as the scan index changes"
        scans_in_list = get_strings_from_table_widget(self.staged_for_removal_table)
        is_checked = str(self.array_viewer.slider.value()) in scans_in_list
        self.mark_for_removal_check_box.setChecked(is_checked)

    def update_staged_for_removal_list(self):
        index = self.array_viewer.slider.value()
        if self.mark_for_removal_check_box.isChecked():
            # add the scan to the list widget
            sorted_index = self.array_viewer.sort_idx[index]
            row_index = self.staged_for_removal_table.rowCount()
            self.staged_for_removal_table.insertRow(row_index)
            # add index
            self.staged_for_removal_table.setItem(row_index, 0, QTableWidgetItem(str(index)))
            # add scan number
            self.staged_for_removal_table.setItem(
                row_index,
                self.scan_column + 1,
                QTableWidgetItem(str(self.projections.scan_numbers[sorted_index])),
            )
            # add angle
            self.staged_for_removal_table.setItem(
                row_index,
                self.angle_column + 1,
                QTableWidgetItem(f"{self.projections.angles[sorted_index]:.3f}"),
            )
            # add file path
            if self.projections.file_paths is not None:
                self.staged_for_removal_table.setItem(
                    row_index,
                    self.file_path_column + 1,
                    QTableWidgetItem(self.projections.file_paths[sorted_index]),
                )
        else:
            # find row and remove it
            for row in range(self.staged_for_removal_table.rowCount()):
                current_scan_index = int(self.staged_for_removal_table.item(row, 0).text())
                if index == current_scan_index:
                    self.staged_for_removal_table.removeRow(row)
                    return

    def closeEvent(self, event):
        # Hide the window instead of closing it
        self.hide()
        event.ignore()


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
        self.checkbox_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        # wrap the button layout in a QGroupBox
        button_group_box = QGroupBox("Select shifts to display")
        button_group_box.setStyleSheet("QGroupBox { font-size: 13pt; }")
        button_group_box.setLayout(self.checkbox_layout)

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


def get_projection_title_strings(scan_numbers: np.ndarray, angles: np.ndarray) -> list[str]:
    whitespace = "&nbsp;" * 3

    def return_angle_string(angle):
        return f"<span style='color:pink'>Angle {angle:.3f}<sup>o</sup></span>"

    def return_scan_string(scan_number):
        return f"<span style='color:#9FEDB9'>Scan {scan_number}</span>"

    title_strings = [
        f"{whitespace}{return_scan_string(scan)}{whitespace}{return_angle_string(angle)}"
        for scan, angle in zip(scan_numbers, angles)
    ]
    return title_strings


@switch_to_matplotlib_qt_backend
def launch_projection_viewer(
    projections: "p.Projections",
    display_only: bool = False,
    wait_until_closed: bool = False,
) -> ProjectionViewer:
    """Launch a GUI for interactively viewing and updating a `Projections`
    object.

    Args:
        projections (Projections): The projections to display.
        display_only (bool): If enabled, interactive features like scan
            removal will not be available. Defaults to false.
        wait_until_closed (bool): if `True`, the application starts a
            blocking call until the GUI window is closed.

    Example:
        Launch a GUI for interactively viewing a `ComplexProjections`
        object::

            gui = pyxalign.gui.launch_projection_viewer(task.complex_projections)
    """
    app = QApplication.instance() or QApplication([])
    gui = ProjectionViewer(projections, display_only=display_only)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui


@switch_to_matplotlib_qt_backend
def launch_volume_viewer(
    array_3d: np.ndarray,
    wait_until_closed: bool = False,
) -> VolumeViewer:
    """Launch the volume viewer GUI. This viewer shows three interactive plots
    where you can index through each layer of the 3D array. Each of the three
    interactive plots indexes through a different dimension of the input array.

    Args:
        array_3d (np.ndarray): A 3-dimensional array.
        wait_until_closed (bool): if `True`, the application starts a
            blocking call until the GUI window is closed.

    Example:
        Reconstruct the 3D volume and display it
        interactively::

            task.phase_projections.get_3D_reconstruction()
            gui = pyxalign.gui.launch_volume_viewer(task.phase_projections.volume.data)
    """
    app = QApplication.instance() or QApplication([])
    gui = VolumeViewer(volume=array_3d)
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui
