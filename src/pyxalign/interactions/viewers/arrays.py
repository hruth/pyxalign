from typing import Callable, Optional
import cupy as cp
import h5py
from pyxalign.api.maps import get_process_func_by_enum
from pyxalign.api.options import ProjectionViewerOptions
from pyxalign.api.options.plotting import ArrayViewerOptions, ProjectionViewerOptions
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.options_utils import print_options
import pyxalign.data_structures.projections as p
from pyxalign.api import enums
from pyxalign.gpu_utils import return_cpu_array
from pyxalign.interactions.mask import launch_mask_builder
from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.reconstruction_parameter_tuner import ReconstructionParameterTuner
from pyxalign.interactions.roi_selector import launch_mask_selection_from_roi
from pyxalign.interactions.utils.loading_display_tools import loading_bar_wrapper
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend
from pyxalign.io.utils import load_list_of_arrays_or_str
from pyxalign.interactions.viewers.base import (
    ArrayViewer,
    IndexSelectorWidget,
    MultiThreadedWidget,
)
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
    QDialog,
    QLineEdit,
    QFileDialog,
    QComboBox,
    QFormLayout,
    QFrame,
    QMessageBox,
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
import pyqtgraph as pg

color_list = list(matplotlib.colors.XKCD_COLORS.values())



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
            hide_climit_controls=True,
        )
        self.side_viewer_2 = ArrayViewer(
            array3d=volume,
            options=ArrayViewerOptions(
                slider_axis=2,
                start_index=int(volume.shape[2] / 2),
            ),
            hide_climit_controls=True,
        )
        # connect climit spinboxes
        main_climit_widget = self.depth_viewer.climit_window
        for climit_widget in [
            self.side_viewer_1.climit_window,
            self.side_viewer_2.climit_window,
        ]:
            main_climit_widget.lower_limit_spinbox.valueChanged.connect(
                climit_widget.lower_limit_spinbox.setValue
            )
            main_climit_widget.upper_limit_spinbox.valueChanged.connect(
                climit_widget.upper_limit_spinbox.setValue
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


class ApplySavedAlignmentShiftDialog(QDialog):
    """Dialog window for applying a saved alignment shift."""

    def __init__(
        self,
        projections: "p.Projections",
        array_viewer: ArrayViewer,
        parent: Optional[QWidget] = None,
        refresh_callback: Optional[Callable] = None,
    ):
        super().__init__(parent)
        self.projections = projections
        self.array_viewer = array_viewer
        self.refresh_callback = refresh_callback
        self.device_options = DeviceOptions()
        self.setWindowTitle("Apply Alignment Shift from File")

        # Store geometry parameters from the file
        self.tilt_angle = None
        self.skew_angle = None
        self.lamino_angle = None
        self.sample_thickness = None

        self.setup_ui()

    def setup_ui(self):
        """Build the user interface."""
        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # File path selection
        file_path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select HDF5 file containing alignment shifts...")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file_path)
        read_file_button = QPushButton("Read File")
        read_file_button.clicked.connect(self.read_file_from_textbox)
        file_path_layout.addWidget(self.file_path_edit)
        file_path_layout.addWidget(browse_button)
        file_path_layout.addWidget(read_file_button)
        form_layout.addRow("Aligned Task/Shift File Path:", file_path_layout)

        # Staged function type dropdown
        self.function_type_combo = QComboBox()
        for shift_type in enums.ShiftType:
            self.function_type_combo.addItem(shift_type.value, shift_type)
        # Set default to FFT
        fft_index = self.function_type_combo.findData(enums.ShiftType.FFT)
        if fft_index >= 0:
            self.function_type_combo.setCurrentIndex(fft_index)
        form_layout.addRow("Staged Function Type:", self.function_type_combo)

        # Drop unshared scans checkbox
        self.drop_unshared_checkbox = QCheckBox()
        self.drop_unshared_checkbox.setChecked(False)
        form_layout.addRow("Drop Unshared Scans:", self.drop_unshared_checkbox)

        main_layout.addLayout(form_layout)

        # Add Device Options editor
        self.device_options_editor = BasicOptionsEditor(
            data=self.device_options.gpu,
            label="GPU Options for Array Shifting",
            parent=self,
        )
        main_layout.addWidget(self.device_options_editor)

        # Add Geometry Parameters Display section
        self.geometry_display_group = QGroupBox("Geometry Parameters from File")
        geometry_layout = QVBoxLayout()

        # Parameters form layout
        params_layout = QFormLayout()
        self.tilt_angle_label = QLabel("N/A")
        self.skew_angle_label = QLabel("N/A")
        self.lamino_angle_label = QLabel("N/A")
        self.sample_thickness_label = QLabel("N/A")

        params_layout.addRow("Tilt Angle:", self.tilt_angle_label)
        params_layout.addRow("Skew Angle:", self.skew_angle_label)
        params_layout.addRow("Laminography Angle:", self.lamino_angle_label)
        params_layout.addRow("Sample Thickness:", self.sample_thickness_label)

        geometry_layout.addLayout(params_layout)

        # Add matplotlib plot for alignment shifts
        self.shift_figure = Figure(figsize=(8, 4), layout="compressed")
        self.shift_canvas = FigureCanvas(self.shift_figure)
        self.shift_ax = [self.shift_figure.add_subplot(211), self.shift_figure.add_subplot(212)]
        self.shift_ax[0].set_title("Horizontal Shift")
        self.shift_ax[1].set_title("Vertical Shift")
        for ax in self.shift_ax:
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Shift (pixels)")
            ax.grid(linestyle=":")

        geometry_layout.addWidget(self.shift_canvas)

        self.geometry_display_group.setLayout(geometry_layout)
        main_layout.addWidget(self.geometry_display_group)

        # Apply button
        apply_button = QPushButton("Apply Alignment Shift from File")
        apply_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        apply_button.clicked.connect(self.apply_shift)
        main_layout.addWidget(apply_button)

        self.setLayout(main_layout)
        self.resize(800, 800)

    def browse_file_path(self):
        """Open a file dialog to select the alignment shift file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Alignment Shift File",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)
            self.load_geometry_parameters(file_path)

    def read_file_from_textbox(self):
        """Read geometry parameters from the file path in the textbox."""
        file_path = self.file_path_edit.text()
        if file_path:
            self.load_geometry_parameters(file_path)
        else:
            print("Error: No file path specified.")

    def load_geometry_parameters(self, task_file_path):
        """Load geometry parameters from the HDF5 file and display them."""
        try:
            with h5py.File(task_file_path, "r") as F:
                # Determine which group to use
                if "phase_projections" in F.keys():
                    group = "phase_projections"
                elif "complex_projections" in F.keys():
                    group = "complex_projections"
                else:
                    print("Warning: No phase_projections or complex_projections group found in file.")
                    return

                # Extract geometry parameters
                self.tilt_angle = F[group]["options/reconstruct/geometry/tilt_angle"][()]
                self.skew_angle = F[group]["options/reconstruct/geometry/skew_angle"][()]
                self.lamino_angle = F[group]["options/experiment/laminography_angle"][()]
                self.sample_thickness = F[group]["options/experiment/sample_thickness"][()]

                # Update the display labels
                self.tilt_angle_label.setText(f"{self.tilt_angle:.6f}")
                self.skew_angle_label.setText(f"{self.skew_angle:.6f}")
                self.lamino_angle_label.setText(f"{self.lamino_angle:.6f}")
                self.sample_thickness_label.setText(f"{self.sample_thickness:.6e}")

                # Load and plot alignment shifts
                angles = F[group]["angles"][()]
                applied_shifts = load_list_of_arrays_or_str(F[group], "applied_shifts")

                if applied_shifts is not None:
                    # Sum all applied shifts and sort by angle
                    total_shifts = np.sum(applied_shifts, 0).astype(np.float32)
                    sort_idx = np.argsort(angles)

                    # Clear previous plots
                    for ax in self.shift_ax:
                        ax.clear()

                    # Plot horizontal shift (first column)
                    self.shift_ax[0].plot(angles[sort_idx], total_shifts[sort_idx, 0])
                    self.shift_ax[0].set_title("Horizontal Shift")
                    self.shift_ax[0].set_xlabel("Angle (deg)")
                    self.shift_ax[0].set_ylabel("Shift (pixels)")
                    self.shift_ax[0].grid(linestyle=":")
                    self.shift_ax[0].autoscale(enable=True, axis="x", tight=True)

                    # Plot vertical shift (second column)
                    self.shift_ax[1].plot(angles[sort_idx], total_shifts[sort_idx, 1])
                    self.shift_ax[1].set_title("Vertical Shift")
                    self.shift_ax[1].set_xlabel("Angle (deg)")
                    self.shift_ax[1].set_ylabel("Shift (pixels)")
                    self.shift_ax[1].grid(linestyle=":")
                    self.shift_ax[1].autoscale(enable=True, axis="x", tight=True)

                    self.shift_canvas.draw()

                print(f"Loaded geometry parameters from: {task_file_path}")
        except Exception as e:
            print(f"Error loading geometry parameters: {e}")
            self.tilt_angle = None
            self.skew_angle = None
            self.lamino_angle = None
            self.sample_thickness = None
            self.tilt_angle_label.setText("N/A")
            self.skew_angle_label.setText("N/A")
            self.lamino_angle_label.setText("N/A")
            self.sample_thickness_label.setText("N/A")

    def apply_shift(self):
        """Apply the saved alignment shift with the selected parameters."""
        task_file_path = self.file_path_edit.text()

        # Validate that a file path was selected
        if not task_file_path:
            print("Error: Please select an alignment shift file.")
            return

        # Get the selected shift type
        staged_function_type = self.function_type_combo.currentData()

        # Get the checkbox state
        drop_unshared_scans = self.drop_unshared_checkbox.isChecked()

        # Call the load_and_stage_shift method
        try:
            load_and_stage_wrapped = loading_bar_wrapper(load_message="Staging shift...")(
                func=self.projections.load_and_stage_shift
            )
            load_and_stage_wrapped(
                task_file_path=task_file_path,
                staged_function_type=staged_function_type,
                drop_unshared_scans=drop_unshared_scans,
            )
            # Apply the staged shift with the configured device options
            apply_shift_wrapped = loading_bar_wrapper(load_message="Applying shift...")(
                func=self.projections.apply_staged_shift
            )
            apply_shift_wrapped(device_options=self.device_options)
            print(f"Successfully applied alignment shift from: {task_file_path}")

            # Apply geometry parameters if they were loaded from the file
            if all(param is not None for param in [self.tilt_angle, self.skew_angle, self.lamino_angle, self.sample_thickness]):
                self.projections.options.reconstruct.geometry.tilt_angle = self.tilt_angle
                self.projections.options.reconstruct.geometry.skew_angle = self.skew_angle
                self.projections.options.experiment.laminography_angle = self.lamino_angle
                self.projections.options.experiment.sample_thickness = self.sample_thickness
                print("Applied geometry parameters from file to projections.")

            # Refresh the array_viewer
            self.array_viewer.reinitialize_all(
                array3d=self.projections.data,
                sort_idx=np.argsort(self.projections.angles),
                extra_title_strings_list=get_projection_title_strings(
                    self.projections.scan_numbers, self.projections.angles
                ),
                new_additional_spinbox_indexing=[self.projections.scan_numbers],
            )

            # Refresh the applied shifts tab if callback is provided
            if self.refresh_callback is not None:
                self.refresh_callback()

            self.accept()  # Close the dialog
        except Exception as e:
            print(f"Error applying alignment shift: {e}")


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
        include_array_saving_widget: bool = False,
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
        # self.projection_dropping_widget = None
        self.options_editor = None
        self.reconstruction_parameter_tuner = None
        self.apply_saved_shift_dialog = None
        self.fsc_calculation_window = None
        self.mask_gui = None
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
            hide_axis_controls=True,
            include_array_saving_widget=include_array_saving_widget,
        )

        # build the array selection widget
        self.build_array_selector()
        # create button for launch the scan removal tool
        if not display_only:
            # create button for scan removal tool
            open_scan_removal_button = QPushButton("Open Scan Removal Window")
            open_scan_removal_button.clicked.connect(self.open_scan_removal_window)
            # create button for the mask creation tools
            open_mask_creation_button = QPushButton("Get Masks from Probe Positions")
            open_mask_creation_button.clicked.connect(self.open_mask_creation_window)
            if self.projections.probe_positions is None:
                open_mask_creation_button.setDisabled(True)
            open_mask_from_roi_button = QPushButton("Get Masks from ROI")
            open_mask_from_roi_button.clicked.connect(self.open_mask_from_roi_window)
            # create button for updating reconstruction parameters
            open_reconstruction_tuner_button = QPushButton("3D Volume Reconstruction Tool")
            open_reconstruction_tuner_button.clicked.connect(self.open_reconstruction_parameter_tuner)
            # Only enable for PhaseProjections
            if self.projections.__class__.__qualname__ != "PhaseProjections":
                open_reconstruction_tuner_button.setDisabled(True)
            # create button for inverting projections
            invert_projections_button = QPushButton("Invert Projections")
            invert_projections_button.clicked.connect(self.invert_projections)
            # create button for applying saved alignment shift
            apply_shift_from_file_button = QPushButton("Apply Alignment Shift from File")
            apply_shift_from_file_button.clicked.connect(self.open_apply_saved_shift_dialog)
            # create button for pinning array memory
            pin_array_memory_button = QPushButton("Pin Array Memory")
            pin_array_memory_button.clicked.connect(self.pin_array_memory)
            # create button for FSC calculation
            calculate_fsc_button = QPushButton("Calculate FSC")
            calculate_fsc_button.clicked.connect(self.open_fsc_calculation_window)
            # Disable for complex projections
            if np.iscomplexobj(self.projections.data):
                calculate_fsc_button.setDisabled(True)
                calculate_fsc_button.setToolTip("FSC calculation is not available for complex projections")

            push_button_layout = QVBoxLayout()
            push_button_layout.addWidget(
                QLabel("Alignment and Reconstruction:"), alignment=Qt.AlignCenter
            )
            push_button_layout.addWidget(open_reconstruction_tuner_button)
            push_button_layout.addWidget(apply_shift_from_file_button)
            push_button_layout.addWidget(
                QLabel("Projection Array Manipulation:"), alignment=Qt.AlignCenter
            )
            push_button_layout.addWidget(open_scan_removal_button)
            push_button_layout.addWidget(invert_projections_button)
            push_button_layout.addWidget(pin_array_memory_button)
            push_button_layout.addWidget(
                QLabel("Mask Creation:"), alignment=Qt.AlignCenter
            )
            push_button_layout.addWidget(open_mask_creation_button)
            push_button_layout.addWidget(open_mask_from_roi_button)
            push_button_layout.addWidget(
                QLabel("Analysis:"), alignment=Qt.AlignCenter
            )
            push_button_layout.addWidget(calculate_fsc_button)

        # setup tabs and layout
        tabs = QTabWidget()
        layout = QHBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)
        # setup array view layout
        array_view_layout = QHBoxLayout()

        # setup control panel on the left
        left_panel = QWidget()
        self.left_panel_layout = QVBoxLayout()
        left_panel.setLayout(self.left_panel_layout)
        array_view_layout.addWidget(left_panel)
        self.left_panel_layout.addWidget(self.button_group_box)
        if not display_only:
            self.left_panel_layout.addLayout(push_button_layout)
        self.left_panel_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        array_view_layout.addWidget(self.array_viewer)
        array_view_widget = QWidget()
        array_view_widget.setLayout(array_view_layout)
        # add tabs
        tabs.addTab(array_view_widget, "Array Viewer")
        # add tab showing past shifts
        self.all_shifts_viewer = None
        if include_shifts:
            self.all_shifts_viewer = AllShiftsViewer(projections)
            tabs.addTab(self.all_shifts_viewer, "Applied Shifts")
            self.all_shifts_viewer.shift_operation_performed.connect(self.refresh_array_viewer)
        if include_options:
            # create options viewer
            self.options_display = OptionsDisplayWidget(projections.options)
            tabs.addTab(self.options_display, "Projection Options")

        # Connect tab change signal to update options display when tab is opened
        self.tabs = tabs
        tabs.currentChanged.connect(self.on_tab_changed)

        # create the scan removal tool
        self.projection_dropping_widget = ScanRemovalTool(
            self.projections,
            self.array_viewer,
            projection_drop_function=self.projections.drop_projections,
        )

    def invert_projections(self):
        """Invert the projection data and refresh the display."""
        def _invert():
            if np.iscomplexobj(self.projections.data):
                self.projections.data[:] = np.conj(self.projections.data)
            else:
                self.projections.data[:] = -self.projections.data
        invert_wrapped = loading_bar_wrapper("Inverting projections...")(_invert)
        invert_wrapped()

        self.array_viewer.refresh_frame()

    def pin_array_memory(self):
        """Pin the projection array memory, which enables faster movement to GPU"""
        pin_wrapped = loading_bar_wrapper("Pinning array memory...")(self.projections.pin_arrays)
        pin_wrapped()

    def open_reconstruction_parameter_tuner(self):
        """Open the reconstruction parameter tuner window."""
        # Check if the window exists and hasn't been deleted
        try:
            if self.reconstruction_parameter_tuner is not None:
                # Try to access a property to see if it's been deleted
                self.reconstruction_parameter_tuner.isVisible()
        except RuntimeError:
            # Window was deleted, set to None so we recreate it
            self.reconstruction_parameter_tuner = None

        if self.reconstruction_parameter_tuner is None:
            self.reconstruction_parameter_tuner = ReconstructionParameterTuner(
                phase_projections=self.projections,
            )
        self.reconstruction_parameter_tuner.show()

    def open_fsc_calculation_window(self):
        """Open the FSC calculation window."""
        # Check if the window exists and hasn't been deleted
        try:
            if self.fsc_calculation_window is not None:
                # Try to access a property to see if it's been deleted
                self.fsc_calculation_window.isVisible()
        except RuntimeError:
            # Window was deleted, set to None so we recreate it
            self.fsc_calculation_window = None

        if self.fsc_calculation_window is None:
            from pyxalign.interactions.fsc_window import FSCCalculationWindow
            self.fsc_calculation_window = FSCCalculationWindow(
                projections=self.projections
            )
        self.fsc_calculation_window.show()

    def open_scan_removal_window(self):
        if self.projection_dropping_widget is None:
            self.projection_dropping_widget = ScanRemovalTool(
                self.projections,
                self.array_viewer,
                projection_drop_function=self.projections.drop_projections,
            )
        self.projection_dropping_widget.show()

    def open_apply_saved_shift_dialog(self):
        """Open the dialog for applying a saved alignment shift."""
        # Check if there are already applied shifts
        if len(self.projections.shift_manager.past_shifts) > 0:
            QMessageBox.warning(
                self,
                "Cannot Apply Saved Alignment Shift",
                "Cannot apply alignment shift if the projections have already been shifted. "
                "If you want to apply a saved alignment shift, you must first undo all previously applied shifts.",
            )
            return

        if self.apply_saved_shift_dialog is None:
            self.apply_saved_shift_dialog = ApplySavedAlignmentShiftDialog(
                self.projections,
                array_viewer=self.array_viewer,
                parent=self,
                refresh_callback=self.refresh_applied_shifts_tab,
            )
        self.apply_saved_shift_dialog.show()

    def open_mask_creation_window(self):
        self._close_mask_gui()
        self.mask_gui = launch_mask_builder(
            self.projections,
            wait_until_closed=False,
        )
        self.mask_gui.masks_created.connect(self.on_masks_created)
        # Auto-clear the reference when Qt destroys the widget (WA_DeleteOnClose).
        self.mask_gui.destroyed.connect(lambda: setattr(self, "mask_gui", None))

    def open_mask_from_roi_window(self):
        self._close_mask_gui()
        self.mask_gui = launch_mask_selection_from_roi(
            self.projections,
            wait_until_closed=False,
        )
        self.mask_gui.masks_created.connect(self.on_masks_created)
        self.mask_gui.destroyed.connect(lambda: setattr(self, "mask_gui", None))

    def _close_mask_gui(self):
        """Close and discard any open mask GUI, handling the case where the
        C++ widget has already been deleted by WA_DeleteOnClose."""
        if self.mask_gui is None:
            return
        try:
            self.mask_gui.close()
        except RuntimeError:
            pass  # C++ object already deleted by Qt
        self.mask_gui = None

    def on_masks_created(self):
        # update viewer so that new masks are shown
        self.update_array_selector()
        self.update_arrays()
        self.array_viewer.refresh_frame()
        print("Updated settings of mask_from_positions:")
        print_options(self.projections.options.mask_from_positions)

    def update_array_selector(self):
        add_masks = self.projections.masks is not None and (
            self.masks_name not in self.array_names
        )
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

    def refresh_array_viewer(self):
        """Reinitialize the array viewer after a shift has been applied or undone."""
        self.array_viewer.reinitialize_all(
            array3d=self.projections.data,
            sort_idx=np.argsort(self.projections.angles),
            extra_title_strings_list=get_projection_title_strings(
                self.projections.scan_numbers, self.projections.angles
            ),
            new_additional_spinbox_indexing=[self.projections.scan_numbers],
        )

    def refresh_applied_shifts_tab(self):
        """Refresh the Applied Shifts tab if it exists."""
        if self.all_shifts_viewer is not None:
            self.all_shifts_viewer.refresh_data()

    def apply_staged_shift(self):
        """Apply the staged shift via the Applied Shifts tab viewer."""
        if self.all_shifts_viewer is not None:
            self.all_shifts_viewer.apply_staged_shift()

    def on_tab_changed(self, index):
        """Handle tab change event to refresh content when tabs are opened."""
        tab_text = self.tabs.tabText(index)
        if tab_text == "Projection Options" and hasattr(self, 'options_display'):
            # Update the options display when the Projection Options tab is opened
            self.options_display.update_display()

    def start(self):
        self.show()


class ScanRemovalTool(QWidget):
    scan_column = 0
    angle_column = 1
    file_path_column = 2

    range_start_column = 0
    range_end_column = 1
    range_type_column = 2

    SELECTION_MODE_INDIVIDUAL = "Select individual scans"
    SELECTION_MODE_SCAN_NUMBER_RANGE = "Select scans by range: scan numbers"
    SELECTION_MODE_ANGLE_RANGE = "Select scans by range: angles"

    _STAGED_BUTTON_STYLE = (
        "QPushButton { background-color: #ADD8E6; border: 1px solid #87CEEB; "
        "border-radius: 3px; padding: 5px 8px; }"
        "QPushButton:disabled { background-color: #D3D3D3; color: #888888; "
        "border: 1px solid #BEBEBE; }"
        "QPushButton:pressed { background-color: #87CEEB; }"
    )

    # Signal emitted when projections are removed
    projections_removed = pyqtSignal()

    def __init__(
        self,
        projections: "p.Projections",
        array_viewer: ArrayViewer,
        projection_drop_function: Callable,  # self.projections.drop_projections
        parent=None,
    ):
        super().__init__(parent=parent)
        self.projection_drop_function = projection_drop_function
        self.setWindowTitle("Scan Removal Tool")
        self.projections = projections
        self.array_viewer = array_viewer
        projection_dropping_widget = self.build_projection_dropper()

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.addWidget(projection_dropping_widget)

    def build_projection_dropper(self) -> QWidget:
        widget_layout = QVBoxLayout()

        # Selection mode dropdown
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Selection mode:", self))
        self.selection_mode_combo = QComboBox(self)
        self.selection_mode_combo.addItems([
            self.SELECTION_MODE_INDIVIDUAL,
            self.SELECTION_MODE_SCAN_NUMBER_RANGE,
            self.SELECTION_MODE_ANGLE_RANGE,
        ])
        self.selection_mode_combo.currentTextChanged.connect(self._on_selection_mode_changed)
        mode_layout.addWidget(self.selection_mode_combo)
        widget_layout.addLayout(mode_layout)

        # Individual mode controls (index selector)
        self.individual_mode_widget = QWidget(self)
        individual_controls_layout = QVBoxLayout(self.individual_mode_widget)
        individual_controls_layout.setContentsMargins(0, 0, 0, 0)
        index_selector_widget = IndexSelectorWidget(
            self.array_viewer.num_frames,
            self.array_viewer.slider.value(),
            include_play_button=False,
            parent=self,
        )
        index_selector_widget.slider.setMinimum(0)
        index_selector_widget.slider.setMaximum(self.array_viewer.slider.maximum())
        index_selector_widget.slider.setValue(self.array_viewer.slider.value())
        index_selector_widget.slider.valueChanged.connect(self.array_viewer.slider.setValue)
        self.array_viewer.slider.valueChanged.connect(index_selector_widget.slider.setValue)
        individual_controls_layout.addWidget(index_selector_widget)
        self.individual_mode_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        widget_layout.addWidget(self.individual_mode_widget)

        # Range mode controls (start/end value inputs)
        self.range_mode_widget = QWidget(self)
        range_controls_layout = QVBoxLayout(self.range_mode_widget)
        range_controls_layout.setContentsMargins(0, 0, 0, 0)
        range_form = QFormLayout()
        self.range_start_input = QLineEdit(self)
        self.range_end_input = QLineEdit(self)
        range_form.addRow("Start:", self.range_start_input)
        range_form.addRow("End:", self.range_end_input)
        range_controls_layout.addLayout(range_form)
        self.range_mode_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        widget_layout.addWidget(self.range_mode_widget)
        self.range_mode_widget.setVisible(False)

        # Add / Remove staged buttons (shared between modes; behavior depends on current mode)
        staged_buttons_layout = QHBoxLayout()
        self.add_to_staged_button = QPushButton("Add to scans staged for removal", self)
        self.add_to_staged_button.setStyleSheet(self._STAGED_BUTTON_STYLE)
        self.add_to_staged_button.pressed.connect(self._add_to_staged)
        self.remove_from_staged_button = QPushButton(
            "Remove from scans staged for removal", self
        )
        self.remove_from_staged_button.setStyleSheet(self._STAGED_BUTTON_STYLE)
        self.remove_from_staged_button.setEnabled(False)
        self.remove_from_staged_button.pressed.connect(self._remove_from_staged)
        staged_buttons_layout.addWidget(self.add_to_staged_button)
        staged_buttons_layout.addWidget(self.remove_from_staged_button)
        widget_layout.addLayout(staged_buttons_layout)

        # Individual scan staging section
        self.individual_staged_section = QWidget(self)
        individual_staged_layout = QVBoxLayout(self.individual_staged_section)
        individual_staged_layout.setContentsMargins(0, 0, 0, 0)
        individual_staged_layout.addWidget(QLabel("Scans staged for removal", self))
        self.staged_for_removal_table = QTableWidget(self)
        self.staged_for_removal_table.setColumnCount(4)
        self.staged_for_removal_table.setHorizontalHeaderLabels(
            ["Index", "Scan Number", "Angle (deg)", "File Path"]
        )
        self.staged_for_removal_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.staged_for_removal_table.currentCellChanged.connect(self.table_item_selected)
        self.staged_for_removal_table.itemSelectionChanged.connect(
            self._update_remove_button_state
        )
        individual_staged_layout.addWidget(self.staged_for_removal_table)
        widget_layout.addWidget(self.individual_staged_section)

        # Range staging section
        self.range_staged_section = QWidget(self)
        range_staged_layout = QVBoxLayout(self.range_staged_section)
        range_staged_layout.setContentsMargins(0, 0, 0, 0)
        range_staged_layout.addWidget(QLabel("Scan ranges staged for removal", self))
        self.scan_ranges_table = QTableWidget(self)
        self.scan_ranges_table.setColumnCount(3)
        self.scan_ranges_table.setHorizontalHeaderLabels(["Start", "End", "Range Type"])
        self.scan_ranges_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.scan_ranges_table.itemSelectionChanged.connect(self._update_remove_button_state)
        range_staged_layout.addWidget(self.scan_ranges_table)
        widget_layout.addWidget(self.range_staged_section)
        self.range_staged_section.setVisible(False)

        drop_projections_button = QPushButton("Permanently Remove Scans", self)
        drop_projections_button.setStyleSheet(
            "QPushButton { background-color: #CC3333; border: 1px solid #991111; "
            "border-radius: 3px; padding: 5px 8px; color: white; font-weight: bold; }"
            "QPushButton:pressed { background-color: #991111; }"
        )
        drop_projections_button.pressed.connect(self.remove_staged_projections)
        widget_layout.addWidget(drop_projections_button)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        widget_layout.addWidget(separator)

        # Previously removed scans table
        widget_layout.addWidget(QLabel("Previously removed scans", self))
        self.removed_scans_table = QTableWidget(self)
        self.removed_scans_table.setColumnCount(3)
        self.removed_scans_table.setHorizontalHeaderLabels(
            ["Scan Number", "Angle (deg)", "File Path"]
        )
        for row_index, scan in enumerate(np.sort(self.projections.dropped_scan_numbers)):
            self.removed_scans_table.insertRow(row_index)
            self.removed_scans_table.setItem(
                row_index, self.scan_column, QTableWidgetItem(str(scan))
            )
            if scan in self.projections.dropped_angles.keys():
                angle = self.projections.dropped_angles[scan]
                self.removed_scans_table.setItem(
                    row_index, self.angle_column, QTableWidgetItem(str(angle))
                )
            if scan in self.projections.dropped_file_paths.keys():
                file_path = self.projections.dropped_file_paths[scan]
                self.removed_scans_table.setItem(
                    row_index, self.file_path_column, QTableWidgetItem(file_path)
                )
        widget_layout.addWidget(self.removed_scans_table)

        widget_group_box = QGroupBox()
        widget_group_box.setStyleSheet("QGroupBox { font-size: 13pt; }")
        widget_group_box.setLayout(widget_layout)
        self.setStyleSheet("QLabel { font-size: 11pt;}")

        return widget_group_box

    def _on_selection_mode_changed(self, mode: str):
        is_individual = mode == self.SELECTION_MODE_INDIVIDUAL
        self.individual_mode_widget.setVisible(is_individual)
        self.range_mode_widget.setVisible(not is_individual)
        self.individual_staged_section.setVisible(is_individual)
        self.range_staged_section.setVisible(not is_individual)
        if not is_individual:
            self.range_start_input.clear()
            self.range_end_input.clear()
        self._update_remove_button_state()

    def _update_remove_button_state(self):
        mode = self.selection_mode_combo.currentText()
        if mode == self.SELECTION_MODE_INDIVIDUAL:
            has_selection = len(self.staged_for_removal_table.selectedItems()) > 0
        else:
            has_selection = len(self.scan_ranges_table.selectedItems()) > 0
        self.remove_from_staged_button.setEnabled(has_selection)

    def _add_to_staged(self):
        mode = self.selection_mode_combo.currentText()
        if mode != self.SELECTION_MODE_INDIVIDUAL:
            self._stage_range_for_removal()
        else:
            self._add_individual_scan_to_staged()

    def _add_individual_scan_to_staged(self):
        index = self.array_viewer.slider.value()
        sorted_index = self.array_viewer.sort_idx[index]
        scan_number = str(self.projections.scan_numbers[sorted_index])
        for row in range(self.staged_for_removal_table.rowCount()):
            if self.staged_for_removal_table.item(row, self.scan_column + 1).text() == scan_number:
                return
        row_index = self.staged_for_removal_table.rowCount()
        self.staged_for_removal_table.insertRow(row_index)
        self.staged_for_removal_table.setItem(
            row_index, 0, QTableWidgetItem(str(index))
        )
        self.staged_for_removal_table.setItem(
            row_index,
            self.scan_column + 1,
            QTableWidgetItem(str(self.projections.scan_numbers[sorted_index])),
        )
        self.staged_for_removal_table.setItem(
            row_index,
            self.angle_column + 1,
            QTableWidgetItem(f"{self.projections.angles[sorted_index]:.3f}"),
        )
        if self.projections.file_paths is not None:
            self.staged_for_removal_table.setItem(
                row_index,
                self.file_path_column + 1,
                QTableWidgetItem(self.projections.file_paths[sorted_index]),
            )

    def _remove_from_staged(self):
        mode = self.selection_mode_combo.currentText()
        table = (
            self.staged_for_removal_table
            if mode == self.SELECTION_MODE_INDIVIDUAL
            else self.scan_ranges_table
        )
        selected_rows = sorted(
            set(item.row() for item in table.selectedItems()), reverse=True
        )
        for row in selected_rows:
            table.removeRow(row)

    def _stage_range_for_removal(self):
        try:
            start = float(self.range_start_input.text())
            end = float(self.range_end_input.text())
        except ValueError:
            return
        mode = self.selection_mode_combo.currentText()
        range_type = "angle" if mode == self.SELECTION_MODE_ANGLE_RANGE else "scan number"
        for row in range(self.scan_ranges_table.rowCount()):
            if (
                float(self.scan_ranges_table.item(row, self.range_start_column).text()) == start
                and float(self.scan_ranges_table.item(row, self.range_end_column).text()) == end
                and self.scan_ranges_table.item(row, self.range_type_column).text() == range_type
            ):
                return
        row_index = self.scan_ranges_table.rowCount()
        self.scan_ranges_table.insertRow(row_index)
        self.scan_ranges_table.setItem(
            row_index, self.range_start_column, QTableWidgetItem(str(start))
        )
        self.scan_ranges_table.setItem(
            row_index, self.range_end_column, QTableWidgetItem(str(end))
        )
        self.scan_ranges_table.setItem(
            row_index, self.range_type_column, QTableWidgetItem(range_type)
        )

    def _get_scan_numbers_from_ranges(self) -> list:
        scan_numbers = []
        for row in range(self.scan_ranges_table.rowCount()):
            start = float(self.scan_ranges_table.item(row, self.range_start_column).text())
            end = float(self.scan_ranges_table.item(row, self.range_end_column).text())
            range_type = self.scan_ranges_table.item(row, self.range_type_column).text()
            if range_type == "scan number":
                matches = [
                    int(sn) for sn, angle in zip(
                        self.projections.scan_numbers, self.projections.angles
                    )
                    if start <= sn <= end
                ]
            else:  # angle
                matches = [
                    int(sn) for sn, angle in zip(
                        self.projections.scan_numbers, self.projections.angles
                    )
                    if start <= angle <= end
                ]
            scan_numbers.extend(matches)
        return list(set(scan_numbers))

    def remove_staged_projections(self):
        remove_scan_numbers = []
        for row in range(self.staged_for_removal_table.rowCount()):
            remove_scan_numbers.append(
                int(self.staged_for_removal_table.item(row, self.scan_column + 1).text())
            )
        remove_scan_numbers.extend(self._get_scan_numbers_from_ranges())
        remove_scan_numbers = list(set(remove_scan_numbers))
        if not remove_scan_numbers:
            return
        drop_projections_wrapped = loading_bar_wrapper("Removing projections...")(
            self.projection_drop_function
        )
        drop_projections_wrapped(remove_scan_numbers)
        self.staged_for_removal_table.blockSignals(True)
        self.staged_for_removal_table.setRowCount(0)
        self.staged_for_removal_table.blockSignals(False)
        self.scan_ranges_table.setRowCount(0)
        for scan in remove_scan_numbers:
            row_index = self.removed_scans_table.rowCount()
            self.removed_scans_table.insertRow(row_index)
            self.removed_scans_table.setItem(
                row_index, self.scan_column, QTableWidgetItem(str(scan))
            )
            if scan in self.projections.dropped_angles.keys():
                angle = self.projections.dropped_angles[scan]
                self.removed_scans_table.setItem(
                    row_index, self.angle_column, QTableWidgetItem(str(angle))
                )
            if scan in self.projections.dropped_file_paths.keys():
                file_path = self.projections.dropped_file_paths[scan]
                self.removed_scans_table.setItem(
                    row_index, self.file_path_column, QTableWidgetItem(file_path)
                )
        sort_idx = np.argsort(self.projections.angles)
        self.array_viewer.reinitialize_all(
            array3d=self.projections.data,
            sort_idx=sort_idx,
            extra_title_strings_list=get_projection_title_strings(
                self.projections.scan_numbers, self.projections.angles
            ),
            new_additional_spinbox_indexing=[self.projections.scan_numbers],
        )
        self.projections_removed.emit()
        print("signal sent")

    def table_item_selected(self, row: int):
        if row < 0:
            return
        item = self.staged_for_removal_table.item(row, 0)
        if item is None:
            return
        index = int(item.text())
        self.array_viewer.update_index_externally(index)

    def closeEvent(self, event):
        # Hide the window instead of closing it
        self.hide()
        event.ignore()


class AllShiftsViewer(MultiThreadedWidget):
    # Signal emitted when a shift operation (apply or undo) is performed
    shift_operation_performed = pyqtSignal()

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

        self.projections = projections
        self.shifts_list = projections.shift_manager.past_shifts
        self.staged_shift = projections.shift_manager.staged_shift
        self.angles = projections.angles
        self.scan_numbers = projections.scan_numbers
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

        # === X-axis radio buttons ===
        x_axis_group_box = QGroupBox("X-axis")
        x_axis_group_box.setStyleSheet("QGroupBox { font-size: 13pt; }")
        x_axis_layout = QVBoxLayout()
        self.x_axis_button_group = QButtonGroup(self)
        self.angle_radio = QRadioButton("Angle")
        self.angle_radio.setChecked(True)
        self.scan_number_radio = QRadioButton("Scan number")
        for btn in (self.angle_radio, self.scan_number_radio):
            btn.setStyleSheet("font-size: 12pt;")
            x_axis_layout.addWidget(btn)
            self.x_axis_button_group.addButton(btn)
        self.angle_radio.toggled.connect(self.update_plot)
        x_axis_group_box.setLayout(x_axis_layout)
        control_layout.addWidget(x_axis_group_box)

        # === Action buttons ===
        action_buttons_layout = QVBoxLayout()

        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_data)
        action_buttons_layout.addWidget(self.refresh_button)

        # Apply staged shift button
        self.apply_staged_shift_button = QPushButton("Apply Staged Shift")
        self.apply_staged_shift_button.clicked.connect(self.apply_staged_shift)
        action_buttons_layout.addWidget(self.apply_staged_shift_button)

        # Undo last shift button
        self.undo_last_shift_button = QPushButton("Undo Last Shift")
        self.undo_last_shift_button.clicked.connect(self.undo_last_shift)
        action_buttons_layout.addWidget(self.undo_last_shift_button)

        # wrap the action buttons in a QGroupBox
        action_buttons_group_box = QGroupBox("Actions")
        action_buttons_group_box.setStyleSheet("QGroupBox { font-size: 13pt; }")
        action_buttons_group_box.setLayout(action_buttons_layout)

        control_layout.addWidget(action_buttons_group_box)

        # === Right panel: pyqtgraph plots ===
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_horizontal = self.plot_widget.addPlot(row=0, col=0, title="Horizontal Shifts")
        self.plot_vertical = self.plot_widget.addPlot(row=1, col=0, title="Vertical Shifts")
        pixel_label = f"Shift ({self.pixel_size * 1e9:.0f} nm px)"
        for plot in (self.plot_horizontal, self.plot_vertical):
            plot.setLabel("left", pixel_label)
            plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot_vertical.setXLink(self.plot_horizontal)

        main_layout.addLayout(control_layout, 1)
        main_layout.addWidget(self.plot_widget, 4)

    def _get_x_axis_data(self):
        """Return (x_values, sort_idx, x_label) based on the selected radio button."""
        if self.angle_radio.isChecked():
            sort_idx = np.argsort(self.angles)
            return self.angles[sort_idx], sort_idx, "Angle (deg)"
        else:
            sort_idx = np.argsort(self.scan_numbers)
            return self.scan_numbers[sort_idx], sort_idx, "Scan number"

    def update_plot(self):
        x_values, sort_idx, x_label = self._get_x_axis_data()
        for plot in (self.plot_horizontal, self.plot_vertical):
            plot.clear()
            legend = plot.addLegend(offset=(10, 10))
            legend.setBrush(pg.mkBrush(30, 30, 30, 160))
            plot.setLabel("bottom", x_label)
        for i, cb in enumerate(self.checkboxes):
            if cb.isChecked():
                array = self.shifts_list[i]
                pen = pg.mkPen(color=color_list[i], width=2)
                self.plot_horizontal.plot(
                    x_values, array[sort_idx, 0], pen=pen, name=cb.text()
                )
                self.plot_vertical.plot(
                    x_values, array[sort_idx, 1], pen=pen, name=cb.text()
                )

    def refresh_data(self):
        """Refresh the shift data from the projections object and update the UI."""
        # Get updated data from projections
        self.shifts_list = self.projections.shift_manager.past_shifts
        self.staged_shift = self.projections.shift_manager.staged_shift
        self.angles = self.projections.angles
        self.scan_numbers = self.projections.scan_numbers

        # Clear existing checkboxes
        for cb in self.checkboxes:
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self.checkboxes.clear()

        # Rebuild checkboxes with updated shift data
        if len(self.shifts_list) > 0:
            # Add checkbox for total shift
            self.shifts_list = [np.sum(self.shifts_list, 0)] + self.shifts_list
            cb = QCheckBox("Total of applied shifts")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_plot)
            self.checkboxes.append(cb)
            self.checkbox_layout.insertWidget(
                self.checkbox_layout.count() - 1, cb
            )  # Insert before spacer
            # Add checkboxes for the rest of the shifts
            for i in range(1, len(self.shifts_list)):
                cb = QCheckBox(f"Applied shift {i}")
                cb.setChecked(True)
                cb.stateChanged.connect(self.update_plot)
                self.checkboxes.append(cb)
                self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, cb)

        if np.any(self.staged_shift != 0):
            # Add checkbox for the staged shift
            self.shifts_list = self.shifts_list + [self.staged_shift]
            cb = QCheckBox("Staged shift")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_plot)
            self.checkboxes.append(cb)
            self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, cb)

        # Format checkboxes
        for cb in self.checkboxes:
            cb.setStyleSheet("font-size: 12pt;")

        # Update the plot with new data
        self.update_plot()

    def apply_staged_shift(self):
        """Apply the staged shift and refresh the display."""
        apply_staged_shift_wrapped = loading_bar_wrapper(load_message="Applying shift...")(
            func=self.projections.apply_staged_shift
        )
        apply_staged_shift_wrapped()
        self.refresh_data()
        # Emit signal to notify that a shift operation was performed
        self.shift_operation_performed.emit()

    def undo_last_shift(self):
        """Undo the last applied shift and refresh the display."""
        undo_last_shift_wrapped = loading_bar_wrapper(load_message="Undoing shift...")(
            func=self.projections.undo_last_shift
        )
        undo_last_shift_wrapped()
        self.refresh_data()
        # Emit signal to notify that a shift operation was performed
        self.shift_operation_performed.emit()

    def start(self):
        self.show()


def get_projection_title_strings(
    scan_numbers: np.ndarray, angles: np.ndarray
) -> list[str]:
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
    include_array_saving_widget: bool = True,
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
    gui = ProjectionViewer(
        projections,
        display_only=display_only,
        include_array_saving_widget=include_array_saving_widget,
    )
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
