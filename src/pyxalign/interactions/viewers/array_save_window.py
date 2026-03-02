"""Array save dialog for ArrayViewer."""

from typing import Optional
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QWidget,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
import numpy as np


class ArraySaveWindow(QDialog):
    """Dialog for saving arrays from ArrayViewer."""

    def __init__(self, array_viewer, parent: Optional[QWidget] = None):
        """
        Initialize the array save dialog.

        Args:
            array_viewer: The ArrayViewer instance to save data from
            parent: Parent widget
        """
        super().__init__(parent)
        self.array_viewer = array_viewer
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Save Array")

        # Create main layout for the dialog
        dialog_layout = QVBoxLayout()
        self.setLayout(dialog_layout)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create a widget to hold the content
        content_widget = QWidget()
        main_layout = QVBoxLayout()
        content_widget.setLayout(main_layout)

        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)
        dialog_layout.addWidget(scroll_area)

        # Section 1: Save mode selection
        mode_group = QGroupBox("Save Mode")
        mode_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        mode_layout = QVBoxLayout()

        self.radio_3d_tiff = QRadioButton("Save 3D array as TIFF")
        self.radio_3d_h5 = QRadioButton("Save 3D array as H5")
        self.radio_current_frame = QRadioButton("Save current frame as image")
        self.radio_3d_tiff.setChecked(True)

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_3d_tiff)
        self.button_group.addButton(self.radio_3d_h5)
        self.button_group.addButton(self.radio_current_frame)
        self.button_group.buttonClicked.connect(self.on_save_mode_changed)

        mode_layout.addWidget(self.radio_3d_tiff)
        mode_layout.addWidget(self.radio_3d_h5)
        mode_layout.addWidget(self.radio_current_frame)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # Section 1.5: Sort order selection (only shown if sort_idx is present)
        self.sort_checkbox = QCheckBox("Save sorted array")
        self.sort_checkbox.setChecked(True)  # Default to sorted

        # Only show if array_viewer has a sort_idx
        if self.array_viewer.sort_idx is not None:
            main_layout.addWidget(self.sort_checkbox)
        else:
            self.sort_checkbox.hide()

        # Section 1.6: Crop into single file option (only shown for 3D TIFF saves)
        self.crop_checkbox = QCheckBox("crop into single file?")
        self.crop_checkbox.setChecked(False)  # Default to no cropping
        main_layout.addWidget(self.crop_checkbox)

        # Section 2: Format selection (for single frame)
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["TIFF", "PNG", "JPG"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        self.format_widget = QWidget()
        self.format_widget.setLayout(format_layout)
        self.format_widget.hide()  # Hidden by default
        main_layout.addWidget(self.format_widget)

        # Section 3: File path selection
        path_layout = QHBoxLayout()
        self.path_line_edit = QLineEdit()
        self.path_line_edit.setPlaceholderText("Select save path...")
        path_layout.addWidget(self.path_line_edit)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file_path)
        path_layout.addWidget(self.browse_button)
        main_layout.addLayout(path_layout)

        # Add stretch to push content to the top
        main_layout.addStretch()

        # Section 4: Action buttons (outside scroll area)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.save_button = QPushButton("Save Array")
        self.save_button.clicked.connect(self.perform_save)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        dialog_layout.addLayout(button_layout)

    def on_save_mode_changed(self):
        """Show/hide format dropdown and sort checkbox based on selected mode."""
        if self.radio_current_frame.isChecked():
            self.format_widget.show()
            # Hide sort checkbox for single frame saves
            self.sort_checkbox.hide()
            # Hide crop checkbox for single frame saves
            self.crop_checkbox.hide()
        elif self.radio_3d_h5.isChecked():
            self.format_widget.hide()
            # Show sort checkbox only if sort_idx exists
            if self.array_viewer.sort_idx is not None:
                self.sort_checkbox.show()
            else:
                self.sort_checkbox.hide()
            # Hide crop checkbox for H5 saves (not applicable)
            self.crop_checkbox.hide()
        else:  # 3D TIFF
            self.format_widget.hide()
            # Show sort checkbox only if sort_idx exists
            if self.array_viewer.sort_idx is not None:
                self.sort_checkbox.show()
            else:
                self.sort_checkbox.hide()
            # Show crop checkbox for 3D TIFF saves
            self.crop_checkbox.show()

    def browse_file_path(self):
        """Open file dialog to select save path."""
        # Determine file filter and default name
        if self.radio_3d_tiff.isChecked():
            filter_str = "TIFF Files (*.tif *.tiff)"
            default_name = "array_3d.tif"
        elif self.radio_3d_h5.isChecked():
            filter_str = "HDF5 Files (*.h5 *.hdf5)"
            default_name = "array_3d.h5"
        else:
            format_type = self.format_combo.currentText()
            frame_idx = self.array_viewer.slider.value()
            if format_type == "PNG":
                filter_str = "PNG Files (*.png)"
                default_name = f"frame_{frame_idx}.png"
            elif format_type == "JPG":
                filter_str = "JPEG Files (*.jpg *.jpeg)"
                default_name = f"frame_{frame_idx}.jpg"
            else:  # TIFF
                filter_str = "TIFF Files (*.tif *.tiff)"
                default_name = f"frame_{frame_idx}.tif"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Array", default_name, filter_str
        )

        if file_path:
            self.path_line_edit.setText(file_path)

    def perform_save(self):
        """Execute the save operation."""
        file_path = self.path_line_edit.text()

        if not file_path:
            QMessageBox.warning(self, "No Path", "Please select a save path.")
            return

        try:
            from pyxalign.io.save import (
                save_array_as_tiff,
                save_2d_array_as_png,
                save_2d_array_as_jpg,
                save_2d_array_as_tiff,
            )
            import h5py

            if self.radio_3d_tiff.isChecked():
                # Save full 3D array with proper orientation
                array = self.array_viewer.array3d
                # Convert from GPU if needed
                if hasattr(array, "get"):  # CuPy array
                    array = array.get()

                # Reorganize array to match displayed orientation
                slider_axis = self.array_viewer.options.slider_axis

                # Move slider axis to position 0 (stack direction)
                if slider_axis != 0:
                    array = np.moveaxis(array, slider_axis, 0)

                # Apply sorting if checkbox is checked and sort_idx exists
                if self.sort_checkbox.isChecked() and self.array_viewer.sort_idx is not None:
                    # Reorder the array using sort_idx along axis 0 (after moveaxis)
                    array = array[self.array_viewer.sort_idx]

                # Transpose each 2D slice to match displayed orientation
                # This transposes the last two axes (height, width) for all slices
                array = np.transpose(array, (0, 2, 1))

                # Rotate 90 degrees counterclockwise
                array = np.rot90(array, k=1, axes=(1, 2))

                # Get crop option
                crop_to_single = self.crop_checkbox.isChecked()

                save_array_as_tiff(array, file_path, crop_to_single_file=crop_to_single)
            elif self.radio_3d_h5.isChecked():
                # Save full 3D array as H5
                array = self.array_viewer.array3d
                # Convert from GPU if needed
                if hasattr(array, "get"):  # CuPy array
                    array = array.get()

                # Reorganize array to match displayed orientation
                slider_axis = self.array_viewer.options.slider_axis

                # Move slider axis to position 0 (stack direction)
                if slider_axis != 0:
                    array = np.moveaxis(array, slider_axis, 0)

                # Apply sorting if checkbox is checked and sort_idx exists
                if self.sort_checkbox.isChecked() and self.array_viewer.sort_idx is not None:
                    # Reorder the array using sort_idx along axis 0 (after moveaxis)
                    array = array[self.array_viewer.sort_idx]

                # Transpose each 2D slice to match displayed orientation
                # This transposes the last two axes (height, width) for all slices
                array = np.transpose(array, (0, 2, 1))

                # Rotate 90 degrees counterclockwise
                array = np.rot90(array, k=1, axes=(1, 2))

                # Save as H5
                with h5py.File(file_path, "w") as F:
                    F.create_dataset(name="data", data=array)
                print(f"File saved to: {file_path}")
            else:
                # Save current frame
                current_frame = self.array_viewer.get_current_frame_data()

                # Rotate 90 degrees counterclockwise
                current_frame = np.rot90(current_frame, k=1)

                format_type = self.format_combo.currentText()

                if format_type == "PNG":
                    save_2d_array_as_png(current_frame, file_path)
                elif format_type == "JPG":
                    save_2d_array_as_jpg(current_frame, file_path)
                else:  # TIFF
                    save_2d_array_as_tiff(current_frame, file_path)

            QMessageBox.information(self, "Success", f"Array saved to {file_path}")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving array: {str(e)}")
