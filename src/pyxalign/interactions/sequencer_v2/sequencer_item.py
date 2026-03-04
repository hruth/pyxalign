"""
SequencerItemV2: A single alignment run block containing multiple parameter modifications.

This version groups all parameter changes for a single alignment run into one cohesive block,
making it clearer that all parameters within the block are applied together for one alignment execution.
"""

import sys
import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Optional, TypeVar
import copy

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QFrame,
    QSizePolicy,
    QStyledItemDelegate,
    QPushButton,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from pyxalign.api.types import OptionsClass
from pyxalign.interactions.options.options_editor import SingleOptionEditor

T = TypeVar("T")


class BoldHeaderDelegate(QStyledItemDelegate):
    """Custom delegate to make section headers bold in combo boxes."""

    def paint(self, painter, option, index):
        text = index.data()
        option_copy = option
        font = QFont(option.font)

        if text and text.startswith("---") and text.endswith("---"):
            font.setBold(True)
        else:
            font.setBold(False)
        option_copy.font = font
        super().paint(painter, option_copy, index)


class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()  # Prevent changing value on scroll


class ParameterRow(QWidget):
    """A single parameter selection row within a sequencer block."""

    remove_requested = pyqtSignal(object)  # Emits self

    def __init__(
        self,
        options: OptionsClass,
        initial_state: Optional[tuple[str, T]] = None,
        basic_options_list: Optional[list[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.options = copy.deepcopy(options)
        self.options_editor = None
        self.basic_options_list = basic_options_list or []

        # Main layout for this row
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.main_layout)

        # Combo boxes for nested selection
        self.combo_boxes: list[tuple[NoScrollComboBox, OptionsClass]] = []

        # Create the initial combo box for the top-level dataclass
        self.add_combo_box(self.options)

        # Add remove button
        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedSize(20, 20)
        self.remove_button.setStyleSheet("QPushButton { font-size: 10px; padding: 0px; background-color: #ffcccc; }")
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        self.main_layout.addWidget(self.remove_button)

        # Initialize with state if provided
        if initial_state is not None:
            initial_field = initial_state[0]
            initial_value = initial_state[1]

            field_list = initial_field.split(".")
            for i, field in enumerate(field_list):
                if i == len(field_list) - 1 and initial_value is not None:
                    parent_options = self.combo_boxes[i][1]
                    setattr(parent_options, field, initial_value)
                self.combo_boxes[i][0].setCurrentText(field)

    def full_field_path(self) -> str:
        """Return the full dotted path to the selected option."""
        path_parts = []
        for combo_box, obj in self.combo_boxes:
            idx = combo_box.currentIndex()
            if idx == 0:
                break
            attr_name = combo_box.currentText()
            if attr_name.startswith("--"):
                break
            path_parts.append(attr_name)
        return ".".join(path_parts)

    def value(self) -> T:
        """Return the current value of the selected option."""
        if self.options_editor is not None:
            return self.options_editor.value()
        return None

    def on_combo_box_changed(self):
        """Handle combo box selection changes."""
        combo_box = self.sender()
        combo_index = next((i for i, (cb, _) in enumerate(self.combo_boxes) if cb == combo_box), -1)

        # Remove any combo boxes that come after this one
        self.remove_combo_boxes_after(combo_index)

        # If placeholder selected, do nothing
        if combo_box.currentIndex() == 0:
            return

        attr_name = combo_box.currentText()

        # Skip section headers
        if attr_name.startswith("---") and attr_name.endswith("---"):
            combo_box.setCurrentIndex(0)
            return

        parent_obj = self.combo_boxes[combo_index][1]
        selected_obj = getattr(parent_obj, attr_name)

        # If it's a dataclass instance, create a new combo box
        if dataclasses.is_dataclass(selected_obj):
            self.add_combo_box(selected_obj)
        else:
            self.add_options_selector(parent_obj, attr_name)

    def add_options_selector(self, parent_obj, attr_name: str):
        """Add the value editor widget for the selected parameter."""
        self.options_editor = SingleOptionEditor(copy.deepcopy(parent_obj), attr_name, parent=self)
        # Insert before the remove button
        self.main_layout.insertWidget(len(self.combo_boxes), self.options_editor)

    def add_combo_box(self, obj):
        """Add a new combo box for selecting from the dataclass fields."""
        combo_box = NoScrollComboBox()
        combo_box.setEditable(False)
        combo_box.addItem("--Select an attribute--")

        # Get current path for categorization
        current_path = self._build_current_path_prefix()
        basic_fields, advanced_fields = self._get_categorized_fields(obj, current_path)

        # Add basic fields section
        if basic_fields:
            combo_box.addItem("---BASIC---")
            for field_name in sorted(basic_fields):
                combo_box.addItem(field_name)

        # Add advanced fields section
        if advanced_fields:
            combo_box.addItem("---ADVANCED---")
            for field_name in sorted(advanced_fields):
                combo_box.addItem(field_name)

        combo_box.setItemDelegate(BoldHeaderDelegate())

        self.combo_boxes.append((combo_box, obj))
        combo_box.currentIndexChanged.connect(self.on_combo_box_changed)

        combo_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        # Insert before the remove button (which is always last)
        insert_position = len(self.combo_boxes) - 1
        self.main_layout.insertWidget(insert_position, combo_box)

    def remove_combo_boxes_after(self, combo_index):
        """Remove combo boxes after the given index."""
        while len(self.combo_boxes) > combo_index + 1:
            cb, _ = self.combo_boxes.pop()
            self.main_layout.removeWidget(cb)
            cb.deleteLater()

        if self.options_editor is not None:
            self.main_layout.removeWidget(self.options_editor)
            self.options_editor.deleteLater()
            self.options_editor.setParent(None)
            self.options_editor = None

    def _get_categorized_fields(
        self, dataclass_obj, path_prefix: str = ""
    ) -> tuple[list[str], list[str]]:
        """Categorize fields into basic and advanced."""
        basic_fields = []
        advanced_fields = []

        for field in fields(dataclass_obj):
            field_name = field.name
            full_path = f"{path_prefix}.{field_name}" if path_prefix else field_name

            if full_path in self.basic_options_list:
                basic_fields.append(field_name)
            else:
                advanced_fields.append(field_name)

        return basic_fields, advanced_fields

    def _build_current_path_prefix(self) -> str:
        """Build the current path prefix based on selected combo boxes."""
        path_parts = []
        for combo_box, obj in self.combo_boxes:
            idx = combo_box.currentIndex()
            if idx == 0:
                break
            attr_name = combo_box.currentText()
            if attr_name.startswith("--"):
                break
            path_parts.append(attr_name)
        return ".".join(path_parts)


class SequencerItemV2(QWidget):
    """
    A single alignment run block that can contain multiple parameter modifications.

    This represents one call to get_projection_matching_shift with all the
    parameter changes grouped together in a single visual block.
    """

    # Signals for parent widget to handle operations
    insert_above_requested = pyqtSignal(object)  # Emits self
    insert_below_requested = pyqtSignal(object)  # Emits self
    duplicate_requested = pyqtSignal(object)     # Emits self
    remove_requested = pyqtSignal(object)        # Emits self

    def __init__(
        self,
        options: OptionsClass,
        initial_parameters: Optional[list[tuple[str, T]]] = None,
        basic_options_list: Optional[list[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.options = copy.deepcopy(options)
        self.basic_options_list = basic_options_list or []
        self.parameter_rows: list[ParameterRow] = []

        # Main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.main_layout)

        # Frame to visually group this alignment block
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Raised)
        self.frame.setLineWidth(2)
        self.frame.setStyleSheet("""
            QFrame {
                background-color: #e8f4f8;
                border: 2px solid #4a90a4;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.main_layout.addWidget(self.frame)

        # Layout inside the frame
        self.frame_layout = QVBoxLayout()
        self.frame.setLayout(self.frame_layout)

        # Add block header with controls
        self.setup_block_header()

        # Container for parameter rows
        self.parameters_container = QWidget()
        self.parameters_layout = QVBoxLayout()
        self.parameters_layout.setContentsMargins(0, 0, 0, 0)
        self.parameters_layout.setSpacing(5)
        self.parameters_container.setLayout(self.parameters_layout)
        self.frame_layout.addWidget(self.parameters_container)

        # Add parameter buttons at the bottom of the block
        self.setup_parameter_buttons()

        # Initialize with provided parameters or add one empty row
        if initial_parameters:
            for field_path, value in initial_parameters:
                self.add_parameter_row(initial_state=(field_path, value))
        else:
            self.add_parameter_row()

    def setup_block_header(self):
        """Setup the header with block control buttons."""
        header_layout = QHBoxLayout()

        # Block label
        block_label = QLabel("Alignment Run Block")
        block_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12px; }")
        header_layout.addWidget(block_label)

        header_layout.addStretch()

        # Block control buttons
        self.insert_above_button = QPushButton("↑ Insert Above")
        self.insert_below_button = QPushButton("↓ Insert Below")
        self.duplicate_button = QPushButton("⧉ Duplicate")
        self.remove_button = QPushButton("✕ Remove")

        button_style = "QPushButton { font-size: 10px; padding: 2px 4px; }"
        self.insert_above_button.setStyleSheet(button_style)
        self.insert_below_button.setStyleSheet(button_style)
        self.duplicate_button.setStyleSheet(button_style)
        self.remove_button.setStyleSheet(button_style + "QPushButton { background-color: #ffcccc; }")

        self.insert_above_button.clicked.connect(lambda: self.insert_above_requested.emit(self))
        self.insert_below_button.clicked.connect(lambda: self.insert_below_requested.emit(self))
        self.duplicate_button.clicked.connect(lambda: self.duplicate_requested.emit(self))
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))

        header_layout.addWidget(self.insert_above_button)
        header_layout.addWidget(self.insert_below_button)
        header_layout.addWidget(self.duplicate_button)
        header_layout.addWidget(self.remove_button)

        self.frame_layout.addLayout(header_layout)

    def setup_parameter_buttons(self):
        """Setup buttons for adding/removing parameters."""
        button_layout = QHBoxLayout()

        self.add_param_button = QPushButton("+ Add Parameter")
        self.add_param_button.setStyleSheet("QPushButton { background-color: #d0f0d0; }")
        self.add_param_button.clicked.connect(lambda: self.add_parameter_row())

        button_layout.addWidget(self.add_param_button)
        button_layout.addStretch()

        self.frame_layout.addLayout(button_layout)

    def add_parameter_row(self, initial_state: Optional[tuple[str, T]] = None):
        """Add a new parameter row to this block."""
        param_row = ParameterRow(
            self.options,
            initial_state=initial_state,
            basic_options_list=self.basic_options_list,
            parent=self,
        )
        param_row.remove_requested.connect(self.remove_parameter_row)
        self.parameter_rows.append(param_row)
        # Insert before the button layout (which is always last in frame_layout)
        self.parameters_layout.addWidget(param_row)

    def remove_parameter_row(self, row: ParameterRow):
        """Remove a parameter row from this block."""
        # Keep at least one parameter row
        if len(self.parameter_rows) <= 1:
            return

        self.parameter_rows.remove(row)
        self.parameters_layout.removeWidget(row)
        row.deleteLater()

    def get_parameter_changes(self) -> dict[str, Any]:
        """
        Get all parameter changes in this block as a dictionary.

        Returns
        -------
        dict
            Dictionary mapping field paths to values for all parameters in this block.
        """
        changes = {}
        for row in self.parameter_rows:
            field_path = row.full_field_path()
            value = row.value()
            if field_path and value is not None:
                changes[field_path] = value
        return changes

    def get_parameter_list(self) -> list[tuple[str, Any]]:
        """
        Get all parameters as a list of (field_path, value) tuples.

        Returns
        -------
        list
            List of (field_path, value) tuples for all parameters in this block.
        """
        params = []
        for row in self.parameter_rows:
            field_path = row.full_field_path()
            value = row.value()
            if field_path and value is not None:
                params.append((field_path, value))
        return params


def main():
    app = QApplication(sys.argv)
    import pyxalign.api.options as opts

    selector = SequencerItemV2(
        opts.ProjectionMatchingOptions(),
        basic_options_list=["high_pass_filter", "downsample", "downsample.scale"],
    )
    selector.setWindowTitle("Sequencer Block V2")
    selector.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
