import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Optional, Union, TypeVar, get_origin, get_args, Any
import cupy as cp
import numpy as np
import pyxalign.api.options as opts

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QToolButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QComboBox,
    QPushButton,
    QDialogButtonBox,
    QSizePolicy,
    QHBoxLayout,
    QLayout,
    QScrollArea,
    QSpacerItem,
    QFrame,
)
from PyQt5.QtCore import Qt

OptionsClass = TypeVar("OptionsClass")


class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()  # Prevent changing value on scroll


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()  # Prevent changing value on scroll


class MinimalDecimalSpinBox(NoScrollDoubleSpinBox):
    def textFromValue(self, value):
        # Format to suppress trailing zeros, but respect min/max decimals
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        if text == "-0":  # Optional: fix "-0" to "0"
            text = "0"
        return text


class CollapsiblePanel(QWidget):
    """
    A simple collapsible panel that starts collapsed by default.
    Clicking the toggle button shows or hides the panel's contents.
    """

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        # Remove visible borders to "blend in"
        self.toggle_button.setStyleSheet(
            """
            QToolButton {
                border: none;
                background: transparent;
                padding: 0px;
            }
            """
        )

        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QWidget()
        self.content_area.setMaximumHeight(0)  # fully collapse

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

        self.setLayout(main_layout)

    def on_toggle(self):
        if self.toggle_button.isChecked():
            # Expand
            self.toggle_button.setArrowType(Qt.DownArrow)
            self.content_area.setMaximumHeight(16777215)  # effectively no limit
        else:
            # Collapse
            self.toggle_button.setArrowType(Qt.RightArrow)
            self.content_area.setMaximumHeight(0)

    def setContentLayout(self, content_layout: QLayout):
        """
        Set the layout containing the actual child widgets that we will
        expand or collapse.
        """
        self.content_area.setLayout(content_layout)


def string_to_tuple(input_str: str) -> tuple[int]:
    """
    Convert a string representation of a tuple (e.g., "(1,2,3)")
    into an actual Python tuple of integers (1, 2, 3).
    """
    stripped_str = input_str.strip().strip("()")
    elements = stripped_str.split(",")
    return tuple(int(x) for x in elements if x.strip() != "")


class SingleOptionEditor(QWidget):
    """
    A widget for editing a single field in the parent dataclass.
    It determines the field type, creates an appropriate editor,
    and updates the parent dataclass field value whenever changes occur.
    """

    def __init__(
        self,
        data_obj,
        field_name: str,
        skip_fields: Optional[list[str]] = None,
        parent=None,
    ):
        """
        :param data_obj: Parent dataclass instance (the 'options' object).
        :param field_name: Name of the field to edit.
        :param skip_fields: List of fully qualified field names to skip.
        :param parent: Optional parent QWidget.
        """
        super().__init__(parent)
        self.data_obj = data_obj
        self.field_name = field_name
        self.skip_fields = skip_fields

        # Find the field's declared type (from the dataclass fields)
        self.field_type = None
        for f in fields(type(data_obj)):
            if f.name == field_name:
                self.field_type = f.type
                break

        # If we couldn't find a declared field type, fall back to value-based
        if not self.field_type:
            self.field_type = type(getattr(data_obj, field_name, None))

        # Main layout for this widget
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Build and add the core input widget
        input_widget = self._create_editor_widget()
        layout.addWidget(input_widget)

        # Add a spacer to push the input widget to the left
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def value(self) -> Any:
        return getattr(self.data_obj, self.field_name)

    def _create_editor_widget(self) -> QWidget:
        """
        Creates and returns the actual input widget for this single field
        based on the type of the field.
        """
        field_value = getattr(self.data_obj, self.field_name, None)

        # Handle booleans
        if self._is_bool_type(self.field_type):
            cb = QCheckBox()
            cb.setChecked(bool(field_value))
            cb.toggled.connect(lambda checked: setattr(self.data_obj, self.field_name, checked))
            return cb

        # Handle ints
        elif self._is_int_type(self.field_type):
            spin = NoScrollSpinBox()
            spin.setRange(-999999, 999999)
            try:
                spin.setValue(int(field_value))
            except (ValueError, TypeError):
                spin.setValue(0)
            spin.valueChanged.connect(lambda val: setattr(self.data_obj, self.field_name, val))
            return spin

        # Handle floats
        elif self._is_float_type(self.field_type):
            dspin = MinimalDecimalSpinBox()
            dspin.setRange(-999999.0, 999999.0)
            dspin.setDecimals(20)
            try:
                dspin.setValue(float(field_value))
            except (ValueError, TypeError):
                dspin.setValue(0.0)
            dspin.valueChanged.connect(lambda val: setattr(self.data_obj, self.field_name, val))
            return dspin

        # Handle Enums
        elif self._is_enum_type(self.field_type):
            combo = QComboBox()
            enum_cls = self._extract_enum_class(self.field_type)
            current_index = 0
            for i, e_val in enumerate(enum_cls):
                combo.addItem(e_val.value, e_val)
                if field_value == e_val:
                    current_index = i
            combo.setCurrentIndex(current_index)
            combo.currentIndexChanged.connect(
                lambda idx: setattr(self.data_obj, self.field_name, combo.itemData(idx))
            )
            return combo

        # Handle tuple of int
        elif self._is_tuple_of_int(self.field_type):
            return self._create_tuple_of_int_widget(field_value)

        # Otherwise, treat as string
        else:
            line = QLineEdit()
            if field_value is not None:
                line.setText(str(field_value))
            line.textChanged.connect(lambda txt: setattr(self.data_obj, self.field_name, txt))
            return line

    def _create_tuple_of_int_widget(self, field_value):
        """
        Creates a widget of checkboxes for a tuple of ints (e.g., GPU device indices).
        If the number of boxes can't be determined, it reverts to a freeform string.
        """
        container = QWidget()
        checkbox_layout = QHBoxLayout()
        container.setLayout(checkbox_layout)
        container.setContentsMargins(0, 0, 0, 0)

        active_indices = set(field_value)

        checkboxes = []

        def update_tuple():
            new_indices = [i for (i, ch) in enumerate(checkboxes) if ch.isChecked()]
            setattr(self.data_obj, self.field_name, tuple(new_indices))

        n_boxes, box_labels, corresponding_values = self.get_n_boxes_and_labels(
            self.data_obj, self.field_name
        )

        if n_boxes is not None:
            for i in range(n_boxes):
                cb = QCheckBox(box_labels[i])
                cb.setChecked(corresponding_values[i] in active_indices)
                cb.toggled.connect(update_tuple)
                checkboxes.append(cb)
                checkbox_layout.addWidget(cb)
        else:
            # Fallback: let user type a string => parse as tuple
            line = QLineEdit()
            line.setText(str(field_value) if field_value is not None else "")
            line.textChanged.connect(
                lambda txt: setattr(self.data_obj, self.field_name, string_to_tuple(txt))
            )
            checkbox_layout.addWidget(line)

        return container

    ########################################################################
    # Helper methods for type checking
    ########################################################################

    def _is_bool_type(self, t):
        if t == bool:
            return True
        if get_origin(t) is Union:
            args = get_args(t)
            if len(args) == 2 and bool in args and type(None) in args:
                return True
        return False

    def _is_int_type(self, t):
        if t == int:
            return True
        if get_origin(t) is Union:
            args = get_args(t)
            if len(args) == 2 and int in args and type(None) in args:
                return True
        return False

    def _is_float_type(self, t):
        if t == float:
            return True
        if get_origin(t) is Union:
            args = get_args(t)
            if len(args) == 2 and float in args and type(None) in args:
                return True
        return False

    def _is_enum_type(self, t):
        if self._safe_issubclass(t, Enum):
            return True
        if get_origin(t) is Union:
            for arg in get_args(t):
                if self._safe_issubclass(arg, Enum):
                    return True
        return False

    def _extract_enum_class(self, t):
        if self._safe_issubclass(t, Enum):
            return t
        if get_origin(t) is Union:
            for arg in get_args(t):
                if self._safe_issubclass(arg, Enum):
                    return arg
        return t

    def _is_tuple_of_int(self, t):
        origin = get_origin(t)
        if origin is tuple:
            args = get_args(t)
            if len(args) > 0 and args[0] == int:
                return True
        if origin is Union:
            for arg in get_args(t):
                if get_origin(arg) is tuple:
                    a = get_args(arg)
                    if len(a) > 0 and a[0] == int:
                        return True
        return False

    def _safe_issubclass(self, cls, base):
        try:
            return issubclass(cls, base)
        except TypeError:
            return False

    def get_n_boxes_and_labels(
        self, data_obj: OptionsClass, field_name: str
    ) -> tuple[Optional[int], Optional[list[str]], Optional[list[int]]]:
        """
        Determines if we can display a fixed number of checkboxes
        for the current field. Returns (n_boxes, box_labels, corresponding_values).
        """
        gpu_list_field_names = [
            "gpu_indices",
            "back_project_gpu_indices",
            "forward_project_gpu_indices",
        ]
        if field_name in gpu_list_field_names:
            n_boxes = cp.cuda.runtime.getDeviceCount()
            box_labels = [str(i) for i in range(n_boxes)]
            corresponding_values = [i for i in range(n_boxes)]
        elif (
            hasattr(opts, "ProjectionMatchingOptions")
            and isinstance(data_obj, opts.ProjectionMatchingOptions)
            and field_name == "filter_directions"
        ):
            n_boxes = 2
            box_labels = ["x", "y"]
            # For "filter_directions", the values are 1 for x and 2 for y
            corresponding_values = [i + 1 for i in range(n_boxes)]
        else:
            n_boxes = None
            box_labels = None
            corresponding_values = None

        return n_boxes, box_labels, corresponding_values


class BasicOptionsEditor(QWidget):
    """
    A widget that displays a GUI for editing a (nested) dataclass instance.
    Updates to widgets immediately update the fields in the dataclass.
    """

    def __init__(self, data: OptionsClass, skip_fields: list[str]=[], parent=None):
        super().__init__(parent)
        self._data = data  # The root dataclass instance
        self.skip_fields = skip_fields

        # Top-level layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create a QScrollArea so that form expansions do not resize the window
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Container widget that will hold the form
        scroll_widget = QWidget()
        self.form_layout = QFormLayout()
        self.form_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        scroll_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scroll_widget.setLayout(self.form_layout)

        # Put that container inside the scroll area
        scroll_area.setWidget(scroll_widget)
        title = QLabel("Options Editor")
        title.setStyleSheet("QLabel {font-size: 16px;}")

        main_layout.addWidget(title)
        main_layout.addWidget(scroll_area)

        # Populate the form
        self._add_dataclass_fields(data, self.form_layout)

        # Optionally, set a default window size for the editor
        # self.resize(600, 400)

    def _add_dataclass_fields(
        self,
        data_obj: OptionsClass,
        form_layout: QFormLayout,
        parent_name: str = "",
        level: int = 0,
    ):
        """Add editors for each field of the given dataclass object into form_layout."""
        if not is_dataclass(data_obj):
            return  # Not a dataclass, do nothing

        for f in fields(data_obj):
            field_name = f.name
            field_value = getattr(data_obj, field_name)

            full_field_name = parent_name + field_name

            # Skip fields if instructed
            if self._check_if_skipped_field(full_field_name):
                continue

            # If this is a nested dataclass, create a collapsible panel
            if is_dataclass(field_value):
                panel = CollapsiblePanel(title=field_name)
                nested_layout = QFormLayout()
                panel.setContentLayout(nested_layout)

                # Recursively add items to nested_layout
                self._add_dataclass_fields(
                    field_value,
                    nested_layout,
                    parent_name=parent_name + field_name + ".",
                    level=level + 1,
                )

                if level == 0:
                    form_layout.addRow(field_name, self.wrap_in_frame(panel))
                else:
                    form_layout.addRow(field_name, panel)
                continue

            # Otherwise, make a row for this field by adding a SingleOptionEditor
            editor = SingleOptionEditor(data_obj, field_name, self.skip_fields, self)
            if level == 0:
                form_layout.addRow(field_name, self.wrap_in_frame(editor))
            else:
                form_layout.addRow(field_name, editor)

    def wrap_in_frame(self, widget: QWidget) -> QFrame:
        # Create a frame to wrap the panel
        frame = QFrame()
        # frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        frame.setFrameShape(QFrame.Panel)
        frame.setLineWidth(1)  # optional: adjust thickness
        frame.setMidLineWidth(0)  # optional
        frame.setStyleSheet("QFrame { background-color:lightGray; border:lightGray}")

        # Add the panel to the frame's layout
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(4, 4, 4, 4)  # optional: remove padding
        frame_layout.addWidget(widget)
        return frame

    def _check_if_skipped_field(self, current_full_field_name: str) -> bool:
        return current_full_field_name in self.skip_fields


def return_parent_option(options: OptionsClass, field_path: str) -> OptionsClass:
    field_names = field_path.split(".")
    current_item = options
    for i, name in enumerate(field_names):
        if i == len(field_names) - 1:
            return current_item
        current_item = getattr(current_item, name)


def get_option_from_field_path(options: OptionsClass, field_path: str) -> Any:
    parent_options = return_parent_option(options, field_path)
    field_name = field_path.split(".")[-1]
    return getattr(parent_options, field_name)


def set_option_from_field_path(options: OptionsClass, field_path: str, value: Any) -> OptionsClass:
    parent_options = return_parent_option(options, field_path)
    field_name = field_path.split(".")[-1]
    setattr(parent_options, field_name, value)
    # return parent_options
    return options


if __name__ == "__main__":
    # p = opts.ProjectionMatchingOptions()
    # set_option_from_field_path(p, "device.gpu.gpu_indices", (1, 2, 4))
    # print(p)
    # print("\n")
    # print(p.device)

    app = QApplication(sys.argv)

    # Use your own dataclass or any nested structure from opts
    config_instance = opts.ProjectionMatchingOptions()

    editor = BasicOptionsEditor(
        config_instance, skip_fields=["plot", "interactive_viewer.update.enabled"]
    )
    editor.setWindowTitle("Nested Dataclass Editor with Scroll and Hidden Borders")

    # Use the left half of the screen
    screen_geometry = app.desktop().availableGeometry(editor)
    # editor.setGeometry(
    #     screen_geometry.x(),
    #     screen_geometry.y(),
    #     int(screen_geometry.width() / 2),
    #     int(screen_geometry.height() * 0.9),
    # )

    editor.show()
    sys.exit(app.exec_())
