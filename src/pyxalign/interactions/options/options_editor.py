import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Optional, Union, TypeVar, get_origin, get_args
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
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        main_layout = QVBoxLayout(self)
        # main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

        self.setLayout(main_layout)

    def on_toggle(self):
        if self.toggle_button.isChecked():
            # Expand
            self.toggle_button.setArrowType(Qt.DownArrow)
            self.content_area.setMaximumHeight(16777215)  # effectively "no limit"
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


class BasicOptionsEditor(QWidget):
    """
    A widget that displays a GUI for editing a (nested) dataclass instance.
    Updates to widgets immediately update the fields in the dataclass.
    """

    def __init__(
        self, data: OptionsClass, skip_fields: list[str], parent=None
    ):
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
        self.form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        scroll_widget.setLayout(self.form_layout)

        # Put that container inside the scroll area
        scroll_area.setWidget(scroll_widget)

        main_layout.addWidget(scroll_area)

        # Populate the form
        self._add_dataclass_fields(data, self.form_layout)

        # Optionally, set a default window size for the editor
        self.resize(600, 400)

    def _add_dataclass_fields(self, data_obj: OptionsClass, form_layout: QFormLayout, parent_name: str = ""):
        """Add editors for each field of the given dataclass object into form_layout."""
        if not is_dataclass(data_obj):
            return  # Not a dataclass, do nothing

        for f in fields(data_obj):
            field_name = f.name
            field_type = f.type
            field_value = getattr(data_obj, field_name)

            full_field_name = parent_name + field_name
            print(full_field_name)

            if self._check_if_skipped_field(full_field_name):
                continue

            # If this is a nested dataclass, create a collapsible panel
            if is_dataclass(field_value):
                panel = CollapsiblePanel(title=field_name)
                # Use a QFormLayout for nested fields as well
                nested_layout = QFormLayout()
                panel.setContentLayout(nested_layout)

                # Recursively add items to nested_layout
                self._add_dataclass_fields(field_value, nested_layout, parent_name + field_name + ".")
                # Insert the collapsible panel as a row in the form
                form_layout.addRow(field_name, panel)
                # By default, the panel is collapsed
                continue

            # Otherwise, make a row for this field
            editor = self._create_editor_widget(data_obj, field_name, field_type, field_value)
            label = QLabel(field_name)
            form_layout.addRow(label, editor)

    def _check_if_skipped_field(self, current_full_field_name: str) -> bool:
        return current_full_field_name in self.skip_fields

    def _create_editor_widget(
        self,
        data_obj,
        field_name: str,
        field_type: type,
        field_value: Union[tuple, int, float, Enum, OptionsClass],
    ) -> QWidget:
        """
        Create an appropriate widget for the field type, connect signals to keep
        data_obj[field_name] updated in real time.
        """
        # Handle booleans
        if self._is_bool_type(field_type):
            cb = QCheckBox()
            cb.setChecked(bool(field_value))
            cb.toggled.connect(lambda checked, o=data_obj, fn=field_name: setattr(o, fn, checked))
            # return cb
            input_widget = cb

        # Handle ints
        elif self._is_int_type(field_type):
            # ensure the value is in the proper type
            try:
                field_value = int(field_value)
            except Exception as e:
                # fallback or log
                print(f"[Warning] Field '{field_name}' (int) has value {field_value} => forcing 0.")
                field_value = 0

            spin = NoScrollSpinBox()
            spin.setRange(-999999, 999999)
            spin.setValue(field_value)
            spin.valueChanged.connect(lambda val, o=data_obj, fn=field_name: setattr(o, fn, val))
            input_widget = spin

        # Handle floats
        elif self._is_float_type(field_type):
            try:
                field_value = float(field_value)
            except Exception as e:
                # fallback or log
                print(
                    f"[Warning] Field '{field_name}' (float) has value {field_value} => forcing 0."
                )
                field_value = 0.0

            dspin = MinimalDecimalSpinBox()
            dspin.setValue(field_value)
            dspin.setRange(-999999.0, 999999.0)
            dspin.setDecimals(20)
            dspin.valueChanged.connect(lambda val, o=data_obj, fn=field_name: setattr(o, fn, val))
            input_widget = dspin

        # Handle Enums
        elif self._is_enum_type(field_type):
            combo = QComboBox()
            enum_cls = self._extract_enum_class(field_type)
            current_index = 0
            for i, e_val in enumerate(enum_cls):
                combo.addItem(e_val.value, e_val)
                if field_value == e_val:
                    current_index = i
            combo.setCurrentIndex(current_index)

            def on_combo_changed(idx, o=data_obj, fn=field_name, c=combo):
                chosen_enum_member = c.itemData(idx)
                setattr(o, fn, chosen_enum_member)

            combo.currentIndexChanged.connect(on_combo_changed)
            input_widget = combo

        # Handle tuple of int
        elif self._is_tuple_of_int(field_type):
            container = QWidget()
            checkbox_layout = QHBoxLayout()
            container.setLayout(checkbox_layout)
            container.setContentsMargins(0, 0, 0, 0)

            if isinstance(field_value, tuple):
                active_indices = set(field_value)
            else:
                active_indices = set()

            checkboxes = []

            def update_tuple():
                new_indices = [i for (i, ch) in enumerate(checkboxes) if ch.isChecked()]
                setattr(data_obj, field_name, tuple(new_indices))

            n_boxes, box_labels, corresponding_values = self.get_n_boxes_and_labels(
                data_obj, field_name
            )

            if n_boxes is not None:
                for i in range(n_boxes):
                    cb = QCheckBox(box_labels[i])
                    # cb.setChecked(i in active_indices)
                    cb.setChecked(corresponding_values[i] in active_indices)
                    cb.toggled.connect(update_tuple)
                    checkboxes.append(cb)
                    checkbox_layout.addWidget(cb)
                # hbox.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
            else:
                # Fallback: let user type a string => parse as tuple
                line = QLineEdit()
                line.setText(str(field_value) if field_value is not None else "")
                line.textChanged.connect(
                    lambda txt, o=data_obj, fn=field_name: setattr(o, fn, string_to_tuple(txt))
                )
                checkbox_layout.addWidget(line)

            # return container
            input_widget = container

        # Otherwise, handle everything else as a string
        else:
            print(f"WARNING: {field_name} being handled as string")
            line = QLineEdit()
            line.setText(str(field_value) if field_value is not None else "")
            line.textChanged.connect(lambda txt, o=data_obj, fn=field_name: setattr(o, fn, txt))
            input_widget = line

        formatted_widget = QWidget()
        hbox = QHBoxLayout()
        formatted_widget.setLayout(hbox)
        hbox.addWidget(input_widget)
        # input_widget.setLayout(hbox)
        hbox.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # return input_widget
        return formatted_widget

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
            # e.g., Union[int, NoneType] => Optional[int]
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
    ) -> tuple[int, list[str], list[Union[int, float]]]:
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
            corresponding_values = [i + 1 for i in range(n_boxes)]
        else:
            n_boxes = None
            box_labels = None
            corresponding_values = None
        return n_boxes, box_labels, corresponding_values


def string_to_tuple(input_str: str) -> tuple[int]:
    """
    Convert a string representation of a tuple (e.g., "(1,2,3)")
    into an actual Python tuple of integers (1, 2, 3).
    """
    stripped_str = input_str.strip().strip("()")
    elements = stripped_str.split(",")
    return tuple(int(x) for x in elements if x.strip() != "")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Use your own dataclass or any nested structure from opts
    config_instance = opts.ProjectionMatchingOptions()

    # editor = BasicOptionsEditor(config_instance, skip_fields=[(opts.DeviceOptions, "gpu")])
    editor = BasicOptionsEditor(config_instance, skip_fields=["plot", "interactive_viewer.update.enabled"])
    editor.setWindowTitle("Nested Dataclass Editor with Scroll and Hidden Borders")

    # Use the left half of the screen
    screen_geometry = app.desktop().availableGeometry(editor)
    editor.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() / 2),
        int(screen_geometry.height() * 0.9),
    )

    editor.show()

    sys.exit(app.exec_())
