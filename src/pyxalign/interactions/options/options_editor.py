from functools import wraps
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from tkinter import filedialog
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
    QFileDialog,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QTimer
from pyxalign.interactions.custom import NoScrollSpinBox, MinimalDecimalSpinBox

from pyxalign.api.options_utils import get_all_attribute_names
from pyxalign.io.utils import OptionsClass
from pyxalign.plotting.interactive.utils import OptionsDisplayWidget


class IntTupleInputWidget(QWidget):
    def __init__(self, field_value, field_name: str, data_obj: OptionsClass):
        super().__init__()

        checkbox_layout = QHBoxLayout()
        self.setLayout(checkbox_layout)
        self.setContentsMargins(0, 0, 0, 0)

        active_indices = set(field_value or ())

        checkboxes = []

        def update_tuple():
            new_indices = [i for (i, ch) in enumerate(checkboxes) if ch.isChecked()]
            setattr(data_obj, field_name, tuple(new_indices))

        n_boxes, box_labels, corresponding_values = self.get_n_boxes_and_labels(field_name)

        if n_boxes is not None:
            for i in range(n_boxes):
                cb = QCheckBox(box_labels[i])
                cb.setChecked(corresponding_values[i] in active_indices)
                cb.toggled.connect(update_tuple)
                checkboxes.append(cb)
                checkbox_layout.addWidget(cb)
        else:
            # Fallback: freeform string => parse as tuple
            self.line = QLineEdit()
            self.line.setText(str(field_value) if field_value is not None else "")

            @update_options_error_handler
            def on_text_changed(txt: str):
                setattr(data_obj, field_name, string_to_tuple(txt))

            self.line.textChanged.connect(on_text_changed)
            checkbox_layout.addWidget(self.line)

    ########################################################################
    # GPU and direction checkboxes
    ########################################################################
    def get_n_boxes_and_labels(
        self, field_name: str
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
            field_name == "filter_directions"
            # hasattr(opts, "ProjectionMatchingOptions")
            # and isinstance(data_obj, opts.ProjectionMatchingOptions)
            # and field_name == "filter_directions"
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


class CustomFileDialog(QWidget):
    def __init__(self, use_folder_dialog: bool = False):
        super().__init__()
        self.use_folder_dialog = use_folder_dialog
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.input_bar = QLineEdit(self)
        self.input_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.input_bar.setMinimumWidth(300)
        self.input_bar.setPlaceholderText("Type or paste file path here...")
        layout.addWidget(self.input_bar)

        if self.use_folder_dialog:
            self.open_dialog_button = QPushButton("Open Folder Dialog", self)
        else:
            self.open_dialog_button = QPushButton("Open File Dialog", self)
        self.open_dialog_button.clicked.connect(self.open_file_dialog)
        self.open_dialog_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout.addWidget(self.open_dialog_button)

        self.setLayout(layout)
        self.setWindowTitle("File Dialog Example")

    def open_file_dialog(self):
        if self.use_folder_dialog:
            path = QFileDialog.getExistingDirectory(self, "Select a Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select a File")

        if path:
            self.input_bar.setText(path)


class CollapsiblePanel(QWidget):
    """
    A simple collapsible panel that starts collapsed by default.
    Clicking the toggle button shows or hides the panel's contents.
    """

    def __init__(self, title: str = "", keep_open: bool = False, parent: Optional[QWidget] = None):
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
        if keep_open:
            self.toggle_button.setChecked(True)
            self.on_toggle()

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
    If the field is Optional[...] type, an extra checkbox is shown on the right:
       - if checked, the widget is enabled, and updates the field
       - if unchecked, the field is set to None and the widget is disabled
    """

    def __init__(
        self,
        data_obj,
        field_name: str,
        skip_fields: Optional[list[str]] = None,
        use_file_dialog: bool = False,
        use_folder_dialog: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.data_obj = data_obj
        self.field_name = field_name
        self.skip_fields = skip_fields
        self.use_file_dialog = use_file_dialog
        self.use_folder_dialog = use_folder_dialog
        if use_file_dialog and use_folder_dialog:
            raise Exception("use_file_dialog and use_folder_dialog cannot both be true!")

        # Find the field's declared type
        self.field_type = None
        for f in fields(type(data_obj)):
            if f.name == field_name:
                self.field_type = f.type
                break
        # Fall back if not found
        if not self.field_type:
            self.field_type = type(getattr(data_obj, field_name, None))

        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Create the editor (or "wrapped" editor if optional)
        editor_widget = self._create_editor_widget()
        layout.addWidget(editor_widget)

        # Add a spacer to push the input widget to the left
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def value(self) -> Any:
        return getattr(self.data_obj, self.field_name)

    def _create_editor_widget(self) -> QWidget:
        """
        Decides whether the field is optional or not. If optional, we wrap
        the 'real' editor in a container that includes a checkbox. If not,
        we return the 'real' editor directly.
        """
        field_value = getattr(self.data_obj, self.field_name, None)

        # Check if it's optional (Union[..., None])
        if self._is_optional_type(self.field_type):
            # extract the underlying type (the real type without None)
            underlying_type = self._extract_non_none_type(self.field_type)
            real_editor = self._create_nonoptional_editor_widget(underlying_type, field_value)
            return self._wrap_optional(real_editor, field_value)
        else:
            # Non-optional field, just create the editor
            return self._create_nonoptional_editor_widget(self.field_type, field_value)

    def _wrap_optional(self, real_editor: QWidget, field_value: Any) -> QWidget:
        """
        Wraps the given editor widget in another widget that has a
        checkbox on the right. If the checkbox is unchecked => None,
        else => the editor's value is used.
        """
        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(container_layout)

        container_layout.addWidget(real_editor)

        optional_cb = QCheckBox()
        container_layout.addWidget(optional_cb)

        # Initialize checkbox state: checked if not None
        is_non_none = field_value is not None
        optional_cb.setChecked(is_non_none)
        real_editor.setEnabled(is_non_none)

        # If toggled OFF => set to None, disable editor
        # If toggled ON  => set to current editor value, enable editor
        def on_cb_toggled(checked: bool):
            real_editor.setEnabled(checked)
            if not checked:
                setattr(self.data_obj, self.field_name, None)
            else:
                # Force an initial update from the real_editor's current value
                self._force_editor_value_update(real_editor)

        optional_cb.toggled.connect(on_cb_toggled)

        return container

    def _force_editor_value_update(self, editor_widget: QWidget):
        """
        Reads the editor_widget's current value and assigns it to data_obj[field_name].
        This is invoked when the optional checkbox is turned on.
        """
        # This part mirrors the data-updating logic from each widget's connect(...).
        # We'll do a quick check to handle the known widget types:
        if isinstance(editor_widget, QCheckBox):
            setattr(self.data_obj, self.field_name, editor_widget.isChecked())
        elif isinstance(editor_widget, QComboBox):
            idx = editor_widget.currentIndex()
            setattr(self.data_obj, self.field_name, editor_widget.itemData(idx))
        elif isinstance(editor_widget, QLineEdit):
            setattr(self.data_obj, self.field_name, editor_widget.text())
        elif isinstance(editor_widget, QSpinBox) or isinstance(editor_widget, QDoubleSpinBox):
            setattr(self.data_obj, self.field_name, editor_widget.value())
        elif isinstance(editor_widget, QWidget) and hasattr(editor_widget, "input_bar"):
            # Possibly it's the CustomFileDialog
            setattr(self.data_obj, self.field_name, editor_widget.input_bar.text())
        elif isinstance(editor_widget, IntTupleInputWidget):
            setattr(self.data_obj, self.field_name, string_to_tuple(editor_widget.line.text()))
        else:
            # Fallback: do nothing
            pass

    def _create_nonoptional_editor_widget(self, t, field_value: Any) -> QWidget:
        """
        Creates an editor for a non-optional field of type t and sets up signals
        that immediately store changes to self.data_obj[self.field_name].
        """
        if self.use_file_dialog or self.use_folder_dialog:
            # File path input
            file_dialog_widget = CustomFileDialog(self.use_folder_dialog)
            if field_value:
                file_dialog_widget.input_bar.setText(str(field_value))

            def on_text_changed(txt: str):
                setattr(self.data_obj, self.field_name, txt)

            file_dialog_widget.input_bar.textChanged.connect(on_text_changed)
            return file_dialog_widget

        # Handle bool
        if self._is_bool_type(t):
            cb = QCheckBox()
            cb.setChecked(bool(field_value))

            def on_toggled(checked):
                setattr(self.data_obj, self.field_name, checked)

            cb.toggled.connect(on_toggled)
            return cb

        # Handle int
        elif self._is_int_type(t):
            spin = NoScrollSpinBox()
            spin.setRange(-999999, 999999)
            try:
                spin.setValue(int(field_value))
            except (ValueError, TypeError):
                spin.setValue(0)

            def on_value_changed(val):
                setattr(self.data_obj, self.field_name, val)

            spin.valueChanged.connect(on_value_changed)
            return spin

        # Handle float
        elif self._is_float_type(t):
            dspin = MinimalDecimalSpinBox()
            dspin.setRange(-999999.0, 999999.0)
            dspin.setDecimals(20)
            try:
                dspin.setValue(float(field_value))
            except (ValueError, TypeError):
                dspin.setValue(0.0)

            def on_value_changed(val):
                setattr(self.data_obj, self.field_name, val)

            dspin.valueChanged.connect(on_value_changed)
            return dspin

        # Handle Enums
        elif self._is_enum_type(t):
            combo = QComboBox()
            enum_cls = self._extract_enum_class(t)
            current_index = 0
            for i, e_val in enumerate(enum_cls):
                combo.addItem(e_val.value, e_val)
                if field_value == e_val:
                    current_index = i
            combo.setCurrentIndex(current_index)

            def on_index_changed(idx):
                setattr(self.data_obj, self.field_name, combo.itemData(idx))

            combo.currentIndexChanged.connect(on_index_changed)
            return combo

        # Handle tuple of int
        elif self._is_tuple_of_int(t):
            return self._create_tuple_of_int_widget(field_value)

        # Handle list of str
        elif self._is_list_of_str(t):
            list_widget = QLineEdit()
            if field_value is not None:
                list_widget.setText(", ".join(field_value))

            def on_text_changed(txt: str):
                setattr(self.data_obj, self.field_name, [s.strip() for s in txt.split(",")])

            list_widget.textChanged.connect(on_text_changed)
            return list_widget

        # Otherwise, treat as string
        else:
            # Unrecognized => string fallback
            line = QLineEdit()
            if field_value is not None:
                line.setText(str(field_value))

            @update_options_error_handler
            def on_text_changed(txt: str):
                setattr(self.data_obj, self.field_name, txt)

            line.textChanged.connect(on_text_changed)
            return line

    ########################################################################
    # Helper editors
    ########################################################################
    def _create_tuple_of_int_widget(self, field_value):
        """
        Creates a widget of checkboxes for a tuple of ints (e.g., GPU device indices).
        If the number of boxes can't be determined, it reverts to a freeform string.
        """
        return IntTupleInputWidget(field_value, self.field_name, self.data_obj)

    ########################################################################
    # Type-checking Helpers
    ########################################################################
    def _is_bool_type(self, t):
        # covers both bool and Optional[bool] in the union check
        return t == bool or (get_origin(t) is Union and bool in get_args(t))

    def _is_int_type(self, t):
        if t == int:
            return True
        if get_origin(t) is Union:
            args = get_args(t)
            if int in args:
                return True
        return False

    def _is_float_type(self, t):
        if t == float:
            return True
        if get_origin(t) is Union:
            args = get_args(t)
            if float in args:
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
        """If t is Union[SomeEnum, None], return SomeEnum; else return t."""
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

    def _is_list_of_str(self, t):
        origin = get_origin(t)
        if origin is list:
            args = get_args(t)
            if len(args) > 0 and args[0] == str:
                return True
        if origin is Union:
            for arg in get_args(t):
                if get_origin(arg) is list:
                    a = get_args(arg)
                    if len(a) > 0 and a[0] == str:
                        return True
        return False

    def _safe_issubclass(self, cls, base):
        try:
            return issubclass(cls, base)
        except TypeError:
            return False

    def _is_optional_type(self, t) -> bool:
        """Return True if t is Union[..., None]."""
        origin = get_origin(t)
        if origin is Union:
            args = get_args(t)
            if type(None) in args:
                return True
        return False

    def _extract_non_none_type(self, t):
        """
        Given a type t that is Union[S, None], return S.
        If we can't determine a unique non-None, return t as-is.
        """
        if get_origin(t) is Union:
            args = get_args(t)
            # Filter out NoneType
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return non_none_args[0]
        return t


class BasicOptionsEditor(QWidget):
    """
    A widget that displays a GUI for editing a (nested) dataclass instance.
    Updates to widgets immediately update the fields in the dataclass.
    """

    def __init__(
        self,
        data: OptionsClass,
        skip_fields: list[str] = [],
        file_dialog_fields: list[str] = [],
        folder_dialog_fields: list[str] = [],
        open_panels_list: list[str] = [],
        advanced_options_list: Optional[list[str]] = None,
        basic_options_list: Optional[list[str]] = None,
        enable_advanced_tab: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._data = data
        self.skip_fields = skip_fields
        self.advanced_options_list = advanced_options_list or []
        self.basic_options_list = basic_options_list or []
        self.enable_advanced_tab = enable_advanced_tab
        self.options_display = None

        # Classify fields for tab distribution if advanced tab is enabled
        if self.enable_advanced_tab and (self.advanced_options_list or self.basic_options_list):
            all_fields = get_all_attribute_names(self._data)
            
            # Determine basic and advanced fields
            if self.basic_options_list:
                # If basic_options_list is provided, use it directly
                basic_fields = self.basic_options_list
                # Store the advanced fields for backward compatibility
                self.advanced_options_list = list(set(all_fields) - set(basic_fields))
            else:
                # Otherwise, basic fields are everything not in advanced_options_list
                basic_fields = list(set(all_fields) - set(self.advanced_options_list))
            
            self.field_classification = self._classify_fields_for_tabs(all_fields, basic_fields)
        else:
            self.field_classification = None

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        title = QLabel("Options Editor")
        title.setStyleSheet("QLabel {font-size: 16px;}")
        main_layout.addWidget(title)

        if self.enable_advanced_tab and self.advanced_options_list:
            # Create tabbed interface
            self.tab_widget = QTabWidget()
            main_layout.addWidget(self.tab_widget)
            
            # Create basic options tab
            self._create_basic_options_tab(
                file_dialog_fields=file_dialog_fields,
                folder_dialog_fields=folder_dialog_fields,
                open_panels_list=open_panels_list,
            )
            
            # Create advanced options tab
            self._create_advanced_options_tab(
                file_dialog_fields=file_dialog_fields,
                folder_dialog_fields=folder_dialog_fields,
                open_panels_list=open_panels_list,
            )
        else:
            # Create single interface (original behavior)
            self._create_single_options_interface(
                main_layout,
                file_dialog_fields=file_dialog_fields,
                folder_dialog_fields=folder_dialog_fields,
                open_panels_list=open_panels_list,
            )

        self.open_display_button = QPushButton("view selections")
        self.open_display_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.open_display_button.clicked.connect(self.open_options_display_window)
        main_layout.addWidget(self.open_display_button)

        self.initialize_viewer()

    def _classify_fields_for_tabs(self, all_fields: list[str], basic_fields: list[str]) -> dict:
        """
        Classify all fields into basic_only, advanced_only, and split_parent categories.
        
        This method analyzes the relationship between parent and child fields to determine
        how to distribute them across basic and advanced tabs.
        
        Args:
            all_fields: Complete list of all field paths in the dataclass
            basic_fields: List of field paths that should appear in basic tab
            
        Returns:
            Dictionary with classification:
            {
                'basic_only': [...],           # Fields that appear only in basic
                'advanced_only': [...],        # Fields that appear only in advanced  
                'split_parents': {             # Parents that appear in both tabs
                    'parent_name': {
                        'basic_children': [...],
                        'advanced_children': [...]
                    }
                }
            }
        """
        basic_set = set(basic_fields)
        advanced_set = set(all_fields) - basic_set
        
        # Find parent-child relationships
        parent_children_map = {}
        for field in all_fields:
            if '.' in field:
                parent = self._get_parent_field_name(field)
                child = field.split('.')[-1]
                if parent not in parent_children_map:
                    parent_children_map[parent] = []
                parent_children_map[parent].append(child)
        
        # Classify fields
        classification = {
            'basic_only': [],
            'advanced_only': [],
            'split_parents': {}
        }
        
        # Check for split parents - cases where parent is in basic but children are distributed
        for parent, children in parent_children_map.items():
            parent_in_basic = parent in basic_set
            child_paths = [f"{parent}.{child}" for child in children]
            
            basic_children = [child for child in children if f"{parent}.{child}" in basic_set]
            advanced_children = [child for child in children if f"{parent}.{child}" in advanced_set]
            
            # A parent is "split" if:
            # 1. The parent itself is in basic_fields AND
            # 2. Some (but not all) of its children are explicitly listed in basic_fields
            if parent_in_basic and basic_children:
                # This is a split parent case - parent appears in both tabs with different children
                classification['split_parents'][parent] = {
                    'basic_children': basic_children,
                    'advanced_children': advanced_children
                }
                # Remove the parent and its children from basic/advanced only lists
                basic_set.discard(parent)
                for child_path in child_paths:
                    basic_set.discard(child_path)
                    advanced_set.discard(child_path)
        
        # Remaining fields go to basic_only or advanced_only
        classification['basic_only'] = list(basic_set)
        classification['advanced_only'] = list(advanced_set)
        
        return classification

    def _get_parent_field_name(self, field_path: str) -> str:
        """
        Extract parent field name from a dotted path.
        
        Args:
            field_path: Dotted field path like "downsample.scale"
            
        Returns:
            Parent field name like "downsample"
        """
        return field_path.split('.')[0]

    def _should_include_field_in_tab(self, field_path: str, tab_type: str, parent_name: str = "") -> bool:
        """
        Determine if a specific field should be included in the given tab.
        
        Args:
            field_path: Full dotted path of the field
            tab_type: Either "basic" or "advanced"
            parent_name: Name of the parent field (for nested fields)
            
        Returns:
            True if field should be included in the specified tab
        """
        if not self.field_classification:
            return True
            
        # Build the full field path considering parent context
        full_field_path = f"{parent_name}.{field_path}" if parent_name else field_path
        
        # Check basic_only and advanced_only lists
        if tab_type == "basic":
            if full_field_path in self.field_classification['basic_only']:
                return True
        else:  # advanced
            if full_field_path in self.field_classification['advanced_only']:
                return True
        
        # Check split_parents
        if parent_name in self.field_classification['split_parents']:
            split_info = self.field_classification['split_parents'][parent_name]
            if tab_type == "basic":
                return field_path in split_info['basic_children']
            else:  # advanced
                return field_path in split_info['advanced_children']
        
        # If parent is in split_parents but this field isn't explicitly listed, exclude it
        if any(parent_name == sp for sp in self.field_classification['split_parents'].keys()):
            return False
            
        return False

    def _create_basic_options_tab(
        self,
        file_dialog_fields: Optional[list[str]] = None,
        folder_dialog_fields: Optional[list[str]] = None,
        open_panels_list: list[str] = [],
    ):
        """Create the basic options tab with intelligent field filtering."""
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        basic_tab.setLayout(basic_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        form_layout = QFormLayout()
        form_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        scroll_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scroll_widget.setLayout(form_layout)

        scroll_area.setWidget(scroll_widget)
        basic_layout.addWidget(scroll_area)

        # Add fields using the new classification system
        self._add_dataclass_fields(
            self._data,
            form_layout,
            file_dialog_fields=file_dialog_fields,
            folder_dialog_fields=folder_dialog_fields,
            open_panels_list=open_panels_list,
            tab_type="basic",
        )

        self.tab_widget.addTab(basic_tab, "Basic Options")

    def _create_advanced_options_tab(
        self,
        file_dialog_fields: Optional[list[str]] = None,
        folder_dialog_fields: Optional[list[str]] = None,
        open_panels_list: list[str] = [],
    ):
        """Create the advanced options tab with intelligent field filtering."""
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_tab.setLayout(advanced_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        form_layout = QFormLayout()
        form_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        scroll_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scroll_widget.setLayout(form_layout)

        scroll_area.setWidget(scroll_widget)
        advanced_layout.addWidget(scroll_area)

        # Add fields using the new classification system
        self._add_dataclass_fields(
            self._data,
            form_layout,
            file_dialog_fields=file_dialog_fields,
            folder_dialog_fields=folder_dialog_fields,
            open_panels_list=open_panels_list,
            tab_type="advanced",
        )

        self.tab_widget.addTab(advanced_tab, "Advanced Options")

    def _create_single_options_interface(
        self,
        main_layout: QVBoxLayout,
        file_dialog_fields: Optional[list[str]] = None,
        folder_dialog_fields: Optional[list[str]] = None,
        open_panels_list: list[str] = [],
    ):
        """Create the original single-interface layout."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        self.form_layout = QFormLayout()
        self.form_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        scroll_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scroll_widget.setLayout(self.form_layout)

        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        self._add_dataclass_fields(
            self._data,
            self.form_layout,
            file_dialog_fields=file_dialog_fields,
            folder_dialog_fields=folder_dialog_fields,
            open_panels_list=open_panels_list,
        )

    def _add_dataclass_fields(
        self,
        data_obj: OptionsClass,
        form_layout: QFormLayout,
        parent_name: str = "",
        file_dialog_fields: Optional[list[str]] = None,
        folder_dialog_fields: Optional[list[str]] = None,
        open_panels_list: list[str] = [],
        level: int = 0,
        skip_fields_override: Optional[list[str]] = None,
        tab_type: Optional[str] = None,
    ):
        if not is_dataclass(data_obj):
            return

        for f in fields(data_obj):
            field_name = f.name
            field_value = getattr(data_obj, field_name)

            full_field_name = parent_name + field_name

            # Handle tab-aware filtering if we're in advanced tab mode
            if tab_type and self.field_classification:
                # For top-level fields, check if this field should be in this tab
                if level == 0:
                    # Check if this is a split parent
                    if field_name in self.field_classification['split_parents']:
                        # This parent should appear in both tabs, continue processing
                        pass
                    elif tab_type == "basic":
                        if full_field_name not in self.field_classification['basic_only']:
                            continue
                    else:  # advanced
                        if full_field_name not in self.field_classification['advanced_only']:
                            continue
                else:
                    # For nested fields, check against the parent context
                    parent_field = parent_name.rstrip('.')
                    if not self._should_include_field_in_tab(field_name, tab_type, parent_field):
                        continue
            else:
                # Use legacy skip fields logic for backward compatibility
                skip_fields_to_use = skip_fields_override if skip_fields_override is not None else self.skip_fields
                if self._check_if_skipped_field(full_field_name, skip_fields_to_use):
                    continue

            # If nested dataclass => collapsible panel
            if is_dataclass(field_value):
                if field_name in open_panels_list:
                    keep_open = True
                else:
                    keep_open = False
                panel = CollapsiblePanel(title=field_name, keep_open=keep_open)
                nested_layout = QFormLayout()
                panel.setContentLayout(nested_layout)

                # For split parents, we need to check if any children should be visible in this tab
                if tab_type and self.field_classification and field_name in self.field_classification['split_parents']:
                    split_info = self.field_classification['split_parents'][field_name]
                    children_for_this_tab = split_info.get(f'{tab_type}_children', [])
                    
                    # Only add the panel if there are children to show in this tab
                    if children_for_this_tab:
                        # Create a custom skip list that includes all children NOT meant for this tab
                        all_children = split_info.get('basic_children', []) + split_info.get('advanced_children', [])
                        children_to_skip = [child for child in all_children if child not in children_for_this_tab]
                        skip_fields_for_nested = [f"{parent_name}{field_name}.{child}" for child in children_to_skip]
                        
                        self._add_dataclass_fields(
                            field_value,
                            nested_layout,
                            parent_name=parent_name + field_name + ".",
                            level=level + 1,
                            file_dialog_fields=file_dialog_fields,
                            folder_dialog_fields=folder_dialog_fields,
                            skip_fields_override=skip_fields_for_nested,
                            tab_type=None,  # Disable tab filtering for nested calls when using skip_fields_override
                        )

                        if level == 0:
                            form_layout.addRow(field_name, self.wrap_in_frame(panel))
                        else:
                            form_layout.addRow(field_name, panel)
                else:
                    # Normal nested dataclass processing
                    self._add_dataclass_fields(
                        field_value,
                        nested_layout,
                        parent_name=parent_name + field_name + ".",
                        level=level + 1,
                        file_dialog_fields=file_dialog_fields,
                        folder_dialog_fields=folder_dialog_fields,
                        skip_fields_override=skip_fields_override,
                        tab_type=tab_type,
                    )

                    if level == 0:
                        form_layout.addRow(field_name, self.wrap_in_frame(panel))
                    else:
                        form_layout.addRow(field_name, panel)

            else:
                # Check if file-dialog or folder-dialog
                use_file_dialog = file_dialog_fields is not None and (
                    full_field_name in file_dialog_fields
                )
                use_folder_dialog = folder_dialog_fields is not None and (
                    full_field_name in folder_dialog_fields
                )
                editor = SingleOptionEditor(
                    data_obj,
                    field_name,
                    self.skip_fields,
                    use_file_dialog=use_file_dialog,
                    use_folder_dialog=use_folder_dialog,
                    parent=self,
                )
                if level == 0:
                    form_layout.addRow(field_name, self.wrap_in_frame(editor))
                else:
                    form_layout.addRow(field_name, editor)

    def wrap_in_frame(self, widget: QWidget) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Panel)
        frame.setLineWidth(1)
        frame.setStyleSheet("QFrame { background-color:lightGray; border:lightGray}")

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(4, 4, 4, 4)
        frame_layout.addWidget(widget)
        return frame

    def _check_if_skipped_field(self, current_full_field_name: str, skip_fields: Optional[list[str]] = None) -> bool:
        fields_to_check = skip_fields if skip_fields is not None else self.skip_fields
        return current_full_field_name in fields_to_check

    def initialize_viewer(self):
        self.options_display = OptionsDisplayWidget(self._data)
        self.update_display_timer = QTimer(self)
        self.update_display_timer.start(100)  # .5 seconds
        self.update_display_timer.timeout.connect(self.options_display.update_display)

    def open_options_display_window(self):
        self.options_display.resize(550, 700)
        self.options_display.show()


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
    return options


def update_options_error_handler(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as ex:
            print("Incorrect value entered for option.")
            print(f"An error occurred: {type(ex).__name__}: {str(ex)}")
            # traceback.print_exc()
        finally:
            pass

    return wrapped


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Example using PyxAlign's ProjectionMatchingOptions:
    config_instance = opts.ProjectionMatchingOptions()

    editor = BasicOptionsEditor(
        config_instance, skip_fields=["plot", "interactive_viewer.update.enabled"]
    )
    editor.setWindowTitle("Nested Dataclass Editor with Optional Fields")

    editor.show()
    sys.exit(app.exec_())
