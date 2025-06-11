import sys
import dataclasses
from dataclasses import dataclass, fields
from typing import Any
import matplotlib
import copy
import pyxalign.api.options as opts

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QFrame,
    QCheckBox
)
from PyQt5.QtCore import Qt

from pyxalign.interactions.options.options_editor import SingleOptionEditor


class SequencerItem(QWidget):
    """
    A widget that allows you to explore the nested attributes of a dataclass
    by dynamically creating combo boxes for each dataclass attribute that
    contains further nested dataclasses.
    """

    def __init__(self, options):
        super().__init__()
        self.data = copy.deepcopy(options)
        self.options_editor = None

        # Top-level layout for this widget
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # A QFrame to visually enclose the contents of the SequencerItem
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Plain)
        self.frame.setLineWidth(1)
        self.frame.setStyleSheet("QFrame { background-color:#c2d1c9; border:#c2d1c9}")
        self.main_layout.addWidget(self.frame)

        # A layout that goes inside the frame
        self.frame_layout = QVBoxLayout()
        self.frame.setLayout(self.frame_layout)

        # We keep track of each "level" of selection:
        # - A combo box for the dataclass attributes
        # - The corresponding dataclass instance or "leaf" value
        self.combo_boxes = []  # [(QComboBox, associated_object), ...]

        # Create the initial combo box for the top-level dataclass
        self.add_combo_box(self.data)

        # self.selected_label = QLabel("Selected Path: (None)")
        # self.frame_layout.addWidget(self.selected_label)

        self.selection_layout = QHBoxLayout()
        self.selected_label = QLabel("")
        self.selection_layout.addWidget(self.selected_label)
        self.frame_layout.addLayout(self.selection_layout)

    def value(self) -> Any:
        return self.options_editor.value()

    def on_combo_box_changed(self):
        """
        Called whenever the user selects an attribute from any combo box.
        Manages the creation or removal of child combo boxes based on whether
        the selected attribute is a nested dataclass or not.
        """
        # Find which combo box was changed and its index in the combo_boxes list
        combo_box = self.sender()
        combo_index = next((i for i, (cb, _) in enumerate(self.combo_boxes) if cb == combo_box), -1)

        # Remove any combo boxes that come after this one
        self.remove_combo_boxes_after(combo_index)

        # If the user is still on the placeholder (index 0), do nothing more.
        if combo_box.currentIndex() == 0:
            # Reset the currently selected path label to reflect partial selection
            self.update_selected_label(ignore_tail=True)
            return

        # Figure out the new selection inside this combo box
        attr_name = combo_box.currentText()
        parent_obj = self.combo_boxes[combo_index][1]

        # Get the selected object (could be a nested dataclass or leaf)
        selected_obj = getattr(parent_obj, attr_name)

        # If it's a dataclass instance, create a new combo box for its fields
        if dataclasses.is_dataclass(selected_obj):
            self.add_combo_box(selected_obj)
        else:
            self.add_options_selector(parent_obj, attr_name)
        self.update_selected_label()

    def add_options_selector(self, parent_obj, attr_name: str):
        self.options_editor = SingleOptionEditor(copy.deepcopy(parent_obj), attr_name, parent=self)
        self.selection_layout.addWidget(self.options_editor)
        # Add checkbox indicating if you want to run the algo after selection or not
        self.run_alignment_after_checkbox = QCheckBox("Run alignment after value change: ", self)
        self.frame_layout.addWidget(self.run_alignment_after_checkbox)

    def add_combo_box(self, obj):
        """
        Adds a new combo box for the provided dataclass object,
        allowing selection of its fields/attributes.
        """
        combo_box = QComboBox()
        combo_box.setEditable(False)

        # Insert a placeholder entry so the drop-down appears empty initially
        combo_box.addItem("--Select an attribute--")
        # Populate with attributes from 'obj'
        for f in fields(obj):
            combo_box.addItem(f.name)

        # Keep track of this combo in our list, along with the associated object
        self.combo_boxes.append((combo_box, obj))
        combo_box.currentIndexChanged.connect(self.on_combo_box_changed)

        # Insert in the frame layout
        self.frame_layout.insertWidget(len(self.combo_boxes) - 1, combo_box)
        self.adjustSize()

    def remove_combo_boxes_after(self, combo_index):
        """
        Removes any combo boxes after the given index from both the layout
        and the internal tracking list, effectively discarding deeper nested
        selection UI.
        """
        # Remove from the end to combo_index + 1 (exclusive)
        while len(self.combo_boxes) > combo_index + 1:
            cb, _ = self.combo_boxes.pop()
            self.frame_layout.removeWidget(cb)
            cb.deleteLater()

        if self.options_editor is not None:
            self.frame_layout.removeWidget(self.options_editor)
            self.options_editor.deleteLater()
            self.options_editor.setParent(None)
            self.options_editor = None

    def update_selected_label(self, ignore_tail=False) -> str:
        """
        Create a string representation of the fully-selected path of attributes.

        If ignore_tail=True, we'll only show the path up to the last
        properly selected field, ignoring placeholder selections in deeper combos.
        """
        path_parts = []
        current_obj = self.data

        for combo_box, obj in self.combo_boxes:
            idx = combo_box.currentIndex()
            # If the user didn't select a real attribute (i.e., is at index 0),
            # stop building the path here if ignoring tail.
            if idx == 0 and ignore_tail:
                break

            attr_name = combo_box.currentText()
            if attr_name.startswith("--"):
                # It's the placeholder
                break

            path_parts.append(attr_name)
            current_obj = getattr(current_obj, attr_name, None)

        path_str = ".".join(path_parts)
        # if path_parts:
        #     self.selected_label.setText(f"Selected Path: {path_str}")
        # else:
        #     self.selected_label.setText("Selected Path: (None)")
        if path_parts:
            self.selected_label.setText(path_str)
        else:
            self.selected_label.setText("")

        return path_str


def main():
    app = QApplication(sys.argv)
    selector = SequencerItem(opts.ProjectionMatchingOptions())
    selector.setWindowTitle("Nested Dataclass Selector")
    selector.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
