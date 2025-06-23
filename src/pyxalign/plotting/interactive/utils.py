from typing import Optional
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QTableWidget,
)
from PyQt5.QtCore import Qt

from pyxalign.interactions.options.options_editor import OptionsClass


def populate_tree_widget(tree_widget: QTreeWidget, data):
    """
    Recursively populates a QTreeWidget item with fields from nested dataclasses.
    :param tree_widget: The QTreeWidget to populate.
    :param data: The root dataclass instance whose data will be displayed.
    """
    # Clear any existing items in the tree
    tree_widget.clear()

    # Create a top-level item for the root dataclass
    root_item = QTreeWidgetItem(tree_widget, [data.__class__.__name__])
    font = root_item.font(0)
    font.setBold(True)
    root_item.setFont(0, font)

    # Recursively add items for the dataclass fields
    add_dataclass_to_tree(root_item, data)

    # Expand all tree items to make it easier to view
    tree_widget.expandAll()


def add_dataclass_to_tree(parent_item: QTreeWidgetItem, data, path: str = ""):
    """
    Recursively adds children to the QTreeWidgetItem based on dataclass fields.
    For each field, the tooltip is set to "full.path.field_name=field_value".

    :param parent_item: QTreeWidgetItem representing the parent node in the tree.
    :param data: The dataclass instance (or other value) whose fields will be added.
    :param path: The current path prefix used for generating tooltips (e.g. "parent.sub.field").
    """

    # If `data` is not a dataclass, just show it as a leaf node
    if not hasattr(data, "__dataclass_fields__"):
        parent_item.setText(1, str(data))
        # If there's a path, set the tooltip to path=value
        if path:
            parent_item.setToolTip(0, f"{path}={data}")
        return

    # Separate non-dataclass fields and dataclass fields
    non_dataclass_fields = []
    dataclass_fields = []
    for field_name, field_value in data.__dict__.items():
        if hasattr(field_value, "__dataclass_fields__"):
            dataclass_fields.append((field_name, field_value))
        else:
            non_dataclass_fields.append((field_name, field_value))

    # Add non-dataclass fields first
    for field_name, field_value in non_dataclass_fields:
        field_item = QTreeWidgetItem(parent_item, [field_name])
        field_item.setText(1, str(field_value))
        # Build the new "path.field_name". If path is empty, just use field_name
        new_path = f"{path}.{field_name}" if path else field_name
        # field_item.setToolTip(0, f"{new_path}={field_value}")
        field_item.setToolTip(0, f"{new_path}")

    # Add dataclass fields next
    for field_name, field_value in dataclass_fields:
        field_item = QTreeWidgetItem(parent_item, [field_name])
        # Make text bold
        font = field_item.font(0)
        font.setBold(True)
        field_item.setFont(0, font)

        # Build the new path for the nested dataclass
        new_path = f"{path}.{field_name}" if path else field_name
        field_item.setToolTip(0, f"{new_path}")

        # Recurse for the nested dataclass
        add_dataclass_to_tree(field_item, field_value, new_path)


class OptionsDisplayWidget(QWidget):
    def __init__(self, options: Optional[OptionsClass] = None):
        super().__init__()
        self.setWindowTitle("Options")
        self.options = options

        # Create layout
        layout = QVBoxLayout(self)

        # Create our QTreeWidget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setColumnCount(2)
        self.tree_widget.setHeaderLabels(["Field", "Value"])
        self.tree_widget.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        # Populate it with the example data
        if options is not None:
            populate_tree_widget(self.tree_widget, options)

        # Add the tree to our layout
        layout.addWidget(self.tree_widget)

    def update_display(self):#, options):
        populate_tree_widget(self.tree_widget, self.options)


def sync_checkboxes(*checkboxes):
    """
    Synchronize the given QCheckBox widgets so that when one changes its state,
    all of them are updated to match it. Signals are blocked temporarily on
    the other checkboxes to prevent infinite loops.
    """

    def update_states(state, source_checkbox):
        # Convert integer state to boolean
        checked = state == Qt.Checked
        for cb in checkboxes:
            if cb is not source_checkbox:
                cb.setChecked(checked)

    # Connect each checkbox's stateChanged signal to our update function
    for cb in checkboxes:
        cb.stateChanged.connect(lambda state, src=cb: update_states(state, src))

def get_strings_from_table_widget(table_widget: QTableWidget) -> list:
    """
    Returns all the strings from a QTableWidget as a Python list of strings.
    """
    items_text = []
    for i in range(table_widget.rowCount()):
        items_text.append(table_widget.item(i, 0).text())
    return items_text