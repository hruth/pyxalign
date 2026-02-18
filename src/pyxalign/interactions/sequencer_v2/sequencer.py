"""
SequencerWidgetV2: Manages multiple alignment run blocks for projection matching.

This version organizes alignment sequences into discrete blocks, where each block
represents a single alignment run with potentially multiple parameter changes.
This makes it clearer which parameters are applied together in each alignment execution.
"""

import sys
import copy
from typing import Optional, TypeVar

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QPushButton,
    QScrollArea,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from pyxalign.api.options.base import BaseOptions
from pyxalign.interactions.options.options_editor import set_option_from_field_path
from pyxalign.interactions.sequencer_v2.sequencer_item import SequencerItemV2
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend

T = TypeVar("T")


class SequencerWidgetV2(QWidget):
    """
    Widget for managing a sequence of alignment run blocks.

    Each block in the sequence represents one call to get_projection_matching_shift
    with a specific set of parameter modifications. This makes it clear that all
    parameters within a block are applied together for that alignment run.
    """

    def __init__(
        self,
        options: BaseOptions,
        list_of_updated_settings: Optional[list[dict]] = None,
        basic_options_list: Optional[list[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.options = options
        self.basic_options_list = basic_options_list or []

        # Main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Title
        sequencer_title = QLabel("Alignment Sequence")
        sequencer_title.setStyleSheet("QLabel {font-size: 16px; font-weight: bold;}")
        self.main_layout.addWidget(sequencer_title)

        # Info label
        info_label = QLabel(
            "Each block represents one alignment run. "
            "Add multiple parameters to a block to apply them together."
        )
        info_label.setStyleSheet("QLabel {font-size: 10px; color: gray;}")
        info_label.setWordWrap(True)
        self.main_layout.addWidget(info_label)

        # Scroll area for sequencer blocks
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.sequencer_list_layout = QVBoxLayout()
        scroll_widget.setLayout(self.sequencer_list_layout)
        scroll_area.setWidget(scroll_widget)

        # Initialize with one empty block
        self.sequencer_blocks = [
            SequencerItemV2(self.options, basic_options_list=self.basic_options_list)
        ]
        self._connect_block_signals(self.sequencer_blocks[0])
        self.sequencer_list_layout.addWidget(self.sequencer_blocks[0])
        self.sequencer_list_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Preferred, QSizePolicy.Expanding)
        )

        self.main_layout.addWidget(scroll_area)

        # Block management buttons
        button_layout = QHBoxLayout()

        self.add_block_button = QPushButton("Add New Block")
        self.add_block_button.setStyleSheet("QPushButton { background-color: #90EE90; }")
        self.add_block_button.pressed.connect(self.add_new_block)

        self.duplicate_block_button = QPushButton("Duplicate Last Block")
        self.duplicate_block_button.pressed.connect(self.duplicate_last_block)

        self.remove_block_button = QPushButton("Delete Last Block")
        self.remove_block_button.setStyleSheet("QPushButton { background-color: #ffcccc; }")
        self.remove_block_button.pressed.connect(self.remove_last_block)

        button_layout.addWidget(self.add_block_button)
        button_layout.addWidget(self.duplicate_block_button)
        button_layout.addWidget(self.remove_block_button)
        self.main_layout.addLayout(button_layout)

        # Load initial settings if provided
        if list_of_updated_settings is not None:
            self.generate_sequence_from_list_of_dicts(list_of_updated_settings)

    def _connect_block_signals(self, block: SequencerItemV2):
        """Connect signals from a SequencerItemV2 to the appropriate handlers."""
        block.insert_above_requested.connect(self.insert_block_above)
        block.insert_below_requested.connect(self.insert_block_below)
        block.duplicate_requested.connect(self.duplicate_block)
        block.remove_requested.connect(self.remove_block)

    def insert_block_above(self, reference_block: SequencerItemV2):
        """Insert a new block above the reference block."""
        index = self.sequencer_blocks.index(reference_block)
        new_block = SequencerItemV2(self.options, basic_options_list=self.basic_options_list)
        self._connect_block_signals(new_block)
        self.sequencer_blocks.insert(index, new_block)
        self.sequencer_list_layout.insertWidget(index, new_block)

    def insert_block_below(self, reference_block: SequencerItemV2):
        """Insert a new block below the reference block."""
        index = self.sequencer_blocks.index(reference_block)
        new_block = SequencerItemV2(self.options, basic_options_list=self.basic_options_list)
        self._connect_block_signals(new_block)
        self.sequencer_blocks.insert(index + 1, new_block)
        self.sequencer_list_layout.insertWidget(index + 1, new_block)

    def duplicate_block(self, reference_block: SequencerItemV2):
        """Duplicate the reference block and insert it below."""
        index = self.sequencer_blocks.index(reference_block)

        # Get the current parameters from the reference block
        parameters = reference_block.get_parameter_list()

        # Create new block with the same parameters
        new_block = SequencerItemV2(
            self.options,
            initial_parameters=parameters,
            basic_options_list=self.basic_options_list,
        )
        self._connect_block_signals(new_block)
        self.sequencer_blocks.insert(index + 1, new_block)
        self.sequencer_list_layout.insertWidget(index + 1, new_block)

    def remove_block(self, block_to_remove: SequencerItemV2):
        """Remove the specified block."""
        # Prevent removing the last block
        if len(self.sequencer_blocks) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Remove",
                "Cannot remove the last alignment block. At least one block must remain."
            )
            return

        index = self.sequencer_blocks.index(block_to_remove)
        self.sequencer_blocks.pop(index)
        self.sequencer_list_layout.removeWidget(block_to_remove)
        block_to_remove.deleteLater()

    def add_new_block(self, initial_parameters: Optional[list[tuple[str, T]]] = None):
        """Add a new alignment block to the sequence."""
        new_block = SequencerItemV2(
            self.options,
            initial_parameters=initial_parameters,
            basic_options_list=self.basic_options_list,
        )
        self._connect_block_signals(new_block)
        self.sequencer_blocks.append(new_block)
        # Insert before the spacer (which is always last)
        self.sequencer_list_layout.insertWidget(len(self.sequencer_blocks) - 1, new_block)

    def duplicate_last_block(self):
        """Duplicate the last block in the sequence."""
        if not self.sequencer_blocks:
            return

        parameters = self.sequencer_blocks[-1].get_parameter_list()
        new_block = SequencerItemV2(
            self.options,
            initial_parameters=parameters,
            basic_options_list=self.basic_options_list,
        )
        self._connect_block_signals(new_block)
        self.sequencer_blocks.append(new_block)
        self.sequencer_list_layout.insertWidget(len(self.sequencer_blocks) - 1, new_block)

    def remove_last_block(self):
        """Remove the last block from the sequence."""
        if len(self.sequencer_blocks) <= 1:
            QMessageBox.warning(
                self,
                "Cannot Remove",
                "Cannot remove the last alignment block. At least one block must remain."
            )
            return

        if self.sequencer_blocks:
            self.sequencer_blocks[-1].deleteLater()
            self.sequencer_blocks = self.sequencer_blocks[:-1]

    def generate_sequence_from_list_of_dicts(self, list_of_updated_settings: list[dict]):
        """
        Generate alignment blocks from a list of settings dictionaries.

        Each dictionary in the list represents one alignment run block with
        multiple parameter changes.

        Parameters
        ----------
        list_of_updated_settings : list[dict]
            List of dictionaries where each dict contains parameter changes
            for one alignment run.
        """
        self.clear_all_blocks()

        for settings_dict in list_of_updated_settings:
            # Convert dict to list of (field_path, value) tuples
            parameters = get_settings_from_dict(settings_dict, value_pairs=[])
            self.add_new_block(initial_parameters=parameters)

    def clear_all_blocks(self):
        """Remove all blocks from the sequence."""
        while len(self.sequencer_blocks) > 0:
            self.remove_last_block()

    def generate_options_sequence(self, options: T) -> list[T]:
        """
        Generate a sequence of options objects, one for each alignment block.

        Each block's parameters are applied cumulatively to create the options
        for that alignment run.

        Parameters
        ----------
        options : T
            The base options object to start from.

        Returns
        -------
        list[T]
            List of options objects, one for each alignment run.
        """
        options_sequence: list[BaseOptions] = []

        for block in self.sequencer_blocks:
            # Start with a fresh copy of the base options
            options_item = copy.deepcopy(options)

            # Apply all parameter changes in this block
            for field_path, value in block.get_parameter_list():
                options_item = set_option_from_field_path(
                    copy.deepcopy(options_item), field_path, value
                )

            options_sequence.append(options_item)

        # If no valid blocks, return the base options
        if len(options_sequence) == 0:
            options_sequence.append(copy.deepcopy(options))

        return options_sequence

    def get_changed_settings_sequence(self) -> list[dict]:
        """
        Generate a list of dictionaries tracking which settings were changed for each block.

        Returns
        -------
        list[dict]
            List of dictionaries where each dict contains the changed settings
            for that alignment block.
        """
        changed_settings_sequence = []

        for block in self.sequencer_blocks:
            changes = block.get_parameter_changes()
            changed_settings_sequence.append(changes)

        return changed_settings_sequence


def get_settings_from_dict(settings_dict: dict, name="", value_pairs=[]):
    """
    Recursively extract (field_path, value) pairs from a nested settings dictionary.

    Parameters
    ----------
    settings_dict : dict
        Dictionary of settings to extract.
    name : str
        Current path prefix (used in recursion).
    value_pairs : list
        Accumulated list of (field_path, value) tuples.

    Returns
    -------
    list
        List of (field_path, value) tuples.
    """
    for setting_name, setting_value in settings_dict.items():
        if isinstance(setting_value, dict):
            value_pairs = get_settings_from_dict(
                setting_value, f"{name}{setting_name}.", value_pairs
            )
        else:
            value_pairs.append((f"{name}{setting_name}", setting_value))
    return value_pairs


@switch_to_matplotlib_qt_backend
def launch_sequencer_v2(
    wait_until_closed: bool = False,
):
    """Launch the SequencerWidgetV2 for testing."""
    from pyxalign.api.options.alignment import ProjectionMatchingOptions

    app = QApplication.instance() or QApplication([])
    gui = SequencerWidgetV2(
        ProjectionMatchingOptions(),
        basic_options_list=["high_pass_filter", "downsample", "downsample.scale"],
    )
    gui.show()
    gui.setAttribute(Qt.WA_DeleteOnClose)
    if wait_until_closed:
        app.exec_()
    return gui


# For demonstration purposes only:
if __name__ == "__main__":
    app = QApplication(sys.argv)
    from pyxalign.api.options.alignment import ProjectionMatchingOptions

    window = SequencerWidgetV2(
        ProjectionMatchingOptions(),
        basic_options_list=["high_pass_filter", "downsample", "downsample.scale"],
    )
    window.setWindowTitle("Alignment Sequencer V2")
    screen_geometry = app.desktop().availableGeometry(window)
    window.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() / 2),
        int(screen_geometry.height() / 2),
    )
    window.show()
    sys.exit(app.exec_())
