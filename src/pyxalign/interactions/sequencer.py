import sys
import matplotlib
import copy
from typing import Optional
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QGridLayout,
    QSpacerItem,
    QSizePolicy,
    QPushButton,
    QScrollArea,
)
from PyQt5.QtCore import Qt
from pyxalign.api.options.alignment import ProjectionMatchingOptions
from pyxalign.interactions.options.options_editor import OptionsClass, set_option_from_field_path

from pyxalign.interactions.sequencer_item import SequencerItem


class SequencerWidget(QWidget):
    def __init__(
        self, options: OptionsClass, basic_options_list: Optional[list[str]] = None, parent=None
    ):
        super().__init__(parent)
        self.options = options
        self.basic_options_list = basic_options_list or []

        # Main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # scroll area for sequencer items
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.sequencer_list_layout = QVBoxLayout()
        scroll_widget.setLayout(self.sequencer_list_layout)
        scroll_area.setWidget(scroll_widget)

        self.sequencer_items = [
            SequencerItem(self.options, basic_options_list=self.basic_options_list)
        ]
        self.sequencer_list_layout.addWidget(self.sequencer_items[0])
        self.sequencer_list_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Preferred, QSizePolicy.Expanding)
        )

        self.add_sequencer_button = QPushButton("Add New Sequence")
        self.add_sequencer_button.pressed.connect(self.add_new_sequencer)

        self.copy_sequencer_button = QPushButton("Duplicate Last Sequence")
        self.copy_sequencer_button.pressed.connect(self.duplicate_last_sequence)

        self.remove_sequencer_button = QPushButton("Delete Last Sequence")
        self.remove_sequencer_button.pressed.connect(self.remove_last_sequence)

        sequencer_title = QLabel("Options Sequencer")
        sequencer_title.setStyleSheet("QLabel {font-size: 16px;}")
        self.main_layout.addWidget(sequencer_title)
        self.main_layout.addWidget(scroll_area)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_sequencer_button)
        button_layout.addWidget(self.copy_sequencer_button)
        button_layout.addWidget(self.remove_sequencer_button)
        self.main_layout.addLayout(button_layout)

    def add_new_sequencer(self):
        new_item = SequencerItem(self.options, basic_options_list=self.basic_options_list)
        self.sequencer_items += [new_item]
        self.sequencer_list_layout.insertWidget(len(self.sequencer_items) - 1, new_item)

    def duplicate_last_sequence(self):
        initial_field = self.sequencer_items[-1].full_field_path()
        initial_value = self.sequencer_items[-1].value()
        checkbox_state = self.sequencer_items[-1].checkbox_state()
        new_item = SequencerItem(
            self.options,
            initial_state=(initial_field, initial_value, checkbox_state),
            basic_options_list=self.basic_options_list,
        )
        self.sequencer_items += [new_item]
        self.sequencer_list_layout.insertWidget(len(self.sequencer_items) - 1, new_item)

    def remove_last_sequence(self):
        if len(self.sequencer_items) > 0:
            self.sequencer_items[-1].deleteLater()
            self.sequencer_items = self.sequencer_items[:-1]

    def generate_options_sequence(self, options: OptionsClass) -> list[OptionsClass]:
        options_sequence: list[OptionsClass] = []
        options_item = copy.deepcopy(options)
        for item in self.sequencer_items:
            if item.value() is None:
                continue
            options_item = set_option_from_field_path(
                copy.deepcopy(options_item), item.full_field_path(), item.value()
            )
            if item.checkbox_state():
                options_sequence += [options_item]
        if len(options_sequence) == 0:
            options_sequence += [options_item]

        return options_sequence


# For demonstration purposes only:
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = SequencerItem()
    window = SequencerWidget(ProjectionMatchingOptions())
    window.setWindowTitle("Sequence Widget Example")
    screen_geometry = app.desktop().availableGeometry(window)
    window.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() / 2),
        int(screen_geometry.height() / 2),
    )
    window.show()
    sys.exit(app.exec_())
