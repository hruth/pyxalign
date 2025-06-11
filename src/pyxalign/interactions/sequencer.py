import sys
import matplotlib
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

from pyxalign.interactions.sequencer_item import SequencerItem


class SequencerWidget(QWidget):
    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.options = options

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

        self.sequencer_items = [SequencerItem(self.options)]
        self.sequencer_list_layout.addWidget(self.sequencer_items[0])
        self.sequencer_list_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Preferred, QSizePolicy.Expanding)
        )

        self.add_sequencer_button = QPushButton("Add New Sequence")
        self.add_sequencer_button.pressed.connect(self.add_new_sequencer)

        self.copy_sequencer_button = QPushButton("Duplicate Last Sequence")
        self.copy_sequencer_button.pressed.connect(self.add_new_sequencer)

        self.remove_sequencer_button = QPushButton("Delete Last Sequence")
        self.remove_sequencer_button.pressed.connect(self.remove_last_sequence)

        self.main_layout.addWidget(scroll_area)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_sequencer_button)
        button_layout.addWidget(self.remove_sequencer_button)
        self.main_layout.addLayout(button_layout)

    def add_new_sequencer(self):
        new_item = SequencerItem(self.options)
        self.sequencer_items += [new_item]
        self.sequencer_list_layout.insertWidget(len(self.sequencer_items) - 1, new_item)

    def remove_last_sequence(self):
        if len(self.sequencer_items) > 0:
            self.sequencer_items[-1].deleteLater()
            self.sequencer_items = self.sequencer_items[:-1]


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
