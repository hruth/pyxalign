import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Optional, Union, Callable
import cupy as cp
from pyxalign.api.options.task import AlignmentTaskOptions

from pyxalign.interactions.options.options_editor import BasicOptionsEditor
from pyxalign.interactions.sequencer import SequencerWidget

# from pyxalign.io.load import load_task
import pyxalign.io.load as load
from pyxalign.plotting.interactive.base import MultiThreadedWidget
import pyxalign.data_structures.task as t

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
    QTabWidget,
    QGridLayout,
    QTableWidget,
)
from PyQt5.QtCore import Qt


class PMAMasterWidget(MultiThreadedWidget):
    # edit PMA options
    # Features:
    # - set up multi-resolution alignment scans
    # - launch viewer when scan is started
    # - store: alignment shift results, options used
    # First: set up layout where you can set up a multi-resolution scan.
    # keep it simple by making all resolutions run with the same options.
    def __init__(
        self,
        task: Optional["t.LaminographyAlignmentTask"] = None,  # temporarily equal to none
        multi_thread_func: Optional[Callable] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            multi_thread_func=multi_thread_func,
            parent=parent,
        )
        self.task = task

        tabs = QTabWidget()
        layout = QHBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)

        self.generate_start_and_stop_buttons()
        self.generate_options_selection_widget()
        self.generate_sequencer()
        self.make_first_tab_layout(tabs)

    def generate_start_and_stop_buttons(self):
        self.button_widget = QWidget(self)
        # button_layout = QGridLayout()
        button_layout = QHBoxLayout()
        self.button_widget.setLayout(button_layout)

        self.start_sequence_button = QPushButton("Start Alignment")
        self.stop_alignment_button = QPushButton("Stop Current Alignment")
        self.stop_sequence_button = QPushButton("Stop Alignment Sequence")

        self.start_sequence_button.pressed.connect(self.start_alignment)

        self.start_sequence_button.setStyleSheet("QPushButton { background-color: green;}")
        self.stop_alignment_button.setStyleSheet("QPushButton { background-color: red;}")
        self.stop_sequence_button.setStyleSheet("QPushButton { background-color: red;}")

        button_layout.addWidget(self.start_sequence_button)
        button_layout.addWidget(self.stop_alignment_button)
        button_layout.addWidget(self.stop_sequence_button)
        button_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Preferred))

        self.button_widget.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 11pt; color: white; padding: 2px 6px; }"
        )

    def start_alignment(self):
        # add in sequence!!
        self.task.get_projection_matching_shift()

    def generate_options_selection_widget(self):
        self.options_editor = BasicOptionsEditor(
            self.task.options.projection_matching, skip_fields=["plot"]
        )

    def generate_sequencer(self):
        self.sequence_table = SequencerWidget(self)

    def make_first_tab_layout(self, tabs: QTabWidget):
        alignment_setup_widget = QWidget(self)

        v_layout = QVBoxLayout()
        # alignment_setup_widget.setLayout(v_layout)
        v_layout.addWidget(self.options_editor)
        v_layout.addWidget(self.button_widget)

        h_layout = QHBoxLayout()
        h_layout.addLayout(v_layout)
        h_layout.addWidget(self.sequence_table)

        alignment_setup_widget.setLayout(h_layout)

        tabs.addTab(alignment_setup_widget, "Options")


if __name__ == "__main__":
    # dummy_task = t.LaminographyAlignmentTask(options=AlignmentTaskOptions(), phase_projections=1)
    dummy_task = load.load_task(
        "/gpfs/dfnt1/test/hruth/pyxalign_ci_test_data/dummy_inputs/cSAXS_e18044_LamNI_201907_16x_downsampled_pre_pma_task.h5"
    )
    dummy_task.options.projection_matching.iterations = 21
    app = QApplication(sys.argv)
    master_widget = PMAMasterWidget(dummy_task)

    # Use the left half of the screen
    screen_geometry = app.desktop().availableGeometry(master_widget)
    master_widget.setGeometry(
        screen_geometry.x(),
        screen_geometry.y(),
        int(screen_geometry.width() * .75),
        int(screen_geometry.height() * 0.9),
    )

    master_widget.show()
    sys.exit(app.exec_())
