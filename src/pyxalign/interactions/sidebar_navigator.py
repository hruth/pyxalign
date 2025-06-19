import matplotlib
import sys
from functools import partial

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QToolBar,
    QStackedWidget,
    QAction,
    QActionGroup,
    QHBoxLayout,
    QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon


class SidebarNavigator(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout: horizontal box (toolbar on the left, content on the right)
        self._main_layout = QHBoxLayout(self)
        self.setLayout(self._main_layout)

        # 1. Create a vertical QToolBar
        self.toolbar = QToolBar("Navigation")
        self.toolbar.setOrientation(Qt.Vertical)
        # Show text beside icons (so text is visible even if icons fail to load)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Use a QActionGroup to allow only one action (page) to be checked at once
        self.action_group = QActionGroup(self)
        self.action_group.setExclusive(True)

        # 2. Create a QStackedWidget for different "pages" of content
        self.stacked_widget = QStackedWidget()

        # 3. Add the toolbar and stacked widget to the layout
        self._main_layout.addWidget(self.toolbar)
        self._main_layout.addWidget(self.stacked_widget)

        # # Final window settings
        # self.setWindowTitle("QToolBar Navigation Example")
        # self.resize(600, 400)

    def addPage(self, page_widget: QWidget, title: str, icon: QIcon = None):
        """
        Programmatically add a new page to this widget's QStackedWidget
        and a corresponding QAction to the QToolBar.

        :param page_widget: The widget to display as a new page.
        :param title: The text label for the "tab" (toolbar action).
        :param icon: (Optional) A QIcon for the toolbar action.
        """
        # 1. Add the page widget to the QStackedWidget and get its index
        index = self.stacked_widget.addWidget(page_widget)

        # 2. Create the QAction for this page
        action = QAction(icon if icon else QIcon(), title, self)
        action.setCheckable(True)
        self.action_group.addAction(action)

        # 3. Connect the action to switch to the appropriate stacked widget index
        action.triggered.connect(partial(self.stacked_widget.setCurrentIndex, index))

        # 4. Add the action to the toolbar
        self.toolbar.addAction(action)

        # 5. If it's the first page, mark its action as checked
        if self.stacked_widget.count() == 1:
            action.setChecked(True)


def main():
    app = QApplication(sys.argv)

    # Create an instance of MainWidget
    main_widget = SidebarNavigator()

    # Create a few example pages
    for i in range(1, 4):
        page_label = QLabel(f"Content for Page {i}")
        page_label.setAlignment(Qt.AlignCenter)
        main_widget.addPage(page_label, f"Page {i}", QIcon())  # Empty QIcon() for demonstration

    # Show the main widget
    main_widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
