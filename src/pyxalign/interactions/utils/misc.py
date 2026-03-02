from functools import wraps
from IPython import get_ipython
from PyQt5.QtWidgets import QApplication, QWidget


def switch_to_matplotlib_qt_backend(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ipython = get_ipython()
        if ipython:
            ipython = get_ipython()
            ipython.run_line_magic("matplotlib", "qt")
        result = func(*args, **kwargs)
        return result

    return wrap


def center_window_on_screen(widget: QWidget, width_fraction: float = 0.75, height_fraction: float = 0.75):
    """
    Center a widget on the screen and set its size to a fraction of the screen size.

    Parameters
    ----------
    widget : QWidget
        The widget to center and resize
    width_fraction : float, optional
        Fraction of screen width (default: 0.75)
    height_fraction : float, optional
        Fraction of screen height (default: 0.75)
    """
    app = QApplication.instance() or QApplication([])
    screen_geometry = app.desktop().availableGeometry(widget)

    # Calculate window size
    window_width = int(screen_geometry.width() * width_fraction)
    window_height = int(screen_geometry.height() * height_fraction)

    # Calculate centered position
    x = screen_geometry.x() + (screen_geometry.width() - window_width) // 2
    y = screen_geometry.y() + (screen_geometry.height() - window_height) // 2

    # Set geometry
    widget.setGeometry(x, y, window_width, window_height)
