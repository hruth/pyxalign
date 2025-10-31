import functools
import sys
import time
import traceback
from typing import Callable

from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt, QEventLoop
from PyQt5.QtWidgets import QApplication, QProgressDialog, QWidget, QVBoxLayout, QLabel


#############################################################
# Create wrapper that runs loading bar in main thread while 
# the slow function is run in a seperate thread             
#############################################################


class Worker(QObject):
    """
    Worker class to run the target function in a separate thread.
    Emits 'done' when the function finishes.
    """

    done = pyqtSignal(object, object)  # (result, exception)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Executed in the worker thread."""
        result = None
        exception = None
        try:
            result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            exception = e
            traceback.print_exc()
        finally:
            self.done.emit(result, exception)


def loading_bar_wrapper(load_message: str = "Processing...", block_all_windows: bool = False):
    """
    Decorator that:
    1. Spawns a QThread to run `func` in the background.
    2. Displays an indefinite / animated QProgressDialog in the main thread.
    3. Blocks until the background work finishes, then returns the result.

    Works for both instance and static methods; the only catch is that
    if a PyQt button is invoking the decorated method, you must handle
    the extra "checked" argument (see usage examples).

    Args:
        load_message: The message to display in the progress dialog.
        block_all_windows: If True, grays out and disables all other windows in the application.
                          If False, only blocks the parent window (default behavior).

    IMPORTANT NOTE FOR DEVELOPERS: this function will not work as a decorator
    for functions that are triggered via PyQt signals unless the function
    has *args in the signature. This is because the value from the PyQt signal
    gets passed into `wrapper` in the `args` tuple, and then is passed into
    `func`. If `func` doesn't have enough input slots, it will throw an error.

    In general, you should try to avoid using this as a decorator on such functions.
    Instead, it should be used on functions that will not be triggered by signals.
    """
    if isinstance(load_message, Callable):
        raise ValueError(
            "loading_bar_wrapper used incorrectly; the argument of loading_bar_wrapper should be a string"
        )

    def middleman(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create the QProgressDialog in the main (GUI) thread
            progress_dialog = QProgressDialog(load_message, None, 0, 0)
            progress_dialog.setWindowTitle("Please Wait")

            # Set modality based on block_all_windows parameter
            if block_all_windows:
                progress_dialog.setWindowModality(Qt.ApplicationModal)
            else:
                progress_dialog.setWindowModality(Qt.WindowModal)

            progress_dialog.setCancelButton(None)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setRange(0, 0)  # indefinite/animated bar

            # Setup the worker/thread
            thread = QThread()
            worker = Worker(func, *args, **kwargs)
            worker.moveToThread(thread)

            # Local event loop to block the wrapper until worker finishes
            loop = QEventLoop()

            result_container = {"result": None, "exception": None}

            def on_done(result, exception):
                result_container["result"] = result
                result_container["exception"] = exception
                loop.quit()

            # Connect signals
            worker.done.connect(on_done)
            thread.started.connect(worker.run)

            # Start the thread
            thread.start()

            # Show the animated progress dialog
            progress_dialog.show()
            QApplication.processEvents()

            # Block here until done
            loop.exec_()

            # Clean up
            progress_dialog.close()
            thread.quit()
            thread.wait()

            # Re-raise exception if needed
            if result_container["exception"] is not None:
                raise result_container["exception"]

            return result_container["result"]

        return wrapper

    return middleman


#############################################################
# Create overlay widget that displays text over the window
# This is useful when you cannot use the loadbar because the
# spawned thread uses PyQt5 which would cause the kernel to 
# crash
#############################################################


class OverlayWidget(QWidget):
    """
    A semi-transparent overlay widget that can display text
    and block interaction with the underlying window.
    """
    def __init__(self, parent=None, text="Processing..."):
        super().__init__(parent)
        
        # Make the overlay cover the entire parent widget
        if parent:
            self.setGeometry(parent.rect())
        
        # Set up the semi-transparent background
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        # Create layout and label for text
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 180);
                padding: 20px;
                border-radius: 10px;
            }
        """)
        
        layout.addWidget(self.label)
        
        # Set the overall background with transparency
        self.setStyleSheet("background-color: rgba(128, 128, 128, 100);")
        
    def setText(self, text):
        """Update the overlay text."""
        self.label.setText(text)
    
    def paintEvent(self, event):
        """Ensure the overlay fills the parent widget."""
        super().paintEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())