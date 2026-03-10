"""
Example script demonstrating the CombinedViewerWidget.

This example shows how to use the CombinedViewerWidget to view projections
and run projection matching alignment in a unified interface.
"""

import sys
from PyQt5.QtWidgets import QApplication
from pyxalign.interactions.combined_viewer import CombinedViewerWidget
from pyxalign.data_structures.task import load_task


def main():
    """
    Launch the CombinedViewerWidget with a saved task.

    Usage:
        python combined_viewer_example.py /path/to/task.h5
    """
    app = QApplication(sys.argv)

    # Load the task from command line argument or use default path
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
        print(f"Loading task from: {task_path}")
        task = load_task(task_path)
    else:
        print("Usage: python combined_viewer_example.py /path/to/task.h5")
        print("No task file provided. Please provide a task file path.")
        sys.exit(1)

    # Create the combined viewer widget
    widget = CombinedViewerWidget(task)

    # Show the widget
    widget.show()

    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
