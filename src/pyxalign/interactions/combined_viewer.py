"""
Combined widget integrating PMAMasterWidget and ProjectionViewer.

This module provides a unified interface that combines projection viewing and
projection matching alignment capabilities with sidebar navigation.

Key Components:
- CombinedAlignmentWidget: Main widget with sidebar navigation to switch between
  PMAMasterWidget (alignment tools) and ProjectionViewer (visualization tools)
"""

from typing import Optional
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QStyle
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

import pyxalign.data_structures.task as t
from pyxalign.api import enums
from pyxalign.interactions.sidebar_navigator import SidebarNavigator
from pyxalign.interactions.pma_runner import PMAMasterWidget
from pyxalign.interactions.cross_correlation import CrossCorrelationMasterWidget
from pyxalign.interactions.viewers.arrays import ProjectionViewer
from pyxalign.interactions.utils.misc import switch_to_matplotlib_qt_backend


class CombinedAlignmentWidget(SidebarNavigator):
    """
    Combined widget that provides sidebar navigation between PMAMasterWidget
    and ProjectionViewer.

    This widget creates a unified interface for both viewing projections and
    running projection matching alignment. The sidebar allows switching between:
    - Projection Viewer: Interactive projection visualization with tools
    - PMA Runner: Projection matching alignment workflow interface
    - Cross-Correlation Runner: Cross-correlation alignment workflow interface

    Parameters
    ----------
    task : LaminographyAlignmentTask
        The alignment task containing projections and configuration.
        - Passed directly to PMAMasterWidget for alignment operations
        - task.phase_projections passed to ProjectionViewer for visualization
    include_projection_matching : bool, optional
        Whether to include the PMA Runner in the sidebar, by default True
    include_cross_correlation : bool, optional
        Whether to include the Cross-Correlation Runner in the sidebar, by default True
    parent : QWidget, optional
        Parent widget, by default None

    Attributes
    ----------
    task : LaminographyAlignmentTask
        The alignment task being worked with
    projection_viewer : ProjectionViewer
        Widget for viewing and manipulating projections
    pma_widget : PMAMasterWidget, optional
        Widget for running projection matching alignment (if include_projection_matching=True)
    cc_widget : CrossCorrelationMasterWidget, optional
        Widget for running cross-correlation alignment (if include_cross_correlation=True)

    Examples
    --------
    >>> from pyxalign.data_structures.task import LaminographyAlignmentTask
    >>> task = LaminographyAlignmentTask(...)
    >>> widget = CombinedAlignmentWidget(task)
    >>> widget.show()
    """

    def __init__(
        self,
        task: "t.LaminographyAlignmentTask",
        updated_settings_for_pma_widget: Optional[list[dict]] = None,
        include_projection_matching: bool = True,
        include_cross_correlation: bool = True,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the combined viewer widget.

        Parameters
        ----------
        task : LaminographyAlignmentTask
            The alignment task containing projection data and configuration
        updated_settings_for_pma_widget : list[dict], optional
            Updated settings for the PMA widget
        include_projection_matching : bool, optional
            Whether to include the PMA Runner in the sidebar, by default True
        include_cross_correlation : bool, optional
            Whether to include the Cross-Correlation Runner in the sidebar, by default True
        parent : QWidget, optional
            Parent widget
        """
        super().__init__()

        if parent is not None:
            self.setParent(parent)

        # Store the task
        self.task = task

        # Store which widgets to include
        self.include_projection_matching = include_projection_matching
        self.include_cross_correlation = include_cross_correlation

        # Initialize the widgets
        self._initialize_widgets(updated_settings_for_pma_widget)

        # Set window properties
        self.setWindowTitle("Laminography Alignment Tool")
        self.resize(1200, 800)

    def _initialize_widgets(self, updated_settings_for_pma_widget: Optional[list[dict]] = None):
        """
        Initialize and add the ProjectionViewer and alignment widget pages.
        """
        # Determine which projection type to use and override include flags if needed
        has_phase = self.task.phase_projections is not None
        has_complex = self.task.complex_projections is not None

        # Determine the projection type for CrossCorrelationMasterWidget
        if has_complex and not has_phase:
            # Only complex projections available
            projection_type_for_cc = enums.ProjectionType.COMPLEX
            projections_for_viewer = self.task.complex_projections
            # Override: only cross correlation should be available
            self.include_projection_matching = False
        elif has_phase and not has_complex:
            # Only phase projections available
            projection_type_for_cc = enums.ProjectionType.PHASE
            projections_for_viewer = self.task.phase_projections
        elif has_phase and has_complex:
            # Both available - default to phase for viewer, None for CC (will auto-detect)
            projection_type_for_cc = None
            projections_for_viewer = self.task.phase_projections
        else:
            # Neither available
            projection_type_for_cc = None
            projections_for_viewer = None

        # Create ProjectionViewer with the appropriate projections
        if projections_for_viewer is not None:
            self.projection_viewer = ProjectionViewer(
                projections=projections_for_viewer,
                include_options=True,
                include_shifts=True,
                display_only=False,
                include_array_saving_widget=True,
            )
            # Pass projection_viewer to PMAMasterWidget for updating
            projection_viewer_for_pma = self.projection_viewer
        else:
            # If no projections available, create a placeholder
            placeholder = QWidget()
            layout = QVBoxLayout(placeholder)
            label = QLabel("No projections available.\nPlease load or unwrap phase first.")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self.projection_viewer = placeholder
            # Don't pass the placeholder to PMAMasterWidget
            projection_viewer_for_pma = None

        # Create PMAMasterWidget with the full task (if requested)
        if self.include_projection_matching:
            self.pma_widget = PMAMasterWidget(
                task=self.task,
                projection_viewer=projection_viewer_for_pma,
                list_of_updated_settings=updated_settings_for_pma_widget,
            )

            # Connect shift operations in ProjectionViewer to clear PMA results
            if hasattr(self.projection_viewer, 'all_shifts_viewer') and self.projection_viewer.all_shifts_viewer is not None:
                # Connect the apply and undo shift buttons to clear PMA results
                self.projection_viewer.all_shifts_viewer.shift_operation_performed.connect(
                    self.pma_widget.clear_alignment_results
                )
        else:
            self.pma_widget = None

        # Create CrossCorrelationMasterWidget with the full task (if requested)
        if self.include_cross_correlation:
            if projection_type_for_cc is not None:
                self.cc_widget = CrossCorrelationMasterWidget(
                    task=self.task,
                    projection_type=projection_type_for_cc,
                    projection_viewer=projection_viewer_for_pma,
                )
            else:
                self.cc_widget = CrossCorrelationMasterWidget(
                    task=self.task,
                    projection_viewer=projection_viewer_for_pma,
                )

            # Connect shift operations in ProjectionViewer to clear CC results
            if hasattr(self.projection_viewer, 'all_shifts_viewer') and self.projection_viewer.all_shifts_viewer is not None:
                # Connect the apply and undo shift buttons to clear CC results
                self.projection_viewer.all_shifts_viewer.shift_operation_performed.connect(
                    self.cc_widget.clear_alignment_results
                )
        else:
            self.cc_widget = None

        # Add pages to the sidebar navigator
        # You can use standard Qt icons or provide custom icon paths
        # For now, using empty icons (text will still show)
        self.addPage(
            page_widget=self.projection_viewer,
            title="Projection Viewer",
            icon=self._get_icon("view")  # Could use a view/eye icon
        )

        if self.include_projection_matching:
            self.addPage(
                page_widget=self.pma_widget,
                title="PMA Runner",
                icon=self._get_icon("align")  # Could use an alignment/settings icon
            )

        if self.include_cross_correlation:
            self.addPage(
                page_widget=self.cc_widget,
                title="Cross-Correlation Runner",
                icon=self._get_icon("align")  # Could use an alignment/settings icon
            )

    def _get_icon(self, icon_name: str) -> QIcon:
        """
        Get a QIcon based on the icon name.

        This method can be extended to use custom icons or Qt standard icons.
        Currently returns Qt standard icons for common use cases.

        Parameters
        ----------
        icon_name : str
            Name of the icon to retrieve ("view", "align", etc.)

        Returns
        -------
        QIcon
            The requested icon, or an empty icon if not found
        """
        # Using Qt's standard pixmaps/icons
        style = QApplication.style()

        icon_map = {
            # "view": QStyle.SP_FileDialogDetailedView,
            "view": QStyle.SP_ComputerIcon,
            "align": QStyle.SP_FileDialogContentsView,
            "settings": QStyle.SP_FileDialogInfoView,
            "file": QStyle.SP_FileIcon,
            "folder": QStyle.SP_DirIcon,
        }

        if icon_name in icon_map:
            return style.standardIcon(icon_map[icon_name])
        else:
            return QIcon()  # Return empty icon if not found


@switch_to_matplotlib_qt_backend
def launch_combined_alignment_widget(
    task: "t.LaminographyAlignmentTask",
    updated_settings_for_pma_widget: Optional[list[dict]] = None,
    include_projection_matching: bool = True,
    include_cross_correlation: bool = True,
    wait_until_closed: bool = False,
) -> CombinedAlignmentWidget:
    """Launch the combined alignment widget GUI.

    This GUI provides a unified interface for both viewing projections and
    running projection matching alignment. The sidebar allows switching between:
    - Projection Viewer: Interactive projection visualization with tools
    - PMA Runner: Projection matching alignment workflow interface
    - Cross-Correlation Runner: Cross-correlation alignment workflow interface

    Args:
        task: LaminographyAlignmentTask containing projections and configuration.
        updated_settings_for_pma_widget: Updated settings for the PMA widget.
        include_projection_matching: Whether to include the PMA Runner in the sidebar.
        include_cross_correlation: Whether to include the Cross-Correlation Runner in the sidebar.
        wait_until_closed: If True, the application starts a blocking call
            until the GUI window is closed.

    Returns:
        The CombinedAlignmentWidget instance.

    Example:
        Launch the combined alignment GUI::

            gui = pyxalign.gui.launch_combined_alignment_widget(task)
    """
    app = QApplication.instance() or QApplication([])
    gui = CombinedAlignmentWidget(
        task=task,
        updated_settings_for_pma_widget=updated_settings_for_pma_widget,
        include_projection_matching=include_projection_matching,
        include_cross_correlation=include_cross_correlation,
    )
    gui.show()
    if wait_until_closed:
        app.exec_()
    return gui


def main():
    """
    Example main function to test the CombinedAlignmentWidget.
    """
    import sys
    from pyxalign.data_structures.task import LaminographyAlignmentTask

    app = QApplication(sys.argv)

    # For testing, create a minimal task (this would need actual data)
    task = LaminographyAlignmentTask()

    # Create and show the widget
    widget = launch_combined_alignment_widget(task)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
