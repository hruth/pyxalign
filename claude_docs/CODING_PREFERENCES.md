# Coding Preferences and Guidelines for PyXAlign

This document contains coding preferences and guidelines for Claude Code to follow when working on this project.

## File Organization

### Documentation Structure
- **claude_docs/**: Directory for Claude Code-related documentation
  - **implementation_notes/**: Design documents, change logs, architecture notes
  - **tests/**: Test scripts and test documentation
  - Keep generated documentation (like summaries of changes) in appropriate subdirectories
  - Don't create documentation files in the project root

### Code Organization
- When refactoring, preserve old versions in versioned folders (e.g., `module_v1/`, `module_v2/`)
- Use `__init__.py` files to maintain clean import paths
- Prefer absolute imports over relative imports in src code

## Code Style

### Python Conventions
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Use descriptive variable names (no single-letter variables except in comprehensions/loops)
- Keep functions focused and single-purpose

### PyQt5 GUI Code
- Separate UI setup into dedicated methods (e.g., `setup_*_layout()`, `setup_*_buttons()`)
- Connect signals in dedicated `_connect_*_signals()` methods
- Use descriptive widget names that indicate their purpose
- Group related widgets into containers for better organization
- Use StyleSheets for consistent visual styling

### Documentation
- Use docstrings for all public classes and methods (Google style)
- Include parameter descriptions and return types in docstrings
- Add inline comments for complex logic, but prefer self-documenting code
- Keep comments up-to-date with code changes

## Architecture Patterns

### Widget Design
- Separate concerns: data management vs. UI presentation
- Use signals and slots for communication between widgets
- Emit signals for parent widgets to handle rather than directly manipulating parent state
- Provide both list and dict APIs for data structures when appropriate

### Options and Settings
- Use dataclasses for options structures
- Maintain API compatibility when creating new versions
- Provide migration paths for old configurations
- Support both programmatic and file-based configuration

### Testing
- Place test scripts in `claude_docs/tests/`
- Write standalone test scripts that can run without the full test suite
- Include verification assertions in tests
- Test both success and failure cases

## Version Control

### Git Workflow
- Use `git mv` when moving/renaming files to preserve history
- Stage refactoring moves separately from content changes
- Write clear commit messages that explain the "why" not just the "what"
- Group related changes into logical commits

### Backward Compatibility
- Preserve old implementations when creating new versions
- Update imports to use new versions but keep old code accessible
- Document migration paths in change notes
- Consider providing compatibility shims for major API changes

## GUI Development Preferences

### Visual Design
- Use visual grouping (frames, boxes) to show relationships between parameters
- Provide clear visual feedback for user actions
- Use colors meaningfully (green for add/success, red for remove/danger)
- Include helpful tooltips and labels
- Ensure consistent spacing and alignment

### User Experience
- Prevent accidental data loss (e.g., warn before removing the last item)
- Provide multiple ways to accomplish common tasks (buttons + keyboard shortcuts)
- Show information hierarchically (most important info first)
- Use progressive disclosure (basic options visible, advanced options hidden by default)

### Control Patterns
- Use combo boxes with categorized sections (Basic/Advanced)
- Provide insert above/below options for list items
- Include duplicate functionality for easy item copying
- Use scroll areas for variable-length content
- Disable mouse wheel on combo boxes to prevent accidental changes

## Project-Specific Patterns

### Options Sequencing
- Each sequence block should represent one complete operation
- Group related parameters together in blocks
- Make block boundaries visually obvious
- Provide both block-level and parameter-level operations

### Alignment Workflows
- Preserve intermediate results for comparison
- Track parameter changes that produced each result
- Allow using previous results as input for next runs
- Support both interactive and automated workflows

## Communication

### Code Comments
- Don't use emojis in code unless explicitly requested
- Keep professional tone in comments
- Focus on "why" not "what" (code shows what)
- Update comments when code changes

### User Interaction
- Provide clear, concise status messages
- Show progress for long operations
- Give actionable error messages
- Confirm destructive operations

## Performance Considerations

### GUI Performance
- Use lazy loading for large lists
- Implement proper cleanup in deleteLater() calls
- Avoid blocking the main thread for computations
- Use background threads for heavy processing

### Memory Management
- Clean up widgets when removing from layout
- Use deep copies when needed to avoid reference issues
- Clear large data structures when no longer needed
- Be mindful of GPU memory in CUDA operations

## Dependencies

### Import Organization
```python
# Standard library imports
import sys
import copy
from typing import Optional, List, Dict

# Third-party imports
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal

# Local imports
from pyxalign.api.options import BaseOptions
from pyxalign.interactions.custom import style_sheet
```

### Version Compatibility
- Maintain compatibility with current PyQt5 version
- Use NumPy/CuPy patterns consistently
- Test with Python 3.8+ features only

## Future Claude Code Sessions

When starting a new Claude Code session:
1. Read this file to understand project preferences
2. Check `claude_docs/implementation_notes/` for recent changes
3. Follow established patterns in the codebase
4. Ask for clarification when preferences conflict with requirements
5. Update this document if new patterns emerge

## Questions and Clarifications

When uncertain about:
- **Architecture decisions**: Ask before implementing major changes
- **Breaking changes**: Always discuss impact on existing code
- **New dependencies**: Verify necessity and compatibility
- **UI/UX choices**: Present options with trade-offs

## Anti-Patterns to Avoid

- Don't create documentation in the project root
- Don't break existing APIs without migration path
- Don't add features that weren't requested
- Don't over-engineer simple solutions
- Don't use abbreviations in variable names
- Don't leave TODO comments without tracking
- Don't commit commented-out code
- Don't create files without proper organization
