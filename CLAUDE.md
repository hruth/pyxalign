# Coding Preferences and Guidelines for Pyxalign

This document contains coding preferences and guidelines for Claude Code to follow when working on this project.

## File Organization

### Documentation Structure
- Don't create documentation files in the project root

### Code Organization
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
- GUIs that are intended to be used in a stand-alone way (at least some of the time) should have a corresponding "launch" method that users can access when scripting with pyxalign. This "launch" method should be added to `src/pyxalign/gui/__init__.py`.

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

# Claude Code Documentation Directory

This directory contains documentation and resources for Claude Code sessions working on the Pyxalign project.

## Purpose

This directory serves as a knowledge base for Claude Code to:
- Understand project-specific coding conventions
- Learn architectural patterns and preferences
- Access implementation notes and design decisions
- Find test scripts and examples
- Maintain consistency across sessions

## Key Files

### CODING_PREFERENCES.md
The primary reference for how code should be written in this project. Covers:
- File organization patterns
- Code style guidelines
- GUI development preferences
- Architecture patterns
- Version control practices

Claude Code should read this file at the start of each session to understand project conventions.

### implementation notes
Put implementation notes for all changes in `implementation_notes/` 
Design documents and change summaries for significant features or refactorings. Each note should:
- Explain the problem being solved
- Describe the solution approach
- Document API changes
- Provide migration guidance
- Suggest future enhancements

### tests
Test scripts that verify functionality. These are:
- Standalone scripts that can run independently
- Quick verification tools for development
- Examples of how to use new features
- Not part of the main test suite
- Put test scripts in `claude_docs/tests/`

## Maintenance

- Keep documentation up-to-date with code changes
- Remove obsolete test scripts
- Don't commit temporary or session-specific files

## Contributing

When adding documentation:
- Use clear, descriptive filenames
- Include dates in implementation notes
- Provide context and rationale, not just facts
- Link to related code files
- Keep documentation focused and concise

# manual additions
- If the says something along the lines of "in general, you should ___" that means you should make a note in the CODING_PREFERENCES.md file with this preference, if it is not already there.
- You should ask the user for confirmation about any additions to the CODING_PREFERENCES.md file
- In the work you are doing, if you find things in the repository that suggest there is a "code smell", ask the user if they would like a prompt to use to look into the issue more in a later session. If they say yes, update the document claude_docs/to_do/CODE_SMELLS_TO_ADDRESS.md with a short description of the issue and a prompt to start a claude code session where you will obtain (1) a more detailed characterization of the issue/code smell, (2) an explanation of why this is a code smell with examples, and (3) suggestions for how to fix it.
- You should generally keep an eye out for ways to make this repo more maintainable, easy to understand, and more like it has been written by an experienced software developer.
- If anything in this file doesn't make sense and/or is contradictory, notify the user and work with them to update this file appropriately.