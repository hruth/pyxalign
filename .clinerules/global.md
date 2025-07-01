# Global Cline Rules for pyxalign project

## Code Style Guidelines

### Line Length
- Maximum 100 characters per line
- Break long lines at logical points (after commas, before operators)

### Variable Naming
- Use descriptive variable names that clearly communicate purpose
- Prefer longer, clear names over short, cryptic ones
- Exception: common loop variables (i, j, x, y) and mathematical variables are acceptable
- Examples:
  - Good: `projection_initializer_widget`, `alignment_results_list`
  - Avoid: `proj_init`, `res_list` (unless context makes meaning obvious)

### Import Organization
Group imports in this order with blank lines between groups:
1. Standard library imports (sorted alphabetically)
2. Third-party library imports (sorted alphabetically)
3. Local application imports (sorted alphabetically)

Example:
```python
import sys
from dataclasses import fields, is_dataclass
from typing import Optional, Union

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from pyxalign.api.options import ProjectionOptions
from pyxalign.interactions.base import BaseWidget
```

### Type Hints
- Use type hints for all function parameters and return values
- Use `Optional[Type]` for optional parameters
- Import types from `typing` module when needed

### Documentation
- All modules should have a module-level docstring
- All classes should have a class-level docstring
- Complex methods should have docstrings explaining purpose, parameters, and return values
- Use Google-style docstrings

### Comments
- Use comments to explain complex logic, not obvious code
- Prefer explanatory variable names over comments when possible
- Use inline comments sparingly, prefer block comments above the code

### Error Handling
- Use specific exception types when possible
- Include meaningful error messages
- Use try-except blocks for expected failures
- Log errors appropriately for debugging
