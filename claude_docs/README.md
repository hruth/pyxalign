# Claude Code Documentation Directory

⚠️ **IMPORTANT FOR CLAUDE CODE**: Read `CODING_PREFERENCES.md` first to understand project conventions!

This directory contains documentation and resources for Claude Code sessions working on the PyXAlign project.

## Directory Structure

```
claude_docs/
├── README.md                    # This file
├── CODING_PREFERENCES.md        # Coding style and architecture preferences
├── implementation_notes/        # Design docs, change logs, architecture notes
│   └── SEQUENCER_V2_CHANGES.md # Example: Sequencer v2 implementation summary
└── tests/                       # Test scripts and testing documentation
    └── test_sequencer_v2.py    # Example: Sequencer v2 test script
```

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

### implementation_notes/
Design documents and change summaries for significant features or refactorings. Each note should:
- Explain the problem being solved
- Describe the solution approach
- Document API changes
- Provide migration guidance
- Suggest future enhancements

### tests/
Test scripts that verify functionality. These are:
- Standalone scripts that can run independently
- Quick verification tools for development
- Examples of how to use new features
- Not part of the main test suite

## Usage for Claude Code

1. **Starting a new session**: Read `CODING_PREFERENCES.md` first
2. **Understanding recent changes**: Check `implementation_notes/` for context
3. **Verifying implementations**: Use scripts in `tests/` for quick checks
4. **Creating documentation**: Add new docs to appropriate subdirectories
5. **Updating preferences**: Modify `CODING_PREFERENCES.md` as patterns evolve

## Usage for Developers

This directory is also useful for human developers to:
- Understand architectural decisions
- Learn about major refactorings
- Find examples of how to use features
- Share knowledge between team members

## Maintenance

- Keep documentation up-to-date with code changes
- Archive old implementation notes when features are stable
- Remove obsolete test scripts
- Update CODING_PREFERENCES.md when new patterns emerge
- Don't commit temporary or session-specific files

## Contributing

When adding documentation:
- Use clear, descriptive filenames
- Include dates in implementation notes
- Provide context and rationale, not just facts
- Link to related code files
- Keep documentation focused and concise
