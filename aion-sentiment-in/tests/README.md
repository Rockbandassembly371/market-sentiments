# AION Tests

This directory contains unit tests, integration tests, and test utilities for the
AION Sentiment Analysis project.

## Test Structure

```
tests/
├── __init__.py           # Test package initialization
├── conftest.py           # Pytest fixtures and configuration
├── test_prepare_data.py  # Tests for data preparation module
└── README.md             # This file
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/aion_sentiment

# Run specific test file
pytest tests/test_prepare_data.py

# Run with verbose output
pytest tests/ -v

# Run specific test function
pytest tests/test_prepare_data.py::test_clean_headline
```

## Writing Tests

Follow these guidelines when adding tests:

1. **Naming**: Use `test_*.py` for test files, `test_*` for test functions
2. **Fixtures**: Use `conftest.py` for shared fixtures
3. **Assertions**: Use descriptive assertion messages
4. **Coverage**: Aim for >80% code coverage
5. **Independence**: Tests should be independent and order-agnostic

## Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Scheduled CI runs

---

**AION Open-Source Project** | Copyright 2026 AION Contributors
