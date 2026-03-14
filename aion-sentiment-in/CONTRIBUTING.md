# Contributing to AION Sentiment Analysis

Thank you for your interest in contributing to the AION Sentiment Analysis project! This document provides guidelines and instructions for contributing to this open-source project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Attribution](#attribution)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming and inclusive community.

## Getting Started

### Ways to Contribute

We welcome various types of contributions:

- **Bug Reports**: Report bugs through GitHub issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Fix bugs or implement new features
- **Documentation**: Improve documentation, examples, or tutorials
- **Testing**: Add test cases or improve test coverage
- **Community**: Help others in discussions and issues

### First Contributions

If you're new to open source, check out these resources:
- [First Contributions Guide](https://github.com/firstcontributions/first-contributions)
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- pip or conda

### Installation

1. **Fork the repository**:
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/aion-sentiment-in.git
   cd aion-sentiment-in
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify installation**:
   ```bash
   pytest
   ```

## Code Style

### Python Style Guide

We follow these coding standards:

- **PEP 8**: Python style guide
- **Black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **Google Style**: Docstrings

### Formatting

Run Black to format your code:

```bash
black src/ tests/ demo.py
```

### Linting

Run flake8 to check for style violations:

```bash
flake8 src/ tests/
```

### Type Checking

Run mypy to verify type hints:

```bash
mypy src/
```

### Pre-commit Hooks

We use pre-commit hooks to automatically check code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Code Example

```python
# =============================================================================
# AION Open-Source Project: [Component Name]
# File: [file_path.py]
# Description: [Brief description]
# License: Apache License, Version 2.0
#
# Copyright 2026 AION Contributors
# =============================================================================
"""
Module docstring with description of purpose.

Example:
    >>> from aion_sentiment import AIONSentimentIN
    >>> model = AIONSentimentIN()
    >>> result = model.predict("Text")

"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ExampleClass:
    """
    Class docstring with description.

    Attributes:
        name (str): Description of attribute.
        value (int): Description of attribute.

    Example:
        >>> obj = ExampleClass("test", 42)
        >>> print(obj.name)
        'test'

    """

    def __init__(self, name: str, value: int) -> None:
        """
        Initialize ExampleClass.

        Args:
            name: Description of name parameter.
            value: Description of value parameter.

        Raises:
            ValueError: When value is negative.

        """
        if value < 0:
            raise ValueError("Value must be non-negative")
        self.name = name
        self.value = value

    def method(self, text: str) -> Dict[str, float]:
        """
        Method docstring.

        Args:
            text: Input text to process.

        Returns:
            Dictionary with processed results.

        """
        return {"result": 1.0}
```

## Pull Request Process

### Before Submitting

1. **Create an issue** (for significant changes):
   - Describe the problem or feature
   - Discuss the proposed solution
   - Wait for maintainer feedback

2. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-123
   ```

3. **Make changes**:
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation
   - Include license headers

4. **Run tests**:
   ```bash
   pytest --cov=aion_sentiment
   ```

5. **Check code quality**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   pre-commit run --all-files
   ```

### Submitting

1. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation
   - `style:` Formatting
   - `refactor:` Code refactoring
   - `test:` Tests
   - `chore:` Maintenance

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**:
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

### PR Template

Include in your PR description:

```markdown
## Description
Brief description of changes.

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Code coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] License headers included
- [ ] No new warnings
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, PR will be merged
4. Celebrate your contribution! 🎉

## Issue Reporting Guidelines

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for answers
3. **Update to the latest version** to see if it's already fixed

### Bug Reports

Use the bug report template and include:

```markdown
**Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Import '...'
2. Call function '...'
3. See error

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- Python version: 3.x.x
- Package version: 0.1.0
- OS: [e.g., macOS, Linux, Windows]

**Additional Context**
Screenshots, logs, or other context.
```

### Feature Requests

```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
Describe the desired feature.

**Alternatives Considered**
Other solutions you've thought about.

**Use Case**
How would this be used?

**Additional Context**
Any other relevant information.
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aion_sentiment --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test function
pytest tests/test_model.py::test_predict
```

### Writing Tests

- Use pytest framework
- Test both success and failure cases
- Include edge cases
- Aim for high code coverage
- Mock external dependencies

```python
"""Tests for AIONSentimentIN model."""

import pytest
from aion_sentiment import AIONSentimentIN


class TestAIONSentimentIN:
    """Test cases for AIONSentimentIN class."""

    def test_predict_positive(self):
        """Test positive sentiment prediction."""
        model = AIONSentimentIN()
        result = model.predict("Stock market soars")
        assert result["sentiment_label"] == "positive"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_empty_text(self):
        """Test prediction with empty text raises error."""
        model = AIONSentimentIN()
        with pytest.raises(ValueError):
            model.predict("")

    def test_predict_batch(self):
        """Test batch prediction."""
        model = AIONSentimentIN()
        texts = ["Good news", "Bad news"]
        results = model.predict_batch(texts)
        assert len(results) == 2
        assert all("sentiment_label" in r for r in results)
```

## Documentation

### Code Documentation

- Include docstrings for all public modules, classes, and functions
- Use Google-style docstrings
- Add type hints to all functions
- Include examples in docstrings

### Documentation Updates

When adding features, update:

- README.md (if user-facing changes)
- API reference in docstrings
- Demo notebook (demo.py)
- Inline code comments (for complex logic)

## Attribution

When contributing, you agree that your contributions will be licensed under the Apache License, Version 2.0.

Include the license header in all new files:

```python
# =============================================================================
# AION Open-Source Project: [Project Name]
# File: [file_name.py]
# Description: [Brief description]
# License: Apache License, Version 2.0
#
# Copyright 2026 AION Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AION Attribution: This file is part of the AION open-source ecosystem.
# Please visit https://github.com/aion for more information.
# =============================================================================
```

## Questions?

If you have questions about contributing:

1. Check existing [issues](https://github.com/aion/aion-sentiment-in/issues)
2. Start a [discussion](https://github.com/aion/aion-sentiment-in/discussions)
3. Contact: contributors@aion.io

Thank you for contributing to AION Sentiment Analysis! 🎉
