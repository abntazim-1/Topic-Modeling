# Contributing

We welcome contributions to the Topic Modeling Project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes
5. Add tests for new functionality
6. Ensure all tests pass
7. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/topic-modeling-project.git
cd topic-modeling-project

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing

Run these before submitting:

```bash
black src/
flake8 src/
mypy src/
pytest tests/
```

## Documentation

When adding new features:

1. Update docstrings using Google style
2. Add examples to the documentation
3. Update the API reference if needed
4. Test that `mkdocs serve` works correctly

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Submit a clear description of your changes
5. Link any related issues

Thank you for contributing!
