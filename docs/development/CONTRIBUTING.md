# Contributing to QeMLflow

Thank you for your interest in contributing to QeMLflow! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/QeMLflow.git
   cd QeMLflow
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv qemlflow_dev
   source qemlflow_dev/bin/activate  # On Windows: qemlflow_dev\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Run linting and formatting**:
   ```bash
   black src/
   isort src/
   flake8 src/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   ```

6. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Contribution Types

### Bug Reports
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include minimal reproduction example
- Provide environment details

### Feature Requests
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the use case and benefits
- Consider implementation complexity

### Code Contributions
- Follow the development workflow above
- Include tests for new functionality
- Update documentation as needed
- Ensure CI/CD passes

### Documentation
- Improve existing docs or add new content
- Use clear, concise language
- Include code examples where appropriate

## Coding Standards

### Python Style
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 88 characters

### Code Quality
- Write clear, self-documenting code
- Include docstrings for all public functions/classes
- Use type hints where appropriate
- Follow existing patterns in the codebase

### Testing
- Write tests for all new functionality
- Aim for high test coverage
- Use descriptive test names
- Include edge cases and error conditions

### Documentation
- Use Google-style docstrings
- Include examples in docstrings
- Update relevant documentation files
- Ensure documentation builds without errors

## Project Structure

```
QeMLflow/
├── src/qemlflow/           # Main package source
│   ├── core/            # Core functionality
│   ├── models/          # ML models
│   ├── preprocessing/   # Data preprocessing
│   ├── quantum/         # Quantum computing
│   └── utils/           # Utilities
├── tests/               # Test suite
├── docs/                # Documentation source
├── examples/            # Usage examples
├── notebooks/           # Jupyter notebooks
└── scripts/             # Development scripts
```

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat(models): add AutoML ensemble method
fix(preprocessing): handle missing values in molecular descriptors
docs(api): update AutoMLRegressor documentation
```

## Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src/qemlflow

# Run integration tests
pytest tests/integration/
```

### Writing Tests
- Place tests in appropriate `tests/` subdirectories
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Mock external dependencies when appropriate

### Test Structure
```python
def test_feature_description():
    """Test that feature works correctly with valid input."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result.is_valid()
    assert len(result) == expected_length
```

## Documentation Guidelines

### Building Documentation
```bash
# Install doc dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

### Writing Documentation
- Use clear, concise language
- Include practical examples
- Organize content logically
- Cross-reference related topics

## Release Process

Releases are automated through our CI/CD pipeline:

1. **Version bumping**: Update version in `pyproject.toml`
2. **Create release tag**: `git tag v0.x.y`
3. **Push tag**: `git push origin v0.x.y`
4. **Automated pipeline**: Handles testing, building, and publishing

## Getting Help

- **GitHub Discussions**: [Ask questions and discuss ideas](https://github.com/SanjeevaRDodlapati/QeMLflow/discussions)
- **Issues**: [Report bugs or request features](https://github.com/SanjeevaRDodlapati/QeMLflow/issues)
- **Documentation**: Check our [comprehensive docs](https://sanjeevardodlapati.github.io/QeMLflow/)

## Recognition

Contributors are recognized in:
- GitHub contributors list
- Release notes
- Documentation acknowledgments

Thank you for contributing to QeMLflow! 🎉
