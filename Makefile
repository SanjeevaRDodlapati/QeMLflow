# ChemML Development Makefile
# Provides convenient commands for development workflow

.PHONY: help install install-dev test test-fast test-coverage lint format type-check security clean build docs serve-docs pre-commit setup-dev setup-env clean-env recreate-env validate-env

# Default target
help:
	@echo "ChemML Development Commands:"
	@echo "=========================="
	@echo "Environment Setup:"
	@echo "  setup-env          - Create virtual environment only"
	@echo "  setup-dev          - Complete development environment setup"
	@echo "  clean-env          - Remove virtual environment"
	@echo "  recreate-env       - Clean and recreate environment"
	@echo "  validate-env       - Validate environment setup"
	@echo ""
	@echo "Installation:"
	@echo "  install            - Install package in development mode"
	@echo "  install-dev        - Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test               - Run all tests"
	@echo "  test-fast          - Run tests excluding slow ones"
	@echo "  test-coverage      - Run tests with coverage report"
	@echo "  test-unit          - Run only unit tests"
	@echo "  test-integration   - Run only integration tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint               - Run all linting tools"
	@echo "  format             - Format code with black and isort"
	@echo "  type-check         - Run type checking with mypy"
	@echo "  security           - Run security checks with bandit"
	@echo "  pre-commit         - Run pre-commit hooks on all files"
	@echo ""
	@echo "Documentation:"
	@echo "  docs               - Build documentation"
	@echo "  serve-docs         - Serve documentation locally"
	@echo ""
	@echo "Utilities:"
	@echo "  clean              - Clean build artifacts and caches"
	@echo "  build              - Build package distributions"
	@echo "  bootcamp-test      - Test bootcamp notebooks"

# Environment variables
PYTHON := python3
PIP := pip3
PYTEST := python -m pytest
COVERAGE := coverage
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit
PRECOMMIT := pre-commit

# Setup and Installation
setup-env:
	@echo "🌍 Setting up QeMLflow development environment..."
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
	fi
	@echo "Upgrading pip and core tools..."
	@venv/bin/pip install --upgrade pip setuptools wheel
	@echo "✅ Virtual environment ready!"
	@echo "To activate: source venv/bin/activate"

setup-dev: setup-env
	@echo "Installing development dependencies..."
	@venv/bin/pip install -e ".[dev,docs,quantum,molecular]"
	@echo "Setting up pre-commit hooks..."
	@venv/bin/pre-commit install
	@echo "✅ Development environment setup complete!"
	@echo ""
	@echo "🚀 Next steps:"
	@echo "  1. Activate environment: source venv/bin/activate"
	@echo "  2. Verify installation: python -c 'import qemlflow; print(\"✅ Success!\")'"
	@echo "  3. Run tests: make test-fast"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,docs,quantum,molecular]"
	@echo "✅ Development dependencies installed"

clean-env:
	@echo "🧹 Cleaning up virtual environment..."
	@rm -rf venv/
	@echo "✅ Virtual environment removed"

recreate-env: clean-env setup-dev
	@echo "✅ Virtual environment recreated!"

validate-env:
	@echo "🔍 Validating development environment..."
	@python scripts/validate_environment.py

# Testing
test:
	$(PYTEST) tests/ -v --tb=short

test-fast:
	$(PYTEST) tests/ -v --tb=short -m "not slow"

test-coverage:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "📊 Coverage report generated in htmlcov/"

test-unit:
	$(PYTEST) tests/unit/ -v --tb=short

test-integration:
	$(PYTEST) tests/integration/ -v --tb=short

test-performance:
	$(PYTEST) tests/ -v --tb=short -m "slow"

# Code Quality
lint: flake8 bandit pydocstyle
	@echo "✅ All linting checks passed"

flake8:
	$(FLAKE8) src/ tests/
	@echo "✅ Flake8 linting passed"

format:
	$(BLACK) src/ tests/ --line-length=88
	$(ISORT) src/ tests/ --profile=black
	@echo "✅ Code formatting applied"

format-check:
	$(BLACK) src/ tests/ --check --line-length=88
	$(ISORT) src/ tests/ --check-only --profile=black

type-check:
	$(MYPY) src/ --config-file=pyproject.toml
	@echo "✅ Type checking passed"

security:
	$(BANDIT) -r src/ -f json -o bandit-report.json
	@echo "✅ Security scan completed (see bandit-report.json)"

pydocstyle:
	pydocstyle src/ --convention=numpy
	@echo "✅ Documentation style check passed"

pre-commit:
	$(PRECOMMIT) run --all-files
	@echo "✅ Pre-commit hooks completed"

# Pre-commit setup
install-pre-commit:
	$(PRECOMMIT) install
	@echo "✅ Pre-commit hooks installed"

# Documentation
docs:
	cd docs && make html
	@echo "📚 Documentation built in docs/_build/html/"

serve-docs:
	cd docs/_build/html && python -m http.server 8000
	@echo "📚 Documentation served at http://localhost:8000"

docs-clean:
	cd docs && make clean

# Build and Distribution
build: clean
	$(PYTHON) -m build
	@echo "📦 Package built in dist/"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf bandit-report.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "🧹 Cleaned build artifacts and caches"

# Bootcamp-specific commands
bootcamp-test:
	@echo "🔍 Checking for bootcamp tests..."
	@if [ -f "notebooks/quickstart_bootcamp/utils/test_assessment_integration.py" ]; then \
		$(PYTEST) notebooks/quickstart_bootcamp/utils/test_assessment_integration.py -v; \
	else \
		echo "⚠️ Bootcamp test file not found, testing basic notebook imports instead..."; \
		python -c "import os; print('📓 Testing notebook directory structure...'); \
		           assert os.path.exists('notebooks'), 'Notebooks directory missing'; \
		           assert os.path.exists('notebooks/examples'), 'Examples directory missing'; \
		           print('✅ Notebook structure check passed')"; \
	fi
	@echo "🎓 Bootcamp tests completed"

bootcamp-setup:
	cd notebooks/quickstart_bootcamp && ./scripts/bootcamp_maintenance.sh
	@echo "🎓 Bootcamp environment prepared"

# Development workflow shortcuts
dev-check: format-check lint type-check test-fast
	@echo "✅ Development checks completed"

ci-check: format-check lint type-check security test-coverage
	@echo "✅ CI checks completed"

# Data and notebook management
clean-notebooks:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/**/*.ipynb
	@echo "📓 Notebook outputs cleared"

update-deps:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,docs,quantum,molecular]"
	@echo "⬆️ Dependencies updated"

# Performance and profiling
profile-tests:
	$(PYTEST) tests/performance/ --profile
	@echo "📈 Test performance profiling completed"

benchmark:
	$(PYTHON) -m pytest tests/ -m "expensive" --benchmark-only
	@echo "📊 Benchmarking completed"

# Database and data management
setup-data:
	mkdir -p data/raw data/processed data/cache data/results
	@echo "📁 Data directories created"

# Docker commands (if Docker is available)
docker-build:
	docker build -t chemml:latest .
	@echo "🐳 Docker image built"

docker-test:
	docker run --rm chemml:latest make test
	@echo "🐳 Docker tests completed"

# Git hooks and workflow
git-setup:
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	@echo "🔧 Git hooks configured"

# Help for specific components
help-testing:
	@echo "Testing Commands Help:"
	@echo "====================="
	@echo "test           - Run all tests (unit + integration + performance)"
	@echo "test-fast      - Skip slow/expensive tests (for quick feedback)"
	@echo "test-coverage  - Generate HTML coverage report"
	@echo "test-unit      - Only unit tests (isolated component testing)"
	@echo "test-integration - Only integration tests (workflow testing)"

help-quality:
	@echo "Code Quality Commands Help:"
	@echo "=========================="
	@echo "format         - Auto-format code (black + isort)"
	@echo "format-check   - Check if code is properly formatted"
	@echo "lint           - Run all linting tools (flake8 + bandit + pydocstyle)"
	@echo "type-check     - Type checking with mypy"
	@echo "pre-commit     - Run all pre-commit hooks"

# Version management
version:
	@$(PYTHON) -c "import src; print(f'ChemML version: {src.__version__}')"

bump-version:
	@echo "Current version: $$(python -c 'import src; print(src.__version__)')"
	@read -p "Enter new version: " version && \
	sed -i.bak "s/version = \".*\"/version = \"$$version\"/" pyproject.toml && \
	echo "Version updated to $$version"
