# 🌍 QeMLflow Environment Setup Guide

**Complete guide for setting up a clean development environment**

*Version 1.0 | Created: June 20, 2025*

---

## 🎯 **Quick Start**

```bash
# 1. Clone and navigate to repository
git clone <repository-url>
cd QeMLflow

# 2. Run automated setup
make setup-dev

# 3. Activate environment and verify
source venv/bin/activate  # On Unix/macOS
# OR
venv\Scripts\activate     # On Windows
python -c "import qemlflow; print('✅ QeMLflow installed successfully!')"
```

---

## 🔧 **Manual Setup (Detailed)**

### **Step 1: Python Version Check**
```bash
python --version  # Should be 3.8, 3.9, 3.10, or 3.11
```

### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation
which python  # Should point to venv/bin/python
```

### **Step 3: Upgrade pip and core tools**
```bash
pip install --upgrade pip setuptools wheel
```

### **Step 4: Install QeMLflow**

#### **Option A: Development Installation (Recommended)**
```bash
# Install in development mode with all features
pip install -e ".[dev,docs,quantum,molecular]"
```

#### **Option B: Core Installation Only**
```bash
# Install minimal core dependencies
pip install -r requirements-core.txt
pip install -e .
```

#### **Option C: Full Installation**
```bash
# Install all dependencies (heavy)
pip install -r requirements.txt
pip install -e .
```

### **Step 5: Install Pre-commit Hooks**
```bash
pre-commit install
```

### **Step 6: Verify Installation**
```bash
# Run quick test
python -c "import qemlflow; print('✅ Installation successful!')"

# Run test suite (optional)
pytest tests/ -v --tb=short -x
```

---

## 📦 **Requirements Files Overview**

| File | Purpose | Size |
|------|---------|------|
| `requirements.txt` | Full feature set | ~134 packages |
| `requirements-core.txt` | Essential features only | ~37 packages |
| `requirements-minimal.txt` | CI/testing minimal | ~27 packages |
| `pyproject.toml` | Package definition | Core + optional |

### **When to Use Each:**

- **`requirements-core.txt`**: New users, basic workflows
- **`requirements.txt`**: Full development, research work
- **`requirements-minimal.txt`**: CI/CD, automated testing
- **`pyproject.toml`**: Package installation (preferred)

---

## 🚀 **Advanced Setup Options**

### **Conda Environment (Alternative)**
```bash
# Create conda environment
conda create -n qemlflow python=3.10
conda activate qemlflow

# Install QeMLflow
pip install -e ".[dev,docs,quantum,molecular]"
```

### **Docker Development (Isolated)**
```bash
# Build development container
docker build -t qemlflow-dev .

# Run interactive development session
docker run -it --rm -v $(pwd):/workspace qemlflow-dev bash
```

### **Professional Development Setup**
```bash
# Full development environment with all tools
make setup-dev

# This includes:
# - Virtual environment creation
# - Development dependencies
# - Pre-commit hooks
# - Code quality tools
# - Documentation tools
```

---

## 🔍 **Environment Validation**

### **Quick Health Check**
```bash
# Run environment diagnostics
python scripts/validate_environment.py
```

### **Comprehensive Testing**
```bash
# Test all components
make test

# Test specific components
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-fast           # Skip slow tests
```

### **Performance Validation**
```bash
# Check import times (should be <5s)
time python -c "import qemlflow"

# Check memory usage
python -c "import qemlflow; import psutil; print(f'Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB')"
```

---

## 🛠️ **Development Workflow**

### **Daily Development**
```bash
# Activate environment
source venv/bin/activate

# Update dependencies (weekly)
pip install --upgrade -r requirements-core.txt

# Run pre-commit before pushing
pre-commit run --all-files

# Run tests
make test-fast
```

### **Code Quality Checks**
```bash
# Format code
make format

# Check linting
make lint

# Type checking
make type-check

# Security audit
make security
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Reinstall in development mode
pip install -e .

# Check PYTHONPATH
echo $PYTHONPATH
```

#### **Permission Errors**
```bash
# Use user installation
pip install --user -e .

# Or fix permissions
sudo chown -R $USER:$USER venv/
```

#### **Dependency Conflicts**
```bash
# Start fresh
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

#### **Slow Installation**
```bash
# Use faster package manager
pip install --no-cache-dir -e .

# Or use pip-tools for faster resolution
pip install pip-tools
pip-compile requirements.in
pip-sync
```

### **Platform-Specific Issues**

#### **macOS**
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew dependencies
brew install git python
```

#### **Windows**
```bash
# Use Windows Subsystem for Linux (WSL) for best experience
# Or install Visual Studio Build Tools
```

#### **Linux**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-venv build-essential
```

---

## 📊 **Environment Monitoring**

### **Resource Usage**
```bash
# Monitor memory and CPU
htop

# Monitor GPU usage (if available)
nvidia-smi
```

### **Package Audit**
```bash
# Check for security vulnerabilities
pip audit

# Check for outdated packages
pip list --outdated
```

---

## 🔄 **Environment Maintenance**

### **Weekly Maintenance**
```bash
# Update core dependencies
pip install --upgrade pip setuptools wheel

# Update development tools
pip install --upgrade black isort flake8 mypy pytest
```

### **Monthly Maintenance**
```bash
# Full dependency update
pip install --upgrade -r requirements.txt

# Audit and clean
pip audit
pip autoremove  # If available
```

### **Environment Recreation**
```bash
# Save current state
pip freeze > environment_backup.txt

# Recreate environment
deactivate
rm -rf venv/
make setup-dev
```

---

## ✅ **Success Checklist**

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] QeMLflow installed in development mode
- [ ] Pre-commit hooks installed
- [ ] All tests passing
- [ ] Import time < 5 seconds
- [ ] Code quality tools working
- [ ] Documentation builds successfully

---

*For questions or issues, check the [troubleshooting section](#-troubleshooting) or open an issue on GitHub.*
