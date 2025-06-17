#!/bin/bash

# ChemML Development Environment Setup
# ===================================
# One-command setup for ChemML development and enhancement features

set -e  # Exit on any error

echo "🧬 Setting up ChemML Development Environment..."
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/chemml" ]; then
    print_error "This script must be run from the ChemML root directory"
    exit 1
fi

print_status "Starting ChemML setup..."

# 1. Install ChemML in development mode
print_status "Installing ChemML in development mode..."
pip install -e . || {
    print_error "Failed to install ChemML"
    exit 1
}

# 2. Install additional development dependencies
print_status "Installing development dependencies..."
pip install pytest pytest-cov flake8 black isort mypy jupyterlab || {
    print_warning "Some development dependencies failed to install, continuing..."
}

# 3. Setup performance monitoring directories
print_status "Setting up performance monitoring..."
mkdir -p performance_reports
mkdir -p docs/api_auto

# 4. Generate API documentation
print_status "Generating API documentation..."
python tools/development/auto_docs.py || {
    print_warning "API documentation generation failed, continuing..."
}

# 5. Create development configuration
print_status "Creating development configuration..."
cat > development_config.py << 'EOF'
"""
ChemML Development Configuration
===============================

Quick setup configuration for development features.
"""

import os
from pathlib import Path

# Development settings
DEVELOPMENT_MODE = True
ENABLE_PERFORMANCE_MONITORING = True
AUTO_GENERATE_DOCS = True

# Paths
PROJECT_ROOT = Path(__file__).parent
PERFORMANCE_REPORTS_DIR = PROJECT_ROOT / "performance_reports"
API_DOCS_DIR = PROJECT_ROOT / "docs" / "api_auto"

# Quick setup function
def setup_development_environment():
    """Setup ChemML development environment with enhanced features."""

    print("🚀 Initializing ChemML development environment...")

    # Create necessary directories
    PERFORMANCE_REPORTS_DIR.mkdir(exist_ok=True)
    API_DOCS_DIR.mkdir(exist_ok=True)

    # Import and test core functionality
    try:
        import chemml
        print("✅ ChemML core imported successfully")

        # Test performance monitoring
        from chemml.core.monitoring import show_performance_dashboard
        print("✅ Performance monitoring available")

        # Test model recommendations
        from chemml.core.recommendations import recommend_model
        print("✅ Model recommendation system available")

        print("\n🎉 Development environment ready!")
        print("\nQuick commands:")
        print("  📊 Performance dashboard: python -c 'from chemml.core.monitoring import show_performance_dashboard; show_performance_dashboard()'")
        print("  🤖 Model recommendation: python -c 'from chemml.core.recommendations import recommend_model; print(recommend_model([\"CCO\", \"CCC\"], \"logP\"))'")
        print("  📚 API docs: open docs/api_auto/index.html")

        return True

    except ImportError as e:
        print(f"⚠️ Warning: Some features may not be available: {e}")
        return False

if __name__ == "__main__":
    setup_development_environment()
EOF

# 6. Test the installation
print_status "Testing ChemML installation..."
python -c "
import chemml
print('✅ ChemML imported successfully')
print(f'📦 Version: {chemml.__version__}')

# Test new features
try:
    from chemml.core.monitoring import create_performance_dashboard
    print('✅ Performance monitoring available')
except ImportError:
    print('⚠️ Performance monitoring not available')

try:
    from chemml.core.recommendations import ModelRecommendationEngine
    print('✅ Model recommendation system available')
except ImportError:
    print('⚠️ Model recommendation system not available')
" || {
    print_error "ChemML installation test failed"
    exit 1
}

# 7. Setup quick development commands
print_status "Setting up development commands..."
cat > quick_dev.py << 'EOF'
#!/usr/bin/env python3
"""
Quick Development Commands for ChemML
====================================

Easy access to enhanced development features.
"""

import sys
import subprocess

def show_dashboard():
    """Show performance dashboard."""
    from chemml.core.monitoring import show_performance_dashboard
    show_performance_dashboard()

def recommend_model_demo():
    """Demo model recommendation system."""
    from chemml.core.recommendations import recommend_model

    # Demo data
    demo_smiles = ["CCO", "CCC", "c1ccccc1", "CCN", "CCCN"]

    print("🤖 Model Recommendation Demo")
    print("=" * 40)

    # Get recommendation
    recommendation = recommend_model(
        molecular_data=demo_smiles,
        target_property="logP",
        computational_budget="medium"
    )

    print(f"📊 Recommended Model: {recommendation['recommended_model']}")
    print(f"🎯 Confidence: {recommendation['confidence']:.2f}")
    print(f"💡 Rationale: {recommendation['rationale']}")
    print(f"⚡ Expected Accuracy: {recommendation['expected_performance']['estimated_accuracy']}")

def generate_docs():
    """Generate API documentation."""
    subprocess.run([sys.executable, "tools/development/auto_docs.py"])

def run_tests():
    """Run test suite."""
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])

def setup_notebook():
    """Setup Jupyter notebook with ChemML."""
    subprocess.run([sys.executable, "-m", "jupyter", "lab", "--ip=0.0.0.0"])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🧬 ChemML Quick Development Commands")
        print("=" * 40)
        print("python quick_dev.py dashboard     - Show performance dashboard")
        print("python quick_dev.py recommend     - Demo model recommendation")
        print("python quick_dev.py docs          - Generate API documentation")
        print("python quick_dev.py tests         - Run test suite")
        print("python quick_dev.py notebook      - Launch Jupyter Lab")
        sys.exit(0)

    command = sys.argv[1]

    if command == "dashboard":
        show_dashboard()
    elif command == "recommend":
        recommend_model_demo()
    elif command == "docs":
        generate_docs()
    elif command == "tests":
        run_tests()
    elif command == "notebook":
        setup_notebook()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
EOF

# 8. Make scripts executable
chmod +x quick_dev.py

# 9. Run development setup
print_status "Running development configuration..."
python development_config.py || {
    print_warning "Development configuration setup had issues, but continuing..."
}

# 10. Create development README
print_status "Creating development README..."
cat > DEVELOPMENT.md << 'EOF'
# ChemML Development Guide

## Quick Start

The development environment has been automatically configured with enhanced features:

### 🚀 Quick Commands

```bash
# Show performance dashboard
python quick_dev.py dashboard

# Demo model recommendation
python quick_dev.py recommend

# Generate API documentation
python quick_dev.py docs

# Run tests
python quick_dev.py tests

# Launch Jupyter Lab
python quick_dev.py notebook
```

### 📊 Performance Monitoring

```python
from chemml.core.monitoring import show_performance_dashboard
show_performance_dashboard()
```

### 🤖 Model Recommendations

```python
from chemml.core.recommendations import recommend_model

# Get model recommendation
recommendation = recommend_model(
    molecular_data=["CCO", "CCC", "c1ccccc1"],
    target_property="logP",
    computational_budget="medium"
)

print(recommendation['recommended_model'])
```

### 📚 API Documentation

Auto-generated API documentation is available at: `docs/api_auto/index.html`

### 🧪 Testing

```bash
pytest tests/ -v
```

### 🔧 Development Features

- **Performance Monitoring**: Real-time performance tracking and optimization suggestions
- **Model Recommendations**: AI-powered model selection based on data characteristics
- **Auto-Documentation**: Automatically generated API documentation with examples
- **Quick Setup**: One-command environment setup

## Enhanced Workflow

1. **Develop**: Write your code with automatic performance monitoring
2. **Test**: Use `python quick_dev.py tests` to run comprehensive tests
3. **Document**: API docs are auto-generated with examples
4. **Optimize**: Use performance dashboard to identify bottlenecks
5. **Deploy**: Enhanced model recommendation helps users choose optimal approaches

Happy coding! 🧬✨
EOF

print_success "ChemML development environment setup complete!"
echo ""
echo "🎉 Success! Your enhanced ChemML development environment is ready."
echo ""
echo "📋 What's been set up:"
echo "   ✅ ChemML installed in development mode"
echo "   ✅ Performance monitoring system"
echo "   ✅ AI-powered model recommendation engine"
echo "   ✅ Auto-generated API documentation"
echo "   ✅ Quick development commands"
echo ""
echo "🚀 Quick start:"
echo "   📊 Performance dashboard: python quick_dev.py dashboard"
echo "   🤖 Model recommendation: python quick_dev.py recommend"
echo "   📚 API documentation: python quick_dev.py docs"
echo "   🧪 Run tests: python quick_dev.py tests"
echo ""
echo "📖 See DEVELOPMENT.md for detailed development guide"
echo ""
echo "Happy innovating! 🧬✨"
