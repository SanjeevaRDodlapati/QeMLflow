#!/bin/bash
# ChemML Bootcamp Environment Setup Script
# File: setup_chemml_bootcamp.sh

set -e  # Exit on any error

echo "🚀 ChemML Bootcamp Environment Setup"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."

    # Check if we're on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        print_warning "This script is optimized for macOS. Continuing anyway..."
    fi

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi

    python_version=$(python3 --version | cut -d' ' -f2)
    major_version=$(echo $python_version | cut -d'.' -f1)
    minor_version=$(echo $python_version | cut -d'.' -f2)

    if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 8 ]]; then
        print_error "Python 3.8+ required, found $python_version"
        exit 1
    fi
    print_status "Python version OK ($python_version)"

    # Check available memory (macOS specific)
    total_mem_bytes=$(sysctl -n hw.memsize)
    total_gb=$((total_mem_bytes / 1024 / 1024 / 1024))

    if [[ $total_gb -lt 8 ]]; then
        print_warning "8GB+ RAM recommended, found ${total_gb}GB"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_status "Memory OK (${total_gb}GB)"
    fi

    # Check disk space
    available_space=$(df -h . | awk 'NR==2{print $4}' | sed 's/Gi*//')
    if [[ ${available_space%.*} -lt 10 ]]; then
        print_warning "10GB+ disk space recommended, found ${available_space}"
    else
        print_status "Disk space OK (${available_space} available)"
    fi

    # Check for conda/mamba
    if command -v conda &> /dev/null; then
        print_status "Conda found: $(conda --version)"
        CONDA_CMD="conda"
    elif command -v mamba &> /dev/null; then
        print_status "Mamba found: $(mamba --version)"
        CONDA_CMD="mamba"
    else
        print_error "Conda or Mamba required. Please install Anaconda/Miniconda first."
        print_info "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
}

# Create conda environment
create_environment() {
    print_info "Creating ChemML bootcamp environment..."

    ENV_NAME="chemml_bootcamp"

    # Check if environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "Environment '${ENV_NAME}' already exists"
        read -p "Remove and recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n $ENV_NAME -y
        else
            print_info "Using existing environment"
            return 0
        fi
    fi

    # Create environment with Python 3.9
    print_info "Creating new conda environment..."
    $CONDA_CMD create -n $ENV_NAME python=3.9 -y

    print_status "Environment '$ENV_NAME' created"

    # Activate environment for package installation
    print_info "Activating environment and installing packages..."

    # Note: We need to source conda to use it in script
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME

    # Install core scientific packages via conda
    print_info "Installing core scientific packages..."
    $CONDA_CMD install -c conda-forge \
        numpy pandas matplotlib seaborn \
        jupyter jupyterlab ipywidgets \
        scikit-learn scipy \
        rdkit openbabel \
        -y

    # Install deep learning packages
    print_info "Installing deep learning packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install torch-geometric torch-scatter torch-sparse

    # Install chemistry and quantum packages
    print_info "Installing chemistry packages..."
    $CONDA_CMD install -c psi4 psi4 -y || print_warning "Psi4 installation failed - will use mock implementation"
    pip install deepchem
    pip install pyscf || print_warning "PySCF installation failed - will use alternative"

    # Install quantum computing packages
    print_info "Installing quantum computing packages..."
    pip install qiskit qiskit-nature qiskit-optimization

    # Install additional useful packages
    print_info "Installing additional packages..."
    pip install plotly dash
    pip install biopython MDAnalysis
    pip install fastapi uvicorn sqlalchemy
    pip install pytest pytest-cov black flake8
    pip install optuna mlflow wandb

    print_status "All packages installed successfully"
}

# Setup data directories and download sample datasets
setup_data() {
    print_info "Setting up data directories and sample datasets..."

    # Create directory structure
    mkdir -p data/{raw,processed,cache,results}
    mkdir -p notebooks/checkpoints
    mkdir -p models/trained
    mkdir -p logs

    print_status "Directory structure created"

    # Test dataset downloads
    print_info "Testing dataset access..."

    # Create a simple test script
    cat > test_datasets.py << 'EOF'
import deepchem as dc
import warnings
warnings.filterwarnings('ignore')

print("Testing dataset loading...")
try:
    # Test ESOL dataset (small)
    tasks, datasets, transformers = dc.molnet.load_esol(featurizer='ECFP')
    print("✅ ESOL dataset loaded successfully")
except Exception as e:
    print(f"❌ ESOL dataset failed: {e}")

try:
    # Test HIV dataset (medium)
    tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='GraphConv')
    print("✅ HIV dataset loaded successfully")
except Exception as e:
    print(f"❌ HIV dataset failed: {e}")

print("Dataset testing complete!")
EOF

    # Run dataset test
    python test_datasets.py
    rm test_datasets.py

    print_status "Dataset setup completed"
}

# Create configuration files
create_configs() {
    print_info "Creating configuration files..."

    # Create bootcamp configuration
    cat > bootcamp_config.yaml << 'EOF'
# ChemML Bootcamp Configuration
bootcamp:
  name: "ChemML QuickStart Bootcamp"
  version: "1.0.0"
  duration_days: 7

  # Learning tracks
  tracks:
    quick:
      hours_per_day: 3
      total_hours: 21
      description: "Core concepts only"

    standard:
      hours_per_day: 4.5
      total_hours: 31.5
      description: "Comprehensive coverage"

    intensive:
      hours_per_day: 6
      total_hours: 42
      description: "Full implementation with extensions"

    extended:
      hours_per_day: 3
      total_hours: 42
      duration_days: 14
      description: "Self-paced learning"

  # Environment settings
  environment:
    compute:
      cpu_cores: 4
      memory_gb: 8
      gpu_enabled: false

    data:
      cache_size_gb: 5
      datasets:
        - esol
        - hiv
        - qm9
        - tox21

  # Assessment settings
  assessment:
    checkpoints_per_day: 5
    auto_save: true
    progress_tracking: true

  # Paths
  paths:
    data: "./data"
    models: "./models"
    notebooks: "./notebooks/quickstart_bootcamp"
    logs: "./logs"
    results: "./results"
EOF

    # Create Jupyter configuration
    mkdir -p ~/.jupyter
    cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
# Jupyter Lab Configuration for ChemML Bootcamp

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = True
c.ServerApp.notebook_dir = './notebooks'

# Enable extensions
c.LabApp.check_for_updates_frequency = 86400

# Memory and performance
c.NotebookApp.max_buffer_size = 2147483648  # 2GB
c.MappingKernelManager.cull_idle_timeout = 7200  # 2 hours
c.MappingKernelManager.cull_interval = 300  # 5 minutes

# Security
c.ServerApp.allow_remote_access = True
c.ServerApp.disable_check_xsrf = False
EOF

    print_status "Configuration files created"
}

# Setup development tools
setup_dev_tools() {
    print_info "Setting up development tools..."

    # Create pre-commit configuration
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
EOF

    # Create pytest configuration
    cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
EOF

    print_status "Development tools configured"
}

# Create launcher script
create_launcher() {
    print_info "Creating launcher script..."

    cat > start_bootcamp.sh << 'EOF'
#!/bin/bash
# ChemML Bootcamp Launcher

echo "🚀 Starting ChemML Bootcamp"
echo "=========================="

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate chemml_bootcamp
elif [[ -f "chemml_env/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source chemml_env/bin/activate
else
    echo "❌ No environment found. Please run setup first."
    exit 1
fi

# Check if Jupyter is available
if ! command -v jupyter &> /dev/null; then
    echo "❌ Jupyter not found in environment"
    exit 1
fi

# Start Jupyter Lab
echo "🔬 Starting Jupyter Lab..."
echo "📍 Navigate to: http://localhost:8888"
echo "📝 Notebooks location: ./notebooks/quickstart_bootcamp/"
echo ""
echo "Press Ctrl+C to stop the server"

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=./notebooks
EOF

    chmod +x start_bootcamp.sh

    print_status "Launcher script created (start_bootcamp.sh)"
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."

    # Create verification script
    cat > verify_install.py << 'EOF'
#!/usr/bin/env python3
"""
ChemML Bootcamp Installation Verification
"""

import sys
import importlib
import warnings
warnings.filterwarnings('ignore')

def check_package(package_name, import_name=None, optional=False):
    """Check if a package is available"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        if optional:
            print(f"⚠️  {package_name}: Optional package not found")
        else:
            print(f"❌ {package_name}: Required package missing")
        return False

def main():
    print("🔍 ChemML Bootcamp Installation Verification")
    print("=" * 50)

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python: {python_version}")

    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scikit-learn', 'sklearn'),
        ('jupyter', 'jupyter'),
        ('rdkit', 'rdkit'),
        ('torch', 'torch'),
        ('deepchem', 'deepchem'),
    ]

    optional_packages = [
        ('psi4', 'psi4', True),
        ('pyscf', 'pyscf', True),
        ('qiskit', 'qiskit', True),
        ('biopython', 'Bio', True),
        ('MDAnalysis', 'MDAnalysis', True),
    ]

    print("\n📦 Required Packages:")
    all_required = True
    for pkg_info in required_packages:
        if not check_package(*pkg_info):
            all_required = False

    print("\n📦 Optional Packages:")
    for pkg_info in optional_packages:
        check_package(*pkg_info)

    print("\n" + "=" * 50)
    if all_required:
        print("🎉 Installation verification successful!")
        print("🚀 Ready to start the ChemML Bootcamp!")
    else:
        print("❌ Some required packages are missing")
        print("Please run the setup script again or install manually")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

    # Run verification
    python verify_install.py
    verification_result=$?

    # Clean up
    rm verify_install.py

    if [[ $verification_result -eq 0 ]]; then
        print_status "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Create helpful documentation
create_documentation() {
    print_info "Creating helpful documentation..."

    cat > BOOTCAMP_QUICKSTART.md << 'EOF'
# ChemML Bootcamp Quick Start Guide

## Getting Started

### 1. Activate Environment
```bash
# If using conda
conda activate chemml_bootcamp

# If using the setup script's virtual environment
source chemml_env/bin/activate
```

### 2. Start Jupyter Lab
```bash
# Use the launcher script
./start_bootcamp.sh

# Or manually
jupyter lab --notebook-dir=./notebooks
```

### 3. Choose Your Learning Track

#### 🏃‍♂️ Quick Track (3 hours/day)
- Focus on core concepts
- Skip advanced implementations
- Total: 21 hours over 7 days

#### 📚 Standard Track (4.5 hours/day)
- Comprehensive coverage
- Most practical exercises
- Total: 31.5 hours over 7 days

#### 🚀 Intensive Track (6 hours/day)
- Full implementations
- Advanced challenges
- Total: 42 hours over 7 days

#### 🎯 Extended Track (3 hours/day, 14 days)
- Self-paced learning
- Deep exploration time
- Total: 42 hours over 14 days

## Day-by-Day Structure

### Day 1: ML & Cheminformatics Foundations
- **Location**: `notebooks/quickstart_bootcamp/day_01_ml_cheminformatics_project.ipynb`
- **Focus**: RDKit, DeepChem, Property Prediction
- **Time**: 3-6 hours based on track

### Day 2: Deep Learning for Molecules
- **Location**: `notebooks/quickstart_bootcamp/day_02_deep_learning_molecules_project.ipynb`
- **Focus**: GNNs, Transformers, Generative Models
- **Time**: 3-6 hours based on track

### Day 3: Molecular Docking Pipeline
- **Location**: `notebooks/quickstart_bootcamp/day_03_molecular_docking_project.ipynb`
- **Focus**: AutoDock, Gnina, Binding Analysis
- **Time**: 3-6 hours based on track

### Day 4: Quantum Chemistry Practice
- **Location**: `notebooks/quickstart_bootcamp/day_04_quantum_chemistry_project.ipynb`
- **Focus**: Psi4, PySCF, DFT Calculations
- **Time**: 3-6 hours based on track

### Day 5: Quantum ML Integration
- **Location**: `notebooks/quickstart_bootcamp/day_05_quantum_ml_project.ipynb`
- **Focus**: QM9, SchNet, Delta Learning
- **Time**: 3-6 hours based on track

### Day 6: Quantum Computing Algorithms
- **Location**: `notebooks/quickstart_bootcamp/day_06_quantum_computing_project.ipynb`
- **Focus**: VQE, Qiskit, Molecular Simulation
- **Time**: 3-6 hours based on track

### Day 7: End-to-End Pipeline Integration
- **Location**: `notebooks/quickstart_bootcamp/day_07_integration_project.ipynb`
- **Focus**: Production deployment, testing, portfolio
- **Time**: 3-6 hours based on track

## Troubleshooting

### Common Issues

#### Environment Activation Problems
```bash
# Ensure conda is initialized
conda init zsh
# Restart terminal and try again
```

#### Package Import Errors
```bash
# Verify installation
python -c "import rdkit; print('RDKit OK')"
python -c "import deepchem; print('DeepChem OK')"
```

#### Memory Issues
- Close other applications
- Restart Jupyter kernel
- Consider using smaller datasets for practice

#### Performance Issues
- Use CPU-only mode for testing
- Enable parallel processing where available
- Monitor system resources

### Getting Help

1. **Check Documentation**: Review the notebook markdown cells
2. **Search Issues**: Look for similar problems in project issues
3. **Community Support**: Join discussion forums
4. **Office Hours**: Attend virtual office hours (if available)

## Tips for Success

### Time Management
- Start early each day
- Take regular breaks (Pomodoro technique)
- Don't skip the assessment checkpoints
- Save work frequently

### Learning Strategy
- Read through entire notebook before starting
- Understand concepts before coding
- Experiment with parameters
- Build a personal knowledge base

### Technical Best Practices
- Use version control for your work
- Document your modifications
- Test code incrementally
- Keep track of successful approaches

## Assessment and Certification

### Daily Checkpoints
- Complete assessment widgets in each notebook
- Review your understanding scores
- Address gaps before moving forward

### Portfolio Development
- Save your best work examples
- Document your learning journey
- Create project summaries
- Prepare for showcase

### Certification Path
- Complete all 7 days with passing scores
- Submit portfolio for review
- Participate in peer assessments
- Demonstrate practical applications

## Next Steps After Bootcamp

### Advanced Learning
- Explore Week 6-12 checkpoint materials
- Join research projects
- Contribute to open source
- Attend conferences and workshops

### Career Development
- Update your LinkedIn profile
- Apply bootcamp skills to real projects
- Network with other graduates
- Consider advanced certifications

Good luck with your ChemML journey! 🚀🧪
EOF

    print_status "Documentation created (BOOTCAMP_QUICKSTART.md)"
}

# Main execution function
main() {
    echo "Starting ChemML Bootcamp setup process..."
    echo "This will take approximately 10-20 minutes depending on your internet connection."
    echo ""

    check_requirements
    create_environment
    setup_data
    create_configs
    setup_dev_tools
    create_launcher
    verify_installation
    create_documentation

    echo ""
    echo "🎉 ChemML Bootcamp Setup Complete!"
    echo "=================================="
    print_status "Environment created: chemml_bootcamp"
    print_status "Launcher script: ./start_bootcamp.sh"
    print_status "Quick start guide: ./BOOTCAMP_QUICKSTART.md"
    echo ""
    echo "🚀 To get started:"
    echo "   1. ./start_bootcamp.sh"
    echo "   2. Open: notebooks/quickstart_bootcamp/day_01_ml_cheminformatics_project.ipynb"
    echo "   3. Begin your ChemML journey!"
    echo ""
    print_info "For troubleshooting, see BOOTCAMP_QUICKSTART.md"
}

# Run main function
main "$@"
