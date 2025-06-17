# Installation Guide

This guide covers different ways to install QeMLflow based on your needs.

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM recommended
- **Storage**: 1GB free space for full installation

## Installation Methods

### 1. Standard Installation (Recommended)

Install from PyPI with all core dependencies:

```bash
pip install qemlflow
```

This includes:
- Core molecular processing capabilities
- Basic machine learning models
- Essential scientific computing libraries

### 2. Development Installation

For contributors and advanced users:

```bash
# Clone the repository
git clone https://github.com/SanjeevaRDodlapati/QeMLflow.git
cd QeMLflow

# Install in development mode
pip install -e ".[dev]"
```

This includes:
- All core functionality
- Development tools (testing, linting, formatting)
- Pre-commit hooks
- Documentation building tools

### 3. Minimal Installation

For lightweight deployments:

```bash
# Install core dependencies only
pip install -r requirements-core.txt

# Then install QeMLflow
pip install qemlflow --no-deps
```

### 4. Docker Installation

Use our official Docker image:

```bash
# Pull the latest image
docker pull qemlflow/qemlflow:latest

# Run interactive session
docker run -it --rm qemlflow/qemlflow:latest python

# Mount your data directory
docker run -it --rm -v $(pwd)/data:/app/data qemlflow/qemlflow:latest
```

## Optional Dependencies

### RDKit (Molecular Processing)
```bash
# Via conda (recommended)
conda install -c conda-forge rdkit

# Via pip (may require additional setup)
pip install rdkit-pypi
```

### Quantum Computing
```bash
# For Qiskit integration
pip install qiskit qiskit-aer

# For Cirq integration
pip install cirq
```

### Machine Learning Extensions
```bash
# For advanced ML models
pip install xgboost lightgbm catboost

# For deep learning
pip install torch tensorflow
```

### Visualization
```bash
# For enhanced plotting
pip install plotly seaborn

# For molecular visualization
pip install py3Dmol
```

## Verification

Test your installation:

```python
import qemlflow
print(f"QeMLflow version: {qemlflow.__version__}")

# Test core functionality
from qemlflow.preprocessing import MolecularDescriptors
desc = MolecularDescriptors()
print("✅ QeMLflow installed successfully!")
```

## Environment Setup

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv qemlflow_env

# Activate (Linux/Mac)
source qemlflow_env/bin/activate

# Activate (Windows)
qemlflow_env\Scripts\activate

# Install QeMLflow
pip install qemlflow
```

### Conda Environment

```bash
# Create conda environment
conda create -n qemlflow python=3.11

# Activate environment
conda activate qemlflow

# Install dependencies
conda install -c conda-forge rdkit pandas scikit-learn

# Install QeMLflow
pip install qemlflow
```

## Configuration

### Environment Variables

```bash
# Optional: Set data directory
export QEMLFLOW_DATA_DIR="/path/to/your/data"

# Optional: Set cache directory
export QEMLFLOW_CACHE_DIR="/path/to/cache"

# Optional: Enable GPU acceleration
export QEMLFLOW_GPU=true
```

### Configuration File

Create `qemlflow_config.yaml`:

```yaml
environment: production
data_directory: "./data"
cache_directory: "./cache"

models:
  default_model_type: "random_forest"
  validation_split: 0.2

preprocessing:
  default_descriptor_type: "morgan"
  fingerprint_radius: 2
  fingerprint_bits: 2048
```

## Troubleshooting

### Common Issues

#### 1. RDKit Installation Problems
```bash
# Try conda installation
conda install -c conda-forge rdkit

# Or use mamba for faster installation
mamba install -c conda-forge rdkit
```

#### 2. Permission Errors
```bash
# Use user installation
pip install --user qemlflow

# Or create virtual environment
python -m venv venv && source venv/bin/activate
```

#### 3. Memory Issues
```bash
# Install with limited memory
pip install --no-cache-dir qemlflow

# Or install dependencies separately
pip install numpy pandas scikit-learn
pip install qemlflow --no-deps
```

#### 4. Import Errors
```python
# Check installation
import sys
print(sys.path)

# Verify modules
import qemlflow
print(qemlflow.__file__)
```

### Platform-Specific Notes

#### macOS
```bash
# May need Xcode command line tools
xcode-select --install

# For M1/M2 Macs, use conda-forge
conda install -c conda-forge rdkit
```

#### Windows
```bash
# May need Visual C++ Build Tools
# Download from Microsoft website

# Use conda for easier dependency management
conda install -c conda-forge rdkit pandas scikit-learn
```

#### Linux
```bash
# Install development headers
sudo apt-get install python3-dev build-essential

# For RHEL/CentOS
sudo yum install python3-devel gcc gcc-c++
```

## Performance Optimization

### Memory Usage
```python
# Configure for limited memory
import qemlflow
qemlflow.config.set_memory_limit(2048)  # 2GB limit
```

### Parallel Processing
```python
# Set number of CPU cores
import qemlflow
qemlflow.config.set_n_jobs(4)  # Use 4 cores
```

### GPU Acceleration
```python
# Enable GPU if available
import qemlflow
qemlflow.config.enable_gpu(True)
```

## Next Steps

After installation:
1. **[Quick Start](quick-start.md)** - Run your first example
2. **[User Guide](../user-guide/overview.md)** - Learn core concepts
3. **[Examples](../examples/basic.md)** - Explore tutorials
4. **[API Reference](../api/core.md)** - Detailed documentation

## Getting Help

- **GitHub Issues**: [Report installation problems](https://github.com/SanjeevaRDodlapati/QeMLflow/issues)
- **Discussions**: [Ask questions](https://github.com/SanjeevaRDodlapati/QeMLflow/discussions)
- **Documentation**: [Full documentation](https://sanjeevardodlapati.github.io/QeMLflow/)
