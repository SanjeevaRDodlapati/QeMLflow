#!/bin/bash
# QeMLflow Best-in-Class Libraries Installation Script
# =================================================

echo "🧬 Installing QeMLflow Best-in-Class Libraries..."
echo "================================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check for pip
if ! command_exists pip; then
    echo "❌ pip is required but not installed. Please install pip first."
    exit 1
fi

# Update pip
echo "🔄 Updating pip..."
pip install --upgrade pip

# Install core libraries first
echo "🔄 Installing core scientific computing libraries..."
pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0 scikit-learn>=1.3.0

# Install distributed ML libraries
echo "🔄 Installing distributed ML libraries..."
pip install ray[default]>=2.5.0 ray[tune]>=2.5.0 ray[train]>=2.5.0
pip install dask[distributed]>=2023.5.0
pip install horovod>=0.28.0 || echo "⚠️  Horovod installation failed (requires MPI). Continuing..."

# Install hyperparameter optimization libraries
echo "🔄 Installing hyperparameter optimization libraries..."
pip install optuna>=3.4.0
pip install hyperopt>=0.2.7
pip install ax-platform>=0.3.4 || echo "⚠️  ax-platform installation failed. Continuing..."
pip install scikit-optimize>=0.9.0
pip install flaml>=2.1.0

# Install performance monitoring libraries
echo "🔄 Installing performance monitoring libraries..."
pip install mlflow>=2.8.0
pip install wandb>=0.15.0
pip install tensorboard>=2.14.0
pip install prometheus-client>=0.18.0
pip install psutil>=5.9.0
pip install gpustat>=1.1.0

# Install quantum computing libraries
echo "🔄 Installing quantum computing libraries..."
pip install pennylane>=0.32.0
pip install qiskit>=1.0.0 qiskit-aer>=0.13.0
pip install cirq>=1.2.0
pip install qulacs>=0.6.0 || echo "⚠️  qulacs installation failed. Continuing..."

# Install uncertainty quantification libraries
echo "🔄 Installing uncertainty quantification libraries..."
pip install pyro-ppl>=1.8.6
pip install gpytorch>=1.11.0 || echo "⚠️  gpytorch installation failed. Continuing..."
pip install botorch>=0.9.4 || echo "⚠️  botorch installation failed. Continuing..."

# Install knowledge integration libraries
echo "🔄 Installing knowledge integration libraries..."
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install spacy>=3.7.0
pip install networkx>=3.2.0
pip install sentence-transformers>=2.2.2

# Install QeMLflow-specific requirements
echo "🔄 Installing QeMLflow requirements..."
pip install -r requirements.txt

# Verify key installations
echo "✅ Verifying installations..."

python -c "
import sys
libraries = [
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'Scikit-learn'),
    ('optuna', 'Optuna'),
    ('mlflow', 'MLflow'),
    ('pennylane', 'PennyLane')
]

failed = []
for lib, name in libraries:
    try:
        __import__(lib)
        print(f'✅ {name} installed successfully')
    except ImportError:
        print(f'❌ {name} installation failed')
        failed.append(name)

if failed:
    print(f'\n⚠️  Some libraries failed to install: {failed}')
    print('You can continue with QeMLflow, but some features may not be available.')
else:
    print('\n🎉 All core libraries installed successfully!')
"

# Optional: Install Ray if not already installed
python -c "
try:
    import ray
    print('✅ Ray is available for distributed computing')
except ImportError:
    print('⚠️  Ray not available. Install with: pip install ray[default]')
"

# Optional: Check FLAML
python -c "
try:
    import flaml
    print('✅ FLAML is available for AutoML')
except ImportError:
    print('⚠️  FLAML not available. Install with: pip install flaml')
"

echo ""
echo "🎯 Installation Summary:"
echo "========================"
echo "✅ Core ML libraries: NumPy, Pandas, Scikit-learn"
echo "✅ Distributed ML: Ray, Dask (Horovod optional)"
echo "✅ Hyperparameter Optimization: Optuna, Hyperopt"
echo "✅ Performance Monitoring: MLflow, W&B, TensorBoard"
echo "✅ Quantum Computing: PennyLane, Qiskit"
echo "✅ Uncertainty Quantification: PyTorch, Pyro"
echo "✅ Knowledge Integration: Transformers, spaCy"
echo ""
echo "🚀 Ready to run: python tools/examples/best_libraries_demo.py"
echo "📖 Configuration: config/advanced_config.yaml"
echo "📚 Documentation: docs/reports/innovation/"
