# ChemML Documentation

Welcome to the **ChemML** documentation - a quantum-enhanced molecular machine learning framework.

## Quick Start

ChemML is a comprehensive Python package for molecular machine learning, featuring:

- **Quantum-Enhanced Algorithms**: Integration with quantum computing frameworks
- **Advanced ML Models**: State-of-the-art machine learning for molecular property prediction
- **AutoML Capabilities**: Automated model selection and hyperparameter optimization
- **Comprehensive Preprocessing**: Molecular descriptors, fingerprints, and feature engineering
- **Professional Tooling**: Production-ready workflows and monitoring

## Installation

```bash
pip install chemml
```

For development installation:

```bash
git clone https://github.com/SanjeevaRDodlapati/ChemML.git
cd ChemML
pip install -e .
```

## Quick Example

```python
import chemml
from chemml.datasets import load_sample_molecules
from chemml.models import AutoMLRegressor

# Load sample data
molecules, properties = load_sample_molecules()

# Create and train model
model = AutoMLRegressor()
model.fit(molecules, properties)

# Make predictions
predictions = model.predict(new_molecules)
```

## Features

### Core Modules

- **`chemml.preprocessing`**: Molecular preprocessing and feature engineering
- **`chemml.models`**: Machine learning models and AutoML
- **`chemml.quantum`**: Quantum computing integration
- **`chemml.ensemble`**: Ensemble methods and model combination
- **`chemml.monitoring`**: Experiment tracking and model monitoring

### Key Capabilities

- ✅ **Molecular Descriptors**: RDKit integration with 200+ descriptors
- ✅ **Fingerprints**: Morgan, MACCS, topological fingerprints
- ✅ **AutoML**: Automated model selection and optimization
- ✅ **Quantum ML**: Variational quantum eigensolvers and circuits
- ✅ **Ensemble Methods**: Advanced stacking and blending
- ✅ **Experiment Tracking**: W&B and MLflow integration
- ✅ **Production Ready**: Docker, CI/CD, and monitoring

## Navigation

- [Getting Started](getting-started/installation.md) - Installation and basic setup
- [User Guide](user-guide/overview.md) - Comprehensive usage guide
- [API Reference](api/core.md) - Detailed API documentation
- [Examples](examples/basic.md) - Practical examples and tutorials
- [Development](development/contributing.md) - Contributing guidelines

## Community & Support

- **GitHub**: [SanjeevaRDodlapati/ChemML](https://github.com/SanjeevaRDodlapati/ChemML)
- **Issues**: [Report bugs or request features](https://github.com/SanjeevaRDodlapati/ChemML/issues)
- **Discussions**: [Community discussions](https://github.com/SanjeevaRDodlapati/ChemML/discussions)

## License

ChemML is released under the MIT License. See [LICENSE](https://github.com/SanjeevaRDodlapati/ChemML/blob/main/LICENSE) for details.
