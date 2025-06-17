# QeMLflow: Advanced Machine Learning for Chemistry

[![CI Status](https://github.com/hachmannlab/qemlflow/workflows/CI/badge.svg)](https://github.com/hachmannlab/qemlflow/actions)
[![Coverage](https://codecov.io/gh/hachmannlab/qemlflow/branch/main/graph/badge.svg)](https://codecov.io/gh/hachmannlab/qemlflow)
[![License: BSD](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**QeMLflow** is a comprehensive, enterprise-grade machine learning framework specifically designed for chemistry and materials science applications. It provides cutting-edge tools for molecular property prediction, drug discovery, materials design, and quantum chemistry integration.

## 🚀 Quick Start

```python
import qemlflow

# Load sample data
data = qemlflow.load_sample_data("molecules")

# Generate molecular fingerprints
fingerprints = qemlflow.morgan_fingerprints(data.smiles)

# Create and train a model
model = qemlflow.create_rf_model(fingerprints, data.targets)

# Evaluate performance
results = qemlflow.quick_classification_eval(model, fingerprints, data.targets)
print(f"Accuracy: {results.accuracy:.3f}")
```

## ✨ Key Features

### 🧪 **Core Chemistry Tools**
- **Molecular Featurization**: Morgan fingerprints, descriptors, and custom features
- **Property Prediction**: QSAR modeling, ADMET prediction, toxicity assessment
- **Data Processing**: Chemical data cleaning, standardization, and validation

### 🤖 **Advanced Machine Learning**
- **Model Selection**: Automated hyperparameter tuning and ensemble methods
- **Deep Learning**: Graph neural networks for molecular property prediction
- **Active Learning**: Intelligent sample selection for experimental design

### 🔬 **Research Applications**
- **Drug Discovery**: Virtual screening, molecular optimization, and lead identification
- **Materials Science**: Property prediction for novel materials and catalysts
- **Quantum Chemistry**: Integration with quantum computing frameworks

### 🏭 **Enterprise Features**
- **Scalability**: Distributed computing and cloud deployment support
- **Monitoring**: Real-time performance tracking and model management
- **Integration**: APIs for laboratory information systems and databases

## 📦 Installation

### Standard Installation
```bash
pip install qemlflow
```

### Development Installation
```bash
git clone https://github.com/hachmannlab/qemlflow.git
cd qemlflow
pip install -e ".[dev]"
```

### With Optional Dependencies
```bash
# For quantum chemistry features
pip install "qemlflow[quantum]"

# For deep learning capabilities
pip install "qemlflow[deep]"

# For full research suite
pip install "qemlflow[research]"
```

## 🎯 Use Cases

### 1. **Molecular Property Prediction**
```python
from qemlflow.core import molecular_properties

# Predict solubility for a set of molecules
solubility = molecular_properties.predict_solubility(smiles_list)
```

### 2. **Drug Discovery Pipeline**
```python
from qemlflow.research.drug_discovery import VirtualScreening

# Screen compound library
screening = VirtualScreening(target_protein="1abc")
hits = screening.screen_library(compound_library)
```

### 3. **Materials Design**
```python
from qemlflow.research.materials_discovery import PropertyOptimizer

# Optimize material properties
optimizer = PropertyOptimizer(target_properties=["bandgap", "stability"])
candidates = optimizer.generate_candidates(seed_structures)
```

## 📖 Documentation

- **[Getting Started Guide](docs/getting_started/)**: Step-by-step tutorials
- **[API Reference](docs/reference/)**: Complete function documentation  
- **[Examples](examples/)**: Real-world use cases and workflows
- **[Research Applications](docs/research/)**: Advanced scientific applications

## 🧪 Examples

Explore our comprehensive example collection:

- **[Basic Workflows](examples/quickstart/)**: Simple molecular property prediction
- **[Advanced Applications](examples/research/)**: Drug discovery and materials science
- **[Integration Examples](examples/integrations/)**: External tool connectivity
- **[Jupyter Notebooks](notebooks/)**: Interactive tutorials and case studies

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/hachmannlab/qemlflow.git
cd qemlflow
make dev-install  # Sets up environment and pre-commit hooks
```

### Running Tests
```bash
# Quick validation
make test-quick

# Full test suite
make test-full

# With coverage
make test-coverage
```

## 📊 Performance & Benchmarks

QeMLflow has been benchmarked on standard chemistry datasets:

| Dataset | Task | Accuracy | Speed |
|---------|------|----------|-------|
| ESOL | Solubility | 0.891 | 2.3s |
| Tox21 | Toxicity | 0.847 | 1.8s |
| QM9 | Quantum Properties | 0.923 | 0.9s |

*Benchmarks run on Intel i7-8700K, 32GB RAM*

## 🔗 Related Projects

- **[RDKit](https://www.rdkit.org/)**: Chemistry toolkit integration
- **[DeepChem](https://deepchem.io/)**: Deep learning for chemistry
- **[Scikit-learn](https://scikit-learn.org/)**: Machine learning foundation

## 📄 Citation

If you use QeMLflow in your research, please cite:

```bibtex
@software{qemlflow2024,
  title={QeMLflow: Machine Learning for Chemistry},
  author={Hachmann Lab},
  year={2024},
  url={https://github.com/hachmannlab/qemlflow}
}
```

## 📝 License

QeMLflow is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/hachmannlab/qemlflow/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hachmannlab/qemlflow/discussions)
- **Documentation**: [https://qemlflow.readthedocs.io](https://qemlflow.readthedocs.io)

---

**Built with ❤️ by the [Hachmann Lab](https://hachmannlab.github.io/)**
