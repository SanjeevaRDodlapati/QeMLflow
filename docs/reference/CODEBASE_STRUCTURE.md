# 🏗️ QeMLflow Codebase Structure Guide

**Current architecture overview for developers working with the QeMLflow modular structure**

*Last updated: June 2025 | Reflects the implemented modular reorganization*

---

## 🎯 **Quick Navigation**

```
src/qemlflow/
├── core/                    # 🧩 Core framework components
├── research/                # 🔬 Research & experimental modules
├── integrations/            # 🔗 External library integrations
└── tutorials/               # 📚 Learning materials
```

**→ [Jump to specific module documentation](#-detailed-module-breakdown)**

---

## 🚀 **Architecture Overview**

### **Design Philosophy**
- **Modular:** Each component has clear responsibilities
- **Extensible:** Easy to add new models and features
- **Research-Ready:** Supports cutting-edge ML/quantum development
- **Production-Friendly:** Clean APIs for deployment

### **Layer Organization**
```
┌─────────────────────────────────────────────────┐
│            Application Layer                    │
│         (notebooks/, scripts/, tools/)         │
├─────────────────────────────────────────────────┤
│            Research Layer                       │
│      (drug_discovery/, quantum/, advanced/)    │
├─────────────────────────────────────────────────┤
│             Core Layer                          │
│   (models/, featurizers/, utils/, data/)       │
├─────────────────────────────────────────────────┤
│          Integration Layer                      │
│      (external libraries, APIs, formats)       │
└─────────────────────────────────────────────────┘
```

---

## 📁 **Detailed Module Breakdown**

### **🧩 `src/qemlflow/core/`** - Framework Foundation

**Purpose:** Essential components used across all research modules

```
core/
├── __init__.py              # Core API exports
├── data.py                  # Data handling & IO
├── evaluation.py            # Model evaluation metrics
├── featurizers.py           # Molecular featurization (RDKit, DeepChem, hybrid)
├── models.py                # Base model classes
├── utils.py                 # Core utilities
├── common/                  # Shared utilities and constants
├── models/                  # Model implementations
│   ├── classical/           # Traditional ML models
│   └── quantum/             # Quantum ML models
├── preprocessing/           # Data preprocessing pipeline
└── utils/                   # Specialized utility modules
    ├── __init__.py
    ├── data_utils.py        # Data manipulation helpers
    ├── metrics.py           # Custom evaluation metrics
    ├── quantum_utils.py     # Quantum computing utilities
    └── visualization.py     # Plotting and visualization
```

**Key Classes:**
- `HybridMolecularFeaturizer` - Combines RDKit + DeepChem features
- `QeMLflowBaseModel` - Base class for all models
- `MolecularDataset` - Standardized molecular data handling

---

### **🔬 `src/qemlflow/research/`** - Research & Experimental

**Purpose:** Advanced models and specialized research workflows

```
research/
├── __init__.py
├── advanced_models.py       # Cutting-edge ML architectures
├── generative.py            # Molecular generation models
├── modern_quantum.py        # Latest quantum ML algorithms
├── quantum.py               # Core quantum computing methods
└── drug_discovery/          # Modular drug discovery pipeline
    ├── __init__.py
    ├── admet.py             # ADMET prediction models
    ├── generation.py        # Molecular generation
    ├── molecular_optimization.py  # Structure optimization
    ├── optimization.py      # Property optimization
    ├── properties.py        # Property prediction
    ├── qsar.py              # QSAR modeling
    └── screening.py         # Virtual screening
```

**Key Workflows:**
- **Drug Discovery Pipeline:** End-to-end drug design workflows
- **Quantum ML:** VQE, QAOA, hybrid classical-quantum models
- **Generative Models:** VAEs, GANs for molecular generation
- **QSAR:** Quantitative structure-activity relationships

---

### **🔗 `src/qemlflow/integrations/`** - External Integrations

**Purpose:** Interfaces to external libraries and services

```
integrations/
├── __init__.py
├── deepchem_interface.py    # DeepChem integration
├── qiskit_interface.py      # Qiskit quantum computing
├── rdkit_interface.py       # RDKit cheminformatics
└── wandb_interface.py       # Weights & Biases tracking
```

---

### **📚 `src/qemlflow/tutorials/`** - Learning Materials

**Purpose:** Educational content as first-class modules

```
tutorials/
├── __init__.py
├── basics/                  # Fundamental concepts
├── intermediate/            # Intermediate workflows
└── advanced/                # Advanced research methods
```

---

## 🛠️ **Usage Patterns**

### **Quick Start - Basic Usage**
```python
from qemlflow.core.featurizers import HybridMolecularFeaturizer
from qemlflow.core.models import QeMLflowClassifier
from qemlflow.research.drug_discovery import QSARPipeline
```

### **Research - Advanced Usage**
```python
from qemlflow.research.quantum import VQEMolecularOrbitals
from qemlflow.research.drug_discovery.generation import MolecularVAE
from qemlflow.integrations.qiskit_interface import QuantumCircuitBuilder
```

### **Production - Deployment Ready**
```python
from qemlflow.core.evaluation import ModelEvaluator
from qemlflow.core.data import MolecularDataset
```

---

## 🎯 **Where to Add New Features**

| Feature Type | Location | Example |
|--------------|----------|---------|
| New featurizer | `core/featurizers.py` | Custom molecular descriptors |
| ML model | `core/models/classical/` or `core/models/quantum/` | New regression algorithm |
| Research workflow | `research/` | Novel drug discovery pipeline |
| External library | `integrations/` | New quantum backend |
| Utility function | `core/utils/` | Data processing helper |
| Tutorial | `tutorials/` | Learning material |

---

## 📊 **Import Hierarchy**

```
Level 1: core/              # No internal dependencies
Level 2: integrations/      # May import from core/
Level 3: research/          # May import from core/ & integrations/
Level 4: tutorials/         # May import from all levels
```

**Rule:** Higher levels can import from lower levels, but not vice versa.

---

## 🚀 **Migration from Legacy Code**

If you're updating old QeMLflow code:

| Old Import | New Import |
|------------|------------|
| `from qemlflow.drug_design import *` | `from qemlflow.research.drug_discovery import *` |
| `from qemlflow.utils import *` | `from qemlflow.core.utils import *` |
| `from qemlflow.models import *` | `from qemlflow.core.models import *` |

**Migration Script:** Use `scripts/migration/migrate_to_hybrid_architecture.py`

---

## 🔧 **Developer Guidelines**

### **Adding New Models:**
1. Inherit from `QeMLflowBaseModel` in `core/models.py`
2. Add to appropriate submodule (`classical/` or `quantum/`)
3. Include unit tests in `tests/unit/`
4. Update module `__init__.py` exports

### **Adding New Research Workflows:**
1. Create new module in `research/`
2. Import only from `core/` and `integrations/`
3. Add comprehensive example in `tutorials/`
4. Include integration tests

### **External Dependencies:**
- Core dependencies go in `requirements.txt`
- Optional dependencies use defensive imports
- Document installation in `docs/`

---

## 📚 **Related Documentation**

- **[Getting Started](GET_STARTED.md)** - Quick setup guide
- **[Learning Paths](LEARNING_PATHS.md)** - Structured learning
- **[API Reference](REFERENCE.md)** - Detailed API docs
- **[Migration Guide](MIGRATION_GUIDE.md)** - Updating legacy code

---

*This document reflects the current implemented structure. For planned architecture evolution, see [HYBRID_ARCHITECTURE_PLAN.md](HYBRID_ARCHITECTURE_PLAN.md)*
