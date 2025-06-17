# ChemML Integration Documentation

**Comprehensive guide for integrating external models into ChemML**

---

## 📋 Overview

ChemML provides a flexible framework for integrating external machine learning models, quantum computing tools, and scientific libraries. This guide covers the complete integration process from basic adapters to advanced workflows.

## 🚀 Quick Integration

### Basic Integration Steps

1. **Choose Adapter Type**
   - `BaseModelAdapter` - General ML models
   - `TorchModelAdapter` - PyTorch models
   - `HuggingFaceAdapter` - Transformers/NLP models
   - `PaperAdapter` - Research implementations

2. **Create Adapter**
```python
from chemml.integrations.adapters.base import TorchModelAdapter

class MyModelAdapter(TorchModelAdapter):
    def __init__(self, model_path=None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path

    def load_model(self):
        # Your model loading logic
        pass

    def predict(self, input_data):
        # Your prediction logic
        pass
```

3. **Register Model**
```python
from chemml.integrations import get_manager

manager = get_manager()
manager.register_adapter("my_model", MyModelAdapter)
```

## 📁 Integration Categories

### Molecular Modeling
- **Protein Structure**: Boltz, AlphaFold, ChimeraX
- **Quantum Chemistry**: PySCF, Psi4, Gaussian
- **Molecular Dynamics**: OpenMM, GROMACS, AMBER

### Drug Discovery
- **QSAR/QSPR**: ChemProp, DeepChem
- **Generative Models**: MOSES, GuacaMol
- **Docking**: AutoDock, Vina, Glide

### Materials Science
- **Crystal Structure**: VASP, Quantum Espresso
- **Properties**: AFLOW, Materials Project
- **ML Potentials**: ANI, PhysNet

## 🛠 Advanced Features

### Performance Monitoring
```python
from chemml.integrations.core import PerformanceMonitor

monitor = PerformanceMonitor()
with monitor.track_performance():
    result = model.predict(data)

print(f"Prediction time: {monitor.get_stats()['prediction_time']}")
```

### Experiment Tracking
```python
from chemml.integrations.utils import ExperimentTracker

tracker = ExperimentTracker(backend="wandb")
tracker.log_experiment("my_model", parameters, results)
```

### Automated Testing
```python
from chemml.integrations.core import AutomatedTesting

tester = AutomatedTesting()
tester.test_adapter("my_model", test_data)
```

## 📖 Model-Specific Guides

- **[Boltz Integration](model_specific/boltz.md)** - Protein structure prediction
- **[DeepChem Integration](model_specific/deepchem.md)** - Chemical ML toolkit
- **[Custom Model Guide](model_specific/custom_models.md)** - Build your own adapters

## 🔗 See Also

- **[Framework Integration Guide](../FRAMEWORK_INTEGRATION_GUIDE.md)** - Core concepts
- **[Enhanced Features Guide](../ENHANCED_FEATURES_GUIDE.md)** - Advanced capabilities
- **[Examples](../../examples/integrations/)** - Working demonstrations
- **[API Reference](../REFERENCE.md)** - Complete API documentation
