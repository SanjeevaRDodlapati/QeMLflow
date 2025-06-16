# ChemML Examples

**Comprehensive examples and demonstrations for ChemML integration framework**

---

## 🚀 Quick Start

**New to ChemML?** Start here:
1. **[Basic Integration](quickstart/basic_integration.py)** - Your first ChemML integration
2. **[Simple Workflow](quickstart/simple_workflow.py)** - End-to-end example
3. **[Integration Guide](../docs/integrations/README.md)** - Complete documentation

---

## 📁 Example Categories

### 🏃‍♂️ Quick Start Examples
- **[basic_integration.py](quickstart/basic_integration.py)** - Minimal integration example
- **[simple_workflow.py](quickstart/simple_workflow.py)** - Basic workflow demonstration

### 🧬 Integration Examples

#### **Boltz (Protein Structure)**
- **[comprehensive_demo.py](integrations/boltz/comprehensive_demo.py)** - Complete Boltz integration
- **[structure_prediction.py](integrations/boltz/structure_prediction.py)** - Protein folding
- **[ligand_docking.py](integrations/boltz/ligand_docking.py)** - Protein-ligand binding

#### **Framework Features**
- **[registry_demo.py](integrations/framework/registry_demo.py)** - Model discovery and registry
- **[monitoring_demo.py](integrations/framework/monitoring_demo.py)** - Performance monitoring
- **[pipeline_demo.py](integrations/framework/pipeline_demo.py)** - Multi-model pipelines

### 🛠 Utility Examples
- **[experiment_tracking.py](utilities/experiment_tracking.py)** - MLflow/W&B integration
- **[performance_testing.py](utilities/performance_testing.py)** - Benchmarking tools
- **[custom_adapters.py](utilities/custom_adapters.py)** - Building custom integrations

---

## 📊 Example Complexity Levels

### 🌱 **Beginner** (New to ChemML)
```python
# basic_integration.py - Start here!
from chemml.integrations import get_manager

manager = get_manager()
model = manager.get_adapter("my_model")
result = model.predict(data)
```

### 🔬 **Intermediate** (Familiar with basics)
```python
# Advanced features with monitoring
from chemml.integrations.core import PerformanceMonitor

monitor = PerformanceMonitor()
with monitor.track_performance():
    result = model.batch_predict(batch_data)

stats = monitor.get_stats()
```

### 🚀 **Advanced** (Power users)
```python
# Custom adapters and pipelines
class MyCustomAdapter(BaseModelAdapter):
    def predict(self, data):
        # Custom implementation
        pass

# Multi-model pipeline
pipeline = Pipeline([
    ("preprocess", preprocessing_step),
    ("model1", boltz_adapter),
    ("model2", deepchem_adapter),
    ("postprocess", analysis_step)
])
```

---

## 🎯 Examples by Use Case

### **Drug Discovery**
- Molecular property prediction
- Protein-ligand docking
- ADMET property calculation
- Lead compound optimization

### **Protein Science**
- Structure prediction (Boltz, AlphaFold)
- Molecular dynamics simulation
- Protein-protein interactions
- Enzyme design

### **Materials Science**
- Crystal structure prediction
- Property prediction (bandgap, formation energy)
- Catalyst design
- Materials discovery

### **Quantum Chemistry**
- Electronic structure calculations
- Reaction pathway analysis
- Spectroscopy prediction
- Quantum machine learning

---

## 🔧 Running Examples

### **Requirements**
```bash
# Core ChemML installation
pip install chemml

# For specific examples, install additional dependencies:
pip install torch torchvision  # For PyTorch models
pip install rdkit             # For molecular examples
pip install wandb             # For experiment tracking
```

### **Quick Test**
```bash
# Test basic functionality
cd examples/quickstart/
python basic_integration.py

# Test specific integration
cd examples/integrations/boltz/
python comprehensive_demo.py
```

### **Interactive Mode**
```bash
# Launch interactive demo selector
python examples/interactive_demo.py

# This provides:
# 1. Example browser and selector
# 2. Interactive parameter configuration
# 3. Real-time result visualization
# 4. Export functionality
```

---

## 📚 Learning Path

### **Week 1: Foundations**
1. Run `quickstart/basic_integration.py`
2. Explore `integrations/framework/registry_demo.py`
3. Try `utilities/experiment_tracking.py`

### **Week 2: Molecular Modeling**
1. Complete `integrations/boltz/comprehensive_demo.py`
2. Experiment with protein structure prediction
3. Try ligand docking examples

### **Week 3: Advanced Features**
1. Performance monitoring and optimization
2. Custom adapter development
3. Multi-model pipeline creation

### **Week 4: Domain Applications**
1. Choose your domain (drug discovery, materials, etc.)
2. Run domain-specific examples
3. Adapt examples to your research needs

---

## 🤝 Contributing Examples

### **Adding New Examples**
1. Choose appropriate category folder
2. Follow naming convention: `feature_description.py`
3. Include comprehensive docstring and comments
4. Add entry to this README

### **Example Template**
```python
"""
Brief Description
================

What this example demonstrates and when to use it.

Requirements:
- List any special dependencies
- Mention data requirements

Example Output:
- Describe what the user should expect to see
"""

def main():
    """Main example function with clear structure."""
    print("🧬 Example Name")
    print("=" * 50)

    # Implementation with clear steps
    pass

if __name__ == "__main__":
    main()
```

### **Testing Examples**
```bash
# Test all examples
python test_all_examples.py

# Test specific category
python test_examples.py --category integrations

# Validate example format
python validate_examples.py
```

---

## 🔗 Related Resources

- **[Documentation](../docs/)** - Complete ChemML documentation
- **[Notebooks](../notebooks/)** - Interactive learning materials
- **[API Reference](../docs/REFERENCE.md)** - Detailed API documentation
- **[Integration Guide](../docs/integrations/README.md)** - Model integration guide

---

*Last updated: June 16, 2025*
