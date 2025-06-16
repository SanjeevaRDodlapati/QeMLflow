# ChemML Quick Start Guide

**Get started with computational molecular modeling and machine learning in 15 minutes!**

---

## 🎯 Welcome to ChemML

ChemML is your gateway to computational drug discovery, combining machine learning and quantum computing for molecular modeling. This guide gets you running with your first molecular ML model quickly.

### 🏃‍♂️ Quick Start Options

- **👨‍🎓 New to Molecular ML?** → [7-Day Bootcamp](#7-day-bootcamp) *(Most Popular)*
- **🔬 Experienced ML Engineer?** → [Direct Setup](#direct-setup)
- **🚀 Advanced User?** → [Learning Paths](LEARNING_PATHS.md)
- **📖 Need API docs?** → [API Reference](REFERENCE.md)

---

## ⚡ 15-Minute Setup

### Step 1: Installation (5 minutes)

Choose your installation method:

#### **Standard Installation (Recommended)**
```bash
pip install chemml
```

#### **Development Installation**
```bash
git clone https://github.com/SanjeevaRDodlapati/ChemML.git
cd ChemML
pip install -e ".[dev]"
```

#### **Minimal Installation**
```bash
pip install -r requirements-core.txt
```

### Step 2: Verify Installation (2 minutes)

```python
import chemml
print(f"ChemML version: {chemml.__version__}")

# Test basic functionality
from chemml.preprocessing import MolecularDescriptors
desc = MolecularDescriptors()
print("✅ ChemML installed successfully!")
```

### Step 3: Your First Success (8 minutes)

#### **Basic Molecular Property Prediction**

```python
import chemml
from chemml.models import AutoMLRegressor
from chemml.preprocessing import MolecularDescriptors

# Sample SMILES strings (molecular representations)
molecules = [
    "CCO",           # Ethanol
    "CC(C)O",        # Isopropanol
    "CCC",           # Propane
    "C1=CC=CC=C1"    # Benzene
]

# Sample properties (e.g., boiling points)
properties = [78.4, 82.6, -42.1, 80.1]

# Generate molecular descriptors
descriptor_calculator = MolecularDescriptors()
descriptors = descriptor_calculator.calculate(molecules)

# Train a simple model
model = AutoMLRegressor()
model.fit(descriptors, properties)

# Make predictions
predictions = model.predict(descriptors)
print("Predictions:", predictions)
```

#### **Using External Models**

```python
from chemml.integrations import get_manager

# Get integration manager
manager = get_manager()

# Example: Use Boltz for protein structure prediction
if "boltz" in manager.list_available_models():
    boltz = manager.get_adapter("boltz")
    # structure = boltz.predict_structure(protein_sequence)
    print("✅ Boltz integration ready!")
```

---

## 🎓 7-Day Bootcamp

### **For Complete Beginners**

**Time Commitment**: 2-3 hours/day for 7 days

**Your Path**:
1. **Day 1**: [Python basics and environment setup](getting_started/prerequisites.md)
2. **Day 2**: [Basic cheminformatics](../notebooks/learning/fundamentals/01_basic_cheminformatics.ipynb)
3. **Day 3**: [Machine learning for molecules](../notebooks/learning/bootcamp/01_ml_cheminformatics.ipynb)
4. **Day 4**: [Deep learning applications](../notebooks/learning/bootcamp/02_deep_learning_molecules.ipynb)
5. **Day 5**: [Molecular docking](../notebooks/learning/bootcamp/03_molecular_docking.ipynb)
6. **Day 6**: [Quantum chemistry](../notebooks/learning/bootcamp/04_quantum_chemistry.ipynb)
7. **Day 7**: [Integration project](../notebooks/learning/bootcamp/07_integration_project.ipynb)

---

## 🔬 Direct Setup

### **For Experienced Users**

**Quick Integration Test**:
```bash
# Clone and test
git clone https://github.com/SanjeevaRDodlapati/ChemML.git
cd ChemML
python examples/quickstart/basic_integration.py
```

**Advanced Features**:
```python
# Performance monitoring
from chemml.integrations.core import PerformanceMonitor

monitor = PerformanceMonitor()
with monitor.track_performance():
    # Your computations here
    pass

# Experiment tracking
from chemml.integrations.utils import ExperimentTracker

tracker = ExperimentTracker(backend="wandb")
tracker.start_experiment("my_experiment")
```

---

## 🎯 Choose Your Learning Path

### 🌱 **Complete Beginner**
*"I'm new to both programming and computational chemistry"*

**Time**: 20-25 hours/week for 16-20 weeks

**Your Path**:
1. [Prerequisites](getting_started/prerequisites.md) - Programming and chemistry basics
2. [7-Day Bootcamp](#7-day-bootcamp) - Intensive introduction
3. [Beginner Track](../notebooks/learning/fundamentals/) - Comprehensive foundation

### 🧪 **Chemistry/Biology Background**
*"I have wet lab experience but I'm new to computational methods"*

**Time**: 25-30 hours/week for 12-16 weeks

**Your Path**:
1. [Python for Scientists](getting_started/python_for_scientists.md)
2. [Computational Chemistry Concepts](getting_started/comp_chem_basics.md)
3. [Drug Design Track](../notebooks/learning/specialized/drug_design_track.md)

### 💻 **Programming/ML Background**
*"I have technical skills but I'm new to chemistry"*

**Time**: 15-20 hours/week for 8-12 weeks

**Your Path**:
1. [Chemistry for Programmers](getting_started/chemistry_for_programmers.md)
2. [Molecular Representations](getting_started/molecular_data.md)
3. [Advanced Integration Examples](../examples/integrations/)

### 🚀 **Advanced User**
*"I'm familiar with both domains and want to dive deep"*

**Time**: 10-15 hours/week for 4-8 weeks

**Your Path**:
1. [Advanced Features Guide](ENHANCED_FEATURES_GUIDE.md)
2. [Integration Development](integrations/README.md)
3. [Research Innovation](research_innovation_template.md)

---

## 🧪 First Examples

### **Molecular Property Prediction**
```python
# examples/quickstart/molecular_properties.py
from chemml.preprocessing import MolecularDescriptors
from chemml.models import GradientBoostingRegressor

# Load data
molecules = ["CCO", "CC(C)O", "CCC"]
properties = [78.4, 82.6, -42.1]

# Calculate descriptors and train model
descriptors = MolecularDescriptors().calculate(molecules)
model = GradientBoostingRegressor()
model.fit(descriptors, properties)
```

### **Protein Structure Prediction**
```python
# examples/integrations/boltz/basic_demo.py
from chemml.integrations import get_manager

manager = get_manager()
boltz = manager.get_adapter("boltz")

# Predict structure
sequence = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER"
structure = boltz.predict_structure(sequence)
```

### **Drug Discovery Workflow**
```python
# examples/workflows/drug_discovery.py
from chemml.workflows import DrugDiscoveryPipeline

pipeline = DrugDiscoveryPipeline()
results = pipeline.run(
    target_protein="1ABC.pdb",
    compound_library="compounds.sdf"
)
```

---

## 🔧 Troubleshooting

### **Installation Issues**

**ImportError: No module named 'chemml'**
```bash
# Ensure you're in the right environment
which python
pip list | grep chemml

# Reinstall if needed
pip uninstall chemml
pip install chemml
```

**GPU/CUDA Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Memory Errors**
```python
# Use memory-efficient options
from chemml.integrations.core import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.set_memory_limit("4GB")  # Adjust as needed
```

### **Getting Help**

- **📚 Documentation**: [Full Documentation](README.md)
- **💬 Community**: [GitHub Discussions](https://github.com/SanjeevaRDodlapati/ChemML/discussions)
- **🐛 Issues**: [Bug Reports](https://github.com/SanjeevaRDodlapati/ChemML/issues)
- **📧 Email**: [Support Contact](mailto:support@chemml.org)

---

## ✅ Next Steps

### **After Successful Setup**
1. **Explore Examples**: Browse `examples/` folder for working demonstrations
2. **Join Bootcamp**: Follow the 7-day intensive learning program
3. **Read Documentation**: Dive deeper with comprehensive guides
4. **Contribute**: Help improve ChemML for the community

### **Recommended Learning Sequence**
1. **Week 1**: Complete this quick start + basic examples
2. **Week 2**: Follow your chosen learning path
3. **Week 3**: Try domain-specific examples (drug discovery, etc.)
4. **Week 4**: Advanced features and custom integrations

---

**🎉 Congratulations! You're ready to explore computational molecular modeling with ChemML!**

*For the most current information and updates, always refer to the [online documentation](https://chemml.readthedocs.io) and [GitHub repository](https://github.com/SanjeevaRDodlapati/ChemML).*
