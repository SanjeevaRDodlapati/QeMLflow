# Quick Start Guide

Get up and running with QeMLflow in minutes!

## Installation

### Standard Installation

```bash
pip install qemlflow
```

### Development Installation

```bash
git clone https://github.com/SanjeevaRDodlapati/QeMLflow.git
cd QeMLflow
pip install -e ".[dev]"
```

### Core Dependencies Only

For minimal installation:

```bash
pip install -r requirements-core.txt
```

## Verify Installation

```python
import qemlflow
print(f"QeMLflow version: {qemlflow.__version__}")

# Test basic functionality
from qemlflow.preprocessing import MolecularDescriptors
desc = MolecularDescriptors()
print("✅ QeMLflow installed successfully!")
```

## First Example

### 1. Basic Molecular Property Prediction

```python
import qemlflow
from qemlflow.models import AutoMLRegressor
from qemlflow.preprocessing import MolecularDescriptors

# Sample SMILES strings
smiles = [
    'CCO',           # Ethanol
    'CC(=O)O',       # Acetic acid
    'c1ccccc1',      # Benzene
    'CCN(CC)CC'      # Triethylamine
]

# Target properties (e.g., boiling points)
properties = [78.4, 118.1, 80.1, 89.5]

# Create molecular descriptors
desc = MolecularDescriptors()
X = desc.fit_transform(smiles)

# Train AutoML model
model = AutoMLRegressor(time_budget=60)  # 1 minute training
model.fit(X, properties)

# Make predictions
new_smiles = ['CCc1ccccc1']  # Ethylbenzene
X_new = desc.transform(new_smiles)
prediction = model.predict(X_new)

print(f"Predicted boiling point: {prediction[0]:.1f}°C")
```

### 2. Advanced Workflow with Monitoring

```python
import qemlflow
from qemlflow.datasets import load_sample_molecules
from qemlflow.models import EnsembleRegressor
from qemlflow.monitoring import ExperimentTracker

# Load sample dataset
molecules, properties = load_sample_molecules()

# Initialize experiment tracking
tracker = ExperimentTracker(project="qemlflow-quickstart")

# Create ensemble model
model = EnsembleRegressor(
    models=['random_forest', 'xgboost', 'neural_network'],
    voting='soft'
)

# Train with tracking
with tracker.track_experiment("ensemble_model"):
    model.fit(molecules, properties)

    # Log metrics
    score = model.score(molecules, properties)
    tracker.log_metric("r2_score", score)

    # Save model
    tracker.log_model(model, "ensemble_regressor")

print(f"Model R² score: {score:.3f}")
```

## Configuration

### Environment Variables

```bash
# Optional: Configure QeMLflow
export QEMLFLOW_CONFIG_PATH="./my_config.yaml"
export QEMLFLOW_DATA_DIR="./data"
export QEMLFLOW_CACHE_DIR="./cache"

# Optional: Enable GPU acceleration
export QEMLFLOW_GPU=true

# Optional: Experiment tracking
export WANDB_API_KEY="your_wandb_key"
```

### Configuration File

Create `qemlflow_config.yaml`:

```yaml
environment: development
debug_mode: false
data_directory: "./data"
cache_directory: "./cache"

models:
  default_model_type: "random_forest"
  validation_split: 0.2
  random_state: 42

preprocessing:
  default_descriptor_type: "morgan"
  fingerprint_radius: 2
  fingerprint_bits: 2048

visualization:
  default_backend: "matplotlib"
  figure_size: [10, 6]
  style: "seaborn-v0_8"

experiment_tracking:
  enabled: true
  default_project: "qemlflow-experiments"
```

## Next Steps

1. **[User Guide](../user-guide/overview.md)** - Learn about all QeMLflow features
2. **[Examples](../examples/basic.md)** - See more detailed examples
3. **[API Reference](../api/core.md)** - Explore the complete API
4. **[Notebooks](../examples/notebooks.md)** - Interactive Jupyter examples

## Need Help?

- 📖 Check the [User Guide](../user-guide/overview.md)
- 💬 Join [GitHub Discussions](https://github.com/SanjeevaRDodlapati/QeMLflow/discussions)
- 🐛 Report issues on [GitHub](https://github.com/SanjeevaRDodlapati/QeMLflow/issues)
- 📧 Contact the maintainers
