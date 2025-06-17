# User Guide Overview

Welcome to the ChemML User Guide. This comprehensive guide covers all aspects of using ChemML for molecular machine learning.

## What is ChemML?

ChemML is a quantum-enhanced molecular machine learning framework designed for:

- **Molecular Property Prediction**
- **Drug Discovery and Design**
- **Materials Science Applications**
- **Chemical Process Optimization**
- **Quantum-Enhanced Computing**

## Core Capabilities

### 🧬 Molecular Processing
- **SMILES and SDF file support**
- **200+ molecular descriptors**
- **Advanced fingerprinting methods**
- **3D molecular representations**

### 🤖 Machine Learning
- **AutoML for automated model selection**
- **Ensemble methods and stacking**
- **Deep learning integration**
- **Quantum machine learning algorithms**

### 📊 Experiment Tracking
- **W&B and MLflow integration**
- **Automated metric logging**
- **Model versioning and storage**
- **Reproducible experiments**

### ⚡ Performance
- **GPU acceleration support**
- **Distributed computing**
- **Memory-efficient processing**
- **Production-ready deployment**

## Getting Started

### Quick Installation
```bash
pip install chemml
```

### Basic Usage
```python
import chemml
from chemml.models import AutoMLRegressor

# Load data
molecules, properties = chemml.datasets.load_sample_molecules()

# Train model
model = AutoMLRegressor()
model.fit(molecules, properties)

# Make predictions
predictions = model.predict(new_molecules)
```

## Guide Structure

This user guide is organized into the following sections:

### 📖 **Core Concepts**
- **[Data Processing](data-processing.md)** - Loading and preprocessing molecular data
- **[Model Training](model-training.md)** - Training and evaluating models
- **[AutoML](automl.md)** - Automated machine learning workflows
- **[Ensemble Methods](ensemble-methods.md)** - Combining multiple models

### 🔬 **Advanced Topics**
- **Quantum Machine Learning** - Quantum computing integration
- **Custom Models** - Building your own ML models
- **Production Deployment** - Deploying models in production
- **Performance Optimization** - Scaling and optimization

### 🛠️ **Practical Applications**
- **Drug Discovery** - ADMET prediction and lead optimization
- **Materials Science** - Property prediction for new materials
- **Process Optimization** - Chemical process parameter tuning
- **Reaction Prediction** - Predicting reaction outcomes

## Key Features

### AutoML Capabilities
ChemML provides state-of-the-art AutoML for molecular data:

```python
from chemml.models import AutoMLRegressor

model = AutoMLRegressor(
    time_budget=300,  # 5 minutes
    models=['rf', 'xgb', 'nn'],  # Model types to try
    feature_selection=True,      # Automatic feature selection
    hyperparameter_tuning=True   # Automated hyperparameter optimization
)
```

### Molecular Descriptors
Comprehensive molecular representation methods:

```python
from chemml.preprocessing import MolecularDescriptors

# Morgan fingerprints
morgan_desc = MolecularDescriptors(descriptor_type='morgan')

# RDKit descriptors
rdkit_desc = MolecularDescriptors(descriptor_type='rdkit_2d')

# Combined descriptors
combined_desc = MolecularDescriptors(descriptor_type='combined')
```

### Experiment Tracking
Built-in experiment management:

```python
from chemml.monitoring import ExperimentTracker

with ExperimentTracker(project="drug-discovery") as tracker:
    model.fit(X_train, y_train)
    tracker.log_metric("r2_score", model.score(X_test, y_test))
    tracker.log_model(model, "final_model")
```

## Configuration

### Global Configuration
ChemML can be configured globally:

```python
import chemml

# Set global configuration
chemml.config.set_n_jobs(4)           # Use 4 CPU cores
chemml.config.enable_gpu(True)        # Enable GPU acceleration
chemml.config.set_memory_limit(8192)  # 8GB memory limit
```

### Environment Variables
```bash
export CHEMML_CONFIG_PATH="./config.yaml"
export CHEMML_DATA_DIR="./data"
export CHEMML_CACHE_DIR="./cache"
export WANDB_API_KEY="your_wandb_key"
```

## Best Practices

### Data Preparation
1. **Clean your data** - Remove invalid SMILES, handle missing values
2. **Feature engineering** - Use appropriate molecular descriptors
3. **Data splitting** - Use stratified splitting for classification
4. **Validation strategy** - Use cross-validation for robust evaluation

### Model Selection
1. **Start with AutoML** - Let ChemML find the best models
2. **Try ensemble methods** - Often better than single models
3. **Consider domain knowledge** - Use chemical intuition
4. **Validate thoroughly** - Use appropriate metrics and validation sets

### Production Deployment
1. **Version your models** - Use experiment tracking
2. **Monitor performance** - Track model drift
3. **Scale appropriately** - Use distributed computing if needed
4. **Test thoroughly** - Validate in production-like environments

## Common Workflows

### Property Prediction Workflow
```python
# 1. Load and preprocess data
molecules, properties = load_molecules("data.csv")
X = preprocess_molecules(molecules)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, properties)

# 3. Train model
model = AutoMLRegressor()
model.fit(X_train, y_train)

# 4. Evaluate
score = model.score(X_test, y_test)
print(f"R² score: {score:.3f}")

# 5. Make predictions
new_predictions = model.predict(new_molecules)
```

### Drug Discovery Workflow
```python
# 1. Load compound library
compounds = load_compound_library("chembl.sdf")

# 2. Generate ADMET descriptors
admet_descriptors = generate_admet_descriptors(compounds)

# 3. Train ADMET models
solubility_model = train_solubility_model(admet_descriptors)
toxicity_model = train_toxicity_model(admet_descriptors)

# 4. Screen compounds
promising_compounds = screen_compounds(
    compounds, solubility_model, toxicity_model
)
```

## Getting Help

- **[Quick Start](../getting-started/quick-start.md)** - Get up and running quickly
- **[Basic Tutorial](../getting-started/basic-tutorial.md)** - Step-by-step tutorial
- **[API Reference](../api/core.md)** - Detailed API documentation
- **[Examples](../examples/basic.md)** - Practical examples
- **[GitHub Issues](https://github.com/SanjeevaRDodlapati/ChemML/issues)** - Report bugs or ask questions

## Contributing

ChemML is an open-source project. We welcome contributions:

- **[Contributing Guide](../development/contributing.md)** - How to contribute
- **[GitHub Repository](https://github.com/SanjeevaRDodlapati/ChemML)** - Source code
- **[Discussions](https://github.com/SanjeevaRDodlapati/ChemML/discussions)** - Community discussions
