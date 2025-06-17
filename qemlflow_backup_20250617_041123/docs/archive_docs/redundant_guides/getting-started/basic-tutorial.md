# Basic Tutorial

This tutorial will walk you through the fundamental concepts and usage patterns of ChemML.

## Overview

ChemML provides a comprehensive framework for molecular machine learning. This tutorial covers:

1. **Data Loading and Preprocessing**
2. **Molecular Descriptors and Fingerprints**
3. **Model Training and Evaluation**
4. **Making Predictions**
5. **Experiment Tracking**

## Prerequisites

Before starting, ensure you have ChemML installed:

```bash
pip install chemml
```

## Tutorial Steps

### Step 1: Import ChemML

```python
import chemml
import pandas as pd
import numpy as np

print(f"ChemML version: {chemml.__version__}")
```

### Step 2: Load Sample Data

```python
from chemml.datasets import load_sample_molecules

# Load built-in sample dataset
molecules, properties = load_sample_molecules()

print(f"Loaded {len(molecules)} molecules")
print(f"Target properties shape: {properties.shape}")
```

### Step 3: Generate Molecular Descriptors

```python
from chemml.preprocessing import MolecularDescriptors

# Create descriptor generator
descriptor_gen = MolecularDescriptors(
    descriptor_type='morgan',
    radius=2,
    n_bits=2048
)

# Generate descriptors
X = descriptor_gen.fit_transform(molecules)
print(f"Generated descriptors shape: {X.shape}")
```

### Step 4: Split Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, properties, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### Step 5: Train a Model

```python
from chemml.models import AutoMLRegressor

# Create AutoML model
model = AutoMLRegressor(
    time_budget=120,  # 2 minutes
    random_state=42
)

# Train the model
model.fit(X_train, y_train)
print("Model training completed!")
```

### Step 6: Evaluate Performance

```python
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")
```

### Step 7: Visualize Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs Actual')
plt.show()
```

## Advanced Topics

### Experiment Tracking

```python
from chemml.monitoring import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(project="tutorial-project")

# Track experiment
with tracker.track_experiment("automl_model"):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # Log metrics
    tracker.log_metric("r2_score", score)
    tracker.log_metric("mse", mse)

    # Log model
    tracker.log_model(model, "tutorial_model")
```

### Custom Preprocessing Pipeline

```python
from chemml.preprocessing import MolecularPreprocessor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create preprocessing pipeline
preprocessor = Pipeline([
    ('descriptors', MolecularDescriptors()),
    ('scaler', StandardScaler())
])

# Apply preprocessing
X_processed = preprocessor.fit_transform(molecules)
```

### Ensemble Methods

```python
from chemml.ensemble import EnsembleRegressor

# Create ensemble model
ensemble = EnsembleRegressor(
    models=['random_forest', 'xgboost', 'neural_network'],
    voting='soft'
)

# Train ensemble
ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)
print(f"Ensemble R²: {ensemble_score:.4f}")
```

## Next Steps

1. **[User Guide](../user-guide/overview.md)** - Explore advanced features
2. **[API Reference](../api/core.md)** - Detailed API documentation
3. **[Examples](../examples/basic.md)** - More practical examples
4. **[Contributing](../development/contributing.md)** - Contribute to ChemML

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or use feature selection
3. **Performance Issues**: Consider using GPU acceleration or distributed computing

### Getting Help

- **GitHub Issues**: [Report problems](https://github.com/SanjeevaRDodlapati/ChemML/issues)
- **Discussions**: [Ask questions](https://github.com/SanjeevaRDodlapati/ChemML/discussions)
- **Documentation**: [Full documentation](https://sanjeevardodlapati.github.io/ChemML/)
