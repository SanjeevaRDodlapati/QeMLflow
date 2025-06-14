# ChemML Weights & Biases Integration Guide

## 🚀 Quick Start

Your wandb API key has been integrated into the ChemML codebase for comprehensive experiment tracking.

### 1. Basic Setup

```python
import wandb
import os

# Your API key is already configured
os.environ['WANDB_API_KEY'] = 'b4f102d87161194b68baa7395d5862aa3f93b2b7'

# Login to wandb
wandb.login(key='b4f102d87161194b68baa7395d5862aa3f93b2b7', relogin=True)
```

### 2. Start an Experiment

```python
# Initialize experiment
run = wandb.init(
    project="chemml-experiments",
    name="my_experiment_name",
    config={
        "model_type": "random_forest",
        "dataset": "molecular_properties",
        "learning_rate": 0.01
    },
    tags=["chemml", "molecular_ml"]
)
```

### 3. Log Metrics

```python
# Log training metrics
wandb.log({
    "epoch": 1,
    "train_loss": 0.5,
    "val_loss": 0.6,
    "accuracy": 0.85
})

# Log with step
wandb.log({"loss": 0.4}, step=epoch)
```

### 4. Log Model Performance

```python
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Your model predictions
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2])

# Calculate and log metrics
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

wandb.log({
    "r2_score": r2,
    "mae": mae,
    "rmse": np.sqrt(np.mean((y_true - y_pred)**2))
})

# Create and log plots
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predictions vs Actual')

# Log plot to wandb
wandb.log({"prediction_plot": wandb.Image(plt)})
plt.close()
```

### 5. Log Molecular Data

```python
# Log molecular dataset information
molecular_data = {
    "num_molecules": 1000,
    "avg_molecular_weight": 342.5,
    "avg_logp": 2.1,
    "drug_like_percentage": 78.5
}
wandb.log(molecular_data)

# Log SMILES and properties as table
import pandas as pd

df = pd.DataFrame({
    'smiles': ['CCO', 'CCC', 'CC(C)O'],
    'molecular_weight': [46.07, 44.10, 60.10],
    'logp': [-0.31, 0.25, 0.05]
})

wandb.log({"molecular_data": wandb.Table(dataframe=df)})
```

### 6. Save Model Artifacts

```python
# Save model file
import joblib

# Train your model
model = SomeModel()
model.fit(X_train, y_train)

# Save model
model_path = "model.pkl"
joblib.dump(model, model_path)

# Log as wandb artifact
artifact = wandb.Artifact("trained_model", type="model")
artifact.add_file(model_path)
run.log_artifact(artifact)
```

### 7. Finish Experiment

```python
# End the experiment
wandb.finish()
```

## 🧪 Complete Example

```python
import wandb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Setup wandb
wandb.login(key='b4f102d87161194b68baa7395d5862aa3f93b2b7', relogin=True)

# Configuration
config = {
    "model_type": "random_forest",
    "n_estimators": 100,
    "max_depth": 10,
    "dataset": "molecular_properties"
}

# Start experiment
run = wandb.init(
    project="chemml-experiments",
    name="molecular_property_prediction",
    config=config,
    tags=["molecular", "regression", "example"]
)

# Generate sample data
X = np.random.randn(100, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)

# Train model
model = RandomForestRegressor(
    n_estimators=config["n_estimators"],
    max_depth=config["max_depth"]
)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# Log metrics
wandb.log({
    "r2_score": r2,
    "mae": mae,
    "final_performance": r2
})

# Create prediction plot
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Model Predictions (R² = {r2:.3f})')
plt.grid(True, alpha=0.3)

# Log plot
wandb.log({"prediction_scatter": wandb.Image(plt)})

print(f"✅ Experiment completed!")
print(f"📊 View at: {run.url}")

# Finish
wandb.finish()
```

## 📊 Integration Status

✅ **Configured Files:**
- `src/chemml_common/wandb_integration.py` - Main integration module
- `src/chemml_common/tracking.py` - Advanced tracking features
- `src/chemml_common/wandb_config.json` - Configuration file
- `setup_wandb_integration.py` - Setup script

✅ **Modified Files with wandb:**
- `src/drug_design/admet_prediction.py`
- `src/models/classical_ml/regression_models.py`
- `src/models/quantum_ml/quantum_circuits.py`
- `quick_access_demo.py`
- `notebooks/tutorials/03_deepchem_drug_discovery.ipynb`

## 🔗 Links

- **Project Dashboard**: https://wandb.ai/projects/chemml-experiments
- **Your Experiments**: https://wandb.ai/sdodlapa/chemml-experiments
- **Documentation**: https://docs.wandb.ai/

## 💡 Tips

1. **Tag your experiments** for better organization
2. **Use descriptive experiment names**
3. **Log hyperparameters in config**
4. **Save important artifacts** (models, datasets)
5. **Create plots and visualizations**
6. **Use wandb.summary** for final results

## 🎯 Next Steps

1. Run the example script: `python examples/wandb_example.py`
2. Integrate wandb into your existing experiments
3. Use the tracking module for advanced features
4. Explore the wandb dashboard to analyze results
5. Set up experiment sweeps for hyperparameter optimization

Your wandb integration is ready! Start tracking your ChemML experiments today. 🚀
