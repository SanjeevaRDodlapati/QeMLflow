# ChemML Universal Wandb Integration Guide

## 🎯 One-Command Experiment Tracking

ChemML now features a **Universal Tracker** that provides one-command experiment tracking across the entire codebase with automatic context detection and graceful fallbacks.

## 🚀 Quick Start

### New Universal Tracker (Recommended)

```python
from chemml_common.tracker import track_experiment, quick_track

# 1. As a decorator (auto-logs function arguments as hyperparameters)
@track_experiment(project="molecular_optimization")
def optimize_molecule(smiles, target_property, learning_rate=0.01):
    # Your code here
    results = run_optimization(smiles, target_property)
    return {"final_score": results.score}

# 2. As a context manager
with track_experiment(project="drug_discovery") as tracker:
    model = train_model(data)
    tracker.log({"accuracy": 0.95, "loss": 0.23})

# 3. Quick one-liner
tracker = quick_track("my_experiment")
tracker.log({"metric": 0.95})
tracker.finish()

# 4. Global tracking (perfect for notebooks)
from chemml_common.tracker import start_global_tracking, log_global
start_global_tracking("notebook_experiment")
log_global({"step1_result": 0.8})
```

### Traditional Wandb (Still Available)

```python
import wandb
import os

# Your API key is already configured
os.environ['WANDB_API_KEY'] = 'b4f102d87161194b68baa7395d5862aa3f93b2b7'

# Login to wandb
wandb.login(key='b4f102d87161194b68baa7395d5862aa3f93b2b7', relogin=True)

# Initialize experiment
run = wandb.init(
    project="chemml-experiments",
    name="my_experiment_name",
    config={"model_type": "random_forest"},
    tags=["chemml", "molecular_ml"]
)

# Log metrics
wandb.log({"epoch": 1, "train_loss": 0.5, "accuracy": 0.85})
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

## ✨ Universal Tracker Features

### 🧠 Smart Auto-Detection
- **Project names**: Automatically detects from file location (`drug_design/` → `chemml_drug_design`)
- **Experiment names**: Generated from function names and context
- **Framework tags**: Auto-detects PyTorch, TensorFlow, RDKit, DeepChem, etc.
- **Context tags**: Adds `jupyter_notebook`, `testing`, `tutorial` tags automatically

### 🛡️ Graceful Fallbacks
- Works even when wandb is not installed or available
- Handles network issues and API failures gracefully
- Provides informative error messages without breaking code

### 📊 Multiple Usage Patterns
1. **Decorator**: `@track_experiment(project="my_project")`
2. **Context Manager**: `with track_experiment() as tracker:`
3. **Quick Track**: `tracker = quick_track("experiment")`
4. **Global Tracking**: `start_global_tracking()` for notebooks

### 🔧 Advanced Capabilities
- **Model watching**: Automatic gradient and parameter tracking
- **Artifact logging**: Files, models, datasets
- **Hyperparameter logging**: Automatic from function arguments
- **Error tracking**: Automatic error logging and recovery

## 🎮 Usage Examples

### For Model Training

```python
from chemml_common.tracker import track_training

# Specialized training tracker
with track_training("gnn_model", {"lr": 0.001, "layers": 3}) as tracker:
    model = create_model()
    tracker.watch_model(model)  # Auto-track gradients

    for epoch in range(epochs):
        loss = train_epoch(model)
        tracker.log({"epoch": epoch, "loss": loss})
```

### For Model Evaluation

```python
from chemml_common.tracker import track_evaluation

with track_evaluation("gnn_model", "test_dataset") as tracker:
    results = evaluate_model(model, test_data)
    tracker.log({
        "accuracy": results["accuracy"],
        "f1_score": results["f1"],
        "confusion_matrix": results["cm"]
    })
```

### For Optimization Tasks

```python
from chemml_common.tracker import track_optimization

with track_optimization("genetic_algorithm", "binding_affinity") as tracker:
    optimizer = GeneticAlgorithm()

    for generation in range(generations):
        population = optimizer.evolve()
        best_score = max(population.scores)
        tracker.log({"generation": generation, "best_score": best_score})
```

### For Jupyter Notebooks

```python
# Cell 1: Setup
from chemml_common.tracker import start_global_tracking, log_global, finish_global_tracking
start_global_tracking("drug_discovery_analysis")

# Cell 2: Data preprocessing
data = preprocess_molecules(raw_data)
log_global({"data_size": len(data), "preprocessing_time": time_taken})

# Cell 3: Model training
model = train_model(data)
log_global({"training_accuracy": model.score, "model_params": model.n_params})

# Cell 4: Evaluation
test_results = evaluate_model(model, test_data)
log_global({"test_accuracy": test_results["accuracy"]})

# Last cell: Cleanup
finish_global_tracking()
```

## 🔗 Integration with ChemML Components

### Automatic Integration Examples

The universal tracker automatically integrates with existing ChemML code:

```python
# Before (existing ChemML code)
def molecular_property_prediction(smiles_list, target_property):
    model = train_model(smiles_list, target_property)
    predictions = model.predict(test_smiles)
    return evaluate_predictions(predictions)

# After (with one-line addition)
from chemml_common.tracker import track_experiment

@track_experiment(project="molecular_properties")  # Only addition needed!
def molecular_property_prediction(smiles_list, target_property):
    model = train_model(smiles_list, target_property)
    predictions = model.predict(test_smiles)
    return evaluate_predictions(predictions)
# Now automatically tracks: smiles_list, target_property as hyperparameters
# and the returned evaluation results as metrics
```

### Integration Status

The following ChemML components now have wandb integration:

- ✅ **Data Processing**: `src/data_processing/`
- ✅ **Drug Design**: `src/drug_design/`
- ✅ **Model Training**: `src/models/`
- ✅ **Notebooks**: `notebooks/tutorials/`
- ✅ **Quantum ML**: `src/models/quantum_ml/`
- ✅ **Utilities**: `src/utils/`

## 🎯 Best Practices

### 1. Use Descriptive Names

```python
# Good
@track_experiment(
    project="molecular_property_prediction",
    name="random_forest_baseline_v2",
    tags=["baseline", "production", "random_forest"]
)

# Better - auto-detected
@track_experiment(project="molecular_property_prediction")
def random_forest_baseline_v2(dataset, n_estimators=100):
    pass  # Name and hyperparameters auto-detected
```

### 2. Organize by Project

```python
# Organize experiments by research area
track_experiment(project="drug_discovery")      # For drug discovery
track_experiment(project="materials_science")   # For materials research
track_experiment(project="method_development")  # For new algorithms
```

### 3. Log Comprehensive Metrics

```python
with track_experiment(project="comprehensive_logging") as tracker:
    # Training metrics
    tracker.log({"train_loss": loss, "train_acc": acc})

    # Validation metrics
    tracker.log({"val_loss": val_loss, "val_acc": val_acc})

    # Model complexity
    tracker.log({"n_parameters": count_params(model)})

    # Resource usage
    tracker.log({"gpu_memory": get_gpu_memory(), "time_per_epoch": epoch_time})
```

### 4. Use Artifacts for Important Files

```python
with track_experiment(project="artifact_management") as tracker:
    # Log datasets
    tracker.log_artifact("processed_data.csv", name="dataset_v1", type="dataset")

    # Log models
    torch.save(model.state_dict(), "model.pth")
    tracker.log_artifact("model.pth", name="trained_model", type="model")

    # Log plots and visualizations
    create_loss_plot()
    tracker.log_artifact("loss_plot.png", name="training_curves", type="image")
```
