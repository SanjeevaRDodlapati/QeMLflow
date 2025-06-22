# QeMLflow Enhanced Features Guide

## Overview

QeMLflow now includes significantly enhanced capabilities for machine learning in chemistry and drug discovery. This guide covers the new features and how to use them effectively.

## 🚀 New Features

### 1. Enhanced Data Processing

#### Advanced Data Loading
```python
from qemlflow.core.data_processing import QeMLflowDataLoader

# Load popular chemistry datasets
loader = QeMLflowDataLoader()
df = loader.load_dataset('bbbp')  # Blood-brain barrier permeability
df = loader.load_dataset('qm9')   # Quantum machine 9
df = loader.load_dataset('tox21') # Toxicology data

# Load custom datasets
df = loader.load_custom_dataset('my_data.csv', smiles_column='SMILES')
```

#### Intelligent Data Preprocessing
```python
from qemlflow.core.data_processing import AdvancedDataPreprocessor

preprocessor = AdvancedDataPreprocessor()
processed_data = preprocessor.create_preprocessing_pipeline(
    df,
    smiles_column='smiles',
    target_columns=['target']
)

# Access processed features
X = processed_data['X']  # Combined molecular and additional features
y = processed_data['y']  # Processed targets
```

#### Smart Data Splitting
```python
from qemlflow.core.data_processing import IntelligentDataSplitter

splitter = IntelligentDataSplitter()

# Scaffold-based splitting (prevents data leakage)
train_idx, test_idx = splitter.scaffold_split(
    smiles=df['smiles'],
    test_size=0.2
)

# Temporal splitting (for time-series data)
train_idx, test_idx = splitter.temporal_split(
    timestamps=df['timestamp'],
    test_size=0.2
)

# Stratified splitting (maintains class balance)
train_idx, test_idx = splitter.stratified_split(
    targets=df['target'],
    test_size=0.2
)
```

### 2. Extended Model Suite

#### Ensemble Methods
```python
from qemlflow.core.enhanced_models import create_ensemble_model

# Create voting ensemble
ensemble = create_ensemble_model(
    ensemble_method='voting',
    voting_strategy='soft'
)

# Create stacking ensemble
ensemble = create_ensemble_model(
    ensemble_method='stacking',
    cv_folds=5
)

ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

#### Gradient Boosting Models
```python
from qemlflow.core.enhanced_models import create_xgboost_model

# XGBoost with automatic task detection
xgb_model = create_xgboost_model(task_type='regression')
xgb_model.fit(X_train, y_train)

# LightGBM for faster training
from qemlflow.core.enhanced_models import create_lightgbm_model
lgb_model = create_lightgbm_model(task_type='classification')
```

#### AutoML Pipeline
```python
from qemlflow.core.enhanced_models import create_automl_model

# Automated model selection and hyperparameter optimization
automl = create_automl_model(
    task_type='regression',
    n_trials=50,
    model_types=['rf', 'xgb', 'lgb', 'svm'],
    optimization_metric='rmse'
)

# Fit and get best model
automl_results = automl.fit(X_train, y_train)
best_model = automl.get_best_model()
best_params = automl.get_best_params()
```

### 3. Advanced Ensemble Methods

#### Adaptive Ensemble
```python
from qemlflow.core.ensemble_advanced import AdaptiveEnsemble

# Ensemble that adapts weights based on molecular similarity
adaptive_ensemble = AdaptiveEnsemble(
    base_models=[rf_model, xgb_model, nn_model],
    adaptation_strategy='performance_weighted',
    uncertainty_quantification=True
)

adaptive_ensemble.fit(X_train, y_train, molecular_features=fingerprints)
predictions, uncertainties = adaptive_ensemble.predict(X_test, return_uncertainty=True)
```

#### Multi-Modal Ensemble
```python
from qemlflow.core.ensemble_advanced import MultiModalEnsemble

# Combine different molecular representations
multimodal = MultiModalEnsemble(
    modality_models={
        'fingerprints': rf_model,
        'descriptors': xgb_model,
        'images': cnn_model
    },
    fusion_strategy='late_fusion'
)

modality_data = {
    'fingerprints': morgan_fps,
    'descriptors': mol_descriptors,
    'images': mol_images
}

multimodal.fit(modality_data, y_train)
```

### 4. Automated ML Pipelines

#### Quick Pipeline
```python
from qemlflow.core.pipeline import quick_pipeline

# One-line ML pipeline
results = quick_pipeline(
    data_source='bbbp',  # Dataset name or DataFrame
    task_type='classification',
    smiles_column='smiles',
    target_columns=['p_np']
)

print(results[['model', 'type', 'accuracy', 'f1_score']])
```

#### Comprehensive Pipeline
```python
from qemlflow.core.pipeline import create_pipeline

# Create customizable pipeline
pipeline = create_pipeline(
    task_type='regression',
    preprocessing_config={
        'feature_engineering': True,
        'molecular_descriptors': True,
        'fingerprint_type': 'morgan'
    },
    model_config={
        'ensemble_methods': True,
        'automl': True,
        'cross_validation': 5
    },
    experiment_tracking={
        'use_wandb': True,
        'project_name': 'chemical_ml'
    }
)

# Run pipeline
results = pipeline.run(
    df,
    smiles_column='smiles',
    target_columns=['solubility']
)
```

## 🔧 Configuration Options

### Data Processing Configuration
```python
preprocessing_config = {
    'feature_engineering': {
        'molecular_descriptors': True,
        'fingerprint_types': ['morgan', 'topological'],
        'fingerprint_params': {'radius': 2, 'nBits': 2048}
    },
    'data_splitting': {
        'method': 'scaffold',
        'test_size': 0.2,
        'validation_size': 0.1
    },
    'scaling': {
        'method': 'standard',
        'apply_to': ['descriptors', 'fingerprints']
    }
}
```

### Model Configuration
```python
model_config = {
    'ensemble_methods': {
        'voting': True,
        'stacking': True,
        'adaptive': True
    },
    'gradient_boosting': {
        'xgboost': True,
        'lightgbm': True,
        'catboost': False
    },
    'automl': {
        'enabled': True,
        'n_trials': 100,
        'optimization_metric': 'rmse',
        'early_stopping': 10
    }
}
```

## 📊 Performance Optimization

### Import Optimization
QeMLflow now features ultra-fast imports (~0.02s) with lazy loading:

```python
import qemlflow  # Fast import
qemlflow.enable_fast_mode()  # Pre-load for power users
```

### Memory Management
```python
# Clear caches to save memory
qemlflow.clear_cache()

# Use streaming for large datasets
from qemlflow.core.data_processing import QeMLflowDataLoader
loader = QeMLflowDataLoader()
data_stream = loader.load_dataset_stream('large_dataset', chunk_size=1000)
```

### Parallel Processing
```python
# Enable parallel processing
pipeline = create_pipeline(
    task_type='regression',
    parallel_config={
        'n_jobs': -1,  # Use all CPU cores
        'backend': 'multiprocessing'
    }
)
```

## 🧪 Chemistry-Specific Features

### Molecular Feature Engineering
```python
from qemlflow.core.data_processing import AdvancedDataPreprocessor

preprocessor = AdvancedDataPreprocessor()

# Automatic molecular feature generation
molecular_features = preprocessor._generate_molecular_features(smiles_series)

# Features include:
# - Molecular weight, LogP, hydrogen bond donors/acceptors
# - Topological polar surface area (TPSA)
# - Rotatable bonds, aromatic rings
# - Morgan fingerprint bits
```

### Chemical Data Validation
```python
# Automatic SMILES validation and cleanup
valid_df = loader._validate_smiles(df, smiles_column='smiles')

# Remove invalid molecules
# Clean malformed structures
# Report data quality statistics
```

### Drug-like Property Filters
```python
from qemlflow.core.data_processing import apply_drug_filters

# Apply Lipinski's Rule of Five
filtered_df = apply_drug_filters(df, filters=['lipinski', 'veber'])
```

## 🎯 Best Practices

### 1. Data Splitting
- Use scaffold splitting for molecular data to prevent data leakage
- Consider temporal splitting for time-series chemical data
- Always validate your splitting strategy

### 2. Model Selection
- Start with ensemble methods for robust performance
- Use AutoML for quick baseline models
- Consider multi-modal approaches for complex data

### 3. Feature Engineering
- Combine multiple molecular representations
- Use domain knowledge to guide feature selection
- Validate features with chemical intuition

### 4. Experiment Tracking
- Use Weights & Biases integration for experiment tracking
- Save model configurations and results
- Compare multiple approaches systematically

## 🐛 Troubleshooting

### Common Issues

#### RDKit Warnings
```python
# Warnings are now automatically suppressed
# Updated to use latest RDKit APIs (MorganGenerator)
```

#### Cross-Validation Errors
```python
# Enhanced cross-validation with robust error handling
# Automatic fallback strategies for small datasets
# Stratified CV for classification, regular CV for regression
```

#### Memory Issues
```python
# Use data streaming for large datasets
# Enable memory-efficient processing
# Clear caches regularly
```

## 📈 Performance Benchmarks

### Import Speed
- Previous: ~2-5 seconds
- Current: ~0.02 seconds (100x faster)

### Model Training
- Ensemble methods: 2-5x faster with parallel processing
- AutoML: Robust optimization with early stopping
- Cross-validation: Enhanced error handling and speed

### Memory Usage
- Optimized data structures
- Lazy loading of optional dependencies
- Efficient caching strategies

## 🔮 Future Enhancements

### Planned Features
- Graph neural networks for molecular data
- Active learning for experimental design
- Multi-objective optimization
- Federated learning for collaborative research
- Enhanced visualization tools

### Roadmap
- Q2 2025: Graph neural networks
- Q3 2025: Active learning framework
- Q4 2025: Federated learning support

## 📚 Additional Resources

- [API Reference](API_REFERENCE.md)
- [Getting Started Guide](GET_STARTED.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Example Notebooks](../notebooks/examples/)
- [Research Applications](../notebooks/research/)

## 🤝 Contributing

We welcome contributions! See our [contribution guidelines](../CONTRIBUTING.md) for details on:
- Adding new models
- Improving data processing
- Writing documentation
- Reporting bugs and feature requests
