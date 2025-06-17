# Quick Start Guide

Welcome to ChemML! This guide will get you up and running in minutes.

## 🏃‍♂️ 5-Minute Quick Start

### 1. Installation
```bash
pip install chemml
```

### 2. Your First Prediction
```python
import chemml

# Load sample data
data = chemml.load_sample_data("molecules")
print(f"Loaded {len(data)} molecules")

# Generate features
features = chemml.morgan_fingerprints(data.smiles)
print(f"Generated {features.shape[1]} molecular features")

# Train a model
model = chemml.create_rf_model(features, data.targets)
print("Model trained successfully!")

# Make predictions
predictions = model.predict(features[:5])
print(f"Sample predictions: {predictions}")
```

### 3. Evaluate Results
```python
# Quick evaluation
results = chemml.quick_classification_eval(model, features, data.targets)
print(f"Model accuracy: {results.accuracy:.3f}")
print(f"Cross-validation score: {results.cv_score:.3f}")
```

## 🎯 Common Use Cases

### Molecular Property Prediction
```python
from chemml.core import featurizers, models

# Generate descriptors
descriptors = featurizers.molecular_descriptors(smiles_list)

# Train property prediction model
property_model = models.PropertyPredictor()
property_model.fit(descriptors, property_values)

# Predict new molecules
new_properties = property_model.predict(new_descriptors)
```

### Drug Discovery Screening
```python
from chemml.research.drug_discovery import VirtualScreening

# Setup virtual screening
screener = VirtualScreening(
    target="protein_target.pdb",
    compound_library="compounds.sdf"
)

# Run screening
hits = screener.screen(
    filters=["lipinski", "toxicity"],
    top_k=100
)

print(f"Found {len(hits)} potential drug candidates")
```

### Materials Property Optimization
```python
from chemml.research.materials_discovery import MaterialsOptimizer

# Define optimization problem
optimizer = MaterialsOptimizer(
    target_properties={"bandgap": 2.0, "stability": "high"},
    constraints=["non_toxic", "synthesizable"]
)

# Generate optimized materials
candidates = optimizer.optimize(
    starting_materials=seed_structures,
    generations=50
)

print(f"Generated {len(candidates)} optimized candidates")
```

## 🔧 Configuration

### Environment Setup
```python
import chemml

# Configure for your environment
chemml.config.set_backend("sklearn")  # or "xgboost", "tensorflow"
chemml.config.set_n_jobs(4)          # parallel processing
chemml.config.enable_caching(True)    # speed up repeated operations
```

### Performance Tuning
```python
# Enable fast mode for production
chemml.enable_fast_mode()

# Use GPU acceleration (if available)
chemml.config.enable_gpu(True)

# Set memory limits
chemml.config.set_memory_limit("8GB")
```

## ❓ Troubleshooting

### Common Issues

**ImportError: No module named 'rdkit'**
```bash
# Install RDKit dependency
conda install -c conda-forge rdkit
# or
pip install rdkit-pypi
```

**Memory errors with large datasets**
```python
# Use batch processing
for batch in chemml.utils.batch_iterator(large_dataset, batch_size=1000):
    results = process_batch(batch)
```

**Slow performance**
```python
# Enable performance optimizations
chemml.enable_fast_mode()
chemml.config.set_n_jobs(-1)  # use all CPU cores
```

## 🚀 Next Steps

1. **[Complete Tutorial](../tutorials/)**: Comprehensive learning path
2. **[API Reference](../reference/)**: Detailed function documentation
3. **[Examples](../../examples/)**: Real-world applications
4. **[Advanced Features](../advanced/)**: Expert-level functionality

## 💡 Tips for Success

- Start with sample data to understand the workflow
- Use built-in validation functions to check your results
- Leverage ChemML's caching for faster repeated operations
- Check the documentation for optimization tips
- Join our community discussions for help and best practices

Ready to dive deeper? Check out our [comprehensive tutorials](../tutorials/) or explore the [examples](../../examples/) directory!
