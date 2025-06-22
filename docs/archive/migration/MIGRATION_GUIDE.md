# QeMLflow Import Migration Guide

## 🔄 New Modular Import Structure

As of QeMLflow v1.0.0, the drug discovery module has been restructured into focused, modular components for better maintainability and usability.

## 📋 Migration Overview

### Before (Legacy)
```python
# Old monolithic imports
from src.drug_design.property_prediction import predict_properties
from src.drug_design.admet_prediction import ADMETPredictor
from src.drug_design.virtual_screening import VirtualScreener
from src.drug_design.qsar_modeling import QSARModel
```

### After (New Modular Structure)
```python
# New modular imports
from qemlflow.research.drug_discovery.properties import predict_properties
from qemlflow.research.drug_discovery.admet import ADMETPredictor
from qemlflow.research.drug_discovery.screening import VirtualScreener
from qemlflow.research.drug_discovery.qsar import QSARModel
```

## 🗂️ Module Mapping

| **Functionality** | **Legacy Import** | **New Modular Import** |
|-------------------|-------------------|------------------------|
| **Molecular Optimization** | `src.drug_design.molecular_optimization` | `qemlflow.research.drug_discovery.molecular_optimization` |
| **ADMET Prediction** | `src.drug_design.admet_prediction` | `qemlflow.research.drug_discovery.admet` |
| **Virtual Screening** | `src.drug_design.virtual_screening` | `qemlflow.research.drug_discovery.screening` |
| **Property Prediction** | `src.drug_design.property_prediction` | `qemlflow.research.drug_discovery.properties` |
| **Molecular Generation** | `src.drug_design.molecular_generation` | `qemlflow.research.drug_discovery.generation` |
| **QSAR Modeling** | `src.drug_design.qsar_modeling` | `qemlflow.research.drug_discovery.qsar` |

## 🔧 Common Import Patterns

### 1. Class Imports
```python
# Legacy
from src.drug_design.admet_prediction import ADMETPredictor

# New
from qemlflow.research.drug_discovery.admet import ADMETPredictor
```

### 2. Function Imports
```python
# Legacy
from src.drug_design.property_prediction import predict_properties

# New
from qemlflow.research.drug_discovery.properties import predict_properties
```

### 3. Multiple Imports
```python
# Legacy
from src.drug_design.qsar_modeling import build_qsar_model, predict_activity

# New
from qemlflow.research.drug_discovery.qsar import build_qsar_model, predict_activity
```

### 4. Main Module Import (Convenience)
```python
# Still supported for backward compatibility
from qemlflow.research.drug_discovery import (
    MolecularOptimizer,
    ADMETPredictor,
    VirtualScreener,
    QSARModel
)
```

## 📝 Step-by-Step Migration

### Step 1: Update Import Statements
Replace all occurrences of `src.drug_design.*` with the corresponding `qemlflow.research.drug_discovery.*` path:

```bash
# Find and replace in your codebase
sed -i 's/from src\.drug_design\./from qemlflow.research.drug_discovery./g' *.py
sed -i 's/src\.drug_design\./qemlflow.research.drug_discovery./g' *.py
```

### Step 2: Module-Specific Replacements
| **Find** | **Replace** |
|----------|-------------|
| `src.drug_design.admet_prediction` | `qemlflow.research.drug_discovery.admet` |
| `src.drug_design.property_prediction` | `qemlflow.research.drug_discovery.properties` |
| `src.drug_design.virtual_screening` | `qemlflow.research.drug_discovery.screening` |
| `src.drug_design.qsar_modeling` | `qemlflow.research.drug_discovery.qsar` |
| `src.drug_design.molecular_generation` | `qemlflow.research.drug_discovery.generation` |
| `src.drug_design.molecular_optimization` | `qemlflow.research.drug_discovery.molecular_optimization` |

### Step 3: Test Your Changes
```python
# Verify imports work
from qemlflow.research.drug_discovery.admet import predict_admet_properties
from qemlflow.research.drug_discovery.properties import predict_properties

# Test functionality
molecules = ["CCO", "CCC"]
admet_results = predict_admet_properties(molecules)
properties = predict_properties(molecules)
print("✅ Migration successful!")
```

## 🔍 Verification Script

Use this script to verify your migration:

```python
#!/usr/bin/env python3
"""Verify QeMLflow import migration"""

def test_new_imports():
    try:
        # Test all new modular imports
        from qemlflow.research.drug_discovery.admet import ADMETPredictor
        from qemlflow.research.drug_discovery.properties import MolecularPropertyPredictor
        from qemlflow.research.drug_discovery.screening import VirtualScreener
        from qemlflow.research.drug_discovery.qsar import QSARModel
        from qemlflow.research.drug_discovery.generation import MolecularGenerator
        from qemlflow.research.drug_discovery.molecular_optimization import MolecularOptimizer

        print("✅ All new imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_new_imports()
```

## 🎯 Benefits of New Structure

1. **Modularity**: Each module focuses on specific functionality
2. **Maintainability**: Easier to maintain and update individual components
3. **Performance**: Import only what you need
4. **Clarity**: Clear separation of concerns
5. **Testing**: Better test isolation and coverage

## 🔄 Backward Compatibility

The main module import still works for backward compatibility:

```python
# This still works (but prefer modular imports)
from qemlflow.research.drug_discovery import ADMETPredictor, QSARModel
```

## 🆘 Troubleshooting

### Common Issues

1. **Import Error**: Make sure you've updated to QeMLflow v1.0.0+
2. **Module Not Found**: Verify you're using the correct new import paths
3. **Attribute Error**: Some function names may have changed - check the module documentation

### Getting Help
- Check the [QeMLflow Documentation](https://qemlflow.readthedocs.io)
- Visit our [GitHub Issues](https://github.com/qemlflow/qemlflow/issues)
- Join our community discussions

---

**Migration completed?** Run the validation script to ensure everything works:
```bash
python scripts/validation/phase_4_validation.py
```
