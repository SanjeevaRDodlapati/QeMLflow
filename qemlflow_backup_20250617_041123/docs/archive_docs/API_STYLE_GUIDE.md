# ChemML API Style Guide

## Parameter Naming Standards

### Data Parameters
- **Primary data**: Use `data` for main dataset
- **Molecular data**: Use `molecules` or `molecular_data` consistently
- **Training data**: Use `X_train`, `y_train`
- **Validation data**: Use `X_val`, `y_val`
- **Test data**: Use `X_test`, `y_test`

### Model Parameters
- **Primary model**: Use `model`
- **Model type**: Use `model_type`
- **Multiple models**: Use `models`

### File Parameters
- **File path**: Use `filepath` (not `filename`, `file_path`, or `path`)
- **Save location**: Use `save_path`
- **Config file**: Use `config_path`

### Type Parameters
- **Task classification**: Use `task_type`
- **Property type**: Use `property_type`
- **Activity type**: Use `activity_type`

## Method Naming Standards

### Getters and Setters
- Use `get_` prefix for retrieving data/properties
- Use `set_` prefix for setting values
- Use `is_` prefix for boolean checks

### Processing Methods
- Use `process_` for data transformation
- Use `calculate_` for computations
- Use `generate_` for creating new data

### I/O Methods
- Use `load_` for reading data
- Use `save_` for writing data
- Use `export_` for converting formats

## Type Annotation Requirements

All public methods must have:
- Parameter type hints
- Return type annotations
- Docstrings with type information

## Error Handling Standards

### Custom Exceptions
```python
class ChemMLError(Exception):
    """Base exception for ChemML operations"""
    pass

class ChemMLDataError(ChemMLError):
    """Raised when data validation fails"""
    pass

class ChemMLModelError(ChemMLError):
    """Raised when model operations fail"""
    pass
```

### Exception Guidelines
- Never use bare `except:` clauses
- Use specific exception types
- Provide helpful error messages
- Log errors appropriately

## Interface Standards

### ML Classes
All machine learning classes should implement:
- `fit(X, y=None)` method
- `predict(X)` or `transform(X)` method
- `get_params()` and `set_params()` methods

### Context Managers
Use context managers for resource management:
```python
with ChemMLExperiment() as exp:
    exp.run_analysis()
```
