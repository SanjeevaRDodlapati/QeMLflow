"""
QeMLflow Core Package

This package provides the core functionality for QeMLflow including
data processing, machine learning models, and evaluation tools.
"""


def __getattr__(name):
    """Dynamic import for lazy loading."""

    # Direct function mappings (fastest path)
    _direct_map = {
        "quick_clean": ("qemlflow.core.data", "quick_clean"),
        "quick_split": ("qemlflow.core.data", "quick_split"),
        "morgan_fingerprints": ("qemlflow.core.featurizers", "morgan_fingerprints"),
        "molecular_descriptors": ("qemlflow.core.featurizers", "molecular_descriptors"),
        "create_rf_model": ("qemlflow.core.models", "create_rf_model"),
        "create_linear_model": ("qemlflow.core.models", "create_linear_model"),
        "create_svm_model": ("qemlflow.core.models", "create_svm_model"),
        # Enhanced models
        "create_ensemble_model": (
            "qemlflow.core.enhanced_models",
            "create_ensemble_model",
        ),
        "create_automl_model": ("qemlflow.core.enhanced_models", "create_automl_model"),
        "create_xgboost_model": (
            "qemlflow.core.enhanced_models",
            "create_xgboost_model",
        ),
        "create_lightgbm_model": (
            "qemlflow.core.enhanced_models",
            "create_lightgbm_model",
        ),
        "create_cnn_model": ("qemlflow.core.enhanced_models", "create_cnn_model"),
        # Data processing
        "load_chemical_dataset": (
            "qemlflow.core.data_processing",
            "load_chemical_dataset",
        ),
        "preprocess_chemical_data": (
            "qemlflow.core.data_processing",
            "preprocess_chemical_data",
        ),
        "split_chemical_data": ("qemlflow.core.data_processing", "split_chemical_data"),
        # Pipeline
        "create_pipeline": ("qemlflow.core.pipeline", "create_pipeline"),
        "quick_pipeline": ("qemlflow.core.pipeline", "quick_pipeline"),
    }

    if name in _direct_map:
        import importlib

        module_path, attr_name = _direct_map[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr

    # Module mappings
    _module_map = {
        "featurizers": "qemlflow.core.featurizers",
        "models": "qemlflow.core.models",
        "enhanced_models": "qemlflow.core.enhanced_models",
        "data_processing": "qemlflow.core.data_processing",
        "pipeline": "qemlflow.core.pipeline",
        "data": "qemlflow.core.data",
        "evaluation": "qemlflow.core.evaluation",
        "utils": "qemlflow.core.utils",
    }

    if name in _module_map:
        import importlib

        module = importlib.import_module(_module_map[name])
        globals()[name] = module
        return module

    # Heavy modules (lazy load)
    _heavy_map = {
        "ensemble_advanced": "qemlflow.core.ensemble_advanced",
        "monitoring": "qemlflow.core.monitoring",
        "recommendations": "qemlflow.core.recommendations",
        "workflow_optimizer": "qemlflow.core.workflow_optimizer",
    }

    if name in _heavy_map:
        import importlib

        module = importlib.import_module(_heavy_map[name])
        globals()[name] = module
        return module

    # Generic search (fallback)
    for module_path in _module_map.values():
        try:
            import importlib

            module = importlib.import_module(module_path)
            if hasattr(module, name):
                attr = getattr(module, name)
                globals()[name] = attr
                return attr
        except (ImportError, AttributeError):
            continue

    raise AttributeError(f"module 'qemlflow.core' has no attribute '{name}'")


# Version compatibility
__all__ = ["load_sample_data", "quick_classification_eval", "quick_regression_eval"]
