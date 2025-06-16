"""
ChemML Core Package

This package provides the core functionality for ChemML including
data processing, machine learning models, and evaluation tools.
"""

def __getattr__(name):
    """Dynamic import for lazy loading."""
    
    # Direct function mappings (fastest path)
    _direct_map = {
        "quick_clean": ("chemml.core.data", "quick_clean"),
        "quick_split": ("chemml.core.data", "quick_split"),
        "morgan_fingerprints": ("chemml.core.featurizers", "morgan_fingerprints"),
        "molecular_descriptors": ("chemml.core.featurizers", "molecular_descriptors"),
        "create_rf_model": ("chemml.core.models", "create_rf_model"),
        "create_linear_model": ("chemml.core.models", "create_linear_model"),
        "create_svm_model": ("chemml.core.models", "create_svm_model"),
        # Enhanced models
        "create_ensemble_model": ("chemml.core.enhanced_models", "create_ensemble_model"),
        "create_automl_model": ("chemml.core.enhanced_models", "create_automl_model"),
        "create_xgboost_model": ("chemml.core.enhanced_models", "create_xgboost_model"),
        "create_lightgbm_model": ("chemml.core.enhanced_models", "create_lightgbm_model"),
        "create_cnn_model": ("chemml.core.enhanced_models", "create_cnn_model"),
        # Data processing
        "load_chemical_dataset": ("chemml.core.data_processing", "load_chemical_dataset"),
        "preprocess_chemical_data": ("chemml.core.data_processing", "preprocess_chemical_data"),
        "split_chemical_data": ("chemml.core.data_processing", "split_chemical_data"),
        # Pipeline
        "create_pipeline": ("chemml.core.pipeline", "create_pipeline"),
        "quick_pipeline": ("chemml.core.pipeline", "quick_pipeline"),
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
        "featurizers": "chemml.core.featurizers",
        "models": "chemml.core.models",
        "enhanced_models": "chemml.core.enhanced_models",
        "data_processing": "chemml.core.data_processing",
        "pipeline": "chemml.core.pipeline",
        "data": "chemml.core.data",
        "evaluation": "chemml.core.evaluation",
        "utils": "chemml.core.utils",
    }

    if name in _module_map:
        import importlib
        module = importlib.import_module(_module_map[name])
        globals()[name] = module
        return module

    # Heavy modules (lazy load)
    _heavy_map = {
        "ensemble_advanced": "chemml.core.ensemble_advanced",
        "monitoring": "chemml.core.monitoring",
        "recommendations": "chemml.core.recommendations",
        "workflow_optimizer": "chemml.core.workflow_optimizer",
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

    raise AttributeError(f"module 'chemml.core' has no attribute '{name}'")


# Version compatibility
__all__ = ["load_sample_data", "quick_classification_eval", "quick_regression_eval"]
