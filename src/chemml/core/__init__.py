"""
Core ChemML functionality - essential modules for chemistry ML workflows.

This module contains the fundamental building blocks:
- featurizers: Molecular feature extraction
- models: Machine learning models for chemistry
- data: Data processing and handling utilities
- evaluation: Model evaluation and metrics
- utils: Common utilities and helpers
"""

# Import main modules
from . import data, evaluation, featurizers, models, utils
from .data import (
    DataSplitter,
    FeatureScaler,
    MolecularDataProcessor,
    load_sample_data,
    quick_clean,
    quick_split,
)
from .evaluation import (
    ClassificationEvaluator,
    ModelComparator,
    RegressionEvaluator,
    quick_classification_eval,
    quick_regression_eval,
)

# Import key classes and functions for direct access
from .featurizers import (
    CombinedFeaturizer,
    DescriptorCalculator,
    ECFPFingerprint,
    MorganFingerprint,
    comprehensive_features,
    molecular_descriptors,
    morgan_fingerprints,
)
from .models import (
    LinearModel,
    RandomForestModel,
    SVMModel,
    compare_models,
    create_linear_model,
    create_rf_model,
    create_svm_model,
    setup_experiment_tracking,
)
from .utils import (
    check_environment,
    configure_warnings,
    ensure_reproducibility,
    get_sample_data,
    setup_logging,
)

__all__ = [
    "featurizers",
    "models",
    "data",
    "evaluation",
    "utils",
    # Featurizers
    "MorganFingerprint",
    "DescriptorCalculator",
    "ECFPFingerprint",
    "CombinedFeaturizer",
    "morgan_fingerprints",
    "molecular_descriptors",
    "comprehensive_features",
    # Models
    "LinearModel",
    "RandomForestModel",
    "SVMModel",
    "create_linear_model",
    "create_rf_model",
    "create_svm_model",
    "setup_experiment_tracking",
    "compare_models",
    # Data
    "MolecularDataProcessor",
    "DataSplitter",
    "FeatureScaler",
    "load_sample_data",
    "quick_clean",
    "quick_split",
    # Evaluation
    "RegressionEvaluator",
    "ClassificationEvaluator",
    "ModelComparator",
    "quick_regression_eval",
    "quick_classification_eval",
    # Utils
    "setup_logging",
    "check_environment",
    "get_sample_data",
    "ensure_reproducibility",
    "configure_warnings",
]
