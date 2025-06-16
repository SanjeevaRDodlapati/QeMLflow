"""
ChemML: Machine Learning for Chemistry and Drug Discovery
Optimized for fast imports and lazy loading.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")

__version__ = "0.2.0"
__author__ = "ChemML Team"

# Core fast imports only - essential functions
from .core.data import load_sample_data, quick_clean, quick_split
from .core.evaluation import quick_classification_eval, quick_regression_eval

# Lazy imports using our system
from .utils.lazy_imports import lazy_import

# All other modules are lazy-loaded
core = lazy_import("chemml.core")
research = lazy_import("chemml.research")
integrations = lazy_import("chemml.integrations")

# Lazy-loaded specific modules
_featurizers = lazy_import("chemml.core.featurizers")
_models = lazy_import("chemml.core.models")
_utils = lazy_import("chemml.core.utils")


# Provide direct access to most commonly used functions via lazy loading
def morgan_fingerprints(*args, **kwargs) -> Any:
    """Generate Morgan fingerprints (lazy loaded)."""
    return _featurizers.morgan_fingerprints(*args, **kwargs)


def molecular_descriptors(*args, **kwargs) -> Any:
    """Calculate molecular descriptors (lazy loaded)."""
    return _featurizers.molecular_descriptors(*args, **kwargs)


def comprehensive_features(*args, **kwargs) -> Any:
    """Generate comprehensive molecular features (lazy loaded)."""
    return _featurizers.comprehensive_features(*args, **kwargs)


def create_rf_model(*args, **kwargs) -> Any:
    """Create random forest model (lazy loaded)."""
    return _models.create_rf_model(*args, **kwargs)


def create_linear_model(*args, **kwargs) -> Any:
    """Create linear model (lazy loaded)."""
    return _models.create_linear_model(*args, **kwargs)


def create_svm_model(*args, **kwargs) -> Any:
    """Create SVM model (lazy loaded)."""
    return _models.create_svm_model(*args, **kwargs)


def compare_models(*args, **kwargs) -> Any:
    """Compare ML models (lazy loaded)."""
    return _models.compare_models(*args, **kwargs)


def ensure_reproducibility(*args, **kwargs) -> Any:
    """Ensure reproducible results (lazy loaded)."""
    return _utils.ensure_reproducibility(*args, **kwargs)


def setup_logging(*args, **kwargs) -> Any:
    """Setup logging (lazy loaded)."""
    return _utils.setup_logging(*args, **kwargs)


# Quick setup function
def _setup_chemml() -> None:
    """Fast ChemML initialization."""
    print("ChemML initialized successfully!")
    print(f"Version: {__version__}")


# Initialize
_setup_chemml()

__all__ = [
    # Fast-loaded functions
    "load_sample_data",
    "quick_clean",
    "quick_split",
    "quick_classification_eval",
    "quick_regression_eval",
    # Lazy-loaded modules
    "core",
    "research",
    "integrations",
    # Lazy-loaded functions (most commonly used)
    "morgan_fingerprints",
    "molecular_descriptors",
    "comprehensive_features",
    "create_rf_model",
    "create_linear_model",
    "create_svm_model",
    "compare_models",
    "ensure_reproducibility",
    "setup_logging",
]
