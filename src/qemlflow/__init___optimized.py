from typing import Any

"""
QeMLflow: Machine Learning for Chemistry and Drug Discovery
Fast-loading version with optimized imports.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")

# __version__ = "0.2.0"
# __author__ = "QeMLflow Team"

from qemlflow.core.data import load_sample_data, quick_clean, quick_split
from qemlflow.core.evaluation import quick_classification_eval, quick_regression_eval

# Fast imports - core functionality only
from qemlflow.utils.lazy_imports import lazy_import

# Lazy imports for everything else
# _core_featurizers = lazy_import("qemlflow.core.featurizers")
# _core_models = lazy_import("qemlflow.core.models")
research = lazy_import("qemlflow.research")
integrations = lazy_import("qemlflow.integrations")


# Provide quick access to most-used functions
def morgan_fingerprints(*args, **kwargs) -> Any:
    """Generate Morgan fingerprints (lazy loaded)."""
    return _core_featurizers.morgan_fingerprints(*args, **kwargs)


def molecular_descriptors(*args, **kwargs) -> Any:
    """Calculate molecular descriptors (lazy loaded)."""
    return _core_featurizers.molecular_descriptors(*args, **kwargs)


def create_rf_model(*args, **kwargs) -> Any:
    """Create random forest model (lazy loaded)."""
    return _core_models.create_rf_model(*args, **kwargs)


def compare_models(*args, **kwargs) -> Any:
    """Compare ML models (lazy loaded)."""
    return _core_models.compare_models(*args, **kwargs)


# Fast setup
def _setup_qemlflow() -> None:
    """Fast QeMLflow initialization."""
    print("QeMLflow initialized successfully!")
    print(f"Version: {__version__}")
    print("⚡ Fast-loading mode active")


_setup_qemlflow()

__all__ = [
    "morgan_fingerprints",
    "molecular_descriptors",
    "create_rf_model",
    "compare_models",
    "load_sample_data",
    "quick_clean",
    "quick_split",
    "quick_classification_eval",
    "quick_regression_eval",
    "research",
    "integrations",
]
