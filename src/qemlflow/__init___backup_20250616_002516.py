"""
QeMLflow: Machine Learning for Chemistry and Drug Discovery

A comprehensive library providing machine learning tools specifically designed
for chemistry and drug discovery applications.

Main Modules:
- core: Essential functionality (featurization, modeling, data processing)
- research: Advanced research modules (drug discovery, clinical research, environmental chemistry, materials discovery)
- integrations: Third-party library integrations (DeepChem, RDKit, pipeline systems)

Quick Start:
    >>> import qemlflow
    >>> from qemlflow.core import featurizers, models, data
    >>>
    >>> # Create molecular fingerprints
    >>> smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']
    >>> fp = featurizers.morgan_fingerprints(smiles)
    >>>
    >>> # Train a model
    >>> model = models.create_rf_model()
    >>> # ... continue with your workflow
"""

# Suppress common warnings for cleaner import experience
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")

#__version__ = "0.2.0"
#__author__ = "QeMLflow Team"

# Core imports - these are fast and always needed
from .core import data, evaluation, featurizers, models, utils
from .core.data import load_sample_data, quick_clean, quick_split
from .core.evaluation import quick_classification_eval, quick_regression_eval

# Make key functions directly accessible
from .core.featurizers import (
    comprehensive_features,
    molecular_descriptors,
    morgan_fingerprints,
)
from .core.models import (
    compare_models,
    create_linear_model,
    create_rf_model,
    create_svm_model,
)
from .core.utils import ensure_reproducibility, setup_logging

# Import lazy loading utilities
from .utils.lazy_imports import check_dependencies, lazy_import

try:
    from .core.utils import check_environment
except ImportError:
    check_environment = None

# Lazy import for heavy modules
research = lazy_import("qemlflow.research")
integrations = lazy_import("qemlflow.integrations")

# Lazy imports for enhanced features (conditional)
#_enhanced_features = lazy_import("qemlflow.core.ensemble_advanced")
#_monitoring = lazy_import("qemlflow.core.monitoring")
#_recommendations = lazy_import("qemlflow.core.recommendations")
#_workflow_optimizer = lazy_import("qemlflow.core.workflow_optimizer")

# Check what's available
#_available_deps = check_dependencies()
HAS_ENHANCED_FEATURES = True  # Will be checked dynamically


# Setup basic configuration
def _setup_qemlflow() -> None:
    """Setup QeMLflow with sensible defaults."""
    import warnings

    # Configure warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Print welcome message
    print("QeMLflow initialized successfully!")
    print(f"Version: {__version__}")

    # Check environment
    if check_environment is not None:
        try:
            env_info = check_environment()
            missing_core = [
                pkg
                for pkg in ["numpy", "pandas", "sklearn"]
                if pkg in env_info["missing_packages"]
            ]

            if missing_core:
                warnings.warn(f"Missing core packages: {missing_core}")
        except Exception:
            # Environment check failed, continue gracefully
            pass

    # Check lazy-loaded modules
    if not research.is_available():
        print("Note: Research modules not available (install optional dependencies)")

    if not integrations.is_available():
        print("Note: Integration modules not available (install optional dependencies)")


# Initialize QeMLflow
_setup_qemlflow()

#__all__ = [
    # Core modules
    "featurizers",
    "models",
    "data",
    "evaluation",
    "utils",
    # Quick access functions
    "morgan_fingerprints",
    "molecular_descriptors",
    "comprehensive_features",
    "create_linear_model",
    "create_rf_model",
    "create_svm_model",
    "compare_models",
    "quick_clean",
    "quick_split",
    "load_sample_data",
    "quick_regression_eval",
    "quick_classification_eval",
    "setup_logging",
    "check_environment",
    "ensure_reproducibility",
    # Lazy-loaded modules
    "research",
    "integrations",
]
