"""
ChemML: Machine Learning for Chemistry and Drug Discovery

A comprehensive library providing machine learning tools specifically designed
for chemistry and drug discovery applications.

Main Modules:
- core: Essential functionality (featurization, modeling, data processing)
- research: Advanced research modules (drug discovery, clinical research, environmental chemistry, materials discovery)
- integrations: Third-party library integrations (DeepChem, RDKit, pipeline systems)

Quick Start:
    >>> import chemml
    >>> from chemml.core import featurizers, models, data
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

__version__ = "0.2.0"
__author__ = "ChemML Team"

# Core imports for easy access
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

try:
    from .core.utils import check_environment
except ImportError:
    check_environment = None

# Research modules (optional import)
try:
    from . import research

    _has_research = True
except ImportError:
    # Research modules may have additional dependencies
    research = None
    _has_research = False

# Integration modules (optional import)
try:
    from . import integrations

    _has_integrations = True
except ImportError:
    integrations = None
    _has_integrations = False

# Enhanced development features (if available)
try:
    from .core.ensemble_advanced import (
        AdaptiveEnsemble,
        MultiModalEnsemble,
        UncertaintyQuantifiedEnsemble,
        create_adaptive_ensemble,
        create_multimodal_ensemble,
        create_uncertainty_ensemble,
    )
    from .core.monitoring import (
        PerformanceDashboard,
        create_performance_dashboard,
        show_performance_dashboard,
    )
    from .core.recommendations import ModelRecommendationEngine
    from .core.recommendations import compare_models as compare_ml_models
    from .core.recommendations import recommend_model

    # Medium-term enhancements
    from .core.workflow_optimizer import (
        WorkflowOptimizer,
        compare_model_workflows,
        optimize_workflow,
    )

    HAS_ENHANCED_FEATURES = True
except ImportError:
    HAS_ENHANCED_FEATURES = False


# Setup basic configuration
def _setup_chemml():
    """Setup ChemML with sensible defaults."""
    import warnings

    # Configure warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Print welcome message
    print("ChemML initialized successfully!")
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

    if not _has_research:
        print("Note: Research modules not available (install optional dependencies)")

    if not _has_integrations:
        print("Note: Integration modules not available (install optional dependencies)")


# Initialize ChemML
_setup_chemml()

__all__ = [
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
    # Optional modules
    "research",
    "integrations",
]

# Add enhanced features to __all__ if available
if HAS_ENHANCED_FEATURES:
    __all__.extend(
        [
            "show_performance_dashboard",
            "create_performance_dashboard",
            "recommend_model",
            "compare_ml_models",
            "PerformanceDashboard",
            "ModelRecommendationEngine",
        ]
    )
