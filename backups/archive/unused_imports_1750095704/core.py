"""
QeMLflow Tutorial Framework - Core Module
=======================================

Provides environment setup, data loading, and basic tutorial infrastructure.
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Import QeMLflow core modules
from qemlflow.core.utils import check_environment as core_check_environment
from qemlflow.core.utils import setup_logging as core_setup_logging

# Optional imports with fallbacks
try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Some tutorial features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn(
        "Matplotlib/Seaborn not available. Visualization features will be limited."
    )

# Configure logging for tutorials
logger = logging.getLogger(__name__)


def setup_learning_environment(
    level: str = "INFO",
    style: str = "seaborn",
    random_seed: int = 42,
    suppress_warnings: bool = True,
) -> Dict[str, Any]:
    """
    Setup the learning environment for QeMLflow tutorials.

    This function configures logging, plotting style, random seeds,
    and checks available dependencies for a consistent tutorial experience.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        style: Plotting style for matplotlib
        random_seed: Random seed for reproducibility
        suppress_warnings: Whether to suppress common warnings

    Returns:
        Dictionary containing environment status and configuration
    """
    # Setup logging
    core_setup_logging(level=level)
    logger.info("Setting up QeMLflow learning environment")

    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(random_seed)

    # Configure plotting if available
    if PLOTTING_AVAILABLE:
        try:
            plt.style.use(style if style != "seaborn" else "default")
            sns.set_palette("husl")
            logger.info(f"Plotting configured with style: {style}")
        except Exception as e:
            logger.warning(f"Could not set plotting style: {e}")

    # Suppress common warnings if requested
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Check environment status
    env_status = _check_tutorial_environment()

    # Create tutorial directories if they don't exist
    _setup_tutorial_directories()

    logger.info("Learning environment setup complete")
    return {
        "status": "ready",
        "random_seed": random_seed,
        "logging_level": level,
        "plotting_available": PLOTTING_AVAILABLE,
        "environment": env_status,
        "tutorial_directories": _get_tutorial_directories(),
    }


def load_tutorial_data(
    dataset_name: str, cache_dir: Optional[str] = None, force_download: bool = False
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Load educational datasets for QeMLflow tutorials.

    Args:
        dataset_name: Name of the dataset to load
        cache_dir: Directory to cache downloaded data
        force_download: Whether to force re-download of cached data

    Returns:
        Loaded dataset as DataFrame or dictionary
    """
    from .data import EducationalDatasets

    logger.info(f"Loading tutorial dataset: {dataset_name}")

    # Initialize datasets manager
    datasets = EducationalDatasets(cache_dir=cache_dir)

    # Load requested dataset
    if dataset_name == "molecular_properties":
        return datasets.load_molecular_properties()
    elif dataset_name == "drug_molecules":
        return datasets.load_drug_molecules()
    elif dataset_name == "qm9_sample":
        return datasets.load_qm9_sample()
    elif dataset_name == "synthetic_molecules":
        return datasets.create_synthetic_examples()
    else:
        available = datasets.list_available_datasets()
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")


def create_interactive_demo(
    demo_type: str = "molecular_visualization", **kwargs
) -> Any:
    """
    Create an interactive demonstration widget.

    Args:
        demo_type: Type of demo to create
        **kwargs: Additional parameters for the demo

    Returns:
        Interactive demo widget
    """
    from .widgets import MolecularVisualizationWidget

    logger.info(f"Creating interactive demo: {demo_type}")

    if demo_type == "molecular_visualization":
        return MolecularVisualizationWidget(**kwargs)
    else:
        raise ValueError(f"Unknown demo type: {demo_type}")


def _check_tutorial_environment() -> Dict[str, bool]:
    """Check the status of dependencies needed for tutorials."""

    # Use core environment checking
    core_status = core_check_environment()

    # Additional tutorial-specific checks
    tutorial_status = {
        "rdkit": RDKIT_AVAILABLE,
        "plotting": PLOTTING_AVAILABLE,
        "jupyter": _check_jupyter_environment(),
        "interactive": _check_interactive_support(),
    }

    # Combine core and tutorial status
    return {**core_status.get("available_packages", {}), **tutorial_status}


def _check_jupyter_environment() -> bool:
    """Check if running in Jupyter environment."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _check_interactive_support() -> bool:
    """Check if interactive widgets are supported."""
    try:

        return True
    except ImportError:
        return False


def _setup_tutorial_directories() -> None:
    """Create necessary directories for tutorial data and outputs."""
    directories = _get_tutorial_directories()

    for dir_path in directories.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger.debug("Tutorial directories created")


def _get_tutorial_directories() -> Dict[str, str]:
    """Get standard tutorial directory paths."""
    base_dir = Path.cwd()

    return {
        "data": str(base_dir / "tutorial_data"),
        "outputs": str(base_dir / "tutorial_outputs"),
        "cache": str(base_dir / "tutorial_cache"),
        "logs": str(base_dir / "tutorial_logs"),
        "progress": str(base_dir / "tutorial_progress"),
    }


# Convenience function for backwards compatibility
def initialize_tutorial_environment(*args, **kwargs) -> Any:
    """Alias for setup_learning_environment for backwards compatibility."""
    return setup_learning_environment(*args, **kwargs)
