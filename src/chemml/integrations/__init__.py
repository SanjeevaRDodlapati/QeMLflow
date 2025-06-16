"""
Integrations module with enhanced external model capabilities
Enhanced with advanced registry, monitoring, and testing
"""

import sys
from typing import TYPE_CHECKING, Any

# Type checking imports (zero runtime cost)
if TYPE_CHECKING:
    from typing import Dict, List, Optional

# Core eager imports (lightweight only)
# Note: pipeline import temporarily disabled due to dependencies

# Import adapters for discovery
from .adapters import (
    ADAPTER_CATEGORIES,
    discover_models_by_task,
    get_model_info,
    get_models_by_complexity,
    list_adapters_by_category,
    list_all_categories,
    search_models,
    suggest_similar_models,
)
from .core.advanced_registry import ModelCategory, TaskComplexity, get_advanced_registry
from .core.automated_testing import create_adapter_test_suite, quick_adapter_test

# New immediate action features - import from core
from .core.external_models import ExternalModelWrapper, PublicationModelRegistry
from .core.integration_manager import ExternalModelManager, get_manager
from .core.performance_monitoring import (
    get_metrics,
    track_integration,
    track_prediction,
)

# Lazy loading registry - updated paths
_LAZY_MODULES = {
    "deepchem_integration": ".adapters.molecular.deepchem_integration",
    "experiment_tracking": ".utils.experiment_tracking",
    "model_adapters": ".adapters.base.model_adapters",
    "boltz_adapter": ".adapters.molecular.boltz_adapter",
}


def __getattr__(name: str) -> Any:
    """Lazy loading for heavy modules"""
    for module_name, module_path in _LAZY_MODULES.items():
        try:
            module = __import__(module_path, fromlist=[name], level=1)
            if hasattr(module, name):
                # Cache the attribute for future access
                globals()[name] = getattr(module, name)
                return globals()[name]
        except (ImportError, AttributeError):
            continue

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Enhanced discovery functions
def discover_models_by_category(category: str):
    """Discover available models by scientific category."""
    return list_adapters_by_category(category)


def list_available_categories():
    """List all available scientific categories."""
    return list_all_categories()


# Enhanced discovery API
def discover_models_by_task(task: str):
    """Discover models suitable for a specific task."""
    from .adapters import discover_models_by_task as _discover_task

    return _discover_task(task)


def search_models(query: str):
    """Search for models by name or description."""
    from .adapters import search_models as _search

    return _search(query)


def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    from .adapters import get_model_info as _get_info

    return _get_info(model_name)


# Version info
__version__ = "0.2.0"
