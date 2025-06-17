from typing import List\n"""
QeMLflow Integrations Package

This package provides integrations with external libraries and models.
"""

from typing import Any

# Lazy loading registry
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
                globals()[name] = getattr(module, name)
                return globals()[name]
        except (ImportError, AttributeError, KeyError):
            continue

    raise AttributeError(f"module 'qemlflow.integrations' has no attribute '{name}'")


# Discovery functions
def discover_models_by_category(category: str):
    """Discover available models by scientific category."""
    try:
        from .adapters import list_adapters_by_category

        return list_adapters_by_category(category)
    except ImportError:
        return []


def list_available_categories():
    """List all available scientific categories."""
    try:
        from .adapters import list_all_categories

        return list_all_categories()
    except ImportError:
        return []


def discover_models_by_task(task: str):
    """Discover models suitable for a specific task."""
    try:
        from .adapters import discover_models_by_task as _discover_task

        return _discover_task(task)
    except ImportError:
        return []


def search_models(query: str):
    """Search for models by name or description."""
    try:
        from .adapters import search_models as _search

        return _search(query)
    except ImportError:
        return []


def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        from .adapters import get_model_info as _get_info

        return _get_info(model_name)
    except ImportError:
        return None


def get_manager():
    """Get the external model manager instance."""
    try:
        from .core.integration_manager import get_manager as _get_manager

        return _get_manager()
    except ImportError:
        return None


__version__ = "0.2.0"
