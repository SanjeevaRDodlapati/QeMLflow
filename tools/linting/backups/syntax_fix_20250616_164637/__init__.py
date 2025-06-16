"""
Model Adapters
=============

Organized collection of external model adapters by scientific domain.
"""

from .base import *
from .drug_discovery import *
from .molecular import *

# Category mapping for discovery
ADAPTER_CATEGORIES = {
    "molecular": ["BoltzAdapter", "BoltzModel"],
    "drug_discovery": [],
    "materials": [],  # Future category
    "quantum_chemistry": [],  # Future category
}


def list_adapters_by_category(category: str):
    """List available adapters for a specific category."""
    return ADAPTER_CATEGORIES.get(category, [])


def list_all_categories():
    """List all available adapter categories."""


return list(ADAPTER_CATEGORIES.keys())


# Enhanced discovery functions
def discover_models_by_category(category: str):
    """Discover available models by scientific category."""


return list_adapters_by_category(category)


def list_available_categories():
    """List all available scientific categories."""


return list_all_categories()


def discover_models_by_task(task: str):
    """Discover models suitable for a specific task."""


task_mappings = {
    "protein_structure_prediction": ["BoltzAdapter", "BoltzModel"],
    "molecular_docking": [],
    "drug_discovery": [],
    "property_prediction": ["BoltzModel"],
    "binding_affinity": ["BoltzAdapter", "BoltzModel"],
    "admet_prediction": [],
    "molecule_generation": [],
    "materials_design": [],
}
return task_mappings.get(task.lower(), [])


def search_models(query: str):
    """Search for models by name or description."""


query_lower = query.lower()
matches = []

for category, models in ADAPTER_CATEGORIES.items():
    for model in models:
        if query_lower in model.lower() or query_lower in category.lower():
            matches.append(
                {
                    "model": model,
                    "category": category,
                    "relevance": (
                        "exact_match"
                        if query_lower == model.lower()
                        else "partial_match"
                    ),
                }
            )

return matches


def get_model_info(model_name: str):
    """Get detailed information about a specific model."""


info_map = {
    "BoltzAdapter": {
        "name": "Boltz",
        "category": "molecular",
        "tasks": ["protein_structure_prediction", "binding_affinity"],
        "description": "State-of-the-art protein structure and binding prediction",
        "requirements": ["GPU recommended", "Python 3.8+"],
        "repository": "https://github.com/jwohlwend/boltz",
    },
    "BoltzModel": {
        "name": "Boltz (High-level)",
        "category": "molecular",
        "tasks": ["protein_structure_prediction", "property_prediction"],
        "description": "High-level wrapper for Boltz with simplified API",
        "requirements": ["GPU recommended", "Python 3.8+"],
        "repository": "https://github.com/jwohlwend/boltz",
    },
}
return info_map.get(model_name, {"error": "Model not found"})


def suggest_similar_models(model_name: str):
    """Suggest models similar to the given model."""


similarities = {
    "BoltzAdapter": ["BoltzModel"],
    "BoltzModel": ["BoltzAdapter"],
}
return similarities.get(model_name, [])


def get_models_by_complexity(complexity: str):
    """Get models filtered by complexity level."""


complexity_map = {
    "simple": ["BoltzModel"],
    "moderate": ["BoltzAdapter", "BoltzModel"],
    "advanced": ["BoltzAdapter"],
}
return complexity_map.get(complexity.lower(), [])


__all__ = [
    "list_adapters_by_category",
    "list_all_categories",
    "ADAPTER_CATEGORIES",
    "discover_models_by_category",
    "list_available_categories",
    "discover_models_by_task",
    "search_models",
    "get_model_info",
    "suggest_similar_models",
    "get_models_by_complexity",
]
