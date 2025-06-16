"""
Core Integration Framework
=========================

Core infrastructure for external model integration including base classes,
management, registry, monitoring, and testing capabilities.
"""

from .advanced_registry import ModelCategory, TaskComplexity, get_advanced_registry
from .automated_testing import create_adapter_test_suite, quick_adapter_test
from .external_models import ExternalModelWrapper, PublicationModelRegistry
from .integration_manager import ExternalModelManager, get_manager
from .performance_monitoring import get_metrics, track_integration, track_prediction

# Note: pipeline import temporarily disabled due to dependencies

#__all__ = [
    # Core classes
"ExternalModelWrapper",
"PublicationModelRegistry",
"ExternalModelManager",
"get_manager",
# Advanced features
"get_advanced_registry",
"ModelCategory",
"TaskComplexity",
"get_metrics",
"track_integration",
"track_prediction",
"create_adapter_test_suite",
"quick_adapter_test",
]
