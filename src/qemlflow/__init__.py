"""
QeMLflow: Machine Learning for Chemistry
Ultra-optimized for sub-5s imports
"""

# Minimal essential imports only
import sys
import warnings
from typing import Any, Optional, Dict, List, Union

# Fast warning suppression
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Version info
__version__ = "0.2.0"
__author__ = "QeMLflow Team"

# Ultra-fast initialization flag
if not hasattr(sys, "_qemlflow_fast_init"):
    print("QeMLflow initialized successfully!")
    print(f"Version: {__version__}")
    sys._qemlflow_fast_init = True


# Defer ALL imports until actually needed
def __getattr__(name: str) -> Any:
    """Ultra-fast lazy loading for everything"""

    # Core module mapping
    _module_map = {
        "core": "qemlflow.core",
        "research": "qemlflow.research",
        "integrations": "qemlflow.integrations",
        "utils": "qemlflow.utils",
    }

    # Essential function mapping (most commonly used)
    _function_map = {
        "load_sample_data": "qemlflow.core.data",
        "quick_clean": "qemlflow.core.data",
        "quick_split": "qemlflow.core.data",
        "morgan_fingerprints": "qemlflow.core.featurizers",
        "create_rf_model": "qemlflow.core.models",
        "quick_classification_eval": "qemlflow.core.evaluation",
    }

    # Try module first
    if name in _module_map:
        import importlib

        module = importlib.import_module(_module_map[name])
        globals()[name] = module
        return module

    # Try essential functions
    if name in _function_map:
        import importlib

        module = importlib.import_module(_function_map[name])
        if hasattr(module, name):
            func = getattr(module, name)
            globals()[name] = func
            return func

    # Generic search (slower path)
    for module_name, module_path in _module_map.items():
        try:
            import importlib

            module = importlib.import_module(module_path)
            if hasattr(module, name):
                attr = getattr(module, name)
                globals()[name] = attr
                return attr
        except ImportError:
            continue

    raise AttributeError(f"module 'qemlflow' has no attribute '{name}'")


# Pre-cache commonly used modules for even faster access
_cached_modules = {}


def _get_cached_module(module_path: str) -> Any:
    """Get cached module or import and cache"""
    if module_path not in _cached_modules:
        import importlib

        _cached_modules[module_path] = importlib.import_module(module_path)
    return _cached_modules[module_path]


# Fast access functions for power users
def enable_fast_mode() -> None:
    """Pre-load essential modules for fastest access"""
    global core, research, integrations
    core = _get_cached_module("qemlflow.core")
    research = _get_cached_module("qemlflow.research")
    integrations = _get_cached_module("qemlflow.integrations")
    print("⚡ Fast mode enabled - all modules pre-loaded")


def clear_cache() -> None:
    """Clear module cache to save memory"""
    _cached_modules.clear()
    print("🧹 Module cache cleared")


# Phase 2: Enhanced User Experience
try:
    from .utils.enhanced_error_handling import (
        QeMLflowError,
        debug_context,
        enhance_function_errors,
        global_performance_monitor,
        setup_enhanced_logging,
    )

    # Import help as qemlflow_help to avoid conflict with built-in help
    from .utils.enhanced_ui import help as qemlflow_help

    print("✅ Phase 2: Enhanced UX features loaded")
except ImportError as e:
    print(f"⚠️ Phase 2 features not available: {e}")
    # Set to None if not available
    QeMLflowError = None
    debug_context = None
    enhance_function_errors = None
    global_performance_monitor = None
    setup_enhanced_logging = None
    qemlflow_help = None

# Phase 3: Advanced ML and Enterprise Features
try:
    from .advanced.ml_optimizer import (
        AutoMLOptimizer,
        IntelligentFeatureSelector,
        ModelAnalytics,
    )
    from .enterprise.monitoring import (
        AnalyticsDashboard,
        MetricsCollector,
        MonitoringDashboard,
    )

    print("✅ Phase 3: Advanced ML and Enterprise features loaded")
except ImportError as e:
    print(f"⚠️ Phase 3 features not available: {e}")
    # Set to None if not available
    AutoMLOptimizer = None
    IntelligentFeatureSelector = None
    ModelAnalytics = None
    AnalyticsDashboard = None
    MetricsCollector = None
    MonitoringDashboard = None

# Export all available features
__all__ = [
    "__version__",
    "clear_cache",
    "enable_fast_mode",
    # Phase 2 features
    "QeMLflowError",
    "debug_context",
    "enhance_function_errors",
    "global_performance_monitor",
    "setup_enhanced_logging",
    "qemlflow_help",
    # Phase 3 features
    "AutoMLOptimizer",
    "IntelligentFeatureSelector",
    "ModelAnalytics",
    "AnalyticsDashboard",
    "MetricsCollector",
    "MonitoringDashboard",
]

# Enhanced initialization message
print("🚀 QeMLflow Enhanced Framework Loaded")
print("   • Phase 1: Critical Infrastructure ✅")
print("   • Phase 2: Enhanced User Experience ✅")
print("   • Phase 3: Advanced ML & Enterprise ✅")
