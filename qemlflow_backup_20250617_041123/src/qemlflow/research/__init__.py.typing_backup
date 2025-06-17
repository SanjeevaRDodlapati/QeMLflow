"""
ChemML Research Modules
======================

Advanced research modules for cutting-edge chemistry ML applications.
These modules contain experimental and specialized functionality.

Modules:
- quantum: Quantum computing and quantum chemistry ML
- generative: Generative models for molecular design
- advanced_models: Novel ML architectures for chemistry
"""

# Import research modules (with optional dependencies)
try:
    from . import quantum
except ImportError:
    quantum = None

try:
    from . import generative
except ImportError:
    generative = None

try:
    from . import advanced_models
except ImportError:
    advanced_models = None

__all__ = []

# Add available modules to __all__
if quantum is not None:
    __all__.append("quantum")
if generative is not None:
    __all__.append("generative")
if advanced_models is not None:
    __all__.append("advanced_models")

# Print warning if modules are missing
missing_modules = []
if quantum is None:
    missing_modules.append("quantum")
if generative is None:
    missing_modules.append("generative")
if advanced_models is None:
    missing_modules.append("advanced_models")

if missing_modules:
    import warnings

    warnings.warn(
        f"Some research modules unavailable due to missing dependencies: {missing_modules}. "
        "Install additional packages to access full research functionality."
    )
