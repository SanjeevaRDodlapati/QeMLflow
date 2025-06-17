"""
ChemML Integrations
==================

Integration modules for third-party libraries and external tools.
Provides seamless interfaces to popular chemistry and ML libraries.

Modules:
- deepchem: DeepChem integration and wrappers
- rdkit: Advanced RDKit utilities
- huggingface: Hugging Face transformers for chemistry
- wandb: Weights & Biases experiment tracking
"""

__all__ = []

# Import integration modules (with optional dependencies)
try:
    from . import deepchem_integration

    __all__.append("deepchem_integration")
except ImportError:
    pass

try:
    from . import rdkit_utils

    __all__.append("rdkit_utils")
except ImportError:
    pass

try:
    from . import huggingface_chemistry

    __all__.append("huggingface_chemistry")
except ImportError:
    pass

try:
    from . import experiment_tracking

    __all__.append("experiment_tracking")
except ImportError:
    pass
