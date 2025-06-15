"""
ChemML Drug Discovery Research Module
====================================

Advanced drug discovery algorithms and workflows.
This module serves as a compatibility layer for the legacy monolithic implementation.

The actual implementation has been modularized into separate files:
- molecular_optimization.py: Molecular optimization strategies
- admet.py: ADMET property prediction
- screening.py: Virtual screening workflows
- properties.py: Property prediction models
- generation.py: Molecular generation and design
- qsar.py: QSAR modeling tools

For new code, import directly from the submodules.
For backward compatibility, all classes and functions are available at the module level.
"""

from .drug_discovery.admet import *
from .drug_discovery.generation import *

# Import everything from the modular components for backward compatibility
from .drug_discovery.molecular_optimization import *
from .drug_discovery.properties import *
from .drug_discovery.qsar import *
from .drug_discovery.screening import *

# Legacy aliases for full backward compatibility
# These maintain the exact same interface as the original monolithic implementation

# Re-export all the classes and functions
__all__ = [
    # Molecular Optimization
    "MolecularOptimizer",
    "BayesianOptimizer",
    "GeneticAlgorithmOptimizer",
    "optimize_molecule",
    "batch_optimize",
    "compare_optimization_methods",
    "create_optimization_report",
    "calculate_optimization_metrics",
    # ADMET
    "ADMETPredictor",
    "DrugLikenessAssessor",
    "ToxicityPredictor",
    "predict_admet_profile",
    "predict_admet_properties",
    "assess_drug_likeness",
    "apply_admet_filters",
    "calculate_drug_likeness_score",
    "predict_toxicity",
    # Screening
    "VirtualScreener",
    "SimilarityScreener",
    "PharmacophoreScreener",
    # Properties
    "MolecularPropertyPredictor",
    "TrainedPropertyModel",
    "predict_properties",
    # Generation
    "MolecularGenerator",
    "FragmentBasedGenerator",
    "generate_molecular_structures",
    "optimize_structure",
    "save_generated_structures",
    "generate_diverse_library",
    # QSAR
    "DescriptorCalculator",
    "QSARModel",
    "ActivityPredictor",
    "TrainedQSARModel",
    "build_qsar_dataset",
    "evaluate_qsar_model",
    "build_qsar_model",
    "predict_activity",
    "validate_qsar_model",
]

# Add a deprecation notice for users importing this module directly
import warnings


def _show_deprecation_warning():
    """Show deprecation warning for direct import of this module."""
    warnings.warn(
        "Importing from chemml.research.drug_discovery directly is deprecated. "
        "Please import from the specific submodules instead:\n"
        "- from chemml.research.drug_discovery.molecular_optimization import ...\n"
        "- from chemml.research.drug_discovery.admet import ...\n"
        "- from chemml.research.drug_discovery.screening import ...\n"
        "- from chemml.research.drug_discovery.properties import ...\n"
        "- from chemml.research.drug_discovery.generation import ...\n"
        "- from chemml.research.drug_discovery.qsar import ...\n"
        "This compatibility layer will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


# Uncomment the line below to show deprecation warnings
# _show_deprecation_warning()
