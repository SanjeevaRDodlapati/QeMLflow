"""
ChemML Drug Discovery Research Module
====================================

Advanced drug discovery algorithms and workflows.
Modular implementation for better maintainability.

Sub-modules:
- optimization: Molecular optimization using Bayesian, genetic, and other algorithms
- admet: ADMET property prediction and drug-likeness assessment
- screening: Virtual screening workflows
- properties: Property prediction models
- qsar: QSAR modeling tools
- molecular_optimization: Advanced molecular optimization strategies
- generation: Molecular generation and design
"""

# Import key classes and functions for backward compatibility
# Only importing what actually exists in the modules

from .admet import (
    ADMETPredictor,
    DrugLikenessAssessor,
    ToxicityPredictor,
    apply_admet_filters,
    assess_drug_likeness,
    predict_admet_profile,
    predict_admet_properties,
)
from .generation import (
    FragmentBasedGenerator,
    MolecularGenerator,
    generate_diverse_library,
    generate_molecular_structures,
    optimize_structure,
)
from .molecular_optimization import (
    BayesianOptimizer,
    GeneticAlgorithmOptimizer,
    MolecularOptimizer,
    batch_optimize,
    optimize_molecule,
)
from .properties import (
    MolecularPropertyPredictor,
    TrainedPropertyModel,
    predict_properties,
)
from .qsar import (
    ActivityPredictor,
    DescriptorCalculator,
    QSARModel,
    TrainedQSARModel,
    build_qsar_dataset,
    build_qsar_model,
    predict_activity,
    validate_qsar_model,
)
from .screening import PharmacophoreScreener, SimilarityScreener, VirtualScreener

# For backward compatibility, also expose at module level
#__all__ = [
    # ADMET
"ADMETPredictor",
"DrugLikenessAssessor",
"ToxicityPredictor",
"predict_admet_profile",
"predict_admet_properties",
"assess_drug_likeness",
"apply_admet_filters",
# Screening
"VirtualScreener",
"SimilarityScreener",
"PharmacophoreScreener",
# Properties
"MolecularPropertyPredictor",
"TrainedPropertyModel",
"predict_properties",
# Molecular Optimization
"MolecularOptimizer",
"BayesianOptimizer",
"GeneticAlgorithmOptimizer",
"optimize_molecule",
"batch_optimize",
# Generation
"MolecularGenerator",
"FragmentBasedGenerator",
"generate_molecular_structures",
"optimize_structure",
"generate_diverse_library",
# QSAR
"DescriptorCalculator",
"QSARModel",
"ActivityPredictor",
"TrainedQSARModel",
"build_qsar_dataset",
"build_qsar_model",
"predict_activity",
"validate_qsar_model",
]
