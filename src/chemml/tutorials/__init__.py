"""
ChemML Tutorial Framework
========================

A comprehensive framework for creating interactive and progressive learning experiences
in computational chemistry and machine learning.

This module provides the foundation for all ChemML tutorials, including:
- Learning assessment and progress tracking
- Interactive components and widgets
- Educational data management
- Environment setup and validation
- Visualization and demonstration tools

Key Components:
- LearningAssessment: Track student progress and understanding
- ProgressTracker: Monitor learning activities and time spent
- EnvironmentManager: Setup and validate tutorial environments
- EducationalDatasets: Curated datasets for learning
- InteractiveComponents: Widgets and visual demonstrations

Usage:
    from chemml.tutorials import (
        setup_learning_environment,
LearningAssessment,
ProgressTracker,
load_tutorial_data
)

# Setup learning environment
env = setup_learning_environment()

# Initialize assessment
assessment = LearningAssessment(student_id="demo", section="fundamentals")

# Track progress
tracker = ProgressTracker(assessment)

# Load educational data
data = load_tutorial_data("molecular_properties")
"""

from .assessment import ConceptCheckpoint, LearningAssessment, ProgressTracker
from .core import (
    create_interactive_demo,
load_tutorial_data,
setup_learning_environment,
)
from .data import (
    EducationalDatasets,
create_synthetic_examples,
get_sample_datasets,
load_educational_molecules,
)
from .environment import EnvironmentManager, check_dependencies, setup_fallbacks
from .quantum import (
    QuantumChemistryTutorial,
QuantumMachineLearning,
check_quantum_requirements,
create_h2_vqe_tutorial,
create_quantum_ml_demo,
get_quantum_tutorial_overview,
)
from .utils import (
    create_progress_dashboard,
interactive_parameter_tuning,
setup_logging,
visualize_molecules,
)
from .widgets import (
    InteractiveAssessment,
MolecularVisualizationWidget,
ProgressDashboard,
)

# Version information
#__version__ = "1.0.0"
#__author__ = "ChemML Development Team"

# Export all public components
#__all__ = [
    # Core functionality
"setup_learning_environment",
"load_tutorial_data",
"create_interactive_demo",
# Assessment and tracking
"LearningAssessment",
"ProgressTracker",
"ConceptCheckpoint",
# Educational data
"EducationalDatasets",
"get_sample_datasets",
"load_educational_molecules",
"create_synthetic_examples",
# Environment management
"EnvironmentManager",
"check_dependencies",
"setup_fallbacks",
# Interactive components
"InteractiveAssessment",
"ProgressDashboard",
"MolecularVisualizationWidget",
# Utilities
"visualize_molecules",
"interactive_parameter_tuning",
"create_progress_dashboard",
"setup_logging",
# Quantum computing
"QuantumChemistryTutorial",
"QuantumMachineLearning",
"create_h2_vqe_tutorial",
"create_quantum_ml_demo",
"check_quantum_requirements",
"get_quantum_tutorial_overview",
]
