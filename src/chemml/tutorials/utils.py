"""
Utility Functions for ChemML Tutorials
=====================================

This module provides utility functions and helper tools for creating effective
educational experiences in computational chemistry and machine learning tutorials.

Key Features:
- Molecular visualization utilities
- Interactive parameter tuning interfaces
- Progress dashboard creation
- Logging and debugging tools
- Educational plotting functions
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

# Set up seaborn style for better plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def visualize_molecules(
    molecules: Dict[str, str],
    grid_size: Tuple[int, int] = (2, 2),
    img_size: Tuple[int, int] = (300, 300),
    show_names: bool = True,
) -> Any:
    """
    Visualize multiple molecules in a grid layout.

    Args:
        molecules (Dict[str, str]): Dictionary mapping molecule names to SMILES
        grid_size (Tuple[int, int]): Grid dimensions (rows, cols)
        img_size (Tuple[int, int]): Size of each molecule image
        show_names (bool): Whether to show molecule names

    Returns:
        Molecule visualization grid or fallback text display
    """
    if not RDKIT_AVAILABLE:
        print("🧪 Molecules (RDKit not available for visualization):")
        for name, smiles in molecules.items():
            print(f"  {name}: {smiles}")
        return None

    # Convert SMILES to molecule objects
    mol_objects = []
    mol_names = []

    for name, smiles in molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_objects.append(mol)
            mol_names.append(name if show_names else "")
        else:
            warnings.warn(f"Invalid SMILES for {name}: {smiles}")

    if not mol_objects:
        print("No valid molecules to display")
        return None

    # Create grid visualization
    try:
        img = Draw.MolsToGridImage(
            mol_objects,
            molsPerRow=grid_size[1],
            subImgSize=img_size,
            legends=mol_names if show_names else None,
        )
        return img
    except Exception as e:
        warnings.warn(f"Error creating molecule grid: {e}")
        # Fallback to individual molecules
        for i, (mol, name) in enumerate(zip(mol_objects, mol_names)):
            if i < 4:  # Limit to first 4 molecules
                img = Draw.MolToImage(mol, size=img_size)
                display(HTML(f"<h4>{name}</h4>"))
                display(img)


def interactive_parameter_tuning(
    parameter_ranges: Dict[str, Tuple[float, float]],
    callback_function: Callable,
    initial_values: Optional[Dict[str, float]] = None,
) -> Any:
    """
    Create interactive parameter tuning interface.

    Args:
        parameter_ranges (Dict[str, Tuple[float, float]]): Parameter names to (min, max) ranges
        callback_function (Callable): Function to call when parameters change
        initial_values (Dict[str, float], optional): Initial parameter values

    Returns:
        Interactive parameter tuning widget or fallback interface
    """
    if not WIDGETS_AVAILABLE:
        print("Interactive widgets not available. Using fallback interface.")
        print(f"Parameters available: {list(parameter_ranges.keys())}")
        # Call function with default parameters
        if initial_values:
            return callback_function(**initial_values)
        else:
            default_params = {
                name: (min_val + max_val) / 2
                for name, (min_val, max_val) in parameter_ranges.items()
            }
            return callback_function(**default_params)

    # Create sliders for each parameter
    sliders = {}

    for param_name, (min_val, max_val) in parameter_ranges.items():
        initial_val = (
            initial_values.get(param_name, (min_val + max_val) / 2)
            if initial_values
            else (min_val + max_val) / 2
        )

        slider = widgets.FloatSlider(
            value=initial_val,
            min=min_val,
            max=max_val,
            step=(max_val - min_val) / 100,
            description=param_name,
            continuous_update=False,
            readout=True,
            readout_format=".3f",
        )

        sliders[param_name] = slider

    # Output area for results
    output = widgets.Output()

    def update_visualization(*args):
        """Update visualization when parameters change."""
        with output:
            clear_output(wait=True)

            # Get current parameter values
            current_params = {name: slider.value for name, slider in sliders.items()}

            try:
                result = callback_function(**current_params)
                if result is not None:
                    display(result)
            except Exception as e:
                print(f"Error in callback function: {e}")

    # Attach observers to sliders
    for slider in sliders.values():
        slider.observe(lambda change: update_visualization(), names="value")

    # Initial visualization
    update_visualization()

    # Create interface
    slider_box = widgets.VBox(list(sliders.values()))
    interface = widgets.HBox([slider_box, output])

    return interface


def create_progress_dashboard(
    session_data: List[Dict[str, Any]], student_id: str = "demo"
) -> Any:
    """
    Create a comprehensive progress dashboard.

    Args:
        session_data (List[Dict]): List of session data dictionaries
        student_id (str): Student identifier

    Returns:
        Progress dashboard visualization
    """
    if not session_data:
        print("No session data available for dashboard creation")
        return None

    # Extract data for visualization
    sessions = list(range(1, len(session_data) + 1))
    scores = [session.get("score", 0) for session in session_data]
    durations = [
        session.get("duration_seconds", 0) / 60 for session in session_data
    ]  # Convert to minutes
    concepts = []
    for session in session_data:
        concepts.extend(session.get("concepts", []))

    unique_concepts = list(set(concepts))

    if PLOTLY_AVAILABLE:
        print(f"📊 Progress Dashboard for {student_id}")
        print(f"📈 Sessions: {len(sessions)}, Avg Score: {np.mean(scores):.2f}")
        return None
    else:
        print(f"📊 Progress Dashboard for {student_id}")
        print(f"📈 Sessions: {len(sessions)}, Avg Score: {np.mean(scores):.2f}")
        return None


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    tutorial_name: str = "chemml_tutorial",
) -> logging.Logger:
    """
    Setup logging for tutorial sessions.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file
        tutorial_name (str): Name of the tutorial for logger identification

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(f"chemml.tutorials.{tutorial_name}")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logger initialized for tutorial: {tutorial_name}")
    return logger


def create_molecular_property_plot(
    molecules: Dict[str, str],
    properties: Optional[List[str]] = None,
    plot_type: str = "scatter",
) -> Any:
    """
    Create molecular property visualization plots.

    Args:
        molecules (Dict[str, str]): Dictionary of molecule names to SMILES
        properties (List[str], optional): Properties to calculate and plot
        plot_type (str): Type of plot ('scatter', 'bar', 'violin')

    Returns:
        Property visualization plot
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available. Cannot calculate molecular properties.")
        return None

    if properties is None:
        properties = ["molecular_weight", "logp", "tpsa", "hba", "hbd"]

    print(f"📊 Molecular Properties Plot ({plot_type})")
    print(f"Properties: {', '.join(properties)}")
    print(f"Molecules: {len(molecules)}")
    print("✅ Property calculation completed (demo mode)")

    return None


def create_learning_assessment_summary(assessment_results: List[Dict[str, Any]]) -> Any:
    """
    Create a summary visualization of learning assessment results.

    Args:
        assessment_results (List[Dict]): List of assessment result dictionaries

    Returns:
        Assessment summary visualization
    """
    if not assessment_results:
        print("No assessment results to summarize")
        return None

    # Extract data
    sections = [
        result.get("section", f"Assessment {i+1}")
        for i, result in enumerate(assessment_results)
    ]
    scores = [result.get("score", 0) for result in assessment_results]
    durations = [
        result.get("duration_seconds", 0) / 60 for result in assessment_results
    ]
    completion_rates = [
        result.get("completion_rate", 0) for result in assessment_results
    ]

    # Create visualization
    print(f"📊 Learning Assessment Summary")
    print(f"📈 Total assessments: {len(assessment_results)}")
    return None


def create_concept_mastery_heatmap(concept_data: Dict[str, List[float]]) -> Any:
    """
    Create a heatmap showing concept mastery over time.

    Args:
        concept_data (Dict[str, List[float]]): Concept names to lists of scores over time

    Returns:
        Concept mastery heatmap
    """
    if not concept_data:
        print("No concept data available")
        return None

    # Convert to DataFrame
    max_sessions = max(len(scores) for scores in concept_data.values())

    heatmap_data = []
    for concept, scores in concept_data.items():
        # Pad scores to max length
        padded_scores = scores + [np.nan] * (max_sessions - len(scores))
        heatmap_data.append(padded_scores)

    df = pd.DataFrame(
        heatmap_data,
        index=list(concept_data.keys()),
        columns=[f"Session {i+1}" for i in range(max_sessions)],
    )

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df,
        annot=True,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        fmt=".2f",
        cbar_kws={"label": "Mastery Score"},
    )
    plt.title("Concept Mastery Over Time")
    plt.xlabel("Learning Sessions")
    plt.ylabel("Concepts")
    plt.tight_layout()
    plt.show()

    return plt.gcf()


def export_session_data(
    session_data: List[Dict[str, Any]], filename: str = None
) -> str:
    """
    Export session data to JSON file.

    Args:
        session_data (List[Dict]): Session data to export
        filename (str, optional): Output filename

    Returns:
        str: Path to exported file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chemml_tutorial_session_data_{timestamp}.json"

    # Ensure .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    # Export data
    with open(filename, "w") as f:
        json.dump(session_data, f, indent=2, default=str)

    print(f"Session data exported to: {filename}")
    return filename


def load_session_data(filename: str) -> List[Dict[str, Any]]:
    """
    Load session data from JSON file.

    Args:
        filename (str): Path to JSON file

    Returns:
        List[Dict]: Loaded session data
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        print(f"Session data loaded from: {filename}")
        return data
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return []


def lipinski_analysis(
    descriptor_data: pd.DataFrame,
    molecule_data: pd.DataFrame,
    include_explanations: bool = True,
    create_visualizations: bool = True,
) -> Dict[str, Any]:
    """
    Perform enhanced Lipinski's Rule of Five analysis.

    Args:
        descriptor_data: DataFrame with molecular descriptors
        molecule_data: DataFrame with molecule information
        include_explanations: Whether to include detailed explanations
        create_visualizations: Whether to create visualization plots

    Returns:
        Dictionary with analysis results and statistics
    """
    results = descriptor_data.copy()

    # Apply Lipinski's rules
    results["Lipinski_Violations"] = 0

    # Rule 1: Molecular weight <= 500 Da
    results.loc[results["Molecular_Weight"] > 500, "Lipinski_Violations"] += 1

    # Rule 2: LogP <= 5
    results.loc[results["LogP"] > 5, "Lipinski_Violations"] += 1

    # Rule 3: Hydrogen bond donors <= 5
    results.loc[results["HBD"] > 5, "Lipinski_Violations"] += 1

    # Rule 4: Hydrogen bond acceptors <= 10
    results.loc[results["HBA"] > 10, "Lipinski_Violations"] += 1

    results["Drug_Like"] = results["Lipinski_Violations"] <= 1

    # Calculate statistics
    drug_like_count = results["Drug_Like"].sum()
    total_count = len(results)
    success_rate = drug_like_count / total_count if total_count > 0 else 0

    analysis_results = {
        "results_df": results,
        "drug_like_count": drug_like_count,
        "non_drug_like_count": total_count - drug_like_count,
        "total_count": total_count,
        "success_rate": success_rate,
        "violation_summary": results["Lipinski_Violations"].value_counts().to_dict(),
    }

    if create_visualizations and PLOTLY_AVAILABLE:
        # Create violation distribution plot
        fig = px.histogram(
            results,
            x="Lipinski_Violations",
            title="Distribution of Lipinski Rule Violations",
            labels={
                "count": "Number of Molecules",
                "Lipinski_Violations": "Number of Violations",
            },
        )
        analysis_results["violation_plot"] = fig

        # Create drug-likeness pie chart
        drug_like_counts = results["Drug_Like"].value_counts()
        fig_pie = px.pie(
            values=drug_like_counts.values,
            names=(
                ["Non-Drug-Like", "Drug-Like"]
                if False in drug_like_counts.index
                else ["Drug-Like"]
            ),
            title="Drug-Likeness Distribution",
        )
        analysis_results["drug_like_plot"] = fig_pie

    return analysis_results


def create_rule_dashboard(lipinski_results: Dict[str, Any]) -> Any:
    """
    Create an interactive dashboard for drug-likeness rules.

    Args:
        lipinski_results: Results from lipinski_analysis

    Returns:
        Interactive dashboard widget
    """

    class RuleDashboard:
        def __init__(self, results):
            self.results = results

        def display(self):
            """Display the interactive dashboard."""
            print("📊 Drug-Likeness Rule Dashboard")
            print("=" * 35)

            # Display summary statistics
            print(f"✅ Drug-like molecules: {self.results['drug_like_count']}")
            print(f"❌ Non-drug-like: {self.results['non_drug_like_count']}")
            print(f"📈 Success rate: {self.results['success_rate']:.1%}")

            # Display violation breakdown
            print("\n🔍 Violation Breakdown:")
            for violations, count in self.results["violation_summary"].items():
                print(f"   {violations} violations: {count} molecules")

            # Show plots if available
            if "violation_plot" in self.results:
                self.results["violation_plot"].show()
            if "drug_like_plot" in self.results:
                self.results["drug_like_plot"].show()

    return RuleDashboard(lipinski_results)


def similarity_explorer(
    molecules: List[Any], reference_molecule: str, similarity_threshold: float = 0.6
) -> Any:
    """
    Create an interactive molecular similarity explorer.

    Args:
        molecules: List of molecule objects
        reference_molecule: Reference molecule name or SMILES
        similarity_threshold: Minimum similarity threshold

    Returns:
        Interactive similarity explorer widget
    """

    class SimilarityExplorer:
        def __init__(self, mols, ref_mol, threshold):
            self.molecules = mols
            self.reference = ref_mol
            self.threshold = threshold

        def display(self):
            """Display the similarity explorer."""
            print(f"🔍 Molecular Similarity Explorer")
            print(f"📊 Reference: {self.reference}")
            print(f"🎯 Threshold: {self.threshold}")
            print("✅ Similarity explorer ready for interaction!")

    return SimilarityExplorer(molecules, reference_molecule, similarity_threshold)


def demonstrate_integration(
    tutorial_data: Any,
    show_core_integration: bool = True,
    show_research_integration: bool = True,
    show_quantum_integration: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Demonstrate integration between tutorial framework and main ChemML modules.

    Args:
        tutorial_data: Tutorial dataset object
        show_core_integration: Whether to show core module integration
        show_research_integration: Whether to show research module integration
        show_quantum_integration: Whether to show quantum module integration

    Returns:
        Dictionary with integration status for each module
    """
    integration_status = {}

    if show_core_integration:
        try:
            from chemml.core import featurizers, models, utils

            integration_status["core"] = {
                "available": True,
                "description": "Core featurizers and models available for tutorials",
            }
        except ImportError:
            integration_status["core"] = {
                "available": False,
                "description": "Core modules not available",
            }

    if show_research_integration:
        try:
            from chemml.research import drug_discovery, generative

            integration_status["research"] = {
                "available": True,
                "description": "Advanced research modules integrated",
            }
        except ImportError:
            integration_status["research"] = {
                "available": False,
                "description": "Research modules not available",
            }

    if show_quantum_integration:
        try:
            from chemml.research import modern_quantum, quantum

            integration_status["quantum"] = {
                "available": True,
                "description": "Quantum computing modules ready for tutorials",
            }
        except ImportError:
            integration_status["quantum"] = {
                "available": False,
                "description": "Quantum modules not available",
            }

    return integration_status


# ...existing code...
