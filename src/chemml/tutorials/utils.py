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
        return _create_fallback_parameter_interface(parameter_ranges, callback_function)

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
        return _create_plotly_dashboard(
            sessions, scores, durations, unique_concepts, student_id
        )
    else:
        return _create_matplotlib_dashboard(
            sessions, scores, durations, unique_concepts, student_id
        )


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
    molecules: Dict[str, str], properties: List[str] = None, plot_type: str = "scatter"
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

    # Calculate properties
    property_data = []

    for name, smiles in molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_props = {"name": name, "smiles": smiles}

            # Calculate requested properties
            for prop in properties:
                if prop == "molecular_weight":
                    mol_props[prop] = Descriptors.MolWt(mol)
                elif prop == "logp":
                    mol_props[prop] = Descriptors.MolLogP(mol)
                elif prop == "tpsa":
                    mol_props[prop] = Descriptors.TPSA(mol)
                elif prop == "hba":
                    mol_props[prop] = Descriptors.NumHAcceptors(mol)
                elif prop == "hbd":
                    mol_props[prop] = Descriptors.NumHDonors(mol)
                elif prop == "num_atoms":
                    mol_props[prop] = mol.GetNumAtoms()
                elif prop == "num_rings":
                    mol_props[prop] = Descriptors.RingCount(mol)
                else:
                    mol_props[prop] = 0  # Default value for unknown properties

            property_data.append(mol_props)

    if not property_data:
        print("No valid molecules for property calculation")
        return None

    # Create DataFrame
    df = pd.DataFrame(property_data)

    # Create visualization
    if plot_type == "scatter" and len(properties) >= 2:
        return _create_scatter_plot(df, properties)
    elif plot_type == "bar":
        return _create_bar_plot(df, properties)
    elif plot_type == "violin":
        return _create_violin_plot(df, properties)
    else:
        return _create_default_property_plot(df, properties)


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
    if PLOTLY_AVAILABLE:
        return _create_plotly_assessment_summary(
            sections, scores, durations, completion_rates
        )
    else:
        return _create_matplotlib_assessment_summary(
            sections, scores, durations, completion_rates
        )


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


# Helper functions for different visualization backends


def _create_fallback_parameter_interface(
    parameter_ranges: Dict[str, Tuple[float, float]], callback_function: Callable
) -> None:
    """Create fallback parameter interface when widgets are not available."""
    print("🎛️  Parameter Tuning Interface (ipywidgets not available)")
    print("=" * 50)

    print("Available parameters:")
    for param, (min_val, max_val) in parameter_ranges.items():
        default_val = (min_val + max_val) / 2
        print(f"  {param}: {min_val:.3f} - {max_val:.3f} (default: {default_val:.3f})")

    print("\n💡 Install ipywidgets for interactive parameter tuning:")
    print("   pip install ipywidgets")


def _create_plotly_dashboard(
    sessions: List[int],
    scores: List[float],
    durations: List[float],
    concepts: List[str],
    student_id: str,
) -> go.Figure:
    """Create dashboard using Plotly."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Score Progression",
            "Time Per Session",
            "Concept Coverage",
            "Score Distribution",
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Score progression
    fig.add_trace(
        go.Scatter(x=sessions, y=scores, mode="lines+markers", name="Scores"),
        row=1,
        col=1,
    )

    # Duration per session
    fig.add_trace(go.Bar(x=sessions, y=durations, name="Duration (min)"), row=1, col=2)

    # Concept coverage (top concepts)
    concept_counts = pd.Series(concepts).value_counts().head(10)
    fig.add_trace(
        go.Bar(x=concept_counts.index, y=concept_counts.values, name="Concept Count"),
        row=2,
        col=1,
    )

    # Score distribution
    fig.add_trace(
        go.Histogram(x=scores, nbinsx=10, name="Score Distribution"), row=2, col=2
    )

    fig.update_layout(
        title=f"Learning Progress Dashboard - {student_id}", showlegend=False
    )

    return fig


def _create_matplotlib_dashboard(
    sessions: List[int],
    scores: List[float],
    durations: List[float],
    concepts: List[str],
    student_id: str,
) -> plt.Figure:
    """Create dashboard using Matplotlib."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Score progression
    ax1.plot(sessions, scores, "bo-", linewidth=2, markersize=6)
    ax1.set_title("Score Progression")
    ax1.set_xlabel("Session")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3)

    # Duration per session
    ax2.bar(sessions, durations, alpha=0.7, color="green")
    ax2.set_title("Time Per Session")
    ax2.set_xlabel("Session")
    ax2.set_ylabel("Duration (minutes)")
    ax2.grid(True, alpha=0.3)

    # Concept coverage
    if concepts:
        concept_counts = pd.Series(concepts).value_counts().head(8)
        ax3.barh(concept_counts.index, concept_counts.values, alpha=0.7, color="orange")
        ax3.set_title("Most Covered Concepts")
        ax3.set_xlabel("Frequency")
    else:
        ax3.text(
            0.5,
            0.5,
            "No concept data",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Concept Coverage")

    # Score distribution
    ax4.hist(scores, bins=10, alpha=0.7, color="purple")
    ax4.set_title("Score Distribution")
    ax4.set_xlabel("Score")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Learning Progress Dashboard - {student_id}", fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig


def _create_scatter_plot(df: pd.DataFrame, properties: List[str]) -> plt.Figure:
    """Create scatter plot for molecular properties."""
    if len(properties) < 2:
        return _create_bar_plot(df, properties)

    plt.figure(figsize=(10, 8))

    x_prop, y_prop = properties[0], properties[1]

    plt.scatter(df[x_prop], df[y_prop], alpha=0.7, s=100)

    # Add molecule names as labels
    for i, name in enumerate(df["name"]):
        plt.annotate(
            name,
            (df[x_prop].iloc[i], df[y_prop].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel(x_prop.replace("_", " ").title())
    plt.ylabel(y_prop.replace("_", " ").title())
    plt.title(f"Molecular Properties: {x_prop} vs {y_prop}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return plt.gcf()


def _create_bar_plot(df: pd.DataFrame, properties: List[str]) -> plt.Figure:
    """Create bar plot for molecular properties."""
    fig, axes = plt.subplots(1, len(properties), figsize=(5 * len(properties), 6))

    if len(properties) == 1:
        axes = [axes]

    for i, prop in enumerate(properties):
        axes[i].bar(df["name"], df[prop], alpha=0.7)
        axes[i].set_title(prop.replace("_", " ").title())
        axes[i].set_ylabel(prop.replace("_", " ").title())
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


def _create_violin_plot(df: pd.DataFrame, properties: List[str]) -> plt.Figure:
    """Create violin plot for molecular properties."""
    # Melt DataFrame for seaborn
    df_melted = df.melt(
        id_vars=["name"], value_vars=properties, var_name="property", value_name="value"
    )

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_melted, x="property", y="value")
    plt.title("Distribution of Molecular Properties")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return plt.gcf()


def _create_default_property_plot(
    df: pd.DataFrame, properties: List[str]
) -> plt.Figure:
    """Create default property plot."""
    return _create_bar_plot(df, properties)


def _create_plotly_assessment_summary(
    sections: List[str],
    scores: List[float],
    durations: List[float],
    completion_rates: List[float],
) -> go.Figure:
    """Create assessment summary using Plotly."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Scores by Section",
            "Time Taken",
            "Completion Rates",
            "Score vs Time",
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Scores by section
    fig.add_trace(go.Bar(x=sections, y=scores, name="Scores"), row=1, col=1)

    # Time taken
    fig.add_trace(go.Bar(x=sections, y=durations, name="Duration (min)"), row=1, col=2)

    # Completion rates
    fig.add_trace(
        go.Bar(x=sections, y=completion_rates, name="Completion Rate"), row=2, col=1
    )

    # Score vs Time
    fig.add_trace(
        go.Scatter(x=durations, y=scores, mode="markers", name="Score vs Time"),
        row=2,
        col=2,
    )

    fig.update_layout(title="Learning Assessment Summary", showlegend=False)

    return fig


def _create_matplotlib_assessment_summary(
    sections: List[str],
    scores: List[float],
    durations: List[float],
    completion_rates: List[float],
) -> plt.Figure:
    """Create assessment summary using Matplotlib."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Scores by section
    ax1.bar(sections, scores, alpha=0.7, color="blue")
    ax1.set_title("Scores by Section")
    ax1.set_ylabel("Score")
    ax1.tick_params(axis="x", rotation=45)

    # Time taken
    ax2.bar(sections, durations, alpha=0.7, color="green")
    ax2.set_title("Time Taken")
    ax2.set_ylabel("Duration (minutes)")
    ax2.tick_params(axis="x", rotation=45)

    # Completion rates
    ax3.bar(sections, completion_rates, alpha=0.7, color="orange")
    ax3.set_title("Completion Rates")
    ax3.set_ylabel("Completion Rate")
    ax3.tick_params(axis="x", rotation=45)

    # Score vs Time
    ax4.scatter(durations, scores, alpha=0.7, s=100, color="red")
    ax4.set_title("Score vs Time")
    ax4.set_xlabel("Duration (minutes)")
    ax4.set_ylabel("Score")

    plt.suptitle("Learning Assessment Summary", fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig


# Convenience functions for quick setup


def quick_molecule_viewer(smiles_dict: Dict[str, str]) -> Any:
    """Quick setup for molecule visualization."""
    return visualize_molecules(smiles_dict)


def quick_parameter_tuner(
    params: Dict[str, Tuple[float, float]], func: Callable
) -> Any:
    """Quick setup for parameter tuning."""
    return interactive_parameter_tuning(params, func)


def quick_progress_tracker(data: List[Dict[str, Any]]) -> Any:
    """Quick setup for progress tracking."""
    return create_progress_dashboard(data)
