"""
Comprehensive visualization utilities for QeMLflow.

This module provides plotting and visualization functions for molecular data,
machine learning models, and chemical analysis results.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True

    # Set default style
    plt.style.use("default")
    sns.set_palette("husl")
    
    # Configure for testing environments
    if ('pytest' in sys.modules or 
        'unittest' in sys.modules or 
        os.environ.get('TESTING', '').lower() in ('1', 'true')):
        plt.ioff()  # Non-interactive mode for testing
        plt.rcParams['figure.max_open_warning'] = 0
        
except ImportError:
    logging.warning("Matplotlib/Seaborn not available. Visualization will be limited.")
    MATPLOTLIB_AVAILABLE = False


def safe_show(fig=None, save_path=None):
    """Safe replacement for safe_show() that auto-closes figures in testing."""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    try:
        if save_path:
            if fig:
                fig.savefig(save_path, bbox_inches='tight', dpi=100)
            else:
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
        
        # Check if we're in a testing environment
        is_testing = ('pytest' in sys.modules or 
                     'unittest' in sys.modules or 
                     os.environ.get('TESTING', '').lower() in ('1', 'true'))
        
        if not is_testing:
            if fig:
                fig.show()
            else:
                safe_show()
                
    finally:
        # Always close figures in testing, optionally in production
        if ('pytest' in sys.modules or 
            'unittest' in sys.modules or 
            os.environ.get('TESTING', '').lower() in ('1', 'true')):
            if fig:
                plt.close(fig)
            else:
                plt.close()

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, rdDepictor

    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available. Molecular visualization will be limited.")
    RDKIT_AVAILABLE = False

try:
    PLOTLY_AVAILABLE = True
except ImportError:
    logging.warning("Plotly not available. Interactive plots will be limited.")
    PLOTLY_AVAILABLE = False


class MolecularVisualizer:
    """Visualize molecular structures and properties."""

    @staticmethod
    def plot_molecular_structure(
        mol_input, filename: Optional[str] = None, size: Tuple[int, int] = (300, 300)
    ) -> Optional[str]:
        """
        Visualize molecular structure from SMILES string or Mol object.

        Args:
            mol_input: SMILES string or RDKit Mol object
            filename: Output filename (optional)
            size: Image size as (width, height)

        Returns:
            Filename if saved, None otherwise
        """
        if not RDKIT_AVAILABLE:
            logging.warning("RDKit not available for molecular visualization")
            return None

        try:
            # Handle both SMILES strings and Mol objects
            if isinstance(mol_input, str):
                mol = Chem.MolFromSmiles(mol_input)
                if mol is None:
                    logging.warning(f"Invalid SMILES: {mol_input}")
                    return None
            else:
                # Assume it's a Mol object
                mol = mol_input
                if mol is None:
                    logging.warning("Invalid molecule object")
                    return None

            # Generate 2D coordinates
            rdDepictor.Compute2DCoords(mol)

            # Create image
            img = Draw.MolToImage(mol, size=size)

            if filename:
                img.save(filename)
                logging.info(f"Molecular structure saved to {filename}")
                return filename
            else:
                img.show()
                return None

        except Exception as e:
            logging.error(f"Error visualizing molecule {mol_input}: {e}")
            return None

    @staticmethod
    def plot_multiple_molecules(
        smiles_list: List[str],
        titles: Optional[List[str]] = None,
        filename: Optional[str] = None,
        mols_per_row: int = 4,
        mol_size: Tuple[int, int] = (200, 200),
    ) -> Optional[str]:
        """
        Plot multiple molecular structures in a grid.

        Args:
            smiles_list: List of SMILES strings
            titles: Optional titles for each molecule
            filename: Output filename (optional)
            mols_per_row: Number of molecules per row
            mol_size: Size of each molecule image

        Returns:
            Filename if saved, None otherwise
        """
        if not RDKIT_AVAILABLE:
            logging.warning("RDKit not available for molecular visualization")
            return None

        try:
            mols = []
            legends = []

            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols.append(mol)
                    if titles and i < len(titles):
                        legends.append(titles[i])
                    else:
                        legends.append(f"Molecule {i+1}")

            if not mols:
                logging.warning("No valid molecules to display")
                return None

            # Create grid image
            img = Draw.MolsToGridImage(
                mols, molsPerRow=mols_per_row, subImgSize=mol_size, legends=legends
            )

            if filename:
                img.save(filename)
                logging.info(f"Molecular grid saved to {filename}")
                return filename
            else:
                img.show()
                return None

        except Exception as e:
            logging.error(f"Error creating molecular grid: {e}")
            return None

    @staticmethod
    def plot_molecular_properties_distribution(
        molecules_df: pd.DataFrame,
        properties: List[str],
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot distribution of molecular properties.

        Args:
            molecules_df: DataFrame with molecular data
            properties: List of property columns to plot
            filename: Output filename (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available for plotting")
            return

        try:
            n_props = len(properties)
            fig, axes = plt.subplots(2, (n_props + 1) // 2, figsize=(12, 8))
            if n_props == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, prop in enumerate(properties):
                if prop in molecules_df.columns:
                    data = molecules_df[prop].dropna()
                    if len(data) > 0:
                        axes[i].hist(data, bins=30, alpha=0.7, edgecolor="black")
                        axes[i].set_title(f"Distribution of {prop}")
                        axes[i].set_xlabel(prop)
                        axes[i].set_ylabel("Frequency")
                        axes[i].grid(True, alpha=0.3)

            # Remove empty subplots
            for i in range(n_props, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logging.info(f"Property distribution plot saved to {filename}")
            else:
                safe_show()

        except Exception as e:
            logging.error(f"Error plotting property distributions: {e}")


class ModelVisualizer:
    """Visualize model performance and analysis."""

    @staticmethod
    def plot_feature_importance(
        importances: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Importance",
        top_n: int = 20,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot feature importance from a trained model.

        Args:
            importances: Array of feature importances
            feature_names: List of feature names
            title: Plot title
            top_n: Number of top features to display
            filename: Output filename (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available for plotting")
            return

        # Validate inputs
        if len(importances) == 0 or len(feature_names) == 0:
            raise ValueError("Importances and feature names cannot be empty")

        if len(importances) != len(feature_names):
            raise ValueError("Length of importances and feature_names must match")

        try:
            # Convert to numpy array if needed
            importances = np.array(importances)

            # Get top features
            indices = np.argsort(importances)[::-1][:top_n]
            top_importances = importances[indices]
            top_names = [feature_names[i] for i in indices]

            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_importances)), top_importances[::-1])

            # Color bars by importance
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_importances)))
            for bar, color in zip(bars, colors[::-1]):
                bar.set_color(color)

            plt.yticks(range(len(top_importances)), top_names[::-1])
            plt.xlabel("Importance")
            plt.title(title)
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logging.info(f"Feature importance plot saved to {filename}")
            else:
                safe_show()

        except Exception as e:
            logging.error(f"Error plotting feature importance: {e}")

    @staticmethod
    def plot_model_performance(
        history: Dict[str, List[float]],
        title: str = "Model Performance",
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot training history and model performance.

        Args:
            history: Dictionary with metrics history
            title: Plot title
            filename: Output filename (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available for plotting")
            return

        try:
            metrics = list(history.keys())
            n_metrics = len(metrics)

            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
            if n_metrics == 1:
                axes = [axes]

            for i, metric in enumerate(metrics):
                epochs = range(1, len(history[metric]) + 1)
                axes[i].plot(epochs, history[metric], "b-", linewidth=2, label=metric)
                axes[i].set_title(f"{metric.title()} Over Time")
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(metric.title())
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logging.info(f"Model performance plot saved to {filename}")
            else:
                safe_show()

        except Exception as e:
            logging.error(f"Error plotting model performance: {e}")

    @staticmethod
    def plot_predictions_vs_actual(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot predicted vs actual values for regression.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            filename: Output filename (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available for plotting")
            return

        try:
            plt.figure(figsize=(8, 8))

            # Scatter plot
            plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="black", linewidth=0.5)

            # Perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect Prediction",
            )

            # Calculate R²
            from sklearn.metrics import r2_score

            r2 = r2_score(y_true, y_pred)

            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"{title} (R² = {r2:.3f})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis("equal")

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logging.info(f"Predictions plot saved to {filename}")
            else:
                safe_show()

        except Exception as e:
            logging.error(f"Error plotting predictions: {e}")

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrix for classification.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            filename: Output filename (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available for plotting")
            return

        try:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(8, 6))

            if MATPLOTLIB_AVAILABLE:
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels,
                )
            else:
                plt.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.colorbar()

            plt.title(title)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logging.info(f"Confusion matrix saved to {filename}")
            else:
                safe_show()

        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")


class ChemicalSpaceVisualizer:
    """Visualize chemical space and molecular similarity."""

    @staticmethod
    def plot_chemical_space_pca(
        molecules_df: pd.DataFrame,
        smiles_column: str = "SMILES",
        color_column: Optional[str] = None,
        title: str = "Chemical Space (PCA)",
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot chemical space using PCA of molecular descriptors.

        Args:
            molecules_df: DataFrame with molecular data
            smiles_column: Name of SMILES column
            color_column: Column to color points by (optional)
            title: Plot title
            filename: Output filename (optional)
        """
        if not MATPLOTLIB_AVAILABLE or not RDKIT_AVAILABLE:
            logging.warning(
                "Required libraries not available for chemical space plotting"
            )
            return

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Calculate molecular descriptors
            descriptors_list = []
            valid_indices = []

            for idx, smiles in enumerate(molecules_df[smiles_column]):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Calculate basic descriptors
                    desc = {
                        "MW": Descriptors.MolWt(mol),
                        "LogP": Descriptors.MolLogP(mol),
                        "TPSA": Descriptors.TPSA(mol),
                        "HBD": Descriptors.NumHDonors(mol),
                        "HBA": Descriptors.NumHAcceptors(mol),
                        "RB": Descriptors.NumRotatableBonds(mol),
                        "Rings": Descriptors.RingCount(mol),
                    }
                    descriptors_list.append(list(desc.values()))
                    valid_indices.append(idx)

            if len(descriptors_list) < 2:
                logging.warning("Not enough valid molecules for PCA")
                return

            # Perform PCA
            X = np.array(descriptors_list)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Plot
            plt.figure(figsize=(10, 8))

            if color_column and color_column in molecules_df.columns:
                colors = molecules_df.iloc[valid_indices][color_column]
                scatter = plt.scatter(
                    X_pca[:, 0],
                    X_pca[:, 1],
                    c=colors,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )
                plt.colorbar(scatter, label=color_column)
            else:
                plt.scatter(
                    X_pca[:, 0],
                    X_pca[:, 1],
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )

            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            plt.title(title)
            plt.grid(True, alpha=0.3)

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                logging.info(f"Chemical space plot saved to {filename}")
            else:
                safe_show()

        except Exception as e:
            logging.error(f"Error plotting chemical space: {e}")


def plot_molecular_structure(
    mol_input, filename: Optional[str] = None
) -> Optional[str]:
    """
    Convenience function to visualize a molecular structure.

    Args:
        mol_input: SMILES string or RDKit Mol object
        filename: Output filename (optional)

    Returns:
        Filename if saved, None otherwise
    """
    return MolecularVisualizer.plot_molecular_structure(mol_input, filename)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importance",
    filename: Optional[str] = None,
) -> None:
    """
    Convenience function to plot feature importance.

    Args:
        importances: Array of feature importances
        feature_names: List of feature names
        title: Plot title
        filename: Output filename (optional)
    """
    ModelVisualizer.plot_feature_importance(
        importances, feature_names, title, filename=filename
    )


def plot_model_performance(
    history: Dict[str, List[float]],
    title: str = "Model Performance",
    filename: Optional[str] = None,
) -> None:
    """
    Convenience function to plot model performance.

    Args:
        history: Dictionary with metrics history
        title: Plot title
        filename: Output filename (optional)
    """
    ModelVisualizer.plot_model_performance(history, title, filename)


def create_dashboard_plots(
    results_df: pd.DataFrame, output_dir: str = "plots"
) -> Dict[str, str]:
    """
    Create a comprehensive set of visualization plots.

    Args:
        results_df: DataFrame with results to visualize
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot types to filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    created_plots = {}

    try:
        # Property distributions
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            filename = str(output_path / "property_distributions.png")
            MolecularVisualizer.plot_molecular_properties_distribution(
                results_df, numeric_cols.tolist()[:6], filename
            )
            created_plots["property_distributions"] = filename

        # Chemical space (if SMILES available)
        if "SMILES" in results_df.columns:
            filename = str(output_path / "chemical_space.png")
            ChemicalSpaceVisualizer.plot_chemical_space_pca(
                results_df, filename=filename
            )
            created_plots["chemical_space"] = filename

        logging.info(
            f"Created {len(created_plots)} visualization plots in {output_dir}"
        )

    except Exception as e:
        logging.error(f"Error creating dashboard plots: {e}")

    return created_plots
