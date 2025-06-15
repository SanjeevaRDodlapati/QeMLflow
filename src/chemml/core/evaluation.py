"""
ChemML Core Evaluation
=====================

Comprehensive evaluation metrics and tools for chemistry ML models.
Provides domain-specific metrics and visualization tools.

Key Features:
- Standard ML metrics (regression, classification)
- Chemistry-specific evaluation metrics
- Model interpretation and visualization tools
- Cross-validation and statistical testing
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Core ML metrics
try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import KFold, cross_val_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Statistical tests
try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class RegressionEvaluator:
    """
    Comprehensive evaluation for regression tasks.

    Provides standard regression metrics plus chemistry-specific evaluations.
    """

    def __init__(self):
        """Initialize regression evaluator."""
        if not HAS_SKLEARN:
            warnings.warn("sklearn not available. Some metrics may not work.")

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        if HAS_SKLEARN:
            # Standard regression metrics
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)
        else:
            # Manual calculation
            metrics["mse"] = np.mean((y_true - y_pred) ** 2)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = np.mean(np.abs(y_true - y_pred))

            # R² calculation
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Additional metrics
        metrics["max_error"] = np.max(np.abs(y_true - y_pred))
        metrics["mean_error"] = np.mean(y_true - y_pred)  # Bias

        # Pearson correlation
        if HAS_SCIPY:
            correlation, p_value = stats.pearsonr(y_true, y_pred)
            metrics["pearson_r"] = correlation
            metrics["pearson_p"] = p_value
        else:
            # Manual Pearson correlation
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            metrics["pearson_r"] = correlation if not np.isnan(correlation) else 0

        return metrics

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predicted vs True Values",
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Create prediction vs true values plot.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure (if plotting available)
        """
        if not HAS_PLOTTING:
            warnings.warn("Matplotlib not available. Cannot create plots.")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

        # Labels and title
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(title)

        # Add R² to plot
        r2 = (
            r2_score(y_true, y_pred)
            if HAS_SKLEARN
            else np.corrcoef(y_true, y_pred)[0, 1] ** 2
        )
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Plot",
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Create residual plot for error analysis.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure (if plotting available)
        """
        if not HAS_PLOTTING:
            warnings.warn("Matplotlib not available. Cannot create plots.")
            return None

        residuals = y_true - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color="r", linestyle="--")
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Predicted")

        # Residual histogram
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Residual Distribution")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


class ClassificationEvaluator:
    """
    Comprehensive evaluation for classification tasks.
    """

    def __init__(self):
        """Initialize classification evaluator."""
        if not HAS_SKLEARN:
            warnings.warn("sklearn not available. Some metrics may not work.")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            y_prob: Predicted probabilities (for AUC calculation)
            average: Averaging strategy for multi-class metrics

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        if HAS_SKLEARN:
            # Basic metrics
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(
                y_true, y_pred, average=average, zero_division=0
            )
            metrics["recall"] = recall_score(
                y_true, y_pred, average=average, zero_division=0
            )
            metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

            # AUC (for binary classification or with probabilities)
            if y_prob is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        metrics["auc"] = roc_auc_score(y_true, y_prob)
                    else:
                        # Multi-class classification
                        metrics["auc"] = roc_auc_score(
                            y_true, y_prob, multi_class="ovr", average=average
                        )
                except:
                    pass
        else:
            # Manual calculation for basic accuracy
            metrics["accuracy"] = np.mean(y_true == y_pred)

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Create confusion matrix plot.

        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            class_names: Names of classes
            normalize: Whether to normalize the matrix
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure (if plotting available)
        """
        if not HAS_PLOTTING or not HAS_SKLEARN:
            warnings.warn(
                "Plotting libraries not available. Cannot create confusion matrix."
            )
            return None

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            ax=ax,
            square=True,
            xticklabels=class_names,
            yticklabels=class_names,
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


class ModelComparator:
    """
    Compare multiple models using various evaluation strategies.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize model comparator.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state

    def compare_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "r2",
        task_type: str = "regression",
    ) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.

        Args:
            models: Dictionary of model_name -> model_instance
            X: Feature matrix
            y: Target values
            scoring: Scoring metric for comparison
            task_type: 'regression' or 'classification'

        Returns:
            DataFrame with comparison results
        """
        if not HAS_SKLEARN:
            warnings.warn("sklearn not available. Cannot perform cross-validation.")
            return pd.DataFrame()

        results = []
        kfold = KFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        for name, model in models.items():
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

                results.append(
                    {
                        "model": name,
                        "mean_score": cv_scores.mean(),
                        "std_score": cv_scores.std(),
                        "min_score": cv_scores.min(),
                        "max_score": cv_scores.max(),
                    }
                )

            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values("mean_score", ascending=False)

        return df_results

    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        metric_name: str = "Score",
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Create bar plot comparing model performances.

        Args:
            results_df: Results DataFrame from compare_models
            metric_name: Name of the metric being compared
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure (if plotting available)
        """
        if not HAS_PLOTTING or results_df.empty:
            warnings.warn("Cannot create comparison plot.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        x_pos = np.arange(len(results_df))

        # Create bars with error bars
        bars = ax.bar(
            x_pos,
            results_df["mean_score"],
            yerr=results_df["std_score"],
            capsize=5,
            alpha=0.7,
        )

        # Customize plot
        ax.set_xlabel("Models")
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df["model"], rotation=45, ha="right")

        # Add value labels on bars
        for i, (bar, mean_score) in enumerate(zip(bars, results_df["mean_score"])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{mean_score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


# Chemistry-specific evaluation metrics
def calculate_chemical_space_coverage(
    molecules_train: List[str],
    molecules_test: List[str],
    fingerprint_type: str = "morgan",
) -> Dict[str, float]:
    """
    Calculate how well the training set covers the chemical space of the test set.

    Args:
        molecules_train: Training set SMILES
        molecules_test: Test set SMILES
        fingerprint_type: Type of fingerprint to use

    Returns:
        Dictionary with coverage metrics
    """
    try:
        from sklearn.neighbors import NearestNeighbors

        from ..core.featurizers import MorganFingerprint

        # Generate fingerprints
        featurizer = MorganFingerprint(radius=2, n_bits=1024)
        train_fps = featurizer.featurize(molecules_train)
        test_fps = featurizer.featurize(molecules_test)

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=1, metric="jaccard")
        nn.fit(train_fps)
        distances, _ = nn.kneighbors(test_fps)

        # Calculate coverage metrics
        similarities = 1 - distances.flatten()  # Convert distance to similarity

        metrics = {
            "mean_similarity": similarities.mean(),
            "min_similarity": similarities.min(),
            "coverage_at_0.7": (similarities >= 0.7).mean(),
            "coverage_at_0.5": (similarities >= 0.5).mean(),
            "coverage_at_0.3": (similarities >= 0.3).mean(),
        }

        return metrics

    except ImportError:
        warnings.warn(
            "Required dependencies not available for chemical space coverage."
        )
        return {}


def applicability_domain_analysis(
    X_train: np.ndarray, X_test: np.ndarray, threshold_percentile: float = 95
) -> Dict[str, Any]:
    """
    Analyze applicability domain using distance-based methods.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        threshold_percentile: Percentile for defining the domain boundary

    Returns:
        Dictionary with applicability domain metrics
    """
    try:
        from sklearn.neighbors import NearestNeighbors

        # Fit nearest neighbors on training data
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_train)

        # Calculate distances for training and test sets
        train_distances, _ = nn.kneighbors(X_train)
        test_distances, _ = nn.kneighbors(X_test)

        # Define domain boundary
        domain_threshold = np.percentile(train_distances, threshold_percentile)

        # Calculate metrics
        train_in_domain = (train_distances.flatten() <= domain_threshold).mean()
        test_in_domain = (test_distances.flatten() <= domain_threshold).mean()

        metrics = {
            "domain_threshold": domain_threshold,
            "train_in_domain": train_in_domain,
            "test_in_domain": test_in_domain,
            "test_mean_distance": test_distances.mean(),
            "test_max_distance": test_distances.max(),
        }

        return metrics

    except ImportError:
        warnings.warn("sklearn not available for applicability domain analysis.")
        return {}


# Convenience functions
def quick_regression_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Quick regression evaluation with essential metrics."""
    evaluator = RegressionEvaluator()
    return evaluator.evaluate(y_true, y_pred)


def quick_classification_eval(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Quick classification evaluation with essential metrics."""
    evaluator = ClassificationEvaluator()
    return evaluator.evaluate(y_true, y_pred)


# Export main classes and functions
__all__ = [
    "RegressionEvaluator",
    "ClassificationEvaluator",
    "ModelComparator",
    "calculate_chemical_space_coverage",
    "applicability_domain_analysis",
    "quick_regression_eval",
    "quick_classification_eval",
]
