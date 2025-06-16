"""
Comprehensive metrics utilities for ChemML evaluations.

This module provides evaluation metrics for various machine learning tasks
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_recall_curve,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn not available. Some metrics will be limited.")
    SKLEARN_AVAILABLE = False

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available. Molecular metrics will be limited.")
    RDKIT_AVAILABLE = False


class ClassificationMetrics:
    """Comprehensive classification metrics evaluator."""

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}

        if SKLEARN_AVAILABLE:
            metrics.update(
                {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    ),
                    "recall": recall_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    ),
                    "f1_score": f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    ),
                }
            )

            if y_prob is not None and len(np.unique(y_true)) == 2:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                except ValueError:
                    metrics["roc_auc"] = 0.5
        else:
            # Fallback implementations
            metrics.update(
                {
                    "accuracy": accuracy(y_true, y_pred),
                    "precision": precision(y_true, y_pred),
                    "recall": recall(y_true, y_pred),
                    "f1_score": f1_score_manual(y_true, y_pred),
                }
            )

        return metrics

    @staticmethod
    def confusion_matrix_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate confusion matrix and derived metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with confusion matrix and metrics
        """
        if SKLEARN_AVAILABLE:
            cm = confusion_matrix(y_true, y_pred)
        else:
            # Simple confusion matrix for binary classification
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            cm = np.zeros((len(unique_labels), len(unique_labels)))

            for i, true_label in enumerate(unique_labels):
                for j, pred_label in enumerate(unique_labels):
                    cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

        # Calculate sensitivity and specificity for binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            return {
                "confusion_matrix": cm,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "true_positive_rate": sensitivity,
                "false_positive_rate": 1 - specificity,
            }

        return {"confusion_matrix": cm}


class RegressionMetrics:
    """Comprehensive regression metrics evaluator."""

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}

        if SKLEARN_AVAILABLE:
            metrics.update(
                {
                    "mse": mean_squared_error(y_true, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                }
            )
        else:
            # Fallback implementations
            metrics.update(
                {
                    "mse": mean_squared_error_manual(y_true, y_pred),
                    "rmse": np.sqrt(mean_squared_error_manual(y_true, y_pred)),
                    "mae": mean_absolute_error_manual(y_true, y_pred),
                    "r2": r_squared(y_true, y_pred),
                }
            )

        # Additional metrics
        metrics.update(
            {
                "mape": mean_absolute_percentage_error(y_true, y_pred),
                "max_error": np.max(np.abs(y_true - y_pred)),
                "explained_variance": explained_variance_score(y_true, y_pred),
            }
        )

        return metrics


class MolecularMetrics:
    """Metrics specific to molecular analysis."""

    @staticmethod
    def tanimoto_similarity(
        smiles1: str, smiles2: str, fingerprint_type: str = "morgan"
    ) -> float:
        """
        Calculate Tanimoto similarity between two molecules.

        Args:
            smiles1: First molecule SMILES
            smiles2: Second molecule SMILES
            fingerprint_type: Type of fingerprint ('morgan', 'maccs')

        Returns:
            Tanimoto similarity coefficient
        """
        if not RDKIT_AVAILABLE:
            return 0.5  # Default similarity

        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            if fingerprint_type == "morgan":
                fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
                fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)
            elif fingerprint_type == "maccs":
                from rdkit.Chem import MACCSkeys

                fp1 = MACCSkeys.GenMACCSKeys(mol1)
                fp2 = MACCSkeys.GenMACCSKeys(mol2)
            else:
                raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")

            return DataStructs.TanimotoSimilarity(fp1, fp2)

        except Exception as e:
            logging.warning(f"Error calculating Tanimoto similarity: {e}")
            return 0.0

    @staticmethod
    def diversity_metrics(smiles_list: List[str]) -> Dict[str, float]:
        """
        Calculate diversity metrics for a set of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary of diversity metrics
        """
        if not RDKIT_AVAILABLE or len(smiles_list) < 2:
            return {
                "mean_pairwise_similarity": 0.5,
                "diversity_index": 0.5,
                "max_similarity": 1.0,
                "min_similarity": 0.0,
            }

        similarities = []
        valid_molecules = []

        # Calculate all pairwise similarities
        for i, smiles1 in enumerate(smiles_list):
            mol1 = Chem.MolFromSmiles(smiles1)
            if mol1 is None:
                continue
            valid_molecules.append(smiles1)

            for j, smiles2 in enumerate(smiles_list[i + 1 :], i + 1):
                mol2 = Chem.MolFromSmiles(smiles2)
                if mol2 is None:
                    continue

                similarity = MolecularMetrics.tanimoto_similarity(smiles1, smiles2)
                similarities.append(similarity)

        if not similarities:
            return {
                "mean_pairwise_similarity": 0.0,
                "diversity_index": 1.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
            }

        similarities = np.array(similarities)

        return {
            "mean_pairwise_similarity": np.mean(similarities),
            "diversity_index": 1.0 - np.mean(similarities),
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities),
            "std_similarity": np.std(similarities),
        }


# Fallback implementations for when scikit-learn is not available
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy manually."""
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty arrays provided")
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")

    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision manually for binary classification."""
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall manually for binary classification."""
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )


def f1_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score manually."""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


def mean_squared_error_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MSE manually."""
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAE manually."""
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² manually."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total > 0 else 0


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate explained variance score."""
    y_true_var = np.var(y_true)
    if y_true_var == 0:
        return 0.0
    residual_var = np.var(y_true - y_pred)
    return 1.0 - (residual_var / y_true_var)


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "regression",
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation for different task types.

    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        task_type: Type of task ("regression" or "classification")
        y_prob: Predicted probabilities for classification (optional)

    Returns:
        Dictionary of evaluation metrics
    """
    if task_type == "regression":
        return RegressionMetrics.calculate_all_metrics(y_true, y_pred)
    elif task_type == "classification":
        return ClassificationMetrics.calculate_all_metrics(y_true, y_pred, y_prob)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def calculate_enrichment_factor(
    y_true: np.ndarray, y_scores: np.ndarray, fraction: float = 0.1
) -> float:
    """
    Calculate enrichment factor for virtual screening evaluation.

    Args:
        y_true: True binary labels (1 for active, 0 for inactive)
        y_scores: Predicted scores/probabilities
        fraction: Fraction of top-scored compounds to consider

    Returns:
        Enrichment factor
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]

    # Number of compounds to consider
    n_selected = int(len(y_true) * fraction)

    # Calculate enrichment
    actives_in_selected = np.sum(sorted_labels[:n_selected])
    total_actives = np.sum(y_true)

    if total_actives == 0:
        return 0.0

    expected_actives = total_actives * fraction

    return actives_in_selected / expected_actives if expected_actives > 0 else 0.0
