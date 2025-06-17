from typing import Dict\nfrom typing import List\nfrom typing import Optional\nfrom typing import Union\n"""
Machine Learning utilities for QeMLflow

This module provides utilities for ML workflows in computational chemistry
and drug discovery applications.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class DatasetSplitter:
    """Split datasets for machine learning workflows"""

    def __init__(
        self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(
        self, X: np.ndarray, y: np.ndarray, stratify: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """
        Split data into train/validation/test sets

        Args:
            X: Feature matrix
            y: Target vector
            stratify: Whether to stratify the split for classification

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        stratify_y = y if stratify else None

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_y,
        )

        # Second split: separate train and validation
        if self.val_size > 0:
            # Adjust validation size relative to remaining data
            adjusted_val_size = self.val_size / (1 - self.test_size)
            stratify_temp = y_temp if stratify else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=adjusted_val_size,
                random_state=self.random_state,
                stratify=stratify_temp,
            )

            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_temp, X_test, y_temp, y_test

    def temporal_split(
        self,
        df: pd.DataFrame,
        date_column: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data temporally for time series or drug discovery pipelines

        Args:
            df: DataFrame with temporal data
            date_column: Name of the date/time column
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        df_sorted = df.sort_values(date_column)
        n_samples = len(df_sorted)

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        return {
            "train": df_sorted.iloc[:train_end],
            "val": df_sorted.iloc[train_end:val_end],
            "test": df_sorted.iloc[val_end:],
        }


class FeatureScaler:
    """Feature scaling utilities"""

    def __init__(self, method: str = "standard"):
        """
        Initialize scaler

        Args:
            method: 'standard', 'minmax', or 'robust'
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.method = method
        self.fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data"""
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transforming")
        return self.scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        return self.scaler.inverse_transform(X)

    def save(self, filepath: str) -> None:
        """Save fitted scaler"""
        if not self.fitted:
            raise ValueError("Cannot save unfitted scaler")
        joblib.dump(self.scaler, filepath)

    def load(self, filepath: str) -> None:
        """Load fitted scaler"""
        self.scaler = joblib.load(filepath)
        self.fitted = True


class ModelEvaluator:
    """Evaluate machine learning models"""

    def __init__(self):
        self.classification_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
        self.regression_metrics = ["mse", "mae", "r2", "rmse"]

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate classification model"""
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Handle binary vs multiclass
        average = "binary" if len(np.unique(y_true)) == 2 else "macro"

        metrics["precision"] = precision_score(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

        # ROC AUC (if probabilities provided)
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr"
                    )
            except ValueError as e:
                logging.warning(f"Could not calculate ROC AUC: {e}")
                metrics["roc_auc"] = np.nan

        return metrics

    def evaluate_regression(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression model"""
        metrics = {}

        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])

        return metrics

    def confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Detailed confusion matrix analysis"""
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "support": report["macro avg"]["support"],
        }


class CrossValidator:
    """Cross-validation utilities"""

    def __init__(
        self, cv_folds: int = 5, scoring: str = "accuracy", random_state: int = 42
    ):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state

    def cross_validate_model(
        self, model: Any, X: np.ndarray, y: np.ndarray, stratified: bool = True
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform cross-validation

        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            stratified: Use stratified CV for classification

        Returns:
            Dictionary with CV results
        """
        if stratified and self._is_classification_task(y):
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold

            cv = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

        scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring)

        return {
            "scores": scores,
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max(),
        }

    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Determine if task is classification based on target values"""
        unique_values = np.unique(y)

        # If all values are integers and there are relatively few unique values
        if np.all(np.equal(np.mod(y, 1), 0)) and len(unique_values) < len(y) * 0.1:
            return True

        # If target is string/categorical
        if not np.issubdtype(y.dtype, np.number):
            return True

        return False


class ModelPersistence:
    """Save and load models with metadata"""

    @staticmethod
    def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None) -> None:
        """
        Save model with optional metadata

        Args:
            model: Trained model object
            filepath: Path to save model
            metadata: Optional metadata dictionary
        """
        model_data = {"model": model, "metadata": metadata or {}}

        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> Tuple[Any, Dict]:
        """
        Load model with metadata

        Args:
            filepath: Path to saved model

        Returns:
            Tuple of (model, metadata)
        """
        model_data = joblib.load(filepath)

        if isinstance(model_data, dict) and "model" in model_data:
            return model_data["model"], model_data.get("metadata", {})
        else:
            # Backward compatibility for models saved without metadata
            return model_data, {}


def calculate_feature_importance(
    model: Any, feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract feature importance from trained model

    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: Optional list of feature names

    Returns:
        DataFrame with feature importance scores
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not have feature importance information")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    return df


def optimize_hyperparameters(
    model: Any,
    param_grid: Dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
) -> Dict:
    """
    Optimize hyperparameters using GridSearchCV

    Args:
        model: Scikit-learn model
        param_grid: Parameter grid for search
        X: Feature matrix
        y: Target vector
        cv: Number of CV folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary with best parameters and score
    """
    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1
    )

    grid_search.fit(X, y)

    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_,
        "cv_results": grid_search.cv_results_,
    }


def evaluate_model(
    model, X: np.ndarray, y: np.ndarray, task_type: str = "classification"
) -> Dict[str, float]:
    """
    Evaluate a trained model on given data.

    Args:
        model: Trained model with predict method
        X: Feature matrix
        y: True labels/values
        task_type: Type of task ("classification" or "regression")

    Returns:
        Dictionary of evaluation metrics
    """
    predictions = model.predict(X)

    if task_type.lower() == "classification":
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(
                y, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(y, predictions, average="weighted", zero_division=0),
            "f1": f1_score(y, predictions, average="weighted", zero_division=0),
        }

        # Add AUC if binary classification and model has predict_proba
        if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y, y_proba)
            except Exception:
                pass  # Skip if AUC calculation fails

    else:  # regression
        metrics = {
            "r2": r2_score(y, predictions),
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
        }

    return metrics


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Split data into train and test sets.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed
        stratify: Whether to stratify split for classification

    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )


def normalize_features(
    X: np.ndarray, method: str = "standard", return_scaler: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
    """
    Normalize features using specified method.

    Args:
        X: Feature matrix
        method: Normalization method ("standard", "minmax", "robust")
        return_scaler: Whether to return the scaler along with normalized features

    Returns:
        Normalized features (if return_scaler=False) or tuple of (normalized features, scaler)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    X_normalized = scaler.fit_transform(X)

    if return_scaler:
        return X_normalized, scaler
    else:
        return X_normalized
