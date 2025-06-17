"""
QeMLflow Model Pipeline Framework
==============================

Provides end-to-end workflows from data loading to model evaluation.

Key Features:
- Automated data preprocessing and feature engineering
- Model selection and hyperparameter optimization
- Cross-validation and performance evaluation
- Pipeline persistence and reproducibility
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import QeMLflow components
try:
    from .data_processing import (
        AdvancedDataPreprocessor,
        QeMLflowDataLoader,
        IntelligentDataSplitter,
    )
    from .enhanced_models import (
        AutoMLModel,
        EnsembleModel,
        create_automl_model,
        create_ensemble_model,
    )
    from .models import (
        LinearModel,
        RandomForestModel,
        SVMModel,
        create_linear_model,
        create_rf_model,
        create_svm_model,
    )
except ImportError:
    from qemlflow.core.data_processing import (
        AdvancedDataPreprocessor,
        QeMLflowDataLoader,
        IntelligentDataSplitter,
    )
    from qemlflow.core.enhanced_models import (
        AutoMLModel,
        EnsembleModel,
        create_automl_model,
        create_ensemble_model,
    )
    from qemlflow.core.models import (
        LinearModel,
        RandomForestModel,
        SVMModel,
        create_linear_model,
        create_rf_model,
        create_svm_model,
    )

# Optional imports
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class QeMLflowPipeline:
    """
    Comprehensive machine learning pipeline for chemistry applications.

    Automates the entire ML workflow from data loading to model evaluation.
    """

    def __init__(
        self,
        task_type: str = "regression",
        auto_preprocess: bool = True,
        auto_feature_selection: bool = True,
        auto_model_selection: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
        experiment_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize QeMLflow pipeline.

        Args:
            task_type: 'regression' or 'classification'
            auto_preprocess: Enable automatic preprocessing
            auto_feature_selection: Enable automatic feature selection
            auto_model_selection: Enable automatic model selection
            cv_folds: Cross-validation folds
            random_state: Random seed for reproducibility
            experiment_name: Name for experiment tracking
            cache_dir: Directory for caching results
        """
        self.task_type = task_type
        self.auto_preprocess = auto_preprocess
        self.auto_feature_selection = auto_feature_selection
        self.auto_model_selection = auto_model_selection
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.experiment_name = (
            experiment_name
            or f"qemlflow_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Setup caching
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".qemlflow" / "pipelines"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.data_loader = QeMLflowDataLoader()
        self.preprocessor = AdvancedDataPreprocessor()
        self.splitter = IntelligentDataSplitter()

        # Pipeline state
        self.data = {}
        self.models = {}
        self.results = {}
        self.best_model = None
        self.pipeline_fitted = False

        # Experiment tracking
        self.experiment_run = None

    def load_data(
        self,
        source: Union[str, pd.DataFrame, Path],
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "QeMLflowPipeline":
        """
        Load data from various sources.

        Args:
            source: Dataset name, DataFrame, or file path
            smiles_column: Name of SMILES column
            target_columns: Target columns for prediction
            **kwargs: Additional arguments for data loading
        """
        print(f"📊 Loading data from {source}...")

        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, (str, Path)):
            if str(source) in self.data_loader.dataset_urls:
                # Load public dataset
                df = self.data_loader.load_dataset(
                    str(source),
                    smiles_column=smiles_column,
                    target_columns=target_columns,
                    **kwargs,
                )
            else:
                # Load custom dataset
                df = self.data_loader.load_custom_dataset(
                    source,
                    smiles_column=smiles_column,
                    target_columns=target_columns,
                    **kwargs,
                )
        else:
            raise ValueError(f"Unsupported data source type: {type(source)}")

        # Store data info
        self.data["raw"] = df
        self.data["smiles_column"] = smiles_column
        self.data["target_columns"] = target_columns or [
            col for col in df.columns if col != smiles_column
        ]

        print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        return self

    def preprocess(
        self,
        feature_types: Optional[Dict[str, str]] = None,
        preprocessing_config: Optional[Dict] = None,
    ) -> "QeMLflowPipeline":
        """
        Preprocess data with automatic feature engineering.

        Args:
            feature_types: Dictionary mapping column names to types
            preprocessing_config: Custom preprocessing configuration
        """
        if not self.auto_preprocess:
            print("⏭️  Skipping preprocessing (auto_preprocess=False)")
            return self

        print("🔧 Preprocessing data...")

        # Use stored or provided columns
        smiles_column = self.data["smiles_column"]
        target_columns = self.data["target_columns"]

        # Apply preprocessing
        processed_data = self.preprocessor.create_preprocessing_pipeline(
            self.data["raw"],
            smiles_column=smiles_column,
            target_columns=target_columns,
            feature_types=feature_types,
        )

        # Store processed data
        self.data["processed"] = processed_data
        self.data["X"] = processed_data["X"]
        self.data["y"] = processed_data["y"]
        self.data["smiles"] = processed_data["smiles"]
        self.data["feature_names"] = processed_data["feature_names"]

        print(f"✅ Generated {len(processed_data['feature_names'])} features")
        return self

    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        split_method: str = "random",
    ) -> "QeMLflowPipeline":
        """
        Split data into train/validation/test sets.

        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            split_method: 'random', 'scaffold', 'stratified', 'temporal'
        """
        print(f"📊 Splitting data using {split_method} method...")

        X = self.data["X"]
        y = self.data["y"]
        smiles = self.data.get("smiles")

        # Perform split
        split_result = self.splitter.split_dataset(
            X=X,
            y=y,
            smiles=smiles,
            test_size=test_size,
            val_size=val_size,
            split_method=split_method,
            random_state=self.random_state,
        )

        # Store split data
        self.data["splits"] = split_result
        split_info = split_result["split_info"]

        print(
            f"✅ Split completed: {split_info['train_size']} train, "
            f"{split_info['val_size']} val, {split_info['test_size']} test"
        )

        return self

    def add_model(
        self, model_name: str, model_type: str, **model_params
    ) -> "QeMLflowPipeline":
        """
        Add a model to the pipeline.

        Args:
            model_name: Name for the model
            model_type: Type of model ('linear', 'rf', 'svm', 'xgb', 'ensemble', 'automl')
            **model_params: Parameters for model initialization
        """
        print(f"🤖 Adding {model_type} model: {model_name}")

        # Create model based on type
        if model_type == "linear":
            model = create_linear_model(task_type=self.task_type, **model_params)
        elif model_type == "rf":
            model = create_rf_model(task_type=self.task_type, **model_params)
        elif model_type == "svm":
            model = create_svm_model(task_type=self.task_type, **model_params)
        elif model_type == "ensemble":
            model = create_ensemble_model(task_type=self.task_type, **model_params)
        elif model_type == "automl":
            model = create_automl_model(task_type=self.task_type, **model_params)
        else:
            # Try enhanced models
            try:
                from .enhanced_models import (
                    create_cnn_model,
                    create_lightgbm_model,
                    create_xgboost_model,
                )

                if model_type == "xgb":
                    model = create_xgboost_model(
                        task_type=self.task_type, **model_params
                    )
                elif model_type == "lgb":
                    model = create_lightgbm_model(
                        task_type=self.task_type, **model_params
                    )
                elif model_type == "cnn":
                    input_dim = (
                        len(self.data["feature_names"])
                        if "feature_names" in self.data
                        else 100
                    )
                    model = create_cnn_model(
                        input_dim=input_dim, task_type=self.task_type, **model_params
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            except ImportError:
                raise ValueError(f"Model type '{model_type}' not available")

        self.models[model_name] = {
            "model": model,
            "type": model_type,
            "params": model_params,
            "fitted": False,
            "metrics": {},
        }

        return self

    def add_default_models(self) -> "QeMLflowPipeline":
        """Add a default set of models for comparison."""
        print("🤖 Adding default model suite...")

        default_models = [
            ("linear_ridge", "linear", {"regularization": "ridge"}),
            ("random_forest", "rf", {"n_estimators": 100}),
            ("svm_rbf", "svm", {"kernel": "rbf"}),
        ]

        # Add advanced models if available
        try:

            default_models.append(("xgboost", "xgb", {}))
        except ImportError:
            pass

        try:

            default_models.append(("automl", "automl", {"n_trials": 20}))
        except ImportError:
            pass

        for name, model_type, params in default_models:
            try:
                self.add_model(name, model_type, **params)
            except Exception as e:
                print(f"⚠️  Could not add {name}: {e}")

        return self

    def train_models(
        self, model_names: Optional[List[str]] = None, use_cross_validation: bool = True
    ) -> "QeMLflowPipeline":
        """
        Train specified models or all models.

        Args:
            model_names: List of model names to train (None for all)
            use_cross_validation: Perform cross-validation evaluation
        """
        if "splits" not in self.data:
            raise ValueError(
                "Data must be split before training. Call split_data() first."
            )

        models_to_train = model_names or list(self.models.keys())
        print(f"🏋️  Training {len(models_to_train)} models...")

        # Get training data
        X_train = self.data["splits"]["X_train"]
        y_train = self.data["splits"]["y_train"]
        X_val = self.data["splits"]["X_val"]
        y_val = self.data["splits"]["y_val"]

        for model_name in models_to_train:
            if model_name not in self.models:
                print(f"⚠️  Model '{model_name}' not found, skipping")
                continue

            print(f"🔄 Training {model_name}...")

            try:
                model_info = self.models[model_name]
                model = model_info["model"]

                # Train model
                train_metrics = model.fit(X_train.values, y_train.values.ravel())

                # Evaluate on validation set
                val_predictions = model.predict(X_val.values)
                val_metrics = self._calculate_metrics(
                    y_val.values.ravel(), val_predictions
                )

                # Cross-validation
                cv_metrics = {}
                if use_cross_validation:
                    cv_metrics = self._cross_validate_model(
                        model, X_train.values, y_train.values.ravel()
                    )

                # Store results
                model_info["fitted"] = True
                model_info["metrics"] = {
                    "train": train_metrics,
                    "validation": val_metrics,
                    "cross_validation": cv_metrics,
                }

                print(f"✅ {model_name} trained successfully")

                # Log to experiment tracker
                if self.experiment_run:
                    self._log_model_metrics(model_name, model_info["metrics"])

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue

        return self

    def evaluate_models(
        self, model_names: Optional[List[str]] = None, use_test_set: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate models and return comparison results.

        Args:
            model_names: List of model names to evaluate (None for all)
            use_test_set: Use test set for final evaluation

        Returns:
            DataFrame with model comparison results
        """
        models_to_evaluate = model_names or [
            name for name, info in self.models.items() if info["fitted"]
        ]

        if not models_to_evaluate:
            raise ValueError("No fitted models to evaluate")

        print(f"📊 Evaluating {len(models_to_evaluate)} models...")

        results = []

        # Get evaluation data
        if use_test_set and "X_test" in self.data["splits"]:
            X_eval = self.data["splits"]["X_test"]
            y_eval = self.data["splits"]["y_test"]
            eval_set = "test"
        else:
            X_eval = self.data["splits"]["X_val"]
            y_eval = self.data["splits"]["y_val"]
            eval_set = "validation"

        for model_name in models_to_evaluate:
            model_info = self.models[model_name]

            if not model_info["fitted"]:
                print(f"⚠️  Model '{model_name}' not fitted, skipping")
                continue

            try:
                model = model_info["model"]
                predictions = model.predict(X_eval.values)
                metrics = self._calculate_metrics(y_eval.values.ravel(), predictions)

                result = {
                    "model": model_name,
                    "type": model_info["type"],
                    "eval_set": eval_set,
                    **metrics,
                }

                # Add cross-validation metrics if available
                if "cross_validation" in model_info["metrics"]:
                    cv_metrics = model_info["metrics"]["cross_validation"]
                    for key, value in cv_metrics.items():
                        result[f"cv_{key}"] = value

                results.append(result)

            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue

        results_df = pd.DataFrame(results)
        self.results["evaluation"] = results_df

        # Find best model
        if not results_df.empty:
            if self.task_type == "regression":
                best_idx = results_df["r2"].idxmax()
            else:
                best_idx = results_df["accuracy"].idxmax()

            best_model_name = results_df.loc[best_idx, "model"]
            self.best_model = self.models[best_model_name]["model"]

            print(f"🏆 Best model: {best_model_name}")

        return results_df

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Make predictions using specified model or best model.

        Args:
            X: Input features
            model_name: Name of model to use (None for best model)

        Returns:
            Predictions array
        """
        if model_name:
            if model_name not in self.models or not self.models[model_name]["fitted"]:
                raise ValueError(f"Model '{model_name}' not found or not fitted")
            model = self.models[model_name]["model"]
        elif self.best_model:
            model = self.best_model
        else:
            raise ValueError("No model specified and no best model found")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return model.predict(X)

    def save_pipeline(self, filepath: Union[str, Path]) -> None:
        """Save the complete pipeline to disk."""
        filepath = Path(filepath)

        # Create pipeline data
        pipeline_data = {
            "task_type": self.task_type,
            "auto_preprocess": self.auto_preprocess,
            "auto_feature_selection": self.auto_feature_selection,
            "auto_model_selection": self.auto_model_selection,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
            "experiment_name": self.experiment_name,
            "data_info": {
                "smiles_column": self.data.get("smiles_column"),
                "target_columns": self.data.get("target_columns"),
                "feature_names": self.data.get("feature_names"),
            },
            "results": self.results,
            "best_model_name": None,
        }

        # Find best model name
        if self.best_model:
            for name, info in self.models.items():
                if info["model"] is self.best_model:
                    pipeline_data["best_model_name"] = name
                    break

        # Save pipeline metadata
        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(pipeline_data, f, indent=2, default=str)

        # Save models
        models_dir = filepath.parent / f"{filepath.stem}_models"
        models_dir.mkdir(exist_ok=True)

        for model_name, model_info in self.models.items():
            if model_info["fitted"]:
                model_path = models_dir / f"{model_name}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model_info["model"], f)

        # Save preprocessor
        if hasattr(self, "preprocessor"):
            preprocessor_path = filepath.parent / f"{filepath.stem}_preprocessor.pkl"
            with open(preprocessor_path, "wb") as f:
                pickle.dump(self.preprocessor, f)

        print(f"✅ Pipeline saved to {filepath}")

    @classmethod
    def load_pipeline(cls, filepath: Union[str, Path]) -> "QeMLflowPipeline":
        """Load a saved pipeline from disk."""
        filepath = Path(filepath)

        # Load pipeline metadata
        with open(filepath.with_suffix(".json"), "r") as f:
            pipeline_data = json.load(f)

        # Create pipeline instance
        pipeline = cls(
            task_type=pipeline_data["task_type"],
            auto_preprocess=pipeline_data["auto_preprocess"],
            auto_feature_selection=pipeline_data["auto_feature_selection"],
            auto_model_selection=pipeline_data["auto_model_selection"],
            cv_folds=pipeline_data["cv_folds"],
            random_state=pipeline_data["random_state"],
            experiment_name=pipeline_data["experiment_name"],
        )

        # Restore data info
        pipeline.data.update(pipeline_data["data_info"])
        pipeline.results = pipeline_data["results"]

        # Load models
        models_dir = filepath.parent / f"{filepath.stem}_models"
        if models_dir.exists():
            for model_file in models_dir.glob("*.pkl"):
                model_name = model_file.stem
                with open(model_file, "rb") as f:
                    model = pickle.load(f)

                pipeline.models[model_name] = {
                    "model": model,
                    "fitted": True,
                    "type": "loaded",
                    "params": {},
                    "metrics": {},
                }

        # Restore best model
        best_model_name = pipeline_data.get("best_model_name")
        if best_model_name and best_model_name in pipeline.models:
            pipeline.best_model = pipeline.models[best_model_name]["model"]

        # Load preprocessor
        preprocessor_path = filepath.parent / f"{filepath.stem}_preprocessor.pkl"
        if preprocessor_path.exists():
            with open(preprocessor_path, "rb") as f:
                pipeline.preprocessor = pickle.load(f)

        pipeline.pipeline_fitted = True
        print(f"✅ Pipeline loaded from {filepath}")
        return pipeline

    def run_full_pipeline(
        self,
        data_source: Union[str, pd.DataFrame, Path],
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
        models_to_include: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run the complete pipeline from data loading to evaluation.

        Args:
            data_source: Source of data
            smiles_column: Name of SMILES column
            target_columns: Target columns for prediction
            models_to_include: Specific models to include

        Returns:
            DataFrame with model comparison results
        """
        print("🚀 Running full QeMLflow pipeline...")

        # Setup experiment tracking
        self._setup_experiment_tracking()

        try:
            # Load and preprocess data
            self.load_data(data_source, smiles_column, target_columns)
            self.preprocess()
            self.split_data()

            # Add models
            if models_to_include:
                for model_spec in models_to_include:
                    if isinstance(model_spec, tuple):
                        name, model_type, params = model_spec
                        self.add_model(name, model_type, **params)
                    else:
                        # Assume it's just a model type
                        self.add_model(model_spec, model_spec)
            else:
                self.add_default_models()

            # Train and evaluate
            self.train_models()
            results = self.evaluate_models()

            print("🎉 Pipeline completed successfully!")
            return results

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise
        finally:
            self._cleanup_experiment_tracking()

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate appropriate metrics based on task type."""
        metrics = {}

        if self.task_type == "regression":
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = np.mean(np.abs(y_true - y_pred))
            metrics["r2"] = r2_score(y_true, y_pred)
        else:
            # Classification
            y_pred_binary = (
                (y_pred > 0.5).astype(int) if len(np.unique(y_true)) == 2 else y_pred
            )
            metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)

            # AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics["auc"] = roc_auc_score(y_true, y_pred)
                except ValueError:
                    pass

        return metrics

    def _cross_validate_model(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Perform cross-validation evaluation."""
        try:
            if self.task_type == "regression":
                cv_scores = cross_val_score(
                    model.model,
                    X,
                    y,
                    cv=KFold(
                        n_splits=self.cv_folds,
                        shuffle=True,
                        random_state=self.random_state,
                    ),
                    scoring="r2",
                )
                return {"cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std()}
            else:
                cv_scores = cross_val_score(
                    model.model,
                    X,
                    y,
                    cv=StratifiedKFold(
                        n_splits=self.cv_folds,
                        shuffle=True,
                        random_state=self.random_state,
                    ),
                    scoring="accuracy",
                )
                return {
                    "cv_accuracy_mean": cv_scores.mean(),
                    "cv_accuracy_std": cv_scores.std(),
                }
        except Exception as e:
            print(f"⚠️  Cross-validation failed: {e}")
            return {}

    def _setup_experiment_tracking(self):
        """Setup experiment tracking if available."""
        if HAS_WANDB:
            try:
                self.experiment_run = wandb.init(
                    project="qemlflow-pipelines",
                    name=self.experiment_name,
                    config={
                        "task_type": self.task_type,
                        "auto_preprocess": self.auto_preprocess,
                        "cv_folds": self.cv_folds,
                        "random_state": self.random_state,
                    },
                )
                print("✅ Experiment tracking started")
            except Exception as e:
                print(f"⚠️  Could not start experiment tracking: {e}")

    def _cleanup_experiment_tracking(self):
        """Cleanup experiment tracking."""
        if self.experiment_run and HAS_WANDB:
            wandb.finish()

    def _log_model_metrics(self, model_name: str, metrics: Dict):
        """Log model metrics to experiment tracker."""
        if self.experiment_run and HAS_WANDB:
            # Flatten metrics for logging
            log_data = {}
            for split, split_metrics in metrics.items():
                for metric, value in split_metrics.items():
                    log_data[f"{model_name}_{split}_{metric}"] = value
            wandb.log(log_data)


# Convenience functions
def create_pipeline(task_type: str = "regression", **kwargs) -> QeMLflowPipeline:
    """Create a new QeMLflow pipeline."""
    return QeMLflowPipeline(task_type=task_type, **kwargs)


def quick_pipeline(
    data_source: Union[str, pd.DataFrame, Path],
    task_type: str = "regression",
    smiles_column: str = "smiles",
    target_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run a quick pipeline with default settings."""
    pipeline = create_pipeline(task_type=task_type)
    return pipeline.run_full_pipeline(
        data_source=data_source,
        smiles_column=smiles_column,
        target_columns=target_columns,
    )


# Export main classes and functions
# __all__ = ["QeMLflowPipeline", "create_pipeline", "quick_pipeline"]
