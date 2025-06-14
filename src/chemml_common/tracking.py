"""
ChemML Experiment Tracking with Weights & Biases (wandb)
=========================================================

This module provides comprehensive experiment tracking capabilities for ChemML
using Weights & Biases. It handles authentication, experiment initialization,
metric logging, and visualization for machine learning experiments.

Author: ChemML Team
Date: June 14, 2025
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WandBTracker:
    """
    Comprehensive Weights & Biases experiment tracking for ChemML.

    Features:
    - Automatic authentication
    - Experiment initialization with metadata
    - Metric and artifact logging
    - Model versioning and checkpoints
    - Hyperparameter tracking
    - Custom charts and visualizations
    """

    def __init__(
        self,
        project_name: str = "chemml-experiments",
        entity: Optional[str] = None,
        auto_login: bool = True,
    ):
        """
        Initialize wandb tracker.

        Args:
            project_name: wandb project name
            entity: wandb entity/team name
            auto_login: Whether to automatically login
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.is_initialized = False

        if auto_login:
            self.login()

    def login(self, api_key: Optional[str] = None) -> bool:
        """
        Login to wandb with API key.

        Args:
            api_key: wandb API key. If None, will try environment variable.

        Returns:
            bool: True if login successful
        """
        try:
            # Use provided API key or ChemML default
            if api_key is None:
                api_key = os.getenv(
                    "WANDB_API_KEY", "b4f102d87161194b68baa7395d5862aa3f93b2b7"
                )

            # Set environment variable
            os.environ["WANDB_API_KEY"] = api_key

            # Login to wandb
            wandb.login(key=api_key, relogin=True)

            logger.info("✅ Successfully logged into wandb")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to login to wandb: {str(e)}")
            return False

    def init_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        """
        Initialize a new wandb experiment run.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary with hyperparameters
            tags: List of tags for the experiment
            notes: Description/notes for the experiment
            group: Group name for related experiments

        Returns:
            bool: True if initialization successful
        """
        try:
            # Add ChemML-specific metadata
            enhanced_config = {
                **config,
                "framework": "ChemML",
                "timestamp": datetime.now().isoformat(),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            }

            # Initialize wandb run
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=experiment_name,
                config=enhanced_config,
                tags=tags or ["chemml"],
                notes=notes,
                group=group,
                reinit=True,
            )

            self.is_initialized = True
            logger.info(f"✅ Initialized wandb experiment: {experiment_name}")
            logger.info(f"📊 wandb URL: {self.run.url}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize wandb experiment: {str(e)}")
            return False

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metric name -> value
            step: Step number for the metrics
            prefix: Prefix to add to metric names
        """
        if not self.is_initialized:
            logger.warning("⚠️ wandb not initialized. Call init_experiment() first.")
            return

        try:
            # Add prefix if provided
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

            # Log to wandb
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

            logger.info(f"📈 Logged metrics: {list(metrics.keys())}")

        except Exception as e:
            logger.error(f"❌ Failed to log metrics: {str(e)}")

    def log_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        step: Optional[int] = None,
    ) -> None:
        """
        Log comprehensive model performance metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model
            step: Step number
        """
        try:
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            # Log metrics
            metrics = {
                f"{model_name}/mse": mse,
                f"{model_name}/mae": mae,
                f"{model_name}/rmse": rmse,
                f"{model_name}/r2_score": r2,
            }

            self.log_metrics(metrics, step=step)

            # Create scatter plot
            self.log_prediction_plot(y_true, y_pred, model_name, step)

        except Exception as e:
            logger.error(f"❌ Failed to log model performance: {str(e)}")

    def log_prediction_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        step: Optional[int] = None,
    ) -> None:
        """
        Log prediction vs actual scatter plot.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model name for plot title
            step: Step number
        """
        try:
            import matplotlib.pyplot as plt

            # Create scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot(
                [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
            )
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"{model_name} - Predictions vs Actual")
            plt.grid(True, alpha=0.3)

            # Log to wandb
            wandb.log({f"{model_name}/prediction_plot": wandb.Image(plt)}, step=step)
            plt.close()

        except Exception as e:
            logger.error(f"❌ Failed to create prediction plot: {str(e)}")

    def log_molecular_data(
        self,
        smiles: List[str],
        properties: Dict[str, List[float]],
        step: Optional[int] = None,
    ) -> None:
        """
        Log molecular data and properties.

        Args:
            smiles: List of SMILES strings
            properties: Dictionary of property name -> list of values
            step: Step number
        """
        try:
            # Create a wandb table
            columns = ["smiles"] + list(properties.keys())
            data = []

            for i, smile in enumerate(smiles):
                row = [smile]
                for prop_values in properties.values():
                    row.append(prop_values[i] if i < len(prop_values) else None)
                data.append(row)

            table = wandb.Table(columns=columns, data=data)
            wandb.log({"molecular_data": table}, step=step)

            logger.info(
                f"🧪 Logged {len(smiles)} molecules with properties: {list(properties.keys())}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to log molecular data: {str(e)}")

    def log_training_progress(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log training progress metrics.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Current learning rate
            additional_metrics: Additional metrics to log
        """
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
        }

        if val_loss is not None:
            metrics["val_loss"] = val_loss

        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        if additional_metrics:
            metrics.update(additional_metrics)

        self.log_metrics(metrics, step=epoch)

    def save_artifact(
        self,
        file_path: Union[str, Path],
        artifact_name: str,
        artifact_type: str = "model",
        description: Optional[str] = None,
    ) -> None:
        """
        Save file as wandb artifact.

        Args:
            file_path: Path to the file to save
            artifact_name: Name of the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            description: Description of the artifact
        """
        try:
            artifact = wandb.Artifact(
                name=artifact_name, type=artifact_type, description=description
            )

            artifact.add_file(str(file_path))
            self.run.log_artifact(artifact)

            logger.info(f"💾 Saved artifact: {artifact_name}")

        except Exception as e:
            logger.error(f"❌ Failed to save artifact: {str(e)}")

    def log_hyperparameter_sweep(
        self,
        config: Dict[str, Any],
        performance_metric: float,
        metric_name: str = "validation_score",
    ) -> None:
        """
        Log hyperparameter sweep results.

        Args:
            config: Hyperparameter configuration
            performance_metric: Performance metric value
            metric_name: Name of the performance metric
        """
        metrics = {
            metric_name: performance_metric,
            **{f"hp_{k}": v for k, v in config.items()},
        }

        self.log_metrics(metrics)

    def finish_experiment(self) -> None:
        """
        Finish the current wandb run.
        """
        if self.run is not None:
            wandb.finish()
            logger.info("✅ Finished wandb experiment")
            self.is_initialized = False
            self.run = None

    def create_summary_report(self) -> Dict[str, Any]:
        """
        Create a summary report of the experiment.

        Returns:
            Dictionary containing experiment summary
        """
        if not self.is_initialized:
            return {}

        try:
            summary = {
                "run_id": self.run.id,
                "run_name": self.run.name,
                "project": self.run.project,
                "url": self.run.url,
                "config": dict(self.run.config),
                "summary": dict(self.run.summary),
                "tags": self.run.tags,
                "notes": self.run.notes,
            }

            return summary

        except Exception as e:
            logger.error(f"❌ Failed to create summary report: {str(e)}")
            return {}


# Global tracker instance
_global_tracker = None


def get_tracker(project_name: str = "chemml-experiments") -> WandBTracker:
    """
    Get global wandb tracker instance.

    Args:
        project_name: wandb project name

    Returns:
        WandBTracker instance
    """
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = WandBTracker(project_name=project_name)

    return _global_tracker


def quick_start_experiment(
    experiment_name: str,
    config: Dict[str, Any],
    project_name: str = "chemml-experiments",
) -> WandBTracker:
    """
    Quick start a wandb experiment with minimal setup.

    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary
        project_name: wandb project name

    Returns:
        WandBTracker instance
    """
    tracker = get_tracker(project_name)
    tracker.init_experiment(experiment_name, config)
    return tracker


# Example usage functions
def demo_molecular_experiment():
    """
    Demonstration of wandb tracking for molecular experiments.
    """
    # Initialize tracker
    tracker = WandBTracker()

    # Start experiment
    config = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "dataset": "molecular_properties",
        "task": "regression",
    }

    tracker.init_experiment(
        experiment_name="molecular_property_prediction_demo",
        config=config,
        tags=["demo", "molecular", "regression"],
        notes="Demonstration of wandb tracking in ChemML",
    )

    # Simulate training progress
    for epoch in range(10):
        train_loss = 1.0 - (epoch * 0.08) + np.random.normal(0, 0.02)
        val_loss = 1.1 - (epoch * 0.075) + np.random.normal(0, 0.03)

        tracker.log_training_progress(
            epoch=epoch,
            train_loss=max(0.1, train_loss),
            val_loss=max(0.15, val_loss),
            learning_rate=0.001 * (0.95**epoch),
        )

    # Log final model performance
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.normal(0, 0.2, 100)

    tracker.log_model_performance(y_true, y_pred, "random_forest")

    # Log molecular data
    smiles = ["CCO", "CCC", "CC(C)O", "CCCO"]
    properties = {
        "molecular_weight": [46.07, 44.10, 60.10, 60.10],
        "logp": [-0.31, 0.25, 0.05, 0.25],
    }

    tracker.log_molecular_data(smiles, properties)

    # Finish experiment
    summary = tracker.create_summary_report()
    print(f"📊 Experiment completed: {summary.get('url', 'N/A')}")

    tracker.finish_experiment()


if __name__ == "__main__":
    # Run demonstration
    demo_molecular_experiment()
