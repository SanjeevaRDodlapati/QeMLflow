#!/usr/bin/env python3
"""
Example: Using WandB in ChemML Experiments
==========================================

This script demonstrates how to use Weights & Biases experiment tracking
in your ChemML experiments.
"""

import os
import sys

import numpy as np

# Updated ChemML experiment tracking
try:
    from chemml.integrations.experiment_tracking import setup_wandb_tracking

    HAS_WANDB_INTEGRATION = True
except ImportError:
    HAS_WANDB_INTEGRATION = False
    print(
        "⚠️  ChemML experiment tracking not available. Install with: pip install wandb"
    )

# Optional direct wandb import
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ChemML experiment tracking helper functions
def start_experiment(experiment_name, config=None, tags=None):
    """Start a wandb experiment with ChemML integration."""
    if HAS_WANDB_INTEGRATION:
        return setup_wandb_tracking(
            experiment_name, config or {}, project="chemml-examples"
        )
    elif HAS_WANDB:
        return wandb.init(
            project="chemml-examples",
            name=experiment_name,
            config=config or {},
            tags=tags or [],
        )
    else:
        print(f"📊 Demo: Starting experiment {experiment_name}")
        return None


def log_metrics(metrics, step=None):
    """Log metrics to wandb."""
    if HAS_WANDB and wandb.run:
        wandb.log(metrics, step=step)
    else:
        print(f"📈 Demo: Logging metrics {metrics}")


def log_model_results(y_true, y_pred, model_name):
    """Log model evaluation results."""
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    results = {
        f"{model_name}_mse": mse,
        f"{model_name}_r2": r2,
        f"{model_name}_rmse": mse**0.5,
    }

    log_metrics(results)
    return results


def finish_experiment():
    """Finish the wandb experiment."""
    if HAS_WANDB and wandb.run:
        wandb.finish()
    else:
        print("✅ Demo: Experiment finished")


def run_example_experiment():
    """Run an example experiment with wandb tracking."""

    print("🧪 Running example ChemML experiment with wandb...")

    # Configuration for the experiment
    config = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "dataset": "molecular_properties",
        "task": "regression",
        "features": "molecular_descriptors",
    }

    # Start experiment
    run = start_experiment(
        experiment_name="chemml_example_experiment",
        config=config,
        tags=["example", "tutorial", "molecular_ml"],
    )

    # Simulate training progress
    print("📈 Simulating training progress...")
    for epoch in range(10):
        # Simulate metrics
        train_loss = 1.0 - (epoch * 0.08) + np.random.normal(0, 0.02)
        val_loss = 1.1 - (epoch * 0.075) + np.random.normal(0, 0.03)

        metrics = {
            "epoch": epoch,
            "train_loss": max(0.1, train_loss),
            "val_loss": max(0.15, val_loss),
            "learning_rate": 0.001 * (0.95**epoch),
        }

        log_metrics(metrics, step=epoch)
        print(
            f"  Epoch {epoch}: train_loss={metrics['train_loss']:.3f}, val_loss={metrics['val_loss']:.3f}"
        )

    # Simulate final model evaluation
    print("🎯 Simulating final model evaluation...")
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.normal(0, 0.2, 100)

    log_model_results(y_true, y_pred, "random_forest")

    # Log molecular data summary
    molecular_summary = {
        "num_molecules": 1000,
        "avg_molecular_weight": 342.5,
        "avg_logp": 2.1,
        "num_drug_like": 785,
        "drug_like_percentage": 78.5,
    }

    log_metrics(molecular_summary)

    print("✅ Experiment completed!")
    if run and hasattr(run, "url"):
        print(f"🔗 View results: {run.url}")
    else:
        print("📊 Demo mode: No wandb URL available")

    # Finish experiment
    finish_experiment()


if __name__ == "__main__":
    run_example_experiment()
