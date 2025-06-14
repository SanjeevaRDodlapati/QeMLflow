#!/usr/bin/env python3
"""
Example: Using WandB in ChemML Experiments
==========================================

This script demonstrates how to use Weights & Biases experiment tracking
in your ChemML experiments.
"""

import os
import sys

sys.path.append("src")

import numpy as np

from chemml_common.wandb_integration import *


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

    print(f"✅ Experiment completed!")
    print(f"🔗 View results: {run.url}")

    # Finish experiment
    finish_experiment()


if __name__ == "__main__":
    run_example_experiment()
