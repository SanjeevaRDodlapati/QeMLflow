#!/usr/bin/env python3
"""
ChemML Best-in-Class Libraries Demo
==================================

This script demonstrates how ChemML leverages the best available libraries
for distributed ML training, hyperparameter search, and performance monitoring.

Run with: python tools/examples/best_libraries_demo.py
"""

import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_distributed_training():
    """Demonstrate Ray-based distributed training."""
    logger.info("🚀 Demonstrating Ray-based distributed training...")

    try:
        import ray
        from ray import tune
        from ray.train import ScalingConfig

        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        logger.info(f"Ray initialized with {ray.available_resources()} resources")

        # Simple distributed molecular property prediction
        @ray.remote
        class MolecularPredictor:
            def __init__(self, model_params: Dict[str, Any]):
                self.params = model_params

            def train_batch(self, molecules: List[str], properties: List[float]):
                """Simulate training on a batch of molecules."""
                # Simulate training time
                time.sleep(0.1)
                return {
                    "batch_size": len(molecules),
                    "avg_property": np.mean(properties),
                    "training_time": 0.1,
                }

        # Create distributed predictors
        predictors = [MolecularPredictor.remote({"model_type": "rf"}) for _ in range(4)]

        # Simulate molecular data
        molecules = [f"CC{'C' * i}O" for i in range(100)]  # Simple SMILES
        properties = np.random.normal(0, 1, 100).tolist()

        # Distribute training
        batch_size = 25
        futures = []
        for i, predictor in enumerate(predictors):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_molecules = molecules[start_idx:end_idx]
            batch_properties = properties[start_idx:end_idx]

            future = predictor.train_batch.remote(batch_molecules, batch_properties)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        logger.info(
            f"✅ Distributed training completed: {len(results)} batches processed"
        )

        # Cleanup
        ray.shutdown()

    except ImportError:
        logger.warning("⚠️  Ray not installed. Install with: pip install ray[default]")

def demo_hyperparameter_optimization():
    """Demonstrate Optuna-based hyperparameter optimization."""
    logger.info("🎯 Demonstrating Optuna hyperparameter optimization...")

    try:
        import optuna
        from optuna.trial import Trial

        def objective(trial: Trial) -> float:
            """Objective function for molecular property prediction."""
            # Suggest hyperparameters
            _n_estimators = trial.suggest_int("n_estimators", 10, 100)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)

            # Simulate model training and validation
            # In practice, this would train a real model
            score = np.random.random()

            # Add some realistic correlation with hyperparameters
            score *= (
                1.0 - abs(learning_rate - 0.1) / 0.2
            )  # Prefer learning_rate around 0.1
            score *= 1.0 - abs(max_depth - 6) / 6  # Prefer max_depth around 6

            return score

        # Create study with advanced settings
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )

        # Run optimization
        study.optimize(objective, n_trials=20, timeout=30)

        # Results
        logger.info(f"✅ Best parameters: {study.best_params}")
        logger.info(f"✅ Best score: {study.best_value:.4f}")
        logger.info(f"✅ Number of trials: {len(study.trials)}")

        # Show pruning statistics
        pruned = sum(
            1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED
        )
        logger.info(
            f"✅ Pruned trials: {pruned}/{len(study.trials)} ({pruned/len(study.trials)*100:.1f}%)"
        )

    except ImportError:
        logger.warning("⚠️  Optuna not installed. Install with: pip install optuna")

def demo_performance_monitoring():
    """Demonstrate MLflow + W&B performance monitoring."""
    logger.info("📊 Demonstrating MLflow + W&B performance monitoring...")

    # MLflow tracking
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        # Set experiment
        mlflow.set_experiment("chemml_best_libraries_demo")

        with mlflow.start_run():
            # Generate synthetic molecular property data
            X, y = make_regression(
                n_samples=1000, n_features=20, noise=0.1, random_state=42
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            # Predictions and metrics
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "mse": mse,
                    "r2_score": r2,
                    "n_train_samples": len(X_train),
                    "n_test_samples": len(X_test),
                }
            )

            # Log model
            mlflow.sklearn.log_model(model, "random_forest_model")

            logger.info(f"✅ MLflow run completed - MSE: {mse:.4f}, R²: {r2:.4f}")

    except ImportError:
        logger.warning("⚠️  MLflow not installed. Install with: pip install mlflow")

    # Weights & Biases tracking (optional - requires account)
    try:
        import wandb

        # Initialize (will prompt for login if not configured)
        wandb.init(
            project="chemml-demo",
            config={
                "learning_rate": 0.01,
                "epochs": 10,
                "batch_size": 32,
                "model_type": "random_forest",
            },
            mode="disabled",  # Set to 'online' if you have W&B account
        )

        # Log metrics
        for epoch in range(10):
            loss = 1.0 / (epoch + 1)  # Simulated decreasing loss
            accuracy = 1.0 - loss  # Simulated increasing accuracy

            wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})

        wandb.finish()
        logger.info("✅ W&B tracking completed (disabled mode)")

    except ImportError:
        logger.warning("⚠️  W&B not installed. Install with: pip install wandb")

def demo_quantum_integration():
    """Demonstrate PennyLane quantum computing integration."""
    logger.info("⚛️  Demonstrating PennyLane quantum computing...")

    try:
        import pennylane as qml
        import torch

        # Create quantum device
        n_qubits = 4
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def quantum_neural_network(inputs, weights):
            """Simple quantum neural network for molecular features."""
            # Encode classical data
            for i, x in enumerate(inputs):
                qml.RY(x, wires=i % n_qubits)

            # Variational quantum circuit
            for layer in range(2):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Initialize parameters
        weight_shape = (2, n_qubits, 2)
        weights = torch.randn(weight_shape, requires_grad=True)

        # Generate sample molecular descriptors
        molecular_features = torch.randn(4)

        # Forward pass
        output = quantum_neural_network(molecular_features, weights)
        logger.info(f"✅ Quantum circuit output: {[f'{x:.4f}' for x in output]}")

        # Demonstrate gradient computation
        cost = sum(output)
        cost.backward()
        logger.info(f"✅ Quantum gradients computed: {weights.grad is not None}")

    except ImportError:
        logger.warning(
            "⚠️  PennyLane not installed. Install with: pip install pennylane"
        )

def demo_uncertainty_quantification():
    """Demonstrate advanced uncertainty quantification."""
    logger.info("📊 Demonstrating uncertainty quantification...")

    try:
        import scipy.stats as stats
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        # Generate data
        X, y = make_regression(
            n_samples=1000, n_features=10, noise=0.1, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Ensemble for uncertainty estimation
        n_models = 10
        models = []
        predictions = []

        for i in range(n_models):
            # Bootstrap sampling
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]

            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=i)
            model.fit(X_boot, y_boot)
            models.append(model)

            # Predict
            pred = model.predict(X_test)
            predictions.append(pred)

        # Calculate ensemble statistics
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Confidence intervals (assuming normal distribution)
        confidence_level = 0.95
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)

        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred

        # Coverage analysis
        coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
        avg_interval_width = np.mean(upper_bound - lower_bound)

        logger.info("✅ Ensemble uncertainty quantification:")
        logger.info(f"   - Coverage: {coverage:.3f} (target: {confidence_level})")
        logger.info(f"   - Average interval width: {avg_interval_width:.3f}")
        logger.info(f"   - Average prediction uncertainty: {np.mean(std_pred):.3f}")

    except Exception as e:
        logger.warning(f"⚠️  Uncertainty quantification demo failed: {e}")

def demo_automl():
    """Demonstrate AutoML capabilities."""
    logger.info("🤖 Demonstrating AutoML with FLAML...")

    try:
        from flaml import AutoML
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        # Generate molecular property prediction data
        X, y = make_regression(n_samples=500, n_features=15, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize AutoML
        automl = AutoML()

        # Auto-train with time budget
        automl.fit(
            X_train,
            y_train,
            task="regression",
            time_budget=30,  # 30 seconds
            metric="r2",
            verbose=0,
        )

        # Predictions
        y_pred = automl.predict(X_test)

        # Results
        from sklearn.metrics import mean_squared_error, r2_score

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        logger.info("✅ AutoML completed:")
        logger.info(f"   - Best model: {automl.best_estimator}")
        logger.info(f"   - Best config: {automl.best_config}")
        logger.info(f"   - R² score: {r2:.4f}")
        logger.info(f"   - MSE: {mse:.4f}")

    except ImportError:
        logger.warning("⚠️  FLAML not installed. Install with: pip install flaml")

def main():
    """Run all demonstrations."""
    print("🧬 ChemML Best-in-Class Libraries Demonstration")
    print("=" * 50)

    demos = [
        ("Distributed Training (Ray)", demo_distributed_training),
        ("Hyperparameter Optimization (Optuna)", demo_hyperparameter_optimization),
        ("Performance Monitoring (MLflow + W&B)", demo_performance_monitoring),
        ("Quantum Computing (PennyLane)", demo_quantum_integration),
        ("Uncertainty Quantification", demo_uncertainty_quantification),
        ("AutoML (FLAML)", demo_automl),
    ]

    for name, demo_func in demos:
        print(f"\n🔄 Running: {name}")
        print("-" * 30)
        try:
            demo_func()
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
        print()

    print("🎉 All demonstrations completed!")
    print("\n📚 Next Steps:")
    print("   1. Install missing libraries with: pip install -r requirements.txt")
    print("   2. Configure backends in: config/advanced_config.yaml")
    print(
        "   3. Start implementing long-term enhancements following the implementation guide"
    )

if __name__ == "__main__":
    main()
