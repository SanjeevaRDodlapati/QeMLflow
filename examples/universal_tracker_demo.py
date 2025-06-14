"""
ChemML Universal Tracker Demo
============================

This script demonstrates the one-command wandb integration across different usage patterns.
Run this to test the universal tracking system.
"""

import time

import numpy as np

from chemml_common.tracker import (
    finish_global_tracking,
    log_global,
    quick_track,
    start_global_tracking,
    track_experiment,
    track_training,
)


def demo_decorator_pattern():
    """Demo: Using the tracker as a decorator."""
    print("\n=== Demo 1: Decorator Pattern ===")

    @track_experiment(project="demo_project", tags=["demo", "decorator"])
    def molecular_optimization(smiles, learning_rate=0.01, epochs=10):
        """Simulate molecular optimization with automatic tracking."""
        print(f"Optimizing {smiles} with lr={learning_rate} for {epochs} epochs")

        # Simulate training loop
        for epoch in range(epochs):
            loss = np.random.exponential(0.5) * np.exp(-epoch * 0.1)
            accuracy = (
                0.5 + 0.4 * (1 - np.exp(-epoch * 0.2)) + np.random.normal(0, 0.05)
            )
            accuracy = max(0, min(1, accuracy))  # Clamp to [0,1]

            # In the decorator pattern, we can access tracker through function attributes
            # This would typically be handled by the tracking system
            print(f"  Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
            time.sleep(0.1)  # Simulate computation

        # Return results (automatically logged)
        return {
            "final_accuracy": accuracy,
            "final_loss": loss,
            "epochs_trained": epochs,
        }

    # Run the decorated function
    result = molecular_optimization("CCO", learning_rate=0.005, epochs=5)
    print(f"Result: {result}")


def demo_context_manager():
    """Demo: Using the tracker as a context manager."""
    print("\n=== Demo 2: Context Manager Pattern ===")

    with track_experiment(
        project="demo_project",
        name="context_manager_demo",
        tags=["demo", "context_manager"],
    ) as tracker:
        print("Training model with explicit tracking...")

        # Simulate model training
        hyperparams = {"lr": 0.01, "batch_size": 32, "hidden_dim": 128}
        tracker.log_hyperparameters(hyperparams)

        best_accuracy = 0
        for epoch in range(8):
            # Simulate training metrics
            train_loss = np.random.exponential(0.3) * np.exp(-epoch * 0.15)
            val_loss = train_loss * (1 + np.random.normal(0, 0.1))
            accuracy = (
                0.6 + 0.35 * (1 - np.exp(-epoch * 0.25)) + np.random.normal(0, 0.03)
            )
            accuracy = max(0, min(1, accuracy))

            if accuracy > best_accuracy:
                best_accuracy = accuracy

            # Log metrics
            tracker.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy,
                    "best_accuracy": best_accuracy,
                }
            )

            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={accuracy:.4f}"
            )
            time.sleep(0.1)

        # Log final results
        tracker.log({"training_completed": True, "total_epochs": epoch + 1})
        print(f"Training completed. Best accuracy: {best_accuracy:.4f}")


def demo_quick_track():
    """Demo: Quick tracking for simple experiments."""
    print("\n=== Demo 3: Quick Track Pattern ===")

    tracker = quick_track("quick_demo_experiment", project="demo_project")

    print("Running quick experiment...")

    # Simulate a quick experiment
    results = []
    for trial in range(5):
        # Simulate some metric
        score = np.random.beta(2, 2) * 0.8 + 0.1  # Random score between 0.1-0.9
        results.append(score)

        tracker.log(
            {"trial": trial, "score": score, "running_average": np.mean(results)}
        )

        print(f"  Trial {trial}: score={score:.4f}")
        time.sleep(0.1)

    # Log summary
    tracker.log(
        {
            "final_average": np.mean(results),
            "best_score": max(results),
            "std_dev": np.std(results),
        }
    )

    tracker.finish()
    print(f"Quick experiment completed. Average score: {np.mean(results):.4f}")


def demo_specialized_tracking():
    """Demo: Specialized tracking functions."""
    print("\n=== Demo 4: Specialized Tracking ===")

    # Demo training tracker
    with track_training("demo_model", {"lr": 0.001, "layers": 3}) as tracker:
        print("Using specialized training tracker...")

        for epoch in range(4):
            loss = np.random.exponential(0.4) * np.exp(-epoch * 0.2)
            tracker.log({"epoch": epoch, "loss": loss})
            print(f"  Training epoch {epoch}: loss={loss:.4f}")
            time.sleep(0.1)


def demo_global_tracking():
    """Demo: Global tracking for notebook-style usage."""
    print("\n=== Demo 5: Global Tracking (Notebook Style) ===")

    # Start global tracking
    start_global_tracking("global_demo_session", project="demo_project")

    print("Step 1: Data preprocessing")
    data_size = np.random.randint(1000, 5000)
    log_global({"step": 1, "data_size": data_size, "preprocessing_time": 1.2})
    time.sleep(0.2)

    print("Step 2: Feature extraction")
    n_features = np.random.randint(50, 200)
    log_global({"step": 2, "n_features": n_features, "extraction_time": 2.1})
    time.sleep(0.2)

    print("Step 3: Model training")
    accuracy = 0.7 + np.random.uniform(0, 0.25)
    log_global({"step": 3, "final_accuracy": accuracy, "training_time": 15.3})
    time.sleep(0.2)

    print("Step 4: Evaluation")
    test_accuracy = accuracy * (0.9 + np.random.uniform(0, 0.2))
    log_global({"step": 4, "test_accuracy": test_accuracy, "eval_time": 3.1})

    # Finish global tracking
    finish_global_tracking()
    print(f"Global tracking completed. Final test accuracy: {test_accuracy:.4f}")


def main():
    """Run all demos."""
    print("ChemML Universal Tracker Demo")
    print("=" * 40)
    print("Testing different usage patterns for one-command wandb integration")

    try:
        # Run all demo patterns
        demo_decorator_pattern()
        demo_context_manager()
        demo_quick_track()
        demo_specialized_tracking()
        demo_global_tracking()

        print("\n" + "=" * 40)
        print("✅ All demos completed successfully!")
        print("Check your wandb dashboard at: https://wandb.ai/projects/demo_project")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This might be expected if wandb is not configured or available.")
        print("The tracker should still work with graceful fallbacks.")


if __name__ == "__main__":
    main()
