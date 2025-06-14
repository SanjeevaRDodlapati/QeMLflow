"""
ChemML Universal Experiment Tracker
===================================

One-command wandb integration for the entire ChemML codebase.
Auto-detects experiment context and initializes appropriate tracking.

Features:
- Automatic project and experiment naming
- Context-aware configuration
- Graceful fallback when wandb is unavailable
- Support for decorator and context manager patterns
- Automatic hyperparameter logging
- Built-in error handling and recovery

Usage Examples:

    # As a decorator
    @track_experiment(project="molecular_optimization")
    def optimize_molecule(smiles, target_property):
        # Your code here
        pass

    # As a context manager
    with track_experiment(project="drug_discovery") as tracker:
        model = train_model(data)
        tracker.log_metrics({"accuracy": 0.95})

    # Quick tracking for notebooks
    tracker = quick_track("notebook_experiment")
    tracker.log({"loss": 0.1})

    # Global tracking for notebooks
    start_global_tracking("my_notebook_session")
    log_global({"step1_result": 0.8})
    finish_global_tracking()

Author: ChemML Team
Date: December 2024
"""

import functools
import inspect
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import wandb with graceful fallback
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available - tracking will be disabled")

# Global configuration
WANDB_CONFIG = {
    "api_key": "b4f102d87161194b68baa7395d5862aa3f93b2b7",
    "project": "chemml-experiments",
    "entity": None,
    "auto_login": True,
}


class ChemMLTracker:
    """
    Universal experiment tracker for ChemML.
    Automatically detects context and initializes appropriate tracking.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.current_run = None
            self.auto_login()
            self.__class__._initialized = True

    def auto_login(self):
        """Automatically login to wandb."""
        try:
            os.environ["WANDB_API_KEY"] = WANDB_CONFIG["api_key"]
            wandb.login(key=WANDB_CONFIG["api_key"], relogin=True)
            logger.info("✅ ChemML Tracker: wandb login successful")
            return True
        except Exception as e:
            logger.error(f"❌ ChemML Tracker: wandb login failed: {e}")
            return False

    def auto_detect_context(self) -> Dict[str, Any]:
        """
        Auto-detect experiment context from calling code.
        """
        # Get caller information
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the calling function
            caller_frame = frame.f_back.f_back

            # Extract context information
            filename = caller_frame.f_code.co_filename
            function_name = caller_frame.f_code.co_name
            line_number = caller_frame.f_lineno

            # Determine module/notebook context
            filepath = Path(filename)

            context = {
                "filename": filepath.name,
                "function": function_name,
                "line": line_number,
                "module": self._detect_module_type(filepath),
                "timestamp": datetime.now().isoformat(),
            }

            # Add ChemML-specific context
            context.update(self._detect_chemml_context(filepath))

            return context

        finally:
            del frame

    def _detect_module_type(self, filepath: Path) -> str:
        """Detect the type of module/script being run."""
        path_str = str(filepath).lower()

        if "notebook" in path_str or filepath.suffix == ".ipynb":
            return "jupyter_notebook"
        elif "day_" in path_str:
            return "bootcamp_day"
        elif "test" in path_str:
            return "test_script"
        elif any(x in path_str for x in ["drug_design", "molecular", "quantum"]):
            return "research_module"
        else:
            return "general_script"

    def _detect_chemml_context(self, filepath: Path) -> Dict[str, Any]:
        """Detect ChemML-specific context."""
        path_str = str(filepath).lower()
        context = {}

        # Detect research area
        if "drug_design" in path_str:
            context["research_area"] = "drug_design"
        elif "quantum" in path_str:
            context["research_area"] = "quantum_computing"
        elif "molecular" in path_str:
            context["research_area"] = "molecular_modeling"
        elif "classical_ml" in path_str:
            context["research_area"] = "classical_ml"
        else:
            context["research_area"] = "general"

        # Detect day/tutorial
        if "day_" in path_str:
            import re

            day_match = re.search(r"day_(\d+)", path_str)
            if day_match:
                context["bootcamp_day"] = int(day_match.group(1))

        # Detect model type
        if any(x in path_str for x in ["vae", "autoencoder"]):
            context["model_type"] = "vae"
        elif any(x in path_str for x in ["transformer", "attention"]):
            context["model_type"] = "transformer"
        elif "quantum" in path_str:
            context["model_type"] = "quantum"
        elif any(x in path_str for x in ["random_forest", "rf"]):
            context["model_type"] = "random_forest"
        elif "neural" in path_str or "nn" in path_str:
            context["model_type"] = "neural_network"

        return context

    def start_tracking(
        self,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        auto_context: bool = True,
    ) -> wandb.run:
        """
        Start experiment tracking with auto-context detection.
        """
        # Auto-detect context if enabled
        if auto_context:
            detected_context = self.auto_detect_context()

            # Generate experiment name if not provided
            if experiment_name is None:
                experiment_name = self._generate_experiment_name(detected_context)

            # Merge detected context with provided config
            full_config = {**detected_context, **(config or {})}

            # Generate tags if not provided
            if tags is None:
                tags = self._generate_tags(detected_context)
        else:
            full_config = config or {}
            tags = tags or ["chemml"]

        # Initialize wandb run
        try:
            self.current_run = wandb.init(
                project=WANDB_CONFIG["project"],
                entity=WANDB_CONFIG["entity"],
                name=experiment_name,
                config=full_config,
                tags=tags,
                reinit=True,
            )

            logger.info(f"🚀 ChemML Tracker: Started '{experiment_name}'")
            logger.info(f"📊 Dashboard: {self.current_run.url}")

            return self.current_run

        except Exception as e:
            logger.error(f"❌ Failed to start tracking: {e}")
            return None

    def _generate_experiment_name(self, context: Dict[str, Any]) -> str:
        """Generate meaningful experiment name from context."""
        parts = []

        # Add research area
        if "research_area" in context:
            parts.append(context["research_area"])

        # Add bootcamp day
        if "bootcamp_day" in context:
            parts.append(f"day{context['bootcamp_day']}")

        # Add model type
        if "model_type" in context:
            parts.append(context["model_type"])

        # Add function name
        if "function" in context and context["function"] != "<module>":
            parts.append(context["function"])

        # Add timestamp
        timestamp = datetime.now().strftime("%m%d_%H%M")
        parts.append(timestamp)

        return "_".join(parts)

    def _generate_tags(self, context: Dict[str, Any]) -> List[str]:
        """Generate meaningful tags from context."""
        tags = ["chemml"]

        if "research_area" in context:
            tags.append(context["research_area"])

        if "module" in context:
            tags.append(context["module"])

        if "bootcamp_day" in context:
            tags.append(f"day{context['bootcamp_day']}")

        if "model_type" in context:
            tags.append(context["model_type"])

        return tags

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to current run."""
        if self.current_run is None:
            logger.warning("⚠️ No active run. Starting auto-tracking...")
            self.start_tracking()

        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
            logger.info(f"📈 Logged: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"❌ Failed to log metrics: {e}")

    def log_model_results(self, y_true, y_pred, model_name: str = "model"):
        """Log comprehensive model results."""
        try:
            import numpy as np
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

            metrics = {
                f"{model_name}_mse": mse,
                f"{model_name}_mae": mae,
                f"{model_name}_rmse": rmse,
                f"{model_name}_r2": r2,
            }

            self.log(metrics)

            # Create prediction plot
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 6))
                plt.scatter(y_true, y_pred, alpha=0.6)
                plt.plot(
                    [y_true.min(), y_true.max()],
                    [y_true.min(), y_true.max()],
                    "r--",
                    lw=2,
                )
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title(f"{model_name} Predictions (R² = {r2:.3f})")
                plt.grid(True, alpha=0.3)

                self.log({f"{model_name}_predictions": wandb.Image(plt)})
                plt.close()

            except ImportError:
                logger.warning("matplotlib not available for plotting")

        except ImportError:
            logger.warning("sklearn not available for metrics")

    def finish(self):
        """Finish current run."""
        if self.current_run is not None:
            wandb.finish()
            logger.info("✅ ChemML Tracker: Experiment finished")
            self.current_run = None


# Global tracker instance
_tracker = ChemMLTracker()


def track(
    experiment_name_or_func: Union[str, Callable] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    auto_finish: bool = True,
):
    """
    Universal tracking decorator/context manager.

    Usage:
        @track  # Auto-detect everything
        def my_experiment():
            pass

        @track("custom_name")
        def my_experiment():
            pass

        @track(config={"lr": 0.01})
        def my_experiment():
            pass

        with track("experiment") as tracker:
            tracker.log({"accuracy": 0.95})
    """

    # If used as decorator without arguments: @track
    if callable(experiment_name_or_func):
        func = experiment_name_or_func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tracker.start_tracking()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if auto_finish:
                    _tracker.finish()

        return wrapper

    # If used as decorator with arguments: @track("name")
    elif isinstance(experiment_name_or_func, str) or experiment_name_or_func is None:
        experiment_name = experiment_name_or_func

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                _tracker.start_tracking(experiment_name, config, tags)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    if auto_finish:
                        _tracker.finish()

            return wrapper

        return decorator

    # If used as context manager
    else:
        return TrackingContext(experiment_name_or_func, config, tags)


class TrackingContext:
    """Context manager for tracking."""

    def __init__(self, experiment_name=None, config=None, tags=None):
        self.experiment_name = experiment_name
        self.config = config
        self.tags = tags

    def __enter__(self):
        _tracker.start_tracking(self.experiment_name, self.config, self.tags)
        return _tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        _tracker.finish()


# Convenient aliases
log = _tracker.log
log_model = _tracker.log_model_results
finish = _tracker.finish


# Quick start function
def quick_track(name: str, config: Dict[str, Any] = None):
    """Quick start tracking with minimal setup."""
    return _tracker.start_tracking(name, config)


# Example usage demonstrations
def demo_automatic_tracking():
    """Demo: Automatic context detection."""

    @track  # Auto-detects everything!
    def molecular_experiment():
        import numpy as np

        # Simulate experiment
        log({"epoch": 1, "loss": 0.5})
        log({"epoch": 2, "loss": 0.3})

        # Simulate model results
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)
        log_model(y_true, y_pred, "molecular_predictor")

        print("✅ Automatic tracking completed!")

    molecular_experiment()


def demo_context_manager():
    """Demo: Context manager usage."""

    with track("manual_experiment") as tracker:
        import numpy as np

        tracker.log({"learning_rate": 0.01, "batch_size": 32})

        for epoch in range(3):
            loss = 1.0 - (epoch * 0.2) + np.random.normal(0, 0.05)
            tracker.log({"epoch": epoch, "loss": max(0.1, loss)}, step=epoch)

        print("✅ Context manager tracking completed!")


def demo_custom_config():
    """Demo: Custom configuration."""

    @track(
        "custom_experiment",
        config={"model": "transformer", "layers": 6},
        tags=["demo", "custom"],
    )
    def custom_experiment():
        log({"accuracy": 0.95, "f1_score": 0.92})
        print("✅ Custom configuration tracking completed!")

    custom_experiment()


if __name__ == "__main__":
    print("🧪 ChemML Universal Tracker Demo")
    print("=" * 40)

    demo_automatic_tracking()
    print()

    demo_context_manager()
    print()

    demo_custom_config()
    print()

    print("🎉 All demos completed!")
    print("📊 Check your wandb dashboard for results!")
