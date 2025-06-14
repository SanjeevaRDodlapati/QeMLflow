"""
ChemML Universal Experiment Tracker - Enhanced Version
=====================================================

Complete rewrite with robust error handling and comprehensive functionality.

Features:
- Automatic project and experiment naming
- Context-aware configuration
- Graceful fallback when wandb is unavailable
- Support for decorator and context manager patterns
- Automatic hyperparameter logging
- Built-in error handling and recovery
- Global tracking for notebooks
- Automatic context detection

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
from typing import Any, Callable, Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import wandb with graceful fallback
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    logger.warning("wandb not available - tracking will be disabled")

# Global configuration
WANDB_CONFIG = {
    "api_key": "b4f102d87161194b68baa7395d5862aa3f93b2b7",
    "project": "chemml-experiments",
    "entity": None,
    "auto_login": True,
}


class UniversalTracker:
    """Universal experiment tracker with wandb integration and graceful fallbacks."""

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        auto_login: bool = True,
        offline: bool = False,
        silent: bool = False,
    ):
        """
        Initialize the universal tracker.

        Args:
            project: wandb project name (auto-detected if None)
            entity: wandb entity/team name
            name: experiment name (auto-generated if None)
            config: hyperparameters and configuration
            tags: list of tags for the experiment
            notes: experiment description
            auto_login: automatically attempt wandb login
            offline: run in offline mode
            silent: suppress wandb output
        """
        self.project = project or self._detect_project()
        self.entity = entity
        self.name = name or self._generate_name()
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.auto_login = auto_login
        self.offline = offline
        self.silent = silent

        self.run = None
        self.wandb_available = WANDB_AVAILABLE
        self.active = False

        # Add context-aware tags
        self.tags.extend(self._detect_context_tags())

        if self.wandb_available and self.auto_login:
            self._ensure_login()

    def _detect_project(self) -> str:
        """Auto-detect project name from context."""
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                filename = caller_frame.f_globals.get("__file__", "unknown")

                if "notebook" in filename.lower() or "ipynb" in filename.lower():
                    return "chemml_notebooks"
                elif "drug_design" in filename:
                    return "chemml_drug_design"
                elif "quantum" in filename:
                    return "chemml_quantum_ml"
                elif "model" in filename:
                    return "chemml_models"
                elif "data" in filename:
                    return "chemml_data_processing"
                else:
                    return "chemml_experiments"
        except Exception:
            pass
        return "chemml_default"

    def _generate_name(self) -> str:
        """Generate experiment name from context."""
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                function_name = caller_frame.f_code.co_name
                filename = os.path.basename(
                    caller_frame.f_globals.get("__file__", "unknown")
                )

                if function_name != "<module>":
                    return f"{filename}_{function_name}"
                else:
                    return f"{filename}_experiment"
        except Exception:
            pass
        return "auto_generated_experiment"

    def _detect_context_tags(self) -> list:
        """Detect context-specific tags."""
        tags = []
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                filename = caller_frame.f_globals.get("__file__", "")

                if "notebook" in filename.lower() or "ipynb" in filename.lower():
                    tags.append("jupyter_notebook")
                if "test" in filename.lower():
                    tags.append("testing")
                if "tutorial" in filename.lower():
                    tags.append("tutorial")
                if "example" in filename.lower():
                    tags.append("example")

                # Detect ML framework usage
                globals_dict = caller_frame.f_globals
                if "torch" in globals_dict or "pytorch" in str(globals_dict):
                    tags.append("pytorch")
                if "tensorflow" in str(globals_dict) or "tf" in globals_dict:
                    tags.append("tensorflow")
                if "sklearn" in str(globals_dict):
                    tags.append("scikit-learn")
                if "rdkit" in str(globals_dict):
                    tags.append("rdkit")
                if "deepchem" in str(globals_dict):
                    tags.append("deepchem")
        except Exception:
            pass

        return tags

    def _ensure_login(self):
        """Ensure wandb is logged in."""
        if not self.wandb_available:
            return

        try:
            if not wandb.api.api_key:
                if not self.silent:
                    print("Attempting wandb login...")
                wandb.login(key=WANDB_CONFIG.get("api_key"))
        except Exception as e:
            if not self.silent:
                print(f"wandb login failed: {e}")

    def start(self) -> "UniversalTracker":
        """Start the tracking session."""
        if not self.wandb_available:
            if not self.silent:
                print("wandb not available, tracking disabled")
            return self

        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                mode="offline" if self.offline else "online",
                reinit=True,
            )
            self.active = True

            if not self.silent:
                print(f"✓ Started tracking: {self.project}/{self.name}")

        except Exception as e:
            if not self.silent:
                print(f"Failed to start wandb tracking: {e}")
            self.active = False

        return self

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.active and self.run:
            try:
                self.run.log(metrics, step=step)
            except Exception as e:
                if not self.silent:
                    print(f"Failed to log metrics: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Alias for log method."""
        self.log(metrics, step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.active and self.run:
            try:
                wandb.config.update(params)
            except Exception as e:
                if not self.silent:
                    print(f"Failed to log hyperparameters: {e}")

    def log_artifact(
        self, file_path: str, name: Optional[str] = None, type: str = "dataset"
    ):
        """Log an artifact to wandb."""
        if self.active and self.run:
            try:
                artifact = wandb.Artifact(
                    name or os.path.basename(file_path), type=type
                )
                artifact.add_file(file_path)
                self.run.log_artifact(artifact)
            except Exception as e:
                if not self.silent:
                    print(f"Failed to log artifact: {e}")

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100):
        """Watch a model for gradient and parameter tracking."""
        if self.active and self.run:
            try:
                wandb.watch(model, log=log, log_freq=log_freq)
            except Exception as e:
                if not self.silent:
                    print(f"Failed to watch model: {e}")

    def finish(self):
        """Finish the tracking session."""
        if self.active and self.run:
            try:
                self.run.finish()
                if not self.silent:
                    print("✓ Finished tracking session")
            except Exception as e:
                if not self.silent:
                    print(f"Error finishing wandb run: {e}")
            finally:
                self.active = False
                self.run = None

    def __enter__(self):
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        if exc_type and not self.silent:
            print(f"Exception in tracked context: {exc_val}")


def track_experiment(
    project: Optional[str] = None,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    auto_login: bool = True,
    offline: bool = False,
    silent: bool = False,
):
    """
    Universal experiment tracking decorator/context manager.

    Can be used as:
    1. Decorator: @track_experiment(project="my_project")
    2. Context manager: with track_experiment(project="my_project") as tracker:
    3. Direct call: tracker = track_experiment(project="my_project")

    Args:
        project: wandb project name
        entity: wandb entity/team name
        name: experiment name
        config: hyperparameters and configuration
        tags: list of tags for the experiment
        notes: experiment description
        auto_login: automatically attempt wandb login
        offline: run in offline mode
        silent: suppress wandb output

    Returns:
        Decorator function or UniversalTracker instance
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config from function arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            func_config = dict(bound_args.arguments)
            if config:
                func_config.update(config)

            tracker = UniversalTracker(
                project=project,
                entity=entity,
                name=name or f"{func.__name__}_experiment",
                config=func_config,
                tags=tags,
                notes=notes or f"Experiment for function: {func.__name__}",
                auto_login=auto_login,
                offline=offline,
                silent=silent,
            )

            with tracker:
                try:
                    result = func(*args, **kwargs)

                    # Log result if it's a dict-like metric
                    if isinstance(result, dict):
                        tracker.log(result)

                    return result
                except Exception as e:
                    tracker.log({"error": str(e), "error_type": type(e).__name__})
                    raise

        return wrapper

    # If no arguments provided, return the tracker for context manager usage
    tracker = UniversalTracker(
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        auto_login=auto_login,
        offline=offline,
        silent=silent,
    )

    # Check if being called as decorator
    if any(
        arg is not None for arg in [project, entity, name, config, tags, notes]
    ) or any([not auto_login, offline, silent]):
        return tracker

    # Default case - return decorator
    return decorator


def quick_track(name: str, project: Optional[str] = None, **kwargs) -> UniversalTracker:
    """
    Quick one-liner for starting experiment tracking.

    Args:
        name: experiment name
        project: project name (auto-detected if None)
        **kwargs: additional arguments for UniversalTracker

    Returns:
        Started UniversalTracker instance

    Example:
        tracker = quick_track("my_experiment")
        tracker.log({"accuracy": 0.95})
        tracker.finish()
    """
    tracker = UniversalTracker(project=project, name=name, **kwargs)
    return tracker.start()


# Convenience functions for common patterns
def track_training(
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    project: str = "chemml_training",
) -> UniversalTracker:
    """Quick setup for model training experiments."""
    return track_experiment(
        project=project,
        name=f"{model_name}_training",
        config=hyperparameters,
        tags=["training", "model"],
    )


def track_evaluation(
    model_name: str, dataset_name: str, project: str = "chemml_evaluation"
) -> UniversalTracker:
    """Quick setup for model evaluation experiments."""
    return track_experiment(
        project=project,
        name=f"{model_name}_eval_{dataset_name}",
        tags=["evaluation", "testing"],
    )


def track_optimization(
    optimization_type: str, target: str, project: str = "chemml_optimization"
) -> UniversalTracker:
    """Quick setup for optimization experiments."""
    return track_experiment(
        project=project,
        name=f"{optimization_type}_{target}",
        tags=["optimization", optimization_type],
    )


# Global tracking instance for notebook usage
_global_tracker = None


def start_global_tracking(
    project: Optional[str] = None, name: Optional[str] = None, **kwargs
) -> UniversalTracker:
    """Start a global tracking session for notebooks."""
    global _global_tracker
    if _global_tracker and _global_tracker.active:
        _global_tracker.finish()

    _global_tracker = UniversalTracker(
        project=project or "chemml_notebook", name=name or "notebook_session", **kwargs
    )
    return _global_tracker.start()


def log_global(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log to the global tracker."""
    if _global_tracker and _global_tracker.active:
        _global_tracker.log(metrics, step)
    else:
        print("No active global tracker. Use start_global_tracking() first.")


def finish_global_tracking():
    """Finish the global tracking session."""
    global _global_tracker
    if _global_tracker and _global_tracker.active:
        _global_tracker.finish()
        _global_tracker = None


# Backward compatibility aliases
def track(experiment_name: str, **kwargs) -> UniversalTracker:
    """Backward compatibility function."""
    return quick_track(experiment_name, **kwargs)


# Export main interfaces
__all__ = [
    "UniversalTracker",
    "track_experiment",
    "quick_track",
    "track_training",
    "track_evaluation",
    "track_optimization",
    "start_global_tracking",
    "log_global",
    "finish_global_tracking",
    "track",  # backward compatibility
]


if __name__ == "__main__":
    # Demo/test the tracker
    print("Testing Universal Tracker...")

    # Test 1: Context manager
    print("\n1. Testing context manager:")
    with track_experiment(project="test_project", name="context_test") as tracker:
        tracker.log({"test_metric": 42})
        print("Logged test metric")

    # Test 2: Decorator
    print("\n2. Testing decorator:")

    @track_experiment(project="test_project")
    def test_function(x, y):
        return {"result": x + y, "product": x * y}

    result = test_function(3, 4)
    print(f"Function result: {result}")

    # Test 3: Quick track
    print("\n3. Testing quick track:")
    tracker = quick_track("quick_test", project="test_project")
    tracker.log({"quick_metric": 123})
    tracker.finish()

    print("\n✓ All tests completed!")
