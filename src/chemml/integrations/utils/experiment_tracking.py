from typing import Any

"""
ChemML Experiment Tracking Integration
=====================================

Weights & Biases integration for experiment tracking.
"""

import warnings

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def setup_wandb_tracking(
    experiment_name: str, config: dict = None, project: str = "chemml-experiments"
) -> Any:
    """Setup wandb experiment tracking."""
    if not HAS_WANDB:
        warnings.warn("wandb not available. Install with: pip install wandb")
        return None
    try:
        run = wandb.init(
            project=project, name=experiment_name, config=config or {}, tags=["chemml"]
        )
        print(f"✅ Wandb tracking started: {run.url}")
        return run
    except Exception as e:
        print(f"⚠️ Wandb setup failed: {e}")
        return None


def log_metrics(metrics: dict, step: int = None) -> Any:
    """Log metrics to wandb."""
    if HAS_WANDB and wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_run() -> Any:
    """Finish wandb run."""
    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


#__all__ = ["setup_wandb_tracking", "log_metrics", "finish_run"]
