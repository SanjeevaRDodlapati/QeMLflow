#!/usr/bin/env python3
"""
QeMLflow Wandb Integration Setup Script
=====================================

This script integrates Weights & Biases experiment tracking throughout the QeMLflow codebase.
It adds wandb tracking to existing notebooks and Python files with minimal code changes.

Usage:
    python setup_wandb_integration.py

Author: QeMLflow Team
Date: June 14, 2025
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import wandb

# Your wandb API key
WANDB_API_KEY = "b4f102d87161194b68baa7395d5862aa3f93b2b7"
PROJECT_NAME = "qemlflow-experiments"


def setup_wandb():
    """Initialize wandb with API key."""
    try:
        # Set environment variable
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY

        # Login to wandb
        wandb.login(key=WANDB_API_KEY, relogin=True)

        print("✅ Successfully logged into Weights & Biases!")
        return True
    except Exception as e:
        print(f"❌ Failed to setup wandb: {e}")
        return False


def create_wandb_config_template():
    """Create a template for wandb configuration."""
    config_template = {
        "project_name": PROJECT_NAME,
        "api_key": WANDB_API_KEY,
        "default_tags": ["qemlflow"],
        "auto_login": True,
        "log_code": True,
        "log_artifacts": True,
        "save_checkpoints": True,
    }

    # Save to config file
    config_path = Path("src/qemlflow_common/wandb_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config_template, f, indent=2)

    print(f"📄 Created wandb config template: {config_path}")
    return config_path


def add_wandb_to_notebook(notebook_path):
    """Add wandb integration to a Jupyter notebook."""
    try:
        notebook_path = Path(notebook_path)

        if not notebook_path.exists():
            print(f"⚠️ Notebook not found: {notebook_path}")
            return False

        # Read notebook content
        with open(notebook_path, "r") as f:
            content = f.read()

        # Check if wandb is already integrated
        if "wandb.init" in content:
            print(f"✅ Wandb already integrated in: {notebook_path.name}")
            return True

        # Add wandb setup cell (this would need more sophisticated notebook parsing)
        print(f"📝 Manual integration needed for: {notebook_path.name}")
        print(f"   Add the wandb setup code from the tracking module")

        return True

    except Exception as e:
        print(f"❌ Error processing notebook {notebook_path}: {e}")
        return False


def add_wandb_to_python_file(python_file):
    """Add wandb integration to a Python file."""
    try:
        python_file = Path(python_file)

        if not python_file.exists():
            print(f"⚠️ Python file not found: {python_file}")
            return False

        # Read file content
        with open(python_file, "r") as f:
            content = f.read()

        # Check if wandb is already imported
        if "import wandb" in content:
            print(f"✅ Wandb already imported in: {python_file.name}")
            return True

        # Add wandb import at the top (simple approach)
        lines = content.split("\n")

        # Find where to insert wandb import
        import_line = -1
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_line = i

        if import_line >= 0:
            # Insert wandb import after existing imports
            lines.insert(import_line + 1, "import wandb")

            # Add wandb setup function
            setup_code = """
# Wandb experiment tracking setup
def setup_wandb_tracking(experiment_name, config=None):
    \"\"\"Setup wandb experiment tracking.\"\"\"
    try:
        wandb.login(key="b4f102d87161194b68baa7395d5862aa3f93b2b7", relogin=True)
        run = wandb.init(
            project="qemlflow-experiments",
            name=experiment_name,
            config=config or {},
            tags=["qemlflow"]
        )
        print(f"✅ Wandb tracking started: {run.url}")
        return run
    except Exception as e:
        print(f"⚠️ Wandb setup failed: {e}")
        return None
"""

            # Find a good place to insert the function
            lines.insert(import_line + 3, setup_code)

            # Write back to file
            new_content = "\n".join(lines)

            # Create backup
            backup_path = python_file.with_suffix(".py.backup")
            with open(backup_path, "w") as f:
                f.write(content)

            with open(python_file, "w") as f:
                f.write(new_content)

            print(f"✅ Added wandb integration to: {python_file.name}")
            print(f"📄 Backup created: {backup_path.name}")

            return True

    except Exception as e:
        print(f"❌ Error processing Python file {python_file}: {e}")
        return False


def integrate_wandb_in_codebase():
    """Integrate wandb throughout the QeMLflow codebase."""

    print("🚀 Starting QeMLflow-WandB Integration...")
    print("=" * 50)

    # Setup wandb
    if not setup_wandb():
        return False

    # Create config template
    config_path = create_wandb_config_template()

    # Find Python files to integrate
    python_files = []

    # Key directories to check
    directories = [
        "src/",
        "notebooks/quickstart_bootcamp/days/",
        "tools/development/",
    ]

    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            python_files.extend(list(dir_path.rglob("*.py")))

    print(f"\n📂 Found {len(python_files)} Python files to check")

    # Process key files
    key_files = [
        "src/drug_design/admet_prediction.py",
        "src/models/classical_ml/regression_models.py",
        "src/models/quantum_ml/quantum_circuits.py",
        "quick_access_demo.py",
    ]

    print("\n🔧 Integrating wandb in key files...")
    for file_path in key_files:
        if Path(file_path).exists():
            add_wandb_to_python_file(file_path)

    # Find notebooks
    notebooks = (
        list(Path("notebooks").rglob("*.ipynb")) if Path("notebooks").exists() else []
    )

    print(f"\n📓 Found {len(notebooks)} Jupyter notebooks")

    # Process key notebooks
    key_notebooks = [
        "notebooks/tutorials/03_deepchem_drug_discovery.ipynb",
        "notebooks/quickstart_bootcamp/days/day_02/day_02_deep_learning_molecules_project.ipynb",
        "notebooks/quickstart_bootcamp/days/day_06/day_06_quantum_computing_project.ipynb",
        "notebooks/quickstart_bootcamp/days/day_07/day_07_integration_project.ipynb",
    ]

    print("\n📝 Checking notebooks for wandb integration...")
    for notebook_path in key_notebooks:
        if Path(notebook_path).exists():
            add_wandb_to_notebook(notebook_path)

    # Create example usage script
    create_example_script()

    print("\n" + "=" * 50)
    print("🎉 QeMLflow-WandB Integration Complete!")
    print(f"📊 Project: {PROJECT_NAME}")
    print(f"🔗 Dashboard: https://wandb.ai/projects/{PROJECT_NAME}")
    print("\n💡 Usage Tips:")
    print(
        "1. Import the tracking module: from src.qemlflow_common.wandb_integration import *"
    )
    print("2. Start experiment: run = start_experiment('my_experiment', config)")
    print("3. Log metrics: log_metrics({'accuracy': 0.95})")
    print("4. Finish: finish_experiment()")


def create_example_script():
    """Create an example script showing wandb usage."""

    example_code = '''#!/usr/bin/env python3
"""
Example: Using WandB in QeMLflow Experiments
==========================================

This script demonstrates how to use Weights & Biases experiment tracking
in your QeMLflow experiments.
"""

import sys
import os
sys.path.append('src')

from qemlflow_common.wandb_integration import *
import numpy as np

def run_example_experiment():
    """Run an example experiment with wandb tracking."""

    print("🧪 Running example QeMLflow experiment with wandb...")

    # Configuration for the experiment
    config = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "dataset": "molecular_properties",
        "task": "regression",
        "features": "molecular_descriptors"
    }

    # Start experiment
    run = start_experiment(
        experiment_name="qemlflow_example_experiment",
        config=config,
        tags=["example", "tutorial", "molecular_ml"]
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
            "learning_rate": 0.001 * (0.95 ** epoch)
        }

        log_metrics(metrics, step=epoch)
        print(f"  Epoch {epoch}: train_loss={metrics['train_loss']:.3f}, val_loss={metrics['val_loss']:.3f}")

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
        "drug_like_percentage": 78.5
    }

    log_metrics(molecular_summary)

    print(f"✅ Experiment completed!")
    print(f"🔗 View results: {run.url}")

    # Finish experiment
    finish_experiment()

if __name__ == "__main__":
    run_example_experiment()
'''

    example_path = Path("examples/wandb_example.py")
    example_path.parent.mkdir(parents=True, exist_ok=True)

    with open(example_path, "w") as f:
        f.write(example_code)

    print(f"📝 Created example script: {example_path}")


if __name__ == "__main__":
    integrate_wandb_in_codebase()
