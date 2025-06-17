"""
ChemML Notebook Integration Templates
====================================

Standardized templates and utilities for integrating ChemML functionality into Jupyter notebooks.
This module provides:

1. Notebook templates with pre-configured ChemML imports
2. Cell execution helpers and error handling
3. Progress tracking integration
4. Output standardization
5. Interactive widget integration
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class NotebookTemplate:
    """Base class for ChemML notebook templates."""

    def __init__(self, name: str, description: str, level: str = "beginner"):
        self.name = name
        self.description = description
        self.level = level  # beginner, intermediate, advanced
        self.cells = []
        self.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
            "chemml": {
                "template_version": "1.0",
                "level": level,
                "created": datetime.now().isoformat(),
            },
        }

    def add_markdown_cell(self, content: str) -> "NotebookTemplate":
        """Add a markdown cell to the template."""
        cell = {"cell_type": "markdown", "metadata": {}, "source": content.split("\n")}
        self.cells.append(cell)
        return self

    def add_code_cell(
        self, content: str, outputs: Optional[List] = None
    ) -> "NotebookTemplate":
        """Add a code cell to the template."""
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs or [],
            "source": content.split("\n"),
        }
        self.cells.append(cell)
        return self

    def add_setup_cell(self) -> "NotebookTemplate":
        """Add standard ChemML setup cell."""
        setup_code = """# ChemML Setup
import sys
import warnings

warnings.filterwarnings('ignore')

# Core ChemML imports
import chemml
from chemml.models import ChemMLModel
from chemml.preprocessing import MoleculePreprocessor
from chemml.visualization import ChemMLVisualizer

# Optional integrations (with graceful fallbacks)
try:
    from chemml.integrations.experiment_tracking import setup_wandb_tracking
    HAS_TRACKING = True
except ImportError:
    HAS_TRACKING = False
    print("⚠️  Experiment tracking not available")

# Display ChemML info
print(f"🧪 ChemML {chemml.__version__} loaded successfully!")
if HAS_TRACKING:
    print("📊 Experiment tracking available")"""

        return self.add_code_cell(setup_code)

    def add_data_cell(self, dataset_name: str = "molecules") -> "NotebookTemplate":
        """Add data loading cell."""
        data_code = f"""# Load sample data
from chemml.datasets import load_{dataset_name}

# Load dataset with error handling
try:
    data = load_{dataset_name}()
    print(f"✅ Loaded {{len(data)}} samples")
except Exception as e:
    print(f"❌ Could not load data: {{e}}")
    # Fallback to demo data
    data = {{"molecules": ["CCO", "CC(C)O", "CCCCO"], "properties": [1.2, 1.5, 1.8]}}
    print("📊 Using demo data instead")

print(f"Data keys: {{list(data.keys())}}")"""

        return self.add_code_cell(data_code)

    def to_notebook(self) -> Dict[str, Any]:
        """Convert template to notebook JSON format."""
        return {
            "cells": self.cells,
            "metadata": self.metadata,
            "nbformat": 4,
            "nbformat_minor": 4,
        }

    def save(self, filepath: str) -> None:
        """Save notebook to file."""
        notebook_json = self.to_notebook()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(notebook_json, f, indent=2, ensure_ascii=False)
        print(f"📓 Notebook saved to {filepath}")


class BasicChemMLTemplate(NotebookTemplate):
    """Template for basic ChemML introduction."""

    def __init__(self):
        super().__init__(
            "Basic ChemML Introduction",
            "Introduction to ChemML basics with molecular data processing",
            "beginner",
        )
        self._build_template()

    def _build_template(self):
        """Build the basic template structure."""
        # Title
        self.add_markdown_cell(
            """# ChemML Basic Tutorial

Welcome to ChemML! This notebook will introduce you to the basics of computational chemistry and machine learning with ChemML.

## Learning Objectives
- Set up ChemML environment
- Load and explore molecular data
- Perform basic molecular preprocessing
- Create simple predictive models
- Visualize results"""
        )

        # Setup
        self.add_markdown_cell("## 1. Environment Setup")
        self.add_setup_cell()

        # Data loading
        self.add_markdown_cell("## 2. Data Loading")
        self.add_data_cell("molecules")

        # Preprocessing
        self.add_markdown_cell("## 3. Molecular Preprocessing")
        self.add_code_cell(
            """# Initialize preprocessor
preprocessor = MoleculePreprocessor()

# Process molecules with error handling
try:
    processed_data = preprocessor.fit_transform(data["molecules"])
    print(f"✅ Processed {len(processed_data)} molecules")
    print(f"Feature shape: {processed_data.shape}")
except Exception as e:
    print(f"❌ Preprocessing failed: {e}")
    print("💡 Try checking your molecule format (SMILES expected)")"""
        )

        # Model training
        self.add_markdown_cell("## 4. Model Training")
        self.add_code_cell(
            """# Initialize model
model = ChemMLModel(model_type="random_forest")

# Train model with error handling
try:
    model.fit(processed_data, data["properties"])
    print("✅ Model trained successfully")

    # Make predictions
    predictions = model.predict(processed_data)
    print(f"Predictions: {predictions[:5]}")  # Show first 5
except Exception as e:
    print(f"❌ Model training failed: {e}")
    print("💡 Check that data shapes match")"""
        )

        # Visualization
        self.add_markdown_cell("## 5. Visualization")
        self.add_code_cell(
            """# Create visualizations
visualizer = ChemMLVisualizer()

# Plot results with error handling
try:
    visualizer.plot_predictions(data["properties"], predictions)
    print("✅ Visualization created")
except Exception as e:
    print(f"❌ Visualization failed: {e}")
    print("💡 Using matplotlib backend")

    # Fallback to simple matplotlib
    import matplotlib.pyplot as plt
    plt.scatter(data["properties"], predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Model Predictions")
    plt.show()"""
        )

        # Summary
        self.add_markdown_cell(
            """## Summary

Congratulations! You've completed the basic ChemML tutorial. You learned how to:

- ✅ Set up the ChemML environment
- ✅ Load molecular datasets
- ✅ Preprocess molecular data
- ✅ Train predictive models
- ✅ Visualize results

### Next Steps
- Try the intermediate tutorials
- Experiment with different molecular datasets
- Explore advanced model types
- Set up experiment tracking"""
        )


class IntermediateChemMLTemplate(NotebookTemplate):
    """Template for intermediate ChemML features."""

    def __init__(self):
        super().__init__(
            "Intermediate ChemML Features",
            "Advanced molecular modeling and experiment tracking",
            "intermediate",
        )
        self._build_template()

    def _build_template(self):
        """Build intermediate template."""
        self.add_markdown_cell(
            """# ChemML Intermediate Tutorial

This tutorial covers intermediate ChemML features including:
- Advanced molecular descriptors
- Deep learning models
- Experiment tracking with Weights & Biases
- Cross-validation and model selection
- Feature importance analysis"""
        )

        self.add_setup_cell()

        # Experiment tracking setup
        self.add_markdown_cell("## Experiment Tracking Setup")
        self.add_code_cell(
            """# Setup experiment tracking
if HAS_TRACKING:
    experiment = setup_wandb_tracking(
        "intermediate_tutorial",
        project="chemml-tutorials",
        tags=["intermediate", "molecular-modeling"]
    )
    print("📊 Experiment tracking initialized")
else:
    print("📝 Running without experiment tracking")"""
        )

        # Advanced preprocessing
        self.add_markdown_cell("## Advanced Molecular Descriptors")
        self.add_code_cell(
            """# Use advanced molecular descriptors
from chemml.preprocessing import AdvancedMoleculePreprocessor

advanced_preprocessor = AdvancedMoleculePreprocessor(
    descriptor_types=["morgan", "rdkit", "3d"],
    include_3d=True
)

try:
    advanced_features = advanced_preprocessor.fit_transform(data["molecules"])
    print(f"✅ Generated {advanced_features.shape[1]} advanced features")

    # Log to experiment tracker
    if HAS_TRACKING:
        experiment.log({"n_features": advanced_features.shape[1]})

except Exception as e:
    print(f"❌ Advanced preprocessing failed: {e}")
    print("💡 Falling back to basic features")
    advanced_features = processed_data"""
        )


def create_notebook_templates():
    """Create all standard notebook templates."""
    templates_dir = Path("notebooks/templates")
    templates_dir.mkdir(parents=True, exist_ok=True)

    # Create basic template
    basic_template = BasicChemMLTemplate()
    basic_template.save(templates_dir / "chemml_basic_tutorial.ipynb")

    # Create intermediate template
    intermediate_template = IntermediateChemMLTemplate()
    intermediate_template.save(templates_dir / "chemml_intermediate_tutorial.ipynb")

    print(f"✅ Created notebook templates in {templates_dir}")


def integrate_existing_notebooks():
    """Integrate existing notebooks with ChemML standards."""
    notebooks_dir = Path("notebooks")

    for notebook_path in notebooks_dir.rglob("*.ipynb"):
        if "templates" in str(notebook_path):
            continue  # Skip template notebooks

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            # Check if notebook needs ChemML integration
            needs_integration = _check_notebook_needs_integration(notebook)

            if needs_integration:
                print(f"📓 Integrating {notebook_path}")
                _integrate_notebook(notebook, notebook_path)
            else:
                print(f"✅ {notebook_path} already integrated")

        except Exception as e:
            print(f"❌ Failed to process {notebook_path}: {e}")


def _check_notebook_needs_integration(notebook: Dict) -> bool:
    """Check if notebook needs ChemML integration."""
    # Check metadata for ChemML marker
    metadata = notebook.get("metadata", {})
    if "chemml" in metadata:
        return False

    # Check for ChemML imports in code cells
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "import chemml" in source:
                return False

    return True


def _integrate_notebook(notebook: Dict, notebook_path: Path) -> None:
    """Integrate notebook with ChemML standards."""
    # Add ChemML metadata
    if "metadata" not in notebook:
        notebook["metadata"] = {}

    notebook["metadata"]["chemml"] = {
        "integrated": True,
        "integration_date": datetime.now().isoformat(),
        "version": "1.0",
    }

    # Add setup cell at the beginning if not present
    cells = notebook.get("cells", [])
    has_setup = any(
        "import chemml" in "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )

    if not has_setup:
        setup_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ChemML Integration Setup",
                "import chemml",
                "print(f'🧪 ChemML {chemml.__version__} loaded for this notebook')",
            ],
        }
        cells.insert(0, setup_cell)
        notebook["cells"] = cells

    # Save integrated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Create templates and integrate existing notebooks
    create_notebook_templates()
    integrate_existing_notebooks()
