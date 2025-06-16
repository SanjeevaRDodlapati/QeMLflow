"""
ChemML Standardized Import Patterns
==================================

Provides standardized import utilities and patterns for consistent ChemML usage.
This module helps ensure consistent imports across examples, tutorials, and user code.
"""

import warnings
from importlib import import_module
from typing import Any, Dict, List, Optional, Union


class ImportManager:
    """Manages standardized imports with graceful fallbacks."""

    def __init__(self) -> None:
        self.import_registry: Dict[str, Dict[str, Any]] = {}
        self.failed_imports: List[str] = []
        self.import_warnings: List[str] = []

    def register_import(
        self,
        module_name: str,
        import_path: str,
        fallback: Optional[Any] = None,
        required: bool = False,
        warning_message: Optional[str] = None,
    ) -> None:
        """Register an import with optional fallback."""
        self.import_registry[module_name] = {
            "import_path": import_path,
            "fallback": fallback,
            "required": required,
            "warning_message": warning_message,
            "imported": None,
            "success": False,
        }

    def import_module_safe(self, module_name: str) -> Any:
        """Safely import a registered module."""
        if module_name not in self.import_registry:
            raise ValueError(f"Module {module_name} not registered")

        entry = self.import_registry[module_name]

        # Return cached import if available
        if entry["success"] and entry["imported"] is not None:
            return entry["imported"]

        try:
            # Attempt import
            module_parts = entry["import_path"].split(".")
            if len(module_parts) == 1:
                # Simple module import
                imported = import_module(module_parts[0])
            else:
                # Import specific item from module
                module_path = ".".join(module_parts[:-1])
                item_name = module_parts[-1]
                module = import_module(module_path)
                imported = getattr(module, item_name)

            entry["imported"] = imported
            entry["success"] = True
            return imported

        except ImportError as e:
            self.failed_imports.append(module_name)

            # Handle required imports
            if entry["required"]:
                raise ImportError(
                    f"Required module {module_name} could not be imported: {e}"
                )

            # Issue warning if specified
            if entry["warning_message"]:
                warning_msg = entry["warning_message"]
                warnings.warn(warning_msg, UserWarning)
                self.import_warnings.append(warning_msg)

            # Return fallback if available
            if entry["fallback"] is not None:
                entry["imported"] = entry["fallback"]
                return entry["fallback"]

            # Return None if no fallback
            return None

    def get_import_status(self) -> Dict[str, Any]:
        """Get status of all registered imports."""
        status = {
            "successful": [],
            "failed": [],
            "with_fallbacks": [],
            "warnings": self.import_warnings,
        }

        for name, entry in self.import_registry.items():
            if entry["success"]:
                status["successful"].append(name)
            elif name in self.failed_imports:
                if entry["fallback"] is not None:
                    status["with_fallbacks"].append(name)
                else:
                    status["failed"].append(name)

        return status


# Global import manager instance
#_import_manager = ImportManager()


def setup_chemml_imports() -> ImportManager:
    """Setup standard ChemML imports with fallbacks."""

    # Core ChemML imports (required)
    _import_manager.register_import("chemml", "chemml", required=True)

    # Preprocessing
    _import_manager.register_import(
        "MoleculePreprocessor",
        "chemml.preprocessing.MoleculePreprocessor",
        fallback=None,
        warning_message="⚠️  MoleculePreprocessor not available. Some preprocessing features disabled.",
    )

    # Models
    _import_manager.register_import(
        "ChemMLModel",
        "chemml.models.ChemMLModel",
        fallback=None,
        warning_message="⚠️  ChemMLModel not available. Using basic model fallbacks.",
    )

    # Visualization
    _import_manager.register_import(
        "ChemMLVisualizer",
        "chemml.visualization.ChemMLVisualizer",
        fallback=None,
        warning_message="⚠️  ChemMLVisualizer not available. Basic matplotlib fallbacks will be used.",
    )

    # Experiment tracking
    _import_manager.register_import(
        "setup_wandb_tracking",
        "chemml.integrations.experiment_tracking.setup_wandb_tracking",
        fallback=lambda *args, **kwargs: print(
            "📝 Mock experiment tracking (tracking disabled)"
        ),
        warning_message="⚠️  Experiment tracking not available",
    )

    # Optional scientific libraries
    _import_manager.register_import("numpy", "numpy", required=True)

    _import_manager.register_import(
        "pandas",
        "pandas",
        fallback=None,
        warning_message="⚠️  Pandas not available. Some data manipulation features disabled.",
    )

    _import_manager.register_import(
        "matplotlib",
        "matplotlib.pyplot",
        fallback=None,
        warning_message="⚠️  Matplotlib not available. Visualization disabled.",
    )

    _import_manager.register_import(
        "sklearn",
        "sklearn",
        fallback=None,
        warning_message="⚠️  Scikit-learn not available. Some ML features disabled.",
    )

    # RDKit (common in cheminformatics)
    _import_manager.register_import(
        "rdkit",
        "rdkit.Chem",
        fallback=None,
        warning_message="⚠️  RDKit not available. Molecular structure features disabled.",
    )

    # DeepChem
    _import_manager.register_import(
        "deepchem",
        "deepchem",
        fallback=None,
        warning_message="⚠️  DeepChem not available. Deep learning molecular features disabled.",
    )

    return _import_manager


def get_standard_imports() -> Dict[str, Any]:
    """Get dictionary of standard imports for use in notebooks/scripts."""
    manager = setup_chemml_imports()

    imports = {}
    for name in manager.import_registry.keys():
        imports[name] = manager.import_module_safe(name)

    # Filter out None values
    return {k: v for k, v in imports.items() if v is not None}


def create_import_cell_code() -> str:
    """Generate code for a standard import cell."""
    return """# ChemML Standard Imports
import warnings

warnings.filterwarnings('ignore')

# Get standardized imports
from chemml.utils.imports import get_standard_imports

imports = get_standard_imports()

# Extract common imports
np = imports.get('numpy')
pd = imports.get('pandas')
plt = imports.get('matplotlib')
chemml = imports.get('chemml')

# ChemML components
MoleculePreprocessor = imports.get('MoleculePreprocessor')
ChemMLModel = imports.get('ChemMLModel')
ChemMLVisualizer = imports.get('ChemMLVisualizer')
setup_wandb_tracking = imports.get('setup_wandb_tracking')

# Optional scientific libraries
sklearn = imports.get('sklearn')
rdkit = imports.get('rdkit')
deepchem = imports.get('deepchem')

# Display import status
print(f"🧪 ChemML {chemml.__version__ if chemml else 'N/A'} imports ready")
print(f"📊 Available: {len([v for v in imports.values() if v is not None])}/{len(imports)} modules")"""


def create_fallback_helpers() -> Any:
    """Create fallback helper functions for missing imports."""

    def mock_preprocessor() -> List[Any]:
        """Mock preprocessor for when real one is unavailable."""

        class MockPreprocessor:
            def fit_transform(self, data) -> List[Any]:
                print(
                    "⚠️  Using mock preprocessor - install ChemML preprocessing for real functionality"
                )
                # Return dummy data
                if hasattr(data, "__len__"):
                    return [[i] * 10 for i in range(len(data))]
                return [[0] * 10]

        return MockPreprocessor()

    def mock_model() -> List[Any]:
        """Mock model for when real one is unavailable."""

        class MockModel:
            def fit(self, X, y) -> None:
                print(
                    "⚠️  Using mock model - install ChemML models for real functionality"
                )

            def predict(self, X) -> List[Any]:
                print("⚠️  Mock prediction")
                if hasattr(X, "__len__"):
                    return [0.5] * len(X)
                return [0.5]

        return MockModel()

    def mock_visualizer() -> Any:
        """Mock visualizer for when real one is unavailable."""

        class MockVisualizer:
            def plot_predictions(self, true_vals, predictions) -> None:
                print("⚠️  Using mock visualizer - install matplotlib for real plots")
                print(
                    f"True values: {true_vals[:5] if hasattr(true_vals, '__getitem__') else 'N/A'}"
                )
                print(
                    f"Predictions: {predictions[:5] if hasattr(predictions, '__getitem__') else 'N/A'}"
                )

        return MockVisualizer()

    return {
        "mock_preprocessor": mock_preprocessor,
        "mock_model": mock_model,
        "mock_visualizer": mock_visualizer,
    }


class ImportContext:
    """Context manager for temporary import overrides."""

    def __init__(self, **overrides):
        self.overrides = overrides
        self.original_imports = {}

    def __enter__(self) -> Any:
        # Store original imports and apply overrides
        for name, override in self.overrides.items():
            if name in _import_manager.import_registry:
                entry = _import_manager.import_registry[name]
                self.original_imports[name] = entry["imported"]
                entry["imported"] = override
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore original imports
        for name, original in self.original_imports.items():
            if name in _import_manager.import_registry:
                _import_manager.import_registry[name]["imported"] = original


def print_import_status() -> None:
    """Print status of all imports."""
    status = _import_manager.get_import_status()

    print("ChemML Import Status")
    print("=" * 40)

    if status["successful"]:
        print(
            f"✅ Successful ({len(status['successful'])}): {', '.join(status['successful'])}"
        )

    if status["with_fallbacks"]:
        print(
            f"⚠️  With Fallbacks ({len(status['with_fallbacks'])}): {', '.join(status['with_fallbacks'])}"
        )

    if status["failed"]:
        print(f"❌ Failed ({len(status['failed'])}): {', '.join(status['failed'])}")

    if status["warnings"]:
        print("\nWarnings:")
        for warning in status["warnings"]:
            print(f"  • {warning}")


if __name__ == "__main__":
    # Demo the import system
    imports = get_standard_imports()
    print_import_status()

    print("\n" + "=" * 40)
    print("Available imports:")
    for name, module in imports.items():
        print(f"  {name}: {type(module)}")

    print("\nGenerated import cell code:")
    print("-" * 40)
    print(create_import_cell_code())
