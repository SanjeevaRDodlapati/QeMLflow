"""
ChemML Import Optimization Tool
Identifies and fixes import performance bottlenecks.
"""

import argparse
import ast
import importlib
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


class ImportProfiler:
    """Profiles and optimizes import patterns."""

    def __init__(self):
        self.slow_imports = {}
        self.optimization_suggestions = []

    def profile_module_imports(self, module_path: str) -> Dict[str, float]:
        """Profile import times for a specific module."""
        results = {}

        try:
            with open(module_path, "r") as f:
                content = f.read()

            tree = ast.parse(content)

            # Extract all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            full_name = f"{node.module}.{alias.name}"
                            imports.append(full_name)

            # Profile each import
            for imp in imports:
                try:
                    start_time = time.time()
                    importlib.import_module(imp.split(".")[0])
                    end_time = time.time()
                    import_time = end_time - start_time

                    if import_time > 0.1:  # Slow import threshold
                        results[imp] = import_time
                        self.slow_imports[imp] = import_time

                except ImportError:
                    continue

        except Exception as e:
            print(f"Error profiling {module_path}: {e}")

        return results

    def suggest_optimizations(
        self, module_path: str, slow_imports: Dict[str, float]
    ) -> List[str]:
        """Generate optimization suggestions for slow imports."""
        suggestions = []

        for imp, time_taken in slow_imports.items():
            if "tensorflow" in imp.lower() or "torch" in imp.lower():
                suggestions.append(
                    f"Make {imp} lazy-loaded (currently {time_taken:.2f}s)"
                )
            elif "deepchem" in imp.lower():
                suggestions.append(
                    f"Defer {imp} import until needed (currently {time_taken:.2f}s)"
                )
            elif time_taken > 1.0:
                suggestions.append(
                    f"Consider lazy loading {imp} (slow: {time_taken:.2f}s)"
                )

        return suggestions


def optimize_core_imports():
    """Optimize imports in core ChemML modules."""
    print("🚀 Optimizing Core Module Imports")
    print("=" * 40)

    core_modules = [
        "src/chemml/__init__.py",
        "src/chemml/core/__init__.py",
        "src/chemml/core/models.py",
        "src/chemml/core/featurizers.py",
        "src/chemml/research/__init__.py",
    ]

    profiler = ImportProfiler()
    total_optimization_time = 0

    for module_path in core_modules:
        if Path(module_path).exists():
            print(f"\n📁 Analyzing {module_path}")
            slow_imports = profiler.profile_module_imports(module_path)

            if slow_imports:
                print(f"  ⚠️  Found {len(slow_imports)} slow imports:")
                for imp, time_taken in slow_imports.items():
                    print(f"    • {imp}: {time_taken:.3f}s")
                    total_optimization_time += time_taken

                suggestions = profiler.suggest_optimizations(module_path, slow_imports)
                for suggestion in suggestions:
                    print(f"    💡 {suggestion}")
            else:
                print("  ✅ No slow imports detected")

    print(f"\n📊 Total potential optimization: {total_optimization_time:.2f}s")
    return profiler.slow_imports


def create_optimized_init_file():
    """Create an optimized version of ChemML's main __init__.py."""
    optimized_content = '''"""
ChemML: Machine Learning for Chemistry and Drug Discovery
Fast-loading version with optimized imports.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")

__version__ = "0.2.0"
__author__ = "ChemML Team"

# Fast imports - core functionality only
from .utils.lazy_imports import lazy_import
from .core.data import load_sample_data, quick_clean, quick_split
from .core.evaluation import quick_classification_eval, quick_regression_eval

# Lazy imports for everything else
_core_featurizers = lazy_import('chemml.core.featurizers')
_core_models = lazy_import('chemml.core.models')
research = lazy_import('chemml.research')
integrations = lazy_import('chemml.integrations')

# Provide quick access to most-used functions
def morgan_fingerprints(*args, **kwargs):
    """Generate Morgan fingerprints (lazy loaded)."""
    return _core_featurizers.morgan_fingerprints(*args, **kwargs)

def molecular_descriptors(*args, **kwargs):
    """Calculate molecular descriptors (lazy loaded)."""
    return _core_featurizers.molecular_descriptors(*args, **kwargs)

def create_rf_model(*args, **kwargs):
    """Create random forest model (lazy loaded)."""
    return _core_models.create_rf_model(*args, **kwargs)

def compare_models(*args, **kwargs):
    """Compare ML models (lazy loaded)."""
    return _core_models.compare_models(*args, **kwargs)

# Fast setup
def _setup_chemml():
    """Fast ChemML initialization."""
    print("ChemML initialized successfully!")
    print(f"Version: {__version__}")
    print("⚡ Fast-loading mode active")

_setup_chemml()

__all__ = [
    'morgan_fingerprints', 'molecular_descriptors', 'create_rf_model',
    'compare_models', 'load_sample_data', 'quick_clean', 'quick_split',
    'quick_classification_eval', 'quick_regression_eval', 'research', 'integrations'
]
'''

    # Save optimized version
    optimized_path = "src/chemml/__init___optimized.py"
    with open(optimized_path, "w") as f:
        f.write(optimized_content)

    print(f"✅ Created optimized init file: {optimized_path}")
    return optimized_path


def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description="ChemML Import Optimization")
    parser.add_argument(
        "--profile-only", action="store_true", help="Only profile, don't optimize"
    )
    parser.add_argument(
        "--create-optimized", action="store_true", help="Create optimized __init__.py"
    )

    args = parser.parse_args()

    print("🔧 ChemML Import Performance Optimization")
    print("=" * 50)

    # Profile current imports
    slow_imports = optimize_core_imports()

    if args.create_optimized:
        optimized_path = create_optimized_init_file()
        print("\n🚀 Next steps:")
        print(
            f'1. Test: python -c \'import sys; sys.path.insert(0, "."); exec(open("{optimized_path}").read())\''
        )
        print("2. If performance is good, replace the original __init__.py")

    print(f"\n📈 Optimization potential: {len(slow_imports)} slow imports identified")


if __name__ == "__main__":
    main()
