#!/usr/bin/env python3
"""
ChemML Dependency Resolution and Warning Suppression
===================================================

This script helps resolve common dependency issues and suppress warnings
in ChemML environments. It provides tools to:

1. Install missing optional dependencies
2. Suppress common framework warnings
3. Validate the environment
4. Provide recommendations for optimal setup

Author: ChemML Development Team
Version: 1.0.0
"""

import os
import subprocess
import sys
import warnings
from typing import Dict, List, Optional


def suppress_common_warnings():
    """Suppress common warnings from dependencies."""
    # Suppress TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
    warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")
    warnings.filterwarnings("ignore", message=".*reduce_retracing.*")

    # Suppress DeepChem optional dependency warnings
    warnings.filterwarnings("ignore", message=".*pytorch-geometric.*")
    warnings.filterwarnings("ignore", message=".*transformers.*")
    warnings.filterwarnings("ignore", message=".*lightning.*")
    warnings.filterwarnings("ignore", message=".*HuggingFaceModel.*")
    warnings.filterwarnings("ignore", message=".*dgl.*")
    warnings.filterwarnings("ignore", message=".*jax.*")


def check_environment() -> Dict[str, bool]:
    """Check which dependencies are available."""
    results = {}

    # Core dependencies
    core_deps = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "rdkit",
        "deepchem",
        "qiskit",
        "torch",
        "tensorflow",
    ]

    # Optional dependencies (that cause warnings when missing)
    optional_deps = [
        "dgl",
        "transformers",
        "lightning",
        "jax",
        "torch_geometric",
        "pennylane",
        "cirq",
        "pyscf",
        "openmm",
        "mdtraj",
    ]

    print("🔍 Checking ChemML Environment")
    print("=" * 40)

    print("\n✅ Core Dependencies:")
    for dep in core_deps:
        try:
            __import__(dep)
            results[dep] = True
            print(f"   ✅ {dep}")
        except ImportError:
            results[dep] = False
            print(f"   ❌ {dep}")

    print("\n📦 Optional Dependencies:")
    for dep in optional_deps:
        try:
            __import__(dep)
            results[dep] = True
            print(f"   ✅ {dep}")
        except ImportError:
            results[dep] = False
            print(f"   ⚠️ {dep} (optional)")

    return results


def install_optional_dependencies():
    """Install commonly missing optional dependencies."""
    optional_packages = [
        "dgl",  # Deep Graph Library
        "transformers",  # Hugging Face
        "lightning",  # PyTorch Lightning
        "jax[cpu]",  # JAX
        "torch-geometric",  # PyTorch Geometric
        "pyscf",  # Classical quantum chemistry
    ]

    print("\n🚀 Installing Optional Dependencies...")
    print("=" * 40)

    for package in optional_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package} (may require specific setup)")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")


def create_optimized_imports():
    """Create optimized import statements that suppress warnings."""
    import_code = """
# Optimized ChemML Imports with Warning Suppression
# ================================================

import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress common deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*experimental_relax_shapes.*')

# Core ChemML imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ChemML modern quantum suite
try:
    from chemml.research.modern_quantum import (
        ModernVQE, ModernQAOA, QuantumFeatureMap,
        MolecularHamiltonianBuilder, HardwareEfficientAnsatz,
        QuantumChemistryWorkflow
    )
    print("✅ ChemML Modern Quantum Suite loaded")
except ImportError as e:
    print(f"⚠️ Modern quantum suite not available: {e}")

# Optional advanced features
try:
    import deepchem as dc
    print(f"✅ DeepChem {dc.__version__} loaded")
except ImportError:
    print("⚠️ DeepChem not available")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} loaded")
except ImportError:
    print("⚠️ PyTorch not available")

print("🎯 ChemML environment ready!")
"""

    with open("optimized_chemml_imports.py", "w") as f:
        f.write(import_code)

    print("📝 Created optimized_chemml_imports.py")


def main():
    """Main dependency resolution workflow."""
    print("🧬 ChemML Dependency Resolution Tool")
    print("=" * 50)

    # Step 1: Suppress warnings
    suppress_common_warnings()
    print("✅ Common warnings suppressed")

    # Step 2: Check environment
    results = check_environment()

    # Step 3: Count missing dependencies
    missing_core = [
        dep
        for dep, available in results.items()
        if not available and dep in ["numpy", "pandas", "rdkit", "deepchem", "qiskit"]
    ]
    missing_optional = [
        dep
        for dep, available in results.items()
        if not available and dep in ["dgl", "transformers", "lightning", "jax"]
    ]

    # Step 4: Provide recommendations
    print(f"\n📊 Environment Summary:")
    print(f"   Missing core dependencies: {len(missing_core)}")
    print(f"   Missing optional dependencies: {len(missing_optional)}")

    if missing_core:
        print(f"\n❗ Critical: Install missing core dependencies:")
        for dep in missing_core:
            print(f"   pip install {dep}")

    if missing_optional:
        print(f"\n💡 Recommended: Install optional dependencies to reduce warnings:")
        for dep in missing_optional:
            print(f"   pip install {dep}")

    # Step 5: Offer to install optional dependencies
    if missing_optional:
        response = input("\n🤔 Install missing optional dependencies? (y/n): ")
        if response.lower() == "y":
            install_optional_dependencies()

    # Step 6: Create optimized import template
    create_optimized_imports()

    print(f"\n🎉 Dependency resolution complete!")
    print(
        f"💡 Use 'import optimized_chemml_imports' to load ChemML with minimal warnings"
    )


if __name__ == "__main__":
    main()
