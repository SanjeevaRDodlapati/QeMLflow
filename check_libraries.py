#!/usr/bin/env python
"""
Script to check the status of required libraries for the day_04 notebook.
"""
import importlib
import sys


def check_library(name):
    try:
        lib = importlib.import_module(name)
        version = getattr(lib, "__version__", "unknown version")
        return f"✅ {name} is installed (version: {version})"
    except ImportError as e:
        return f"❌ {name} is not installed: {str(e)}"
    except Exception as e:
        return f"⚠️ {name} has issues: {str(e)}"


# List of libraries to check
libraries = [
    "numpy",
    "pandas",
    "matplotlib",
    "rdkit",
    "ase",
    "pyscf",
    "deepchem",
    "torch",
    "sklearn",
]

print("=" * 50)
print("LIBRARY STATUS CHECK")
print("=" * 50)
print(f"Python version: {sys.version}")
print("=" * 50)

for lib in libraries:
    print(check_library(lib))

print("=" * 50)
print("NUMPY COMPATIBILITY CHECK")
print("=" * 50)

# Check NumPy compatibility specifically
try:
    import numpy as np

    print(f"NumPy version: {np.__version__}")
    print("Testing NumPy functionality...")
    try:
        # Test basic NumPy functionality
        arr = np.array([1, 2, 3])
        print(f"Array creation: {arr}")
        print("NumPy appears to be working correctly")
    except Exception as e:
        print(f"NumPy functionality issue: {str(e)}")
except Exception as e:
    print(f"NumPy import error: {str(e)}")

# Try to import and use deepchem
print("=" * 50)
print("DEEPCHEM TEST")
print("=" * 50)
try:
    import deepchem as dc

    print(f"DeepChem version: {dc.__version__}")
    try:
        # Try to use a simple deepchem feature
        featurizer = dc.feat.CircularFingerprint(size=1024)
        print("DeepChem feature creation successful")
    except Exception as e:
        print(f"DeepChem usage error: {str(e)}")
except Exception as e:
    print(f"DeepChem import error: {str(e)}")
