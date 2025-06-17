# ================================================

import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress common deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*experimental_relax_shapes.*")

import matplotlib.pyplot as plt

# Core QeMLflow imports
import numpy as np
import pandas as pd

# QeMLflow modern quantum suite
try:
    from qemlflow.research.modern_quantum import (
        HardwareEfficientAnsatz,
        ModernQAOA,
        ModernVQE,
        MolecularHamiltonianBuilder,
        QuantumChemistryWorkflow,
        QuantumFeatureMap,
    )

    print("✅ QeMLflow Modern Quantum Suite loaded")
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

print("🎯 QeMLflow environment ready!")
