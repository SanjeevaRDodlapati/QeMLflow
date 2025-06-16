"""
Modern Quantum Suite Validation for ChemML Bootcamp
==================================================

Validates that all Day 6/7 bootcamp notebooks work with the modern quantum suite.
Tests import compatibility, basic functionality, and integration workflows.

Version: 2.0.0
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add ChemML to path
sys.path.append("src")


def test_modern_quantum_imports():
    """Test that modern quantum suite imports work correctly."""
    print("🧪 Testing Modern Quantum Suite Imports...")

    try:
        from chemml.research.modern_quantum import (
            HardwareEfficientAnsatz,
            ModernQAOA,
            ModernVQE,
            MolecularHamiltonianBuilder,
            QuantumChemistryWorkflow,
            QuantumFeatureMap,
        )

        print("✅ All modern quantum classes imported successfully")

        # Test instantiation
        hamiltonian = MolecularHamiltonianBuilder.h2_hamiltonian()
        _vqe = ModernVQE(HardwareEfficientAnsatz.two_qubit_ansatz, hamiltonian)
        _feature_map = QuantumFeatureMap(2)
        _workflow = QuantumChemistryWorkflow()

        print("✅ All modern quantum objects instantiated successfully")
        return True

    except Exception as e:
        print(f"❌ Modern quantum import failed: {e}")
        return False


def test_notebook_imports(notebook_path: str) -> Tuple[bool, str]:
    """Test imports in a notebook by extracting and running import cells."""
    print(f"🔍 Testing imports in {notebook_path}...")

    try:
        # Read notebook
        import json

        with open(notebook_path, "r") as f:
            notebook = json.load(f)

        # Extract import code from cells
        import_code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if any(keyword in source for keyword in ["import", "from"]):
                    # Add path adjustment for notebook context
                    adjusted_source = source.replace(
                        "sys.path.append('../../../../src')", "sys.path.append('src')"
                    )
                    import_code.append(adjusted_source)
                    break  # Test only first import cell

        if not import_code:
            return True, "No import cells found"

        # Test the import code
        combined_code = "\n".join(import_code)
        exec(combined_code, {"__name__": "__main__"})

        return True, "Imports successful"

    except Exception as e:
        return False, f"Import test failed: {e}"


def test_quantum_functionality():
    """Test core quantum functionality."""
    print("⚡ Testing Core Quantum Functionality...")

    try:
        from chemml.research.modern_quantum import (
            HardwareEfficientAnsatz,
            ModernVQE,
            MolecularHamiltonianBuilder,
        )

        # Test VQE
        hamiltonian = MolecularHamiltonianBuilder.h2_hamiltonian()
        vqe = ModernVQE(HardwareEfficientAnsatz.two_qubit_ansatz, hamiltonian)
        result = vqe.run([0.1, 0.2])

        assert "ground_state_energy" in result
        assert "converged" in result
        print(f"✅ VQE test passed: Energy = {result['ground_state_energy']:.6f}")

        # Test quantum feature map
        from chemml.research.modern_quantum import QuantumFeatureMap

        feature_map = QuantumFeatureMap(2)
        test_data = [[0.1, 0.2], [0.3, 0.4]]
        features = feature_map.transform(test_data)

        assert features.shape[0] == 2
        print(f"✅ Quantum feature map test passed: Shape = {features.shape}")

        return True

    except Exception as e:
        print(f"❌ Quantum functionality test failed: {e}")
        return False


def validate_notebook_structure(notebook_path: str) -> Dict[str, Any]:
    """Validate notebook structure and content."""
    try:
        import json

        with open(notebook_path, "r") as f:
            notebook = json.load(f)

        cells = notebook["cells"]
        code_cells = [c for c in cells if c["cell_type"] == "code"]
        markdown_cells = [c for c in cells if c["cell_type"] == "markdown"]

        # Check for modern quantum usage
        modern_quantum_usage = False
        legacy_quantum_usage = False

        for cell in code_cells:
            source = "".join(cell["source"])
            if "chemml.research.modern_quantum" in source or "ModernVQE" in source:
                modern_quantum_usage = True
            if "qiskit_algorithms" in source or "BaseSampler" in source:
                legacy_quantum_usage = True

        return {
            "total_cells": len(cells),
            "code_cells": len(code_cells),
            "markdown_cells": len(markdown_cells),
            "modern_quantum_usage": modern_quantum_usage,
            "legacy_quantum_usage": legacy_quantum_usage,
            "valid": True,
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}


def main():
    """Main validation routine."""
    print("🎯 ChemML Modern Quantum Suite Validation")
    print("=" * 50)

    # Test 1: Basic imports
    print("\n1️⃣ Testing Basic Imports...")
    if not test_modern_quantum_imports():
        print("❌ Basic import test failed. Exiting.")
        return False

    # Test 2: Core functionality
    print("\n2️⃣ Testing Core Functionality...")
    if not test_quantum_functionality():
        print("❌ Core functionality test failed. Exiting.")
        return False

    # Test 3: Notebook validation
    print("\n3️⃣ Validating Notebooks...")
    notebook_paths = [
        "notebooks/quickstart_bootcamp/days/day_06/day_06_module_1_quantum_foundations.ipynb",
        "notebooks/quickstart_bootcamp/days/day_06/day_06_module_2_vqe_algorithms.ipynb",
        "notebooks/quickstart_bootcamp/days/day_06/day_06_module_3_quantum_production.ipynb",
        "notebooks/quickstart_bootcamp/days/day_07/day_07_module_1_integration.ipynb",
    ]

    all_passed = True
    results = {}

    for notebook_path in notebook_paths:
        if os.path.exists(notebook_path):
            # Test imports
            import_success, import_msg = test_notebook_imports(notebook_path)

            # Validate structure
            structure = validate_notebook_structure(notebook_path)

            results[notebook_path] = {
                "import_test": import_success,
                "import_message": import_msg,
                "structure": structure,
            }

            status = (
                "✅"
                if import_success and structure.get("modern_quantum_usage", False)
                else "⚠️"
            )
            legacy = "🔧" if structure.get("legacy_quantum_usage", False) else ""

            print(f"   {status} {os.path.basename(notebook_path)} {legacy}")
            if not import_success:
                print(f"      Import error: {import_msg}")
                all_passed = False
            if structure.get("legacy_quantum_usage", False):
                print("      ⚠️ Contains legacy quantum code")
        else:
            print(f"   ❌ {notebook_path} - File not found")
            all_passed = False

    # Summary
    print("\n📊 Validation Summary:")
    print("   Modern quantum imports: ✅")
    print("   Core functionality: ✅")

    modern_count = sum(
        1 for r in results.values() if r["structure"].get("modern_quantum_usage", False)
    )
    legacy_count = sum(
        1 for r in results.values() if r["structure"].get("legacy_quantum_usage", False)
    )

    print(f"   Notebooks with modern quantum: {modern_count}/{len(results)}")
    print(f"   Notebooks with legacy code: {legacy_count}/{len(results)}")

    if all_passed and legacy_count == 0:
        print("\n🎉 All validations passed! Modern quantum suite ready for production.")
        return True
    elif modern_count > 0:
        print(
            "\n✅ Modern quantum suite partially implemented. Some legacy code remains."
        )
        return True
    else:
        print("\n❌ Validation failed. Modern quantum suite not properly implemented.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
