#!/usr/bin/env python3
"""
Phase 4 Legacy Architecture Consolidation - Final Validation Script

This script validates that all import pattern migrations are complete and functional.
"""

import sys
from typing import List, Tuple


def test_core_module_imports() -> Tuple[bool, str]:
    """Test core module imports are working."""
    try:
        from chemml.research.drug_discovery.admet import ADMETPredictor
        from chemml.research.drug_discovery.generation import MolecularGenerator
        from chemml.research.drug_discovery.molecular_optimization import (
            MolecularOptimizer,
        )
        from chemml.research.drug_discovery.properties import MolecularPropertyPredictor
        from chemml.research.drug_discovery.qsar import DescriptorCalculator
        from chemml.research.drug_discovery.screening import VirtualScreener

        return True, "All core modular imports successful"
    except Exception as e:
        return False, f"Core imports failed: {e}"


def test_backward_compatibility() -> Tuple[bool, str]:
    """Test that backward compatibility is maintained."""
    try:
        # Test main module imports still work
        from chemml.research.drug_discovery import (
            ADMETPredictor,
            MolecularGenerator,
            MolecularOptimizer,
            MolecularPropertyPredictor,
            QSARModel,
            VirtualScreener,
        )

        return True, "Backward compatibility maintained"
    except Exception as e:
        return False, f"Backward compatibility failed: {e}"


def test_individual_modules() -> List[Tuple[str, bool, str]]:
    """Test each module individually."""
    results = []

    modules_to_test = [
        (
            "molecular_optimization",
            "chemml.research.drug_discovery.molecular_optimization",
        ),
        ("admet", "chemml.research.drug_discovery.admet"),
        ("screening", "chemml.research.drug_discovery.screening"),
        ("properties", "chemml.research.drug_discovery.properties"),
        ("generation", "chemml.research.drug_discovery.generation"),
        ("qsar", "chemml.research.drug_discovery.qsar"),
    ]

    for module_name, import_path in modules_to_test:
        try:
            exec(f"import {import_path}")
            results.append((module_name, True, "Import successful"))
        except Exception as e:
            results.append((module_name, False, f"Import failed: {e}"))

    return results


def test_function_imports() -> List[Tuple[str, bool, str]]:
    """Test specific function imports."""
    results = []

    function_tests = [
        (
            "predict_properties",
            "from chemml.research.drug_discovery.properties import predict_properties",
        ),
        (
            "predict_admet_properties",
            "from chemml.research.drug_discovery.admet import predict_admet_properties",
        ),
        (
            "perform_virtual_screening",
            "from chemml.research.drug_discovery.screening import perform_virtual_screening",
        ),
        (
            "build_qsar_model",
            "from chemml.research.drug_discovery.qsar import build_qsar_model",
        ),
        (
            "generate_molecular_structures",
            "from chemml.research.drug_discovery.generation import generate_molecular_structures",
        ),
        (
            "optimize_molecule",
            "from chemml.research.drug_discovery.molecular_optimization import optimize_molecule",
        ),
    ]

    for func_name, import_stmt in function_tests:
        try:
            exec(import_stmt)
            results.append((func_name, True, "Function import successful"))
        except Exception as e:
            results.append((func_name, False, f"Function import failed: {e}"))

    return results


def main():
    """Run all validation tests."""
    print("🚀 Phase 4 Legacy Architecture Consolidation - Final Validation")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0

    # Test 1: Core module imports
    print("\n📦 Testing Core Module Imports...")
    success, message = test_core_module_imports()
    total_tests += 1
    if success:
        passed_tests += 1
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

    # Test 2: Backward compatibility
    print("\n🔄 Testing Backward Compatibility...")
    success, message = test_backward_compatibility()
    total_tests += 1
    if success:
        passed_tests += 1
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

    # Test 3: Individual modules
    print("\n🧩 Testing Individual Modules...")
    module_results = test_individual_modules()
    for module_name, success, message in module_results:
        total_tests += 1
        if success:
            passed_tests += 1
            print(f"✅ {module_name}: {message}")
        else:
            print(f"❌ {module_name}: {message}")

    # Test 4: Function imports
    print("\n⚙️ Testing Function Imports...")
    function_results = test_function_imports()
    for func_name, success, message in function_results:
        total_tests += 1
        if success:
            passed_tests += 1
            print(f"✅ {func_name}: {message}")
        else:
            print(f"❌ {func_name}: {message}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Phase 4 validation complete.")
        return 0
    else:
        print(
            f"\n⚠️ {total_tests - passed_tests} tests failed. Please review the errors above."
        )
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Validation script crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
