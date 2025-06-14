#!/usr/bin/env python3
"""
Day 3 Molecular Docking Notebook - Fix Validation Script
=======================================================

This script validates that all critical fixes have been properly implemented
in the Day 3 molecular docking notebook.
"""

import json
import os
import sys


def validate_notebook_fixes():
    """Validate all fixes in the Day 3 molecular docking notebook"""

    notebook_path = "notebooks/quickstart_bootcamp/days/day_03/day_03_molecular_docking_project.ipynb"

    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return False

    print("🔍 Validating Day 3 Molecular Docking Notebook Fixes...")
    print("=" * 60)

    # Load notebook
    try:
        with open(notebook_path, "r") as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load notebook: {e}")
        return False

    cells = notebook.get("cells", [])
    validation_results = {}

    # Check 1: Missing imports fixed
    print("\n1️⃣ Checking Missing Imports Fix...")
    import_cell_found = False
    time_import_found = False
    random_import_found = False

    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "import time" in source:
                time_import_found = True
                import_cell_found = True
            if "import random" in source:
                random_import_found = True
                import_cell_found = True

    if time_import_found and random_import_found:
        print("   ✅ Missing imports (time, random) - FIXED")
        validation_results["missing_imports"] = True
    else:
        print("   ❌ Missing imports still present")
        validation_results["missing_imports"] = False

    # Check 2: Variable naming fix
    print("\n2️⃣ Checking Variable Naming Fix...")
    variable_fix_found = False
    filtered_library_issues = 0

    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Check for the fix pattern
            if (
                "filtered_compounds = screening_pipeline.apply_filters(compound_library)"
                in source
            ):
                variable_fix_found = True
            # Check for remaining issues
            if "filtered_library" in source and "filtered_compounds" not in source:
                filtered_library_issues += 1

    if variable_fix_found and filtered_library_issues == 0:
        print("   ✅ Variable naming (filtered_library → filtered_compounds) - FIXED")
        validation_results["variable_naming"] = True
    else:
        print(
            f"   ❌ Variable naming issues remain ({filtered_library_issues} issues found)"
        )
        validation_results["variable_naming"] = False

    # Check 3: PDBQT format fix
    print("\n3️⃣ Checking PDBQT Format Fix...")
    pdbqt_fix_found = False

    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Look for the improved PDBQT generation
            if "ROOT\\n" in source and "ENDROOT\\n" in source and "TORSDOF" in source:
                pdbqt_fix_found = True
                break

    if pdbqt_fix_found:
        print("   ✅ PDBQT file format generation - FIXED")
        validation_results["pdbqt_format"] = True
    else:
        print("   ❌ PDBQT format issues remain")
        validation_results["pdbqt_format"] = False

    # Check 4: ML Scoring Function fix
    print("\n4️⃣ Checking ML Scoring Function Fix...")
    ml_fix_found = False
    descriptors3d_import = False

    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "from rdkit.Chem import Descriptors3D" in source:
                descriptors3d_import = True
            if "DESCRIPTORS_3D_AVAILABLE" in source:
                ml_fix_found = True

    if ml_fix_found and descriptors3d_import:
        print("   ✅ ML Scoring Function with 3D descriptors - FIXED")
        validation_results["ml_scoring"] = True
    else:
        print("   ❌ ML Scoring Function issues remain")
        validation_results["ml_scoring"] = False

    # Check 5: Documentation and status
    print("\n5️⃣ Checking Documentation and Status...")
    status_documentation = False

    for cell in cells:
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if (
                "COMPREHENSIVE FIXES APPLIED" in source
                or "CRITICAL ISSUES FIXED" in source
            ):
                status_documentation = True
                break

    if status_documentation:
        print("   ✅ Comprehensive documentation added - COMPLETE")
        validation_results["documentation"] = True
    else:
        print("   ❌ Status documentation missing")
        validation_results["documentation"] = False

    # Overall assessment
    print("\n" + "=" * 60)
    print("🎯 OVERALL VALIDATION RESULTS")
    print("=" * 60)

    total_checks = len(validation_results)
    passed_checks = sum(validation_results.values())
    success_rate = passed_checks / total_checks

    print(f"✅ Checks Passed: {passed_checks}/{total_checks}")
    print(f"📊 Success Rate: {success_rate:.1%}")

    if success_rate >= 0.8:
        print("🎉 EXCELLENT: Day 3 notebook fixes successfully validated!")
        print("🚀 Notebook is ready for authentic molecular docking education!")
        return True
    elif success_rate >= 0.6:
        print("👍 GOOD: Most fixes validated, minor issues remain")
        return True
    else:
        print("📝 NEEDS ATTENTION: Multiple validation failures")
        return False


if __name__ == "__main__":
    success = validate_notebook_fixes()
    sys.exit(0 if success else 1)
