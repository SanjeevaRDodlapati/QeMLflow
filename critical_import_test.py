#!/usr/bin/env python3
"""
Critical Import Test for GitHub Actions Workflows
==============================================

This test simulates exactly what GitHub Actions workflows do.
"""

import sys
import traceback
from pathlib import Path


def test_critical_imports():
    """Test the exact imports that cause workflow failures"""
    results = []

    # Add src to path (like workflows do)
    repo_root = Path.cwd()
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    tests = [
        ("Main QeMLflow import", "import qemlflow"),
        (
            "QeMLflow version",
            "import qemlflow; print(f'Version: {qemlflow.__version__}')",
        ),
        ("Research module", "import qemlflow.research"),
        ("Clinical research", "import qemlflow.research.clinical_research"),
        ("Materials discovery", "import qemlflow.research.materials_discovery"),
        ("Quantum module", "import qemlflow.research.quantum"),
        ("Advanced models", "import qemlflow.research.advanced_models"),
        ("Core module", "import qemlflow.core"),
        ("Utils module", "import qemlflow.utils"),
    ]

    print("🔍 CRITICAL IMPORT TESTING")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, test_code in tests:
        try:
            exec(test_code)
            print(f"✅ {test_name}: SUCCESS")
            results.append(f"PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            results.append(f"FAIL: {test_name} - {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")

    # Write results to file for persistence
    with open("import_test_results.txt", "w") as f:
        f.write("Critical Import Test Results\n")
        f.write("===========================\n\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n\n")
        for result in results:
            f.write(result + "\n")

    # Determine if workflows will pass
    critical_failures = failed
    if critical_failures == 0:
        print("🎉 ALL TESTS PASSED - GitHub Actions workflows should succeed!")
        return True
    else:
        print(f"🚨 {critical_failures} CRITICAL FAILURES - Workflows will fail!")
        return False


if __name__ == "__main__":
    success = test_critical_imports()
    sys.exit(0 if success else 1)
