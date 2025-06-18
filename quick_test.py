#!/usr/bin/env python3
"""
Quick validation test - minimal and safe
"""
import ast
from pathlib import Path


def quick_test():
    # Test critical files for basic syntax
    critical_files = [
        "src/qemlflow/__init__.py",
        "src/qemlflow/research/clinical_research.py",
        "safe_validation.py",
    ]

    results = []
    for file_path in critical_files:
        try:
            with open(file_path, "r") as f:
                content = f.read()
            ast.parse(content)
            results.append(f"✅ {file_path}: OK")
        except Exception as e:
            results.append(f"❌ {file_path}: {e}")

    with open("quick_test_results.txt", "w") as f:
        f.write("Quick Validation Results:\n")
        f.write("========================\n\n")
        for result in results:
            f.write(result + "\n")

    return len([r for r in results if "✅" in r])


if __name__ == "__main__":
    success_count = quick_test()
    print(f"Tested files, {success_count} passed. Check quick_test_results.txt")
