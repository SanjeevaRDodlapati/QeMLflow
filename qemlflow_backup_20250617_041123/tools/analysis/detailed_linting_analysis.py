#!/usr/bin/env python3
"""
Comprehensive linting analysis tool for ChemML codebase.
Provides detailed module-wise breakdown of code quality issues.
"""

import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def run_flake8_analysis():
    """Run flake8 and collect detailed statistics."""
    try:
        result = subprocess.run(
            ["flake8", "src/", "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        return [line for line in lines if line.strip()]
    except Exception as e:
        print(f"Error running flake8: {e}")
        return []


def analyze_by_module(flake8_lines):
    """Analyze issues by module/directory."""
    module_stats = defaultdict(
        lambda: {
            "total_issues": 0,
            "syntax_errors": 0,
            "import_issues": 0,
            "unused_imports": 0,
            "complexity_issues": 0,
            "undefined_names": 0,
            "star_imports": 0,
            "other_issues": 0,
            "files_with_issues": set(),
            "error_details": [],
        }
    )

    # Error code categories
    syntax_errors = ["E999"]
    import_issues = ["E402", "F401", "F403", "F405"]
    complexity_issues = ["C901"]
    undefined_names = ["F821"]
    unused_vars = ["F841"]

    for line in flake8_lines:
        if ":" not in line:
            continue

        parts = line.split(":", 4)
        if len(parts) < 4:
            continue

        file_path = parts[0]
        error_code = parts[3].split()[0] if len(parts[3].split()) > 0 else "UNKNOWN"
        error_message = ":".join(parts[3:])

        # Extract module path
        if file_path.startswith("src/chemml/"):
            module_path = file_path[len("src/chemml/") :]
            if "/" in module_path:
                module = module_path.split("/")[0]
            else:
                module = "root"
        else:
            module = "other"

        stats = module_stats[module]
        stats["total_issues"] += 1
        stats["files_with_issues"].add(file_path)
        stats["error_details"].append((file_path, error_code, error_message))

        # Categorize errors
        if error_code in syntax_errors:
            stats["syntax_errors"] += 1
        elif error_code in import_issues:
            stats["import_issues"] += 1
            if error_code == "F401":
                stats["unused_imports"] += 1
            elif error_code in ["F403", "F405"]:
                stats["star_imports"] += 1
        elif error_code in complexity_issues:
            stats["complexity_issues"] += 1
        elif error_code in undefined_names:
            stats["undefined_names"] += 1
        else:
            stats["other_issues"] += 1

    return dict(module_stats)


def analyze_by_error_type(flake8_lines):
    """Analyze issues by error type."""
    error_stats = Counter()

    for line in flake8_lines:
        if ":" not in line:
            continue

        parts = line.split(":", 4)
        if len(parts) < 4:
            continue

        error_code = parts[3].split()[0] if len(parts[3].split()) > 0 else "UNKNOWN"
        error_stats[error_code] += 1

    return error_stats


def get_file_stats():
    """Get basic file statistics."""
    src_path = Path(__file__).parent.parent.parent / "src"

    py_files = list(src_path.glob("**/*.py"))
    total_lines = 0

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                total_lines += len(f.readlines())
        except Exception:
            pass

    return len(py_files), total_lines


def print_detailed_analysis(module_stats, error_stats, file_count, total_lines):
    """Print comprehensive analysis results."""
    print("=" * 80)
    print("📊 COMPREHENSIVE CHEMML LINTING ANALYSIS")
    print("=" * 80)
    print()

    # Overall statistics
    total_issues = sum(stats["total_issues"] for stats in module_stats.values())
    total_files_with_issues = len(
        set().union(*[stats["files_with_issues"] for stats in module_stats.values()])
    )

    print("🔍 OVERALL STATISTICS:")
    print(f"  • Total Python files: {file_count}")
    print(f"  • Total lines of code: {total_lines:,}")
    print(f"  • Files with issues: {total_files_with_issues}")
    print(f"  • Total issues: {total_issues}")
    print(f"  • Issues per 1000 LOC: {total_issues / total_lines * 1000:.1f}")
    print()

    # Error type breakdown
    print("🚨 ERROR TYPE BREAKDOWN:")
    error_categories = {
        "E999": "Syntax Errors",
        "F401": "Unused Imports",
        "F403": "Star Imports (import *)",
        "F405": "Undefined from Star Imports",
        "F821": "Undefined Names",
        "E402": "Module Import Not at Top",
        "C901": "Too Complex",
        "F841": "Unused Variables",
    }

    for error_code, count in error_stats.most_common():
        category = error_categories.get(error_code, error_code)
        percentage = count / total_issues * 100
        print(f"  • {category:<30} {count:>4} ({percentage:>5.1f}%)")
    print()

    # Module-wise breakdown
    print("📁 MODULE-WISE BREAKDOWN:")
    print(
        f"{'Module':<25} {'Issues':<8} {'Files':<6} {'Syntax':<7} {'Import':<7} {'Complex':<8} {'Undef':<6}"
    )
    print("-" * 80)

    # Sort modules by total issues
    sorted_modules = sorted(
        module_stats.items(), key=lambda x: x[1]["total_issues"], reverse=True
    )

    for module, stats in sorted_modules:
        if stats["total_issues"] == 0:
            continue

        print(
            f"{module:<25} {stats['total_issues']:<8} {len(stats['files_with_issues']):<6} "
            f"{stats['syntax_errors']:<7} {stats['import_issues']:<7} "
            f"{stats['complexity_issues']:<8} {stats['undefined_names']:<6}"
        )

    print()

    # Top problematic modules (detailed)
    print("🔥 TOP PROBLEMATIC MODULES (Detailed):")
    print()

    for i, (module, stats) in enumerate(sorted_modules[:5]):
        if stats["total_issues"] == 0:
            continue

        print(f"{i+1}. {module.upper()} MODULE:")
        print(f"   • Total Issues: {stats['total_issues']}")
        print(f"   • Files Affected: {len(stats['files_with_issues'])}")
        print(f"   • Syntax Errors: {stats['syntax_errors']}")
        print(f"   • Import Issues: {stats['import_issues']}")
        print(f"   • Unused Imports: {stats['unused_imports']}")
        print(f"   • Complexity Issues: {stats['complexity_issues']}")
        print(f"   • Undefined Names: {stats['undefined_names']}")

        # Show most common issues in this module
        module_errors = Counter()
        for file_path, error_code, error_message in stats["error_details"]:
            module_errors[error_code] += 1

        print(f"   • Top Error Types:")
        for error_code, count in module_errors.most_common(3):
            category = error_categories.get(error_code, error_code)
            print(f"     - {category}: {count}")
        print()

    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print()

    syntax_error_count = sum(stats["syntax_errors"] for stats in module_stats.values())
    import_issue_count = sum(stats["import_issues"] for stats in module_stats.values())

    if syntax_error_count > 0:
        print(f"🚨 PRIORITY 1: Fix {syntax_error_count} syntax errors")
        print("   These prevent modules from being imported and must be fixed first.")
        print()

    if import_issue_count > 0:
        print(f"📦 PRIORITY 2: Clean up {import_issue_count} import issues")
        print("   Remove unused imports and fix star import patterns.")
        print()

    complexity_count = sum(
        stats["complexity_issues"] for stats in module_stats.values()
    )
    if complexity_count > 0:
        print(f"🔧 PRIORITY 3: Refactor {complexity_count} complex functions")
        print("   Break down complex functions for better maintainability.")
        print()

    undefined_count = sum(stats["undefined_names"] for stats in module_stats.values())
    if undefined_count > 0:
        print(f"🔍 PRIORITY 4: Fix {undefined_count} undefined name references")
        print("   Add proper imports or fix typos in variable names.")
        print()


def main():
    """Main analysis function."""
    print("Running comprehensive linting analysis...")
    print()

    # Get flake8 results
    flake8_lines = run_flake8_analysis()

    if not flake8_lines:
        print("✅ No linting issues found or flake8 failed to run!")
        return

    # Analyze results
    module_stats = analyze_by_module(flake8_lines)
    error_stats = analyze_by_error_type(flake8_lines)
    file_count, total_lines = get_file_stats()

    # Print detailed analysis
    print_detailed_analysis(module_stats, error_stats, file_count, total_lines)


if __name__ == "__main__":
    main()
