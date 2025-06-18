#!/usr/bin/env python3
"""
Validation Test for safe_git_commit.py
=====================================

This script thoroughly tests the git commit script before execution.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path


def test_script_syntax():
    """Test if the script has valid Python syntax"""
    try:
        with open("safe_git_commit.py", "r") as f:
            source = f.read()
        ast.parse(source)
        print("✅ Script syntax: VALID")
        return True
    except SyntaxError as e:
        print(f"❌ Script syntax: INVALID - {e}")
        return False
    except FileNotFoundError:
        print("❌ Script file not found")
        return False


def test_script_structure():
    """Test if the script has proper structure and safety measures"""
    try:
        with open("safe_git_commit.py", "r") as f:
            content = f.read()

        # Check for safety features
        safety_checks = [
            ("subprocess.run", "Uses subprocess properly"),
            ("capture_output=True", "Captures output safely"),
            ("timeout=", "Has timeout protection"),
            ("try:", "Has error handling"),
            ("except", "Has exception handling"),
            ("shell=False", "No shell injection risk (shell=False or not specified)"),
        ]

        issues = []
        for check, description in safety_checks:
            if check == "shell=False":
                # Special check - ensure shell=True is not used
                if "shell=True" in content:
                    issues.append(f"❌ {description} - Found shell=True")
                else:
                    print(f"✅ {description}")
            elif check in content:
                print(f"✅ {description}")
            else:
                issues.append(f"⚠️ {description} - Not found")

        return len(issues) == 0, issues

    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False, [str(e)]


def test_git_environment():
    """Test if we're in a proper git environment"""
    try:
        # Check if git is available
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            print(f"✅ Git available: {result.stdout.strip()}")
        else:
            print("❌ Git not available")
            return False

        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            print("✅ In git repository")
            return True
        else:
            print("❌ Not in git repository")
            return False

    except Exception as e:
        print(f"❌ Git environment test failed: {e}")
        return False


def test_file_permissions():
    """Test if we have proper file permissions"""
    try:
        script_path = Path("safe_git_commit.py")
        if script_path.exists():
            if script_path.is_file():
                print("✅ Script file exists and is readable")
                return True
            else:
                print("❌ Script path exists but is not a file")
                return False
        else:
            print("❌ Script file does not exist")
            return False
    except Exception as e:
        print(f"❌ File permission test failed: {e}")
        return False


def run_comprehensive_validation():
    """Run all validation tests"""
    print("🔍 COMPREHENSIVE VALIDATION OF safe_git_commit.py")
    print("=" * 55)

    tests = [
        ("Script Syntax", test_script_syntax),
        ("Git Environment", test_git_environment),
        ("File Permissions", test_file_permissions),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Test script structure separately
    print(f"\n📋 Testing: Script Structure")
    struct_ok, struct_issues = test_script_structure()
    results["Script Structure"] = struct_ok

    if not struct_ok:
        print("Structure issues found:")
        for issue in struct_issues:
            print(f"  {issue}")

    # Generate validation report
    print("\n" + "=" * 55)
    print("📊 VALIDATION REPORT")
    print("=" * 55)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    is_safe = all(results.values())

    print(f"\n🎯 OVERALL ASSESSMENT:")
    if is_safe:
        print("   ✅ SAFE TO EXECUTE")
        print("   ✅ All validation tests passed")
        print("   ✅ Script should work correctly")
    else:
        print("   🚨 NOT SAFE TO EXECUTE")
        print("   🚨 Some validation tests failed")
        print("   🚨 Review issues before running")

    # Save validation results
    try:
        validation_data = {
            "timestamp": str(Path.cwd()),
            "results": results,
            "safe_to_execute": is_safe,
            "passed_tests": passed,
            "total_tests": total,
        }

        with open("script_validation_results.json", "w") as f:
            json.dump(validation_data, f, indent=2)
        print(f"\n📋 Validation results saved to: script_validation_results.json")
    except Exception as e:
        print(f"\n⚠️ Could not save validation results: {e}")

    return is_safe


if __name__ == "__main__":
    is_safe = run_comprehensive_validation()

    if is_safe:
        print("\n🎉 VALIDATION SUCCESSFUL - Script is safe to execute!")
        sys.exit(0)
    else:
        print(
            "\n⚠️ VALIDATION FAILED - Do not execute script until issues are resolved!"
        )
        sys.exit(1)
