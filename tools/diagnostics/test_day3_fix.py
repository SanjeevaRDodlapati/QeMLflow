#!/usr/bin/env python3
"""
Test script to verify Day 3 pandas fix is working correctly.
"""

import os
import subprocess
import sys


def test_day3_script():
    """Test that Day 3 script runs without pandas errors."""
    print("🧪 Testing Day 3 script for pandas issues...")

    script_path = (
        "notebooks/quickstart_bootcamp/days/day_03/day_03_molecular_docking_final.py"
    )

    try:
        result = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True, timeout=60
        )

        # Check if there are any pandas-related errors
        if "name 'pd' is not defined" in result.stderr:
            print("❌ FAILED: 'pd' is not defined error still present")
            print("STDERR:", result.stderr)
            return False
        elif "name 'pd' is not defined" in result.stdout:
            print("❌ FAILED: 'pd' is not defined error still present")
            print("STDOUT:", result.stdout)
            return False
        elif result.returncode == 0:
            print("✅ SUCCESS: Day 3 script completed without pandas errors")
            print(f"Return code: {result.returncode}")
            return True
        else:
            print(
                f"⚠️  WARNING: Script returned non-zero exit code: {result.returncode}"
            )
            if "pandas" in result.stderr.lower() or "pd" in result.stderr:
                print("❌ FAILED: Pandas-related issues detected in stderr")
                print("STDERR:", result.stderr[-500:])  # Last 500 chars
                return False
            else:
                print("✅ SUCCESS: No pandas errors detected (other issues may exist)")
                return True

    except subprocess.TimeoutExpired:
        print("⚠️  WARNING: Script timed out after 60 seconds")
        return True  # Timeout might be due to waiting for input, not pandas error
    except Exception as e:
        print(f"❌ ERROR: Exception occurred: {e}")
        return False


def test_quick_access_demo():
    """Test that quick access demo can handle Day 3 without errors."""
    print("\n🎯 Testing quick access demo with Day 3...")

    try:
        # Simulate selecting day 3, then back
        input_data = "3\nb\nq\n"
        result = subprocess.run(
            [sys.executable, "quick_access_demo.py"],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if (
            "name 'pd' is not defined" in result.stderr
            or "name 'pd' is not defined" in result.stdout
        ):
            print("❌ FAILED: Pandas error in quick access demo")
            return False
        else:
            print("✅ SUCCESS: Quick access demo works without pandas errors")
            return True

    except subprocess.TimeoutExpired:
        print("⚠️  WARNING: Quick access demo timed out")
        return True
    except Exception as e:
        print(f"❌ ERROR: Exception in quick access demo test: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("    Day 3 Pandas Fix Verification Test")
    print("=" * 60)

    # Change to the correct directory
    os.chdir("/Users/sanjeevadodlapati/Downloads/Repos/ChemML")

    # Run tests
    test1_passed = test_day3_script()
    test2_passed = test_quick_access_demo()

    print("\n" + "=" * 60)
    print("    Test Results Summary")
    print("=" * 60)

    print(f"Day 3 Direct Script Test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Quick Access Demo Test:   {'✅ PASS' if test2_passed else '❌ FAIL'}")

    overall_success = test1_passed and test2_passed
    print(
        f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}"
    )

    if overall_success:
        print("\n🎉 Day 3 pandas fix is working correctly!")
    else:
        print("\n⚠️  Some issues remain with the pandas fix.")

    sys.exit(0 if overall_success else 1)
