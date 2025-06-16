#!/usr/bin/env python3
"""
Final comprehensive test to verify all VAE fixes are working correctly
"""

import subprocess
import sys


def run_test(test_file, description):
    """Run a test file and return the result"""
    print(f"\n🧪 Running: {description}")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, test_file], capture_output=True, text=True, cwd="."
        )

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run all VAE-related tests"""
    print("🚀 FINAL VAE FIX VERIFICATION")
    print("=" * 60)

    tests = [
        ("test_vae_fix.py", "VAE Loss Function Fix (tensor compatibility)"),
        ("test_vae_decode_fix.py", "VAE Decode Method Fix (tensor dimensions)"),
        ("test_notebook_vae.py", "Complete Notebook VAE Implementation"),
    ]

    results = []

    for test_file, description in tests:
        success = run_test(test_file, description)
        results.append((description, success))

    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:<12} - {description}")
        if success:
            passed += 1

    print("\n" + "=" * 60)
    if passed == len(results):
        print("🏆 ALL VAE FIXES VERIFIED SUCCESSFULLY!")
        print("🚀 Day 2 Deep Learning for Molecules VAE is ready!")
        print("✅ Molecular generation functionality is working")
        print("✅ No more RuntimeError in tensor operations")
    else:
        print("⚠️  SOME TESTS FAILED - Further investigation needed")
        print(f"📊 {passed}/{len(results)} tests passed")

    print("=" * 60)

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
