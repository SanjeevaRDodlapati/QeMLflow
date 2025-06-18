#!/usr/bin/env python3
"""
Test Suite for Safe Auto-Fix Framework
=====================================

Comprehensive tests to ensure the auto-fix framework works correctly
and doesn't corrupt files.
"""

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from safe_auto_fix import FixResult, SafeLintingAutoFix


class TestSafeAutoFix(unittest.TestCase):
    """Test suite for SafeLintingAutoFix."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.auto_fixer = SafeLintingAutoFix(root_path=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file with given content."""
        file_path = self.temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    def test_syntax_validation_valid_file(self):
        """Test syntax validation with valid Python file."""
        content = '''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    hello_world()
'''
        file_path = self.create_test_file("valid.py", content)
        is_valid, error = self.auto_fixer.validate_python_syntax(file_path)

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_syntax_validation_invalid_file(self):
        """Test syntax validation with invalid Python file."""
        content = '''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")

return "success"  # This is outside the function!
'''
        file_path = self.create_test_file("invalid.py", content)
        is_valid, error = self.auto_fixer.validate_python_syntax(file_path)

        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("return", error.lower())

    def test_fix_return_outside_function_simple(self):
        """Test fixing simple return outside function."""
        content = '''def test_function():
    """Test function."""
    pass

return "hello"
'''
        expected_fixes = '''def test_function():
    """Test function."""
    pass

# TODO: Fix orphaned return statement: return "hello"
'''
        fixed_content, changes = self.auto_fixer.fix_return_outside_function(content)

        self.assertIn("orphaned return", changes[0])
        self.assertEqual(fixed_content.strip(), expected_fixes.strip())

    def test_fix_return_outside_function_with_proper_function(self):
        """Test fixing return that should be inside a function."""
        content = '''def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    result = a + b

return result
'''
        fixed_content, changes = self.auto_fixer.fix_return_outside_function(content)

        # Should fix the indentation
        self.assertIn("Fixed return statement indentation", changes[0])
        self.assertIn("    return result", fixed_content)

    def test_backup_creation_and_restoration(self):
        """Test backup creation and restoration."""
        content = "print('original content')"
        file_path = self.create_test_file("test.py", content)

        # Create backup
        backup_path = self.auto_fixer.create_backup(file_path)
        self.assertTrue(backup_path.exists())

        # Modify original file
        with open(file_path, "w") as f:
            f.write("print('modified content')")

        # Restore from backup
        success = self.auto_fixer.restore_from_backup(file_path)
        self.assertTrue(success)

        # Verify restoration
        with open(file_path, "r") as f:
            restored_content = f.read()
        self.assertEqual(restored_content, content)

    def test_fix_file_safely_dry_run(self):
        """Test safe file fixing in dry-run mode."""
        content = """def test():
    pass

return "test"  # Outside function
"""
        file_path = self.create_test_file("test.py", content)

        result = self.auto_fixer.fix_file_safely(file_path, dry_run=True)

        # Should identify the syntax error
        self.assertFalse(result.syntax_valid_before)

        # Original file should be unchanged in dry-run
        with open(file_path, "r") as f:
            unchanged_content = f.read()
        self.assertEqual(unchanged_content, content)

    def test_fix_file_safely_live_mode(self):
        """Test safe file fixing in live mode."""
        content = """def test():
    pass

return "test"  # Outside function
"""
        file_path = self.create_test_file("test.py", content)

        result = self.auto_fixer.fix_file_safely(file_path, dry_run=False)

        # Should have created a backup
        self.assertIsNotNone(result.backup_path)
        self.assertTrue(Path(result.backup_path).exists())

        # Should have made changes
        self.assertGreater(len(result.changes_made), 0)

    def test_no_corruption_on_valid_file(self):
        """Test that valid files are not corrupted."""
        content = '''def hello_world():
    """Say hello to the world."""
    print("Hello, World!")
    return "success"

def main():
    result = hello_world()
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
'''
        file_path = self.create_test_file("valid.py", content)

        # Verify it's initially valid
        is_valid_before, _ = self.auto_fixer.validate_python_syntax(file_path)
        self.assertTrue(is_valid_before)

        # Apply fixes
        result = self.auto_fixer.fix_file_safely(file_path, dry_run=False)

        # Should still be valid after fixes
        is_valid_after, _ = self.auto_fixer.validate_python_syntax(file_path)
        self.assertTrue(is_valid_after)
        self.assertTrue(result.syntax_valid_after)

    def test_rollback_on_corruption(self):
        """Test that files are rolled back if corruption is detected."""
        # This is a bit tricky to test since our fixes should be safe
        # But we can test the rollback mechanism
        content = """def test():
    return "valid"
"""
        file_path = self.create_test_file("test.py", content)

        # Manually create a backup
        backup_path = self.auto_fixer.create_backup(file_path)

        # Corrupt the file manually
        with open(file_path, "w") as f:
            f.write("def test(\n    return invalid syntax")

        # Attempt to restore
        success = self.auto_fixer.restore_from_backup(file_path)
        self.assertTrue(success)

        # Verify it's restored to valid state
        is_valid, _ = self.auto_fixer.validate_python_syntax(file_path)
        self.assertTrue(is_valid)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete framework."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.src_dir = self.temp_dir / "src"
        self.src_dir.mkdir(parents=True)
        self.auto_fixer = SafeLintingAutoFix(root_path=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        """Create a set of test files with various issues."""
        # Valid file
        valid_content = '''def hello():
    """Say hello."""
    return "hello"
'''
        (self.src_dir / "valid.py").write_text(valid_content)

        # File with return outside function
        invalid_content = """def test():
    pass

return "outside"  # Problem!
"""
        (self.src_dir / "invalid.py").write_text(invalid_content)

        # File with formatting issues (but valid syntax)
        formatting_content = """def test( ):
    x=1+2
    return x
"""
        (self.src_dir / "formatting.py").write_text(formatting_content)

    def test_full_workflow_dry_run(self):
        """Test the complete workflow in dry-run mode."""
        self.create_test_files()

        report = self.auto_fixer.run_safe_auto_fix(dry_run=True, max_files=5)

        # Should have processed files
        self.assertGreater(report.total_files, 0)
        self.assertEqual(report.files_processed, report.total_files)

        # Should identify syntax errors
        self.assertGreater(len(report.files_with_syntax_errors), 0)

        # No files should be corrupted in dry-run
        self.assertEqual(len(report.corrupted_files), 0)

    def test_full_workflow_live_mode(self):
        """Test the complete workflow in live mode."""
        self.create_test_files()

        report = self.auto_fixer.run_safe_auto_fix(dry_run=False, max_files=5)

        # Should have processed files
        self.assertGreater(report.total_files, 0)

        # Should have made some fixes
        self.assertGreater(report.fixes_applied, 0)

        # Most importantly: NO CORRUPTED FILES
        self.assertEqual(
            len(report.corrupted_files),
            0,
            f"CRITICAL: Files were corrupted: {report.corrupted_files}",
        )


def run_safety_validation():
    """Run comprehensive safety validation."""
    print("🧪 Running Safety Validation Tests...")
    print("=" * 60)

    # Create a test loader
    loader = unittest.TestLoader()

    # Load all tests
    test_suite = unittest.TestSuite()
    test_suite.addTests(loader.loadTestsFromTestCase(TestSafeAutoFix))
    test_suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)

    # Report results
    print("\n" + "=" * 60)
    print("🔍 SAFETY VALIDATION RESULTS")
    print("=" * 60)
    print(
        f"✅ Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}"
    )
    print(f"❌ Tests failed: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")

    if result.failures:
        print("\n🚨 FAILURES:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback}")

    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback}")

    # Overall safety assessment
    if result.wasSuccessful():
        print("\n🛡️ SAFETY ASSESSMENT: ✅ FRAMEWORK IS SAFE TO USE")
        return True
    else:
        print("\n🚨 SAFETY ASSESSMENT: ❌ FRAMEWORK IS NOT SAFE - DO NOT USE")
        return False


if __name__ == "__main__":
    import sys

    # Run safety validation
    is_safe = run_safety_validation()

    if not is_safe:
        print("\n⚠️  Framework failed safety tests. Please fix issues before use.")
        sys.exit(1)
    else:
        print("\n🎉 Framework passed all safety tests!")
        sys.exit(0)
