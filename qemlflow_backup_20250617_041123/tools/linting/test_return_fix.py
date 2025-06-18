#!/usr/bin/env python3
"""
Test Suite for Return Outside Function Fixer
============================================

Tests for the focused syntax error fixer.
"""

import tempfile
import unittest
from pathlib import Path
import sys
import os

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from return_fix import ReturnOutsideFunctionFixer


class TestReturnFixer(unittest.TestCase):
    """Test suite for ReturnOutsideFunctionFixer."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fixer = ReturnOutsideFunctionFixer(
            create_backups=False
        )  # No backups for tests

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
    return "Hello, World!"
'''
        file_path = self.create_test_file("valid.py", content)
        is_valid, error = self.fixer.validate_syntax(file_path)

        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_syntax_validation_invalid_file(self):
        """Test syntax validation with invalid Python file."""
        content = '''
def hello_world():
    """Say hello to the world."""
    pass

return "success"  # This is outside the function!
'''
        file_path = self.create_test_file("invalid.py", content)
        is_valid, error = self.fixer.validate_syntax(file_path)

        self.assertFalse(is_valid)
        self.assertIn("'return' outside function", error)

    def test_fix_return_outside_function_simple(self):
        """Test fixing simple return outside function."""
        content = '''def test_function():
    """Test function."""
    pass

return "hello"
'''
        fixed_content, changes = self.fixer.fix_return_outside_function(content)

        self.assertEqual(len(changes), 1)
        self.assertIn("Fixed return statement indentation", changes[0])
        self.assertIn('    return "hello"', fixed_content)

    def test_fix_return_outside_function_orphaned(self):
        """Test fixing orphaned return (no previous function)."""
        content = """# Some module
import os

return "orphaned"
"""
        fixed_content, changes = self.fixer.fix_return_outside_function(content)

        self.assertEqual(len(changes), 1)
        self.assertIn("Commented out orphaned return", changes[0])
        self.assertIn(
            '# TODO: Fix orphaned return statement: return "orphaned"', fixed_content
        )

    def test_fix_multiple_returns(self):
        """Test fixing multiple return statements outside functions."""
        content = """def func1():
    pass

return "first"

def func2():
    pass

return "second"
"""
        fixed_content, changes = self.fixer.fix_return_outside_function(content)

        self.assertEqual(len(changes), 2)
        lines = fixed_content.split("\n")
        # Both returns should be properly indented
        self.assertIn('    return "first"', fixed_content)
        self.assertIn('    return "second"', fixed_content)

    def test_fix_file_dry_run(self):
        """Test fixing file in dry-run mode."""
        content = """def test():
    pass

return "test"
"""
        file_path = self.create_test_file("test.py", content)

        success, changes, message = self.fixer.fix_file(file_path, dry_run=True)

        self.assertTrue(success)
        self.assertEqual(len(changes), 1)
        self.assertIn("Dry run", message)

        # File should be unchanged
        with open(file_path, "r") as f:
            unchanged_content = f.read()
        self.assertEqual(unchanged_content, content)

    def test_fix_file_live_mode(self):
        """Test fixing file in live mode."""
        content = """def test():
    pass

return "test"
"""
        file_path = self.create_test_file("test.py", content)

        success, changes, message = self.fixer.fix_file(file_path, dry_run=False)

        self.assertTrue(success)
        self.assertEqual(len(changes), 1)
        self.assertIn("successfully", message)

        # File should be changed and syntactically valid
        is_valid, error = self.fixer.validate_syntax(file_path)
        self.assertTrue(is_valid, f"Fixed file should be valid: {error}")

        with open(file_path, "r") as f:
            fixed_content = f.read()
        self.assertIn('    return "test"', fixed_content)

    def test_no_changes_needed(self):
        """Test file that doesn't need changes."""
        content = """def test():
    return "test"
"""
        file_path = self.create_test_file("good.py", content)

        success, changes, message = self.fixer.fix_file(file_path, dry_run=False)

        self.assertTrue(success)
        self.assertEqual(len(changes), 0)
        self.assertIn("No changes needed", message)

    def test_is_inside_function(self):
        """Test the _is_inside_function helper method."""
        lines = [
            "def test():",
            "    pass",
            '    return "inside"',
            "",
            'return "outside"',
        ]

        # Line 2 (return "inside") should be inside function
        self.assertTrue(self.fixer._is_inside_function(lines, 2))

        # Line 4 (return "outside") should not be inside function
        self.assertFalse(self.fixer._is_inside_function(lines, 4))

    def test_find_previous_function(self):
        """Test the _find_previous_function helper method."""
        lines = [
            "import os",
            "def func1():",
            "    pass",
            "",
            "def func2():",
            "    pass",
            "",
            'return "after func2"',
        ]

        # Line 7 should find func2 (line 4)
        func_line = self.fixer._find_previous_function(lines, 7)
        self.assertEqual(func_line, 4)

        # Line 1 should find no function
        func_line = self.fixer._find_previous_function(lines, 1)
        self.assertIsNone(func_line)


class TestRealWorldScenario(unittest.TestCase):
    """Test with the actual problematic pattern found in ChemML."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fixer = ReturnOutsideFunctionFixer(create_backups=False)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file with given content."""
        file_path = self.temp_dir / filename
        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    def test_chemml_adapter_pattern(self):
        """Test the exact pattern found in ChemML adapters/__init__.py."""
        content = '''def list_adapters_by_category(category: str):
    """List available adapters for a specific category."""
    return ADAPTER_CATEGORIES.get(category, [])


def list_all_categories():
    """List all available adapter categories."""


return list(ADAPTER_CATEGORIES.keys())


def discover_models_by_category(category: str):
    """Discover available models by scientific category."""


return list_adapters_by_category(category)
'''

        file_path = self.create_test_file("adapter_test.py", content)

        # First verify it has syntax errors
        is_valid_before, error_before = self.fixer.validate_syntax(file_path)
        self.assertFalse(is_valid_before)
        self.assertIn("'return' outside function", error_before)

        # Fix the file
        success, changes, message = self.fixer.fix_file(file_path, dry_run=False)

        self.assertTrue(success, f"Fix should succeed: {message}")
        self.assertGreater(len(changes), 0, "Should have made changes")

        # Verify it's now syntactically valid
        is_valid_after, error_after = self.fixer.validate_syntax(file_path)
        self.assertTrue(is_valid_after, f"Fixed file should be valid: {error_after}")

        # Check the fixed content
        with open(file_path, "r") as f:
            fixed_content = f.read()

        # Should have proper indentation for both returns
        self.assertIn("    return list(ADAPTER_CATEGORIES.keys())", fixed_content)
        self.assertIn("    return list_adapters_by_category(category)", fixed_content)


if __name__ == "__main__":
    unittest.main()
