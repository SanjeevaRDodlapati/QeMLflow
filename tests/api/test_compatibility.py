"""
Test suite for API compatibility functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.qemlflow.api.compatibility import (
    APISignature,
    APIAnalyzer,
    APICompatibilityChecker,
    APISnapshot
)


class TestAPISignature:
    """Test API signature representation."""
    
    def test_signature_creation(self):
        """Test creating API signature."""
        sig = APISignature(
            name="test_function",
            type="function",
            parameters=[{"name": "arg1", "type": "str"}],
            return_type="bool"
        )
        
        assert sig.name == "test_function"
        assert sig.type == "function"
        assert sig.is_public is True
        assert len(sig.parameters) == 1
    
    def test_private_function_detection(self):
        """Test detection of private functions."""
        private_sig = APISignature(name="_private_func", type="function")
        public_sig = APISignature(name="public_func", type="function")
        dunder_sig = APISignature(name="__init__", type="method")
        
        assert private_sig.is_public is False
        assert public_sig.is_public is True
        assert dunder_sig.is_public is True  # Dunder methods are considered public
    
    def test_signature_serialization(self):
        """Test signature to/from dictionary conversion."""
        sig = APISignature(
            name="test_method",
            type="method",
            parameters=[{"name": "self"}, {"name": "value", "type": "int"}],
            return_type="None",
            docstring="Test method"
        )
        
        sig_dict = sig.to_dict()
        assert sig_dict["name"] == "test_method"
        assert sig_dict["type"] == "method"
        assert len(sig_dict["parameters"]) == 2
        
        # Test round-trip conversion
        restored_sig = APISignature.from_dict(sig_dict)
        assert restored_sig.name == sig.name
        assert restored_sig.type == sig.type
        assert restored_sig.parameters == sig.parameters


class TestAPIAnalyzer:
    """Test API analysis functionality."""
    
    @pytest.fixture
    def sample_module_code(self):
        """Sample Python module code for testing."""
        return '''
"""Sample module for testing."""

def public_function(arg1: str, arg2: int = 10) -> bool:
    """A public function."""
    return True

def _private_function():
    """A private function."""
    pass

class SampleClass:
    """A sample class."""
    
    def __init__(self, value: int):
        self.value = value
    
    def public_method(self) -> int:
        """A public method."""
        return self.value
    
    def _private_method(self):
        """A private method."""
        pass

class _PrivateClass:
    """A private class."""
    pass
'''
    
    def test_module_analysis(self, sample_module_code):
        """Test analyzing a Python module."""
        analyzer = APIAnalyzer()
        
        # Mock file reading
        with patch("builtins.open", mock_open(read_data=sample_module_code)):
            signatures = analyzer.analyze_module("sample.py", "sample")
        
        # Should find public functions and classes
        assert "public_function" in signatures
        assert "SampleClass" in signatures
        assert "SampleClass.public_method" in signatures
        
        # Check function signature details
        func_sig = signatures["public_function"]
        assert func_sig.type == "function"
        assert len(func_sig.parameters) == 2
        assert func_sig.parameters[0]["name"] == "arg1"
        assert func_sig.parameters[1]["name"] == "arg2"
        assert func_sig.parameters[1]["default"] == "10"
    
    def test_class_analysis(self, sample_module_code):
        """Test analyzing class definitions."""
        analyzer = APIAnalyzer()
        
        with patch("builtins.open", mock_open(read_data=sample_module_code)):
            signatures = analyzer.analyze_module("sample.py", "sample")
        
        # Check class signature
        class_sig = signatures["SampleClass"]
        assert class_sig.type == "class"
        assert class_sig.docstring == "A sample class."
        
        # Check method signature
        method_sig = signatures["SampleClass.public_method"]
        assert method_sig.type == "method"
        assert method_sig.return_type == "int"


class TestAPICompatibilityChecker:
    """Test API compatibility checking."""
    
    def test_no_changes(self):
        """Test when APIs are identical."""
        old_api = {
            "func1": APISignature("func1", "function", [{"name": "arg1"}])
        }
        new_api = {
            "func1": APISignature("func1", "function", [{"name": "arg1"}])
        }
        
        checker = APICompatibilityChecker()
        changes = checker.compare_apis(old_api, new_api)
        
        assert len(changes) == 0
    
    def test_added_function(self):
        """Test detection of added functions."""
        old_api = {}
        new_api = {
            "new_func": APISignature("new_func", "function", [])
        }
        
        checker = APICompatibilityChecker()
        changes = checker.compare_apis(old_api, new_api)
        
        assert len(changes) == 1
        change = changes[0]
        assert change.change_type == "added"
        assert change.element_name == "new_func"
        assert change.breaking is False
    
    def test_removed_function(self):
        """Test detection of removed functions."""
        old_api = {
            "old_func": APISignature("old_func", "function", [])
        }
        new_api = {}
        
        checker = APICompatibilityChecker()
        changes = checker.compare_apis(old_api, new_api)
        
        assert len(changes) == 1
        change = changes[0]
        assert change.change_type == "removed"
        assert change.element_name == "old_func"
        assert change.breaking is True  # Removing public API is breaking
    
    def test_modified_function_signature(self):
        """Test detection of modified function signatures."""
        old_api = {
            "func": APISignature(
                "func", "function", 
                [{"name": "arg1", "annotation": "str"}],
                return_type="int"
            )
        }
        new_api = {
            "func": APISignature(
                "func", "function",
                [{"name": "arg1", "annotation": "str"}, {"name": "arg2", "annotation": "int"}],
                return_type="int"
            )
        }
        
        checker = APICompatibilityChecker()
        changes = checker.compare_apis(old_api, new_api)
        
        assert len(changes) == 1
        change = changes[0]
        assert change.change_type == "modified"
        assert change.element_name == "func"
    
    def test_breaking_change_detection(self):
        """Test detection of breaking changes."""
        old_func = APISignature(
            "func", "function",
            [{"name": "arg1", "annotation": "str"}, {"name": "arg2", "annotation": "int", "default": "10"}],
            return_type="bool"
        )
        
        # Remove required parameter (breaking)
        new_func = APISignature(
            "func", "function",
            [{"name": "arg2", "annotation": "int", "default": "10"}],
            return_type="bool"
        )
        
        checker = APICompatibilityChecker()
        is_breaking = checker._is_breaking_change(old_func, new_func)
        
        assert is_breaking is True
    
    def test_non_breaking_change_detection(self):
        """Test detection of non-breaking changes."""
        old_func = APISignature(
            "func", "function",
            [{"name": "arg1", "annotation": "str"}],
            return_type="bool"
        )
        
        # Add optional parameter (non-breaking)
        new_func = APISignature(
            "func", "function",
            [{"name": "arg1", "annotation": "str"}, {"name": "arg2", "annotation": "int", "default": "10"}],
            return_type="bool"
        )
        
        checker = APICompatibilityChecker()
        is_breaking = checker._is_breaking_change(old_func, new_func)
        
        assert is_breaking is False
    
    def test_compatibility_report_generation(self):
        """Test compatibility report generation."""
        old_api = {
            "func1": APISignature("func1", "function", []),
            "func2": APISignature("func2", "function", [])
        }
        new_api = {
            "func1": APISignature("func1", "function", [{"name": "new_arg", "default": "None"}]),  # Modified
            "func3": APISignature("func3", "function", [])  # Added
        }
        
        checker = APICompatibilityChecker()
        changes = checker.compare_apis(old_api, new_api)
        report = checker.generate_compatibility_report(changes)
        
        assert report["total_changes"] == 3  # 1 modified, 1 removed, 1 added
        assert report["summary"]["added"] == 1
        assert report["summary"]["removed"] == 1
        assert report["summary"]["modified"] == 1
        assert "compatibility_level" in report


class TestAPISnapshot:
    """Test API snapshot functionality."""
    
    def test_snapshot_creation(self, tmp_path):
        """Test creating API snapshots."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot = APISnapshot(str(snapshot_dir))
        
        # Create a sample module file
        sample_module = tmp_path / "sample.py"
        sample_module.write_text("""
def sample_function():
    pass
""")
        
        # Create snapshot
        snapshot_file = snapshot.create_snapshot(
            "1.0.0", 
            [str(sample_module)],
            "Test snapshot"
        )
        
        assert Path(snapshot_file).exists()
        
        # Verify snapshot content
        snapshot_data = snapshot.load_snapshot("1.0.0")
        assert snapshot_data is not None
        assert snapshot_data["version"] == "1.0.0"
        assert "sample" in snapshot_data["modules"]
    
    def test_snapshot_comparison(self, tmp_path):
        """Test comparing API snapshots."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot = APISnapshot(str(snapshot_dir))
        
        # Create two different module versions
        module_v1 = tmp_path / "module_v1.py"
        module_v1.write_text("""
def old_function():
    pass
""")
        
        module_v2 = tmp_path / "module_v2.py"
        module_v2.write_text("""
def old_function():
    pass

def new_function():
    pass
""")
        
        # Create snapshots
        snapshot.create_snapshot("1.0.0", [str(module_v1)], "Version 1")
        snapshot.create_snapshot("1.1.0", [str(module_v2)], "Version 2")
        
        # Compare snapshots
        comparison = snapshot.compare_snapshots("1.0.0", "1.1.0")
        
        assert comparison is not None
        assert comparison["total_changes"] == 1
        assert comparison["summary"]["added"] == 1
    
    def test_snapshot_listing(self, tmp_path):
        """Test listing available snapshots."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot = APISnapshot(str(snapshot_dir))
        
        # Create multiple snapshots
        sample_module = tmp_path / "sample.py"
        sample_module.write_text("def func(): pass")
        
        snapshot.create_snapshot("1.0.0", [str(sample_module)], "First")
        snapshot.create_snapshot("1.1.0", [str(sample_module)], "Second")
        
        # List snapshots
        snapshots = snapshot.list_snapshots()
        
        assert len(snapshots) == 2
        versions = [s["version"] for s in snapshots]
        assert "1.0.0" in versions
        assert "1.1.0" in versions


if __name__ == "__main__":
    pytest.main([__file__])
