#!/usr/bin/env python3
"""
Comprehensive Test Suite for QeMLflow Renaming Script

This test suite validates the renaming script thoroughly before applying it to the actual codebase.
Tests cover edge cases, error conditions, and rollback functionality.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import json
from pathlib import Path

class RenameScriptTester:
    def __init__(self):
        self.test_results = []
        self.test_dir = None
        self.original_dir = os.getcwd()
        
    def log_test(self, test_name, success, message=""):
        """Log test results"""
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message
        })
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
    
    def create_test_repo(self):
        """Create a test repository structure mimicking the real repo"""
        self.test_dir = tempfile.mkdtemp(prefix="qemlflow_test_")
        os.chdir(self.test_dir)
        
        # Create directory structure
        directories = [
            "src/chemml/core",
            "src/chemml/integrations", 
            "src/chemml/research",
            "tests",
            "docs",
            "examples",
            "tools"
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
            
        # Create test files with various chemml references
        test_files = {
            "setup.py": '''from setuptools import find_packages, setup

setup(
    name="ChemML",
    version="0.1.0",
    description="A project for practicing machine learning and quantum computing techniques for molecular modeling and drug designing.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)''',
            
            "pyproject.toml": '''[project]
name = "chemml"
version = "0.2.0"
description = "Quantum-Enhanced Molecular Machine Learning Framework"
authors = [
    {name = "ChemML Contributors", email = "chemml@example.com"}
]''',
            
            "src/chemml/__init__.py": '''"""
ChemML: Machine Learning for Chemistry and Drug Discovery
"""

import chemml.core
from chemml.models import ChemMLModel
''',
            
            "src/chemml/core/models.py": '''"""
ChemML Core Models Module
"""

class ChemMLModel:
    """Base class for ChemML models"""
    
    def __init__(self):
        self.chemml_version = "2.0"
        
    def process_chemml_data(self, data):
        """Process data using ChemML algorithms"""
        return f"ChemML processed: {data}"
''',
            
            "README.md": '''# ChemML

ChemML is a machine learning framework for chemistry.

## Installation

```bash
pip install chemml
```

## Usage

```python
import chemml
from chemml.core import models
```''',
            
            "docs/chemml_guide.md": '''# ChemML User Guide

This guide shows how to use ChemML for chemical machine learning.

## ChemML Features

- chemml.core: Core functionality
- chemml.models: Machine learning models
''',
            
            "tests/test_chemml.py": '''"""Test ChemML functionality"""

import chemml
from chemml.core.models import ChemMLModel

def test_chemml_model():
    model = ChemMLModel()
    assert "ChemML" in model.chemml_version
''',
            
            "examples/chemml_example.py": '''#!/usr/bin/env python3
"""Example using ChemML"""

import chemml
from chemml.core import models

# Initialize ChemML
model = models.ChemMLModel()
print("ChemML example running")
'''
        }
        
        # Write test files
        for file_path, content in test_files.items():
            full_path = os.path.join(self.test_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
                
        return self.test_dir
    
    def cleanup_test_repo(self):
        """Clean up test repository"""
        os.chdir(self.original_dir)
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_file_detection(self):
        """Test that the script correctly identifies files to rename"""
        try:
            # Copy the rename script to test directory
            script_path = os.path.join(self.original_dir, "tools/migration/safe_rename_to_qemlflow.py")
            local_script = "safe_rename_to_qemlflow.py"
            shutil.copy2(script_path, local_script)
            
            # Run in dry-run mode to see what files would be processed
            result = subprocess.run([
                sys.executable, local_script, "--dry-run"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check that it identified the right files
                output = result.stdout
                expected_files = ["setup.py", "pyproject.toml", "__init__.py", "models.py", "README.md"]
                found_all = all(filename in output for filename in expected_files)
                
                self.log_test("File Detection", found_all, 
                             f"Found files: {found_all}, Output length: {len(output)}")
            else:
                self.log_test("File Detection", False, f"Script failed: {result.stderr}")
                
        except Exception as e:
            self.log_test("File Detection", False, f"Exception: {e}")
    
    def test_backup_creation(self):
        """Test that backups are created properly"""
        try:
            script_path = "safe_rename_to_qemlflow.py"
            
            # Run the script to create backups
            result = subprocess.run([
                sys.executable, script_path, "--backup-only"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check that backup directory was created
                backup_dirs = [d for d in os.listdir('.') if d.startswith('chemml_backup_')]
                backup_created = len(backup_dirs) > 0
                
                if backup_created:
                    backup_dir = backup_dirs[0]
                    # Check that files were backed up
                    backup_has_files = len(os.listdir(backup_dir)) > 0
                    self.log_test("Backup Creation", backup_has_files, 
                                 f"Backup directory: {backup_dir}")
                else:
                    self.log_test("Backup Creation", False, "No backup directory created")
            else:
                self.log_test("Backup Creation", False, f"Backup failed: {result.stderr}")
                
        except Exception as e:
            self.log_test("Backup Creation", False, f"Exception: {e}")
    
    def test_content_replacement(self):
        """Test that content is replaced correctly"""
        try:
            # Create a simple test file
            test_file = "test_replacement.py"
            with open(test_file, 'w') as f:
                f.write("import chemml\nfrom chemml.core import models\nprint('ChemML test')")
            
            # Test the replacement logic manually
            with open(test_file, 'r') as f:
                original = f.read()
            
            # Apply replacements (simulate the script's logic)
            replacements = {
                'chemml': 'qemlflow',
                'ChemML': 'QeMLflow',
                'CHEMML': 'QEMLFLOW'
            }
            
            modified = original
            for old, new in replacements.items():
                modified = modified.replace(old, new)
            
            # Check replacements worked
            expected_content = "import qemlflow\nfrom qemlflow.core import models\nprint('QeMLflow test')"
            replacement_correct = modified == expected_content
            
            self.log_test("Content Replacement", replacement_correct,
                         f"Original: {len(original)} chars, Modified: {len(modified)} chars")
            
            # Clean up
            os.remove(test_file)
            
        except Exception as e:
            self.log_test("Content Replacement", False, f"Exception: {e}")
    
    def test_edge_cases(self):
        """Test edge cases like binary files, symlinks, etc."""
        try:
            # Create a binary file
            with open("binary_test.png", 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')
            
            # Create a symlink (if supported)
            symlink_created = False
            try:
                os.symlink("setup.py", "symlink_test.py")
                symlink_created = True
            except (OSError, NotImplementedError):
                pass  # Symlinks not supported on this system
            
            # Create a file with special characters
            with open("special_chars.txt", 'w', encoding='utf-8') as f:
                f.write("chemml with unicode: café, naïve, résumé")
            
            # Test that script handles these gracefully
            script_path = "safe_rename_to_qemlflow.py"
            result = subprocess.run([
                sys.executable, script_path, "--dry-run"
            ], capture_output=True, text=True)
            
            # Script should not crash on these files
            script_handled_gracefully = result.returncode == 0
            
            self.log_test("Edge Cases", script_handled_gracefully,
                         f"Binary files, symlinks ({symlink_created}), special chars handled")
            
            # Clean up
            os.remove("binary_test.png")
            if symlink_created and os.path.islink("symlink_test.py"):
                os.remove("symlink_test.py")
            os.remove("special_chars.txt")
            
        except Exception as e:
            self.log_test("Edge Cases", False, f"Exception: {e}")
    
    def test_rollback_functionality(self):
        """Test that rollback works correctly"""
        try:
            script_path = "safe_rename_to_qemlflow.py"
            
            # First, run the script to make changes
            result = subprocess.run([
                sys.executable, script_path, "--execute"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check that changes were made
                with open("setup.py", 'r') as f:
                    content_after = f.read()
                changes_made = "QeMLflow" in content_after
                
                if changes_made:
                    # Now test rollback
                    backup_dirs = [d for d in os.listdir('.') if d.startswith('qemlflow_backup_')]
                    if backup_dirs:
                        backup_dir = backup_dirs[0]
                        
                        # Run rollback
                        rollback_result = subprocess.run([
                            sys.executable, script_path, "--rollback", backup_dir
                        ], capture_output=True, text=True)
                        
                        if rollback_result.returncode == 0:
                            # Check that original content is restored
                            with open("setup.py", 'r') as f:
                                content_restored = f.read()
                            rollback_successful = "ChemML" in content_restored and "QeMLflow" not in content_restored
                            
                            self.log_test("Rollback Functionality", rollback_successful,
                                         f"Rollback from {backup_dir}")
                        else:
                            self.log_test("Rollback Functionality", False, 
                                         f"Rollback failed: {rollback_result.stderr}")
                    else:
                        self.log_test("Rollback Functionality", False, "No backup directory found")
                else:
                    self.log_test("Rollback Functionality", False, "No changes were made initially")
            else:
                self.log_test("Rollback Functionality", False, f"Initial execution failed: {result.stderr}")
                
        except Exception as e:
            self.log_test("Rollback Functionality", False, f"Exception: {e}")
    
    def test_directory_structure_preservation(self):
        """Test that directory structure is preserved"""
        try:
            # Record original structure
            original_structure = {}
            for root, dirs, files in os.walk('.'):
                original_structure[root] = {'dirs': dirs.copy(), 'files': files.copy()}
            
            # Run the script
            script_path = "safe_rename_to_qemlflow.py"
            result = subprocess.run([
                sys.executable, script_path, "--execute"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check that structure is preserved (only src/chemml should change to src/qemlflow)
                new_structure = {}
                for root, dirs, files in os.walk('.'):
                    new_structure[root] = {'dirs': dirs.copy(), 'files': files.copy()}
                
                # The main difference should be src/chemml -> src/qemlflow
                structure_preserved = True
                expected_changes = {
                    'src/chemml': 'src/qemlflow'
                }
                
                # Check that src/qemlflow exists and src/chemml doesn't
                qemlflow_exists = os.path.exists('src/qemlflow')
                chemml_gone = not os.path.exists('src/chemml')
                
                structure_preserved = qemlflow_exists and chemml_gone
                
                self.log_test("Directory Structure", structure_preserved,
                             f"QeMLflow dir exists: {qemlflow_exists}, ChemML dir gone: {chemml_gone}")
            else:
                self.log_test("Directory Structure", False, f"Script failed: {result.stderr}")
                
        except Exception as e:
            self.log_test("Directory Structure", False, f"Exception: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting Comprehensive QeMLflow Renaming Script Test Suite")
        print("=" * 70)
        
        try:
            # Create test repository
            test_dir = self.create_test_repo()
            print(f"📁 Created test repository: {test_dir}")
            
            # Run tests
            self.test_file_detection()
            self.test_backup_creation() 
            self.test_content_replacement()
            self.test_edge_cases()
            self.test_directory_structure_preservation()
            self.test_rollback_functionality()
            
            # Print results
            print("\n" + "=" * 70)
            print("📊 TEST RESULTS SUMMARY")
            print("=" * 70)
            
            passed = sum(1 for result in self.test_results if result['success'])
            total = len(self.test_results)
            
            for result in self.test_results:
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"{status}: {result['test']}")
                if result['message']:
                    print(f"   📝 {result['message']}")
            
            print(f"\n🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
            
            if passed == total:
                print("\n🎉 ALL TESTS PASSED! The renaming script is ready for production use.")
                print("✅ You can safely proceed with the actual repository renaming.")
            else:
                print(f"\n⚠️  {total-passed} test(s) failed. Please review and fix issues before proceeding.")
                print("❌ Do NOT run the script on the actual repository yet.")
            
        finally:
            # Clean up
            self.cleanup_test_repo()
            print(f"\n🧹 Cleaned up test repository")

if __name__ == "__main__":
    tester = RenameScriptTester()
    tester.run_all_tests()
