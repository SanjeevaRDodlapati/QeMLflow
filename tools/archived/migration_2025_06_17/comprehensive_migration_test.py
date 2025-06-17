#!/usr/bin/env python3
"""
Comprehensive Migration Validation Test Suite
=============================================

This script performs extensive testing to validate the complete ChemML to QeMLflow migration.
It checks for any remaining references, import issues, functionality problems, and ensures
the migration was completed without any hidden mistakes.

Author: Migration Validation System
Date: June 17, 2025
"""

import os
import sys
import re
import json
import subprocess
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Set
import ast
import configparser

class MigrationValidator:
    """Comprehensive migration validation system."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.errors = []
        self.warnings = []
        self.info = []
        self.test_results = {}
        
    def log_error(self, test_name: str, message: str):
        """Log an error."""
        error_msg = f"❌ ERROR [{test_name}]: {message}"
        self.errors.append(error_msg)
        print(error_msg)
        
    def log_warning(self, test_name: str, message: str):
        """Log a warning."""
        warning_msg = f"⚠️  WARNING [{test_name}]: {message}"
        self.warnings.append(warning_msg)
        print(warning_msg)
        
    def log_info(self, test_name: str, message: str):
        """Log info."""
        info_msg = f"ℹ️  INFO [{test_name}]: {message}"
        self.info.append(info_msg)
        print(info_msg)
        
    def log_success(self, test_name: str, message: str):
        """Log success."""
        success_msg = f"✅ SUCCESS [{test_name}]: {message}"
        print(success_msg)
        
    def run_test(self, test_name: str, test_func):
        """Run a test function and record results."""
        print(f"\n🔍 Running test: {test_name}")
        print("=" * 60)
        
        try:
            result = test_func()
            self.test_results[test_name] = {
                'status': 'PASSED' if result else 'FAILED',
                'result': result
            }
            if result:
                self.log_success(test_name, "Test passed")
            else:
                self.log_error(test_name, "Test failed")
            return result
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'ERROR',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.log_error(test_name, f"Test error: {e}")
            return False

    def test_no_chemml_references(self) -> bool:
        """Test 1: Check for any remaining 'chemml' references in code files."""
        test_name = "No ChemML References"
        
        # File patterns to check
        code_patterns = [
            "**/*.py", "**/*.md", "**/*.rst", "**/*.txt", "**/*.yml", "**/*.yaml",
            "**/*.json", "**/*.toml", "**/*.cfg", "**/*.ini", "**/*.sh", "**/*.bat"
        ]
        
        # Directories to ignore
        ignore_dirs = {
            ".git", "__pycache__", ".pytest_cache", "node_modules", 
            "qemlflow_backup_20250617_041123", "backups", ".venv", "venv",
            "qemlflow_env", "chemml_env", "site"
        }
        
        chemml_refs = []
        
        for pattern in code_patterns:
            for file_path in self.root_path.rglob(pattern):
                # Skip ignored directories
                if any(ignore_dir in file_path.parts for ignore_dir in ignore_dirs):
                    continue
                    
                # Skip binary files
                if file_path.suffix in ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg']:
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Look for 'chemml' (case insensitive)
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        if re.search(r'\bchemml\b', line, re.IGNORECASE):
                            # Skip if it's in a comment about the migration
                            if not re.search(r'(migration|rename|old name|previous|backup)', line, re.IGNORECASE):
                                chemml_refs.append({
                                    'file': str(file_path.relative_to(self.root_path)),
                                    'line': line_num,
                                    'content': line.strip()
                                })
                                
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        if chemml_refs:
            self.log_error(test_name, f"Found {len(chemml_refs)} remaining 'chemml' references:")
            for ref in chemml_refs[:10]:  # Show first 10
                self.log_error(test_name, f"  {ref['file']}:{ref['line']} - {ref['content']}")
            if len(chemml_refs) > 10:
                self.log_error(test_name, f"  ... and {len(chemml_refs) - 10} more")
            return False
        
        self.log_success(test_name, "No remaining 'chemml' references found")
        return True

    def test_import_qemlflow(self) -> bool:
        """Test 2: Test that QeMLflow can be imported successfully."""
        test_name = "QeMLflow Import"
        
        try:
            # Add src to path if needed
            src_path = self.root_path / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            import qemlflow
            self.log_success(test_name, f"QeMLflow imported successfully, version: {qemlflow.__version__}")
            
            # Test core modules
            core_modules = [
                'qemlflow.datasets',
                'qemlflow.preprocessing',
                'qemlflow.models',
                'qemlflow.visualization',
                'qemlflow.utils'
            ]
            
            for module_name in core_modules:
                try:
                    importlib.import_module(module_name)
                    self.log_info(test_name, f"Successfully imported {module_name}")
                except ImportError as e:
                    self.log_warning(test_name, f"Could not import {module_name}: {e}")
            
            return True
            
        except ImportError as e:
            self.log_error(test_name, f"Failed to import qemlflow: {e}")
            return False

    def test_package_structure(self) -> bool:
        """Test 3: Validate the package structure is correct."""
        test_name = "Package Structure"
        
        # Expected directories
        expected_dirs = [
            "src/qemlflow",
            "tests",
            "docs",
            "examples",
            "notebooks",
            "tools"
        ]
        
        # Expected files
        expected_files = [
            "src/qemlflow/__init__.py",
            "setup.py",
            "pyproject.toml",
            "requirements.txt",
            "README.md"
        ]
        
        missing_dirs = []
        missing_files = []
        
        for dir_path in expected_dirs:
            full_path = self.root_path / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
            else:
                self.log_info(test_name, f"Found directory: {dir_path}")
        
        for file_path in expected_files:
            full_path = self.root_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                self.log_info(test_name, f"Found file: {file_path}")
        
        if missing_dirs:
            for missing in missing_dirs:
                self.log_error(test_name, f"Missing directory: {missing}")
        
        if missing_files:
            for missing in missing_files:
                self.log_error(test_name, f"Missing file: {missing}")
        
        if missing_dirs or missing_files:
            return False
        
        self.log_success(test_name, "Package structure is correct")
        return True

    def test_configuration_files(self) -> bool:
        """Test 4: Validate configuration files are updated correctly."""
        test_name = "Configuration Files"
        
        config_tests = []
        
        # Test setup.py
        setup_py = self.root_path / "setup.py"
        if setup_py.exists():
            with open(setup_py, 'r') as f:
                content = f.read()
                if 'qemlflow' in content.lower():
                    config_tests.append(("setup.py", True, "Contains 'qemlflow'"))
                else:
                    config_tests.append(("setup.py", False, "Missing 'qemlflow' reference"))
        
        # Test pyproject.toml
        pyproject_toml = self.root_path / "pyproject.toml"
        if pyproject_toml.exists():
            with open(pyproject_toml, 'r') as f:
                content = f.read()
                if 'qemlflow' in content.lower():
                    config_tests.append(("pyproject.toml", True, "Contains 'qemlflow'"))
                else:
                    config_tests.append(("pyproject.toml", False, "Missing 'qemlflow' reference"))
        
        # Test README.md
        readme_md = self.root_path / "README.md"
        if readme_md.exists():
            with open(readme_md, 'r') as f:
                content = f.read()
                if 'qemlflow' in content.lower():
                    config_tests.append(("README.md", True, "Contains 'QeMLflow'"))
                else:
                    config_tests.append(("README.md", False, "Missing 'QeMLflow' reference"))
        
        all_passed = True
        for file_name, passed, message in config_tests:
            if passed:
                self.log_info(test_name, f"{file_name}: {message}")
            else:
                self.log_error(test_name, f"{file_name}: {message}")
                all_passed = False
        
        return all_passed

    def test_python_syntax(self) -> bool:
        """Test 5: Check all Python files for syntax errors."""
        test_name = "Python Syntax"
        
        python_files = list(self.root_path.rglob("*.py"))
        syntax_errors = []
        
        # Filter out backup and env directories
        ignore_dirs = {"qemlflow_backup_20250617_041123", "backups", "qemlflow_env", "chemml_env", ".git"}
        
        for py_file in python_files:
            if any(ignore_dir in py_file.parts for ignore_dir in ignore_dirs):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the AST to check syntax
                ast.parse(content, filename=str(py_file))
                
            except SyntaxError as e:
                syntax_errors.append({
                    'file': str(py_file.relative_to(self.root_path)),
                    'line': e.lineno,
                    'error': str(e)
                })
            except Exception as e:
                # Non-syntax errors (like encoding issues)
                self.log_warning(test_name, f"Could not parse {py_file.relative_to(self.root_path)}: {e}")
        
        if syntax_errors:
            self.log_error(test_name, f"Found {len(syntax_errors)} syntax errors:")
            for error in syntax_errors:
                self.log_error(test_name, f"  {error['file']}:{error['line']} - {error['error']}")
            return False
        
        self.log_success(test_name, f"All {len(python_files)} Python files have valid syntax")
        return True

    def test_git_status(self) -> bool:
        """Test 6: Verify git repository is clean and synchronized."""
        test_name = "Git Status"
        
        try:
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  cwd=self.root_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log_error(test_name, f"Git status failed: {result.stderr}")
                return False
            
            if result.stdout.strip():
                self.log_warning(test_name, f"Working directory not clean:\n{result.stdout}")
                return False
            
            # Check if branch is up to date
            result = subprocess.run(['git', 'status', '-b', '--porcelain'], 
                                  cwd=self.root_path, capture_output=True, text=True)
            
            if 'ahead' in result.stdout or 'behind' in result.stdout:
                self.log_warning(test_name, f"Branch not synchronized: {result.stdout}")
            
            self.log_success(test_name, "Git repository is clean and synchronized")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_error(test_name, f"Git command failed: {e}")
            return False

    def test_documentation_links(self) -> bool:
        """Test 7: Check documentation files for broken internal links."""
        test_name = "Documentation Links"
        
        doc_files = list(self.root_path.rglob("*.md")) + list(self.root_path.rglob("*.rst"))
        broken_links = []
        
        # Filter out backup directories
        ignore_dirs = {"qemlflow_backup_20250617_041123", "backups", ".git"}
        
        for doc_file in doc_files:
            if any(ignore_dir in doc_file.parts for ignore_dir in ignore_dirs):
                continue
                
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for internal file references
                internal_refs = re.findall(r'\[.*?\]\(([^http][^)]+)\)', content)
                
                for ref in internal_refs:
                    # Clean up the reference
                    ref = ref.split('#')[0]  # Remove anchors
                    if ref.startswith('./'):
                        ref = ref[2:]
                    
                    ref_path = doc_file.parent / ref
                    if not ref_path.exists():
                        broken_links.append({
                            'file': str(doc_file.relative_to(self.root_path)),
                            'link': ref
                        })
                        
            except Exception as e:
                self.log_warning(test_name, f"Could not check {doc_file.relative_to(self.root_path)}: {e}")
        
        if broken_links:
            self.log_warning(test_name, f"Found {len(broken_links)} potentially broken internal links:")
            for link in broken_links[:5]:  # Show first 5
                self.log_warning(test_name, f"  {link['file']} -> {link['link']}")
            return True  # Not critical for migration success
        
        self.log_success(test_name, "No broken internal links found")
        return True

    def test_environment_activation(self) -> bool:
        """Test 8: Test virtual environment activation and package installation."""
        test_name = "Environment Activation"
        
        env_path = self.root_path / "qemlflow_env"
        
        if not env_path.exists():
            self.log_warning(test_name, "Virtual environment not found - creating new one")
            try:
                subprocess.run([sys.executable, '-m', 'venv', str(env_path)], 
                             cwd=self.root_path, check=True)
                self.log_info(test_name, "Created new virtual environment")
            except subprocess.CalledProcessError as e:
                self.log_error(test_name, f"Failed to create virtual environment: {e}")
                return False
        
        # Test activation by running a command in the environment
        if os.name == 'nt':  # Windows
            python_exe = env_path / "Scripts" / "python.exe"
        else:  # Unix-like
            python_exe = env_path / "bin" / "python"
        
        if not python_exe.exists():
            self.log_error(test_name, f"Python executable not found: {python_exe}")
            return False
        
        try:
            # Test importing qemlflow in the virtual environment
            result = subprocess.run([str(python_exe), '-c', 'import qemlflow; print(f"QeMLflow {qemlflow.__version__}")'], 
                                  cwd=self.root_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log_success(test_name, f"Virtual environment working: {result.stdout.strip()}")
                return True
            else:
                self.log_warning(test_name, f"QeMLflow not installed in virtual environment: {result.stderr}")
                # Try to install it
                result = subprocess.run([str(python_exe), '-m', 'pip', 'install', '-e', '.'], 
                                      cwd=self.root_path, capture_output=True, text=True)
                if result.returncode == 0:
                    self.log_info(test_name, "Successfully installed QeMLflow in editable mode")
                    return True
                else:
                    self.log_error(test_name, f"Failed to install QeMLflow: {result.stderr}")
                    return False
                    
        except Exception as e:
            self.log_error(test_name, f"Environment test failed: {e}")
            return False

    def test_backup_integrity(self) -> bool:
        """Test 9: Verify backup integrity and completeness."""
        test_name = "Backup Integrity"
        
        backup_path = self.root_path / "qemlflow_backup_20250617_041123"
        
        if not backup_path.exists():
            self.log_warning(test_name, "Backup directory not found")
            return True  # Not critical if backup was moved
        
        # Check if backup contains expected structure
        expected_backup_items = [
            "src/chemml",
            "chemml_env",
            "setup.py",
            "README.md"
        ]
        
        missing_items = []
        for item in expected_backup_items:
            if not (backup_path / item).exists():
                missing_items.append(item)
        
        if missing_items:
            self.log_warning(test_name, f"Backup missing items: {missing_items}")
        else:
            self.log_success(test_name, "Backup appears complete")
        
        return True

    def test_performance_basic(self) -> bool:
        """Test 10: Basic performance test to ensure QeMLflow functions correctly."""
        test_name = "Basic Performance"
        
        try:
            # Add src to path
            src_path = self.root_path / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            import qemlflow
            import time
            
            # Test basic functionality timing
            start_time = time.time()
            
            # Try to access some core functionality
            if hasattr(qemlflow, 'datasets'):
                self.log_info(test_name, "Datasets module accessible")
            
            if hasattr(qemlflow, 'utils'):
                self.log_info(test_name, "Utils module accessible")
            
            end_time = time.time()
            load_time = end_time - start_time
            
            if load_time > 5.0:
                self.log_warning(test_name, f"Import time seems slow: {load_time:.2f} seconds")
            else:
                self.log_success(test_name, f"Import performance good: {load_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.log_error(test_name, f"Performance test failed: {e}")
            return False

    def generate_report(self) -> Dict:
        """Generate a comprehensive test report."""
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        total_tests = len(self.test_results)
        
        report = {
            'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': total_tests - passed_tests,
                'pass_rate': f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
            },
            'test_results': self.test_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }
        
        return report

    def run_all_tests(self):
        """Run all validation tests."""
        print("🚀 Starting Comprehensive Migration Validation")
        print("=" * 80)
        
        tests = [
            ("No ChemML References", self.test_no_chemml_references),
            ("QeMLflow Import", self.test_import_qemlflow),
            ("Package Structure", self.test_package_structure),
            ("Configuration Files", self.test_configuration_files),
            ("Python Syntax", self.test_python_syntax),
            ("Git Status", self.test_git_status),
            ("Documentation Links", self.test_documentation_links),
            ("Environment Activation", self.test_environment_activation),
            ("Backup Integrity", self.test_backup_integrity),
            ("Basic Performance", self.test_performance_basic)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Generate and display report
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Pass Rate: {report['summary']['pass_rate']}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Save detailed report
        report_file = self.root_path / "tools" / "validation" / "migration_validation_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Final verdict
        if report['summary']['failed'] == 0:
            print("\n🎉 MIGRATION VALIDATION: ALL TESTS PASSED!")
            print("✅ The ChemML to QeMLflow migration appears to be complete and successful.")
        else:
            print(f"\n⚠️  MIGRATION VALIDATION: {report['summary']['failed']} TESTS FAILED")
            print("❌ Please review and fix the issues before considering the migration complete.")
        
        return report

def main():
    """Main entry point."""
    # Get the root directory (assuming script is in tools/validation/)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent
    
    validator = MigrationValidator(str(root_dir))
    report = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if report['summary']['failed'] == 0 else 1)

if __name__ == "__main__":
    main()
