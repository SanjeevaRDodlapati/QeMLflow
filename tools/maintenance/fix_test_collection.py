#!/usr/bin/env python3
"""
ChemML Test Collection Fixer

This script fixes test collection issues by:
1. Identifying import errors in test files
2. Fixing missing imports and dependencies
3. Resolving module path issues
4. Updating deprecated test patterns

Usage:
    python tools/maintenance/fix_test_collection.py [--dry-run] [--verbose]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import ast
import re
from typing import List, Dict, Set
import json


class TestCollectionFixer:
    """Fixes test collection issues in the ChemML test suite."""
    
    def __init__(self, base_dir: Path, dry_run: bool = False, verbose: bool = False):
        self.base_dir = base_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.fixed_count = 0
        self.errors = []
        
    def fix_all_collection_issues(self):
        """Fix all test collection issues."""
        print("🧪 ChemML Test Collection Fixer")
        print("=" * 40)
        
        # First, identify collection issues
        collection_issues = self.identify_collection_issues()
        print(f"   📋 Found {len(collection_issues)} collection issues")
        
        if not collection_issues:
            print("   ✅ No collection issues found!")
            return
        
        # Fix each category of issues
        fixes = [
            ("Fixing import errors", self.fix_import_errors, collection_issues),
            ("Fixing missing dependencies", self.fix_missing_dependencies, collection_issues),
            ("Fixing module path issues", self.fix_module_paths, collection_issues),
            ("Updating deprecated patterns", self.fix_deprecated_patterns, collection_issues),
        ]
        
        for description, fix_func, issues in fixes:
            print(f"\n📋 {description}...")
            try:
                count = fix_func(issues)
                self.fixed_count += count
                print(f"   ✅ Fixed {count} issues")
            except Exception as e:
                error = f"❌ Error in {description}: {e}"
                print(f"   {error}")
                self.errors.append(error)
        
        # Validate fixes
        self.validate_fixes()
        self.generate_report()
    
    def identify_collection_issues(self) -> Dict[str, List[str]]:
        """Identify test collection issues by running pytest --collect-only."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--collect-only", "-q",
                str(self.base_dir / "tests")
            ], capture_output=True, text=True, cwd=self.base_dir)
            
            issues = {
                'import_errors': [],
                'missing_deps': [],
                'module_path_errors': [],
                'syntax_errors': [],
                'other_errors': []
            }
            
            # Parse errors from stderr
            error_lines = result.stderr.split('\n')
            current_file = None
            
            for line in error_lines:
                if self.verbose:
                    print(f"   Debug: {line}")
                
                # Extract file names
                if '::' in line and 'FAILED' in line:
                    current_file = line.split('::')[0]
                    continue
                
                # Categorize errors
                line_lower = line.lower()
                if 'importerror' in line_lower or 'modulenotfounderror' in line_lower:
                    issues['import_errors'].append(line)
                elif 'no module named' in line_lower:
                    issues['missing_deps'].append(line)
                elif 'syntaxerror' in line_lower:
                    issues['syntax_errors'].append(line)
                elif current_file and ('error' in line_lower or 'failed' in line_lower):
                    issues['other_errors'].append(f"{current_file}: {line}")
            
            return issues
            
        except Exception as e:
            print(f"   Warning: Failed to collect issues: {e}")
            return {}
    
    def fix_import_errors(self, issues: Dict[str, List[str]]) -> int:
        """Fix import errors in test files."""
        count = 0
        import_errors = issues.get('import_errors', [])
        
        # Get all test files
        test_files = list(self.base_dir.glob("tests/**/*.py"))
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix common import issues
                content = self._fix_common_imports(content)
                content = self._fix_relative_imports(content, test_file)
                content = self._add_missing_imports(content)
                
                if content != original_content:
                    if not self.dry_run:
                        with open(test_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                    count += 1
                    if self.verbose:
                        print(f"      Fixed imports in {test_file.name}")
                        
            except Exception as e:
                self.errors.append(f"Error fixing imports in {test_file}: {e}")
        
        return count
    
    def fix_missing_dependencies(self, issues: Dict[str, List[str]]) -> int:
        """Fix missing dependency issues."""
        count = 0
        missing_deps = issues.get('missing_deps', [])
        
        # Common missing dependencies and their fixes
        dep_fixes = {
            'sklearn': 'from sklearn import *',
            'numpy': 'import numpy as np',
            'pandas': 'import pandas as pd',
            'matplotlib': 'import matplotlib.pyplot as plt',
            'scipy': 'import scipy',
            'rdkit': 'try:\n    from rdkit import Chem\nexcept ImportError:\n    pass',
        }
        
        test_files = list(self.base_dir.glob("tests/**/*.py"))
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Add missing imports based on usage
                for dep, import_line in dep_fixes.items():
                    if dep in content and f'import {dep}' not in content:
                        # Add import at the top after existing imports
                        content = self._add_import_after_existing(content, import_line)
                
                if content != original_content:
                    if not self.dry_run:
                        with open(test_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                    count += 1
                    
            except Exception as e:
                self.errors.append(f"Error fixing dependencies in {test_file}: {e}")
        
        return count
    
    def fix_module_paths(self, issues: Dict[str, List[str]]) -> int:
        """Fix module path issues."""
        count = 0
        
        # Add src to Python path in conftest.py if not present
        conftest_path = self.base_dir / "tests" / "conftest.py"
        
        if conftest_path.exists():
            try:
                with open(conftest_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if path setup exists
                if 'sys.path' not in content:
                    path_setup = '''
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
'''
                    content = path_setup + content
                    
                    if not self.dry_run:
                        with open(conftest_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    count += 1
                    
            except Exception as e:
                self.errors.append(f"Error fixing conftest.py: {e}")
        else:
            # Create conftest.py with path setup
            conftest_content = '''"""Test configuration and fixtures."""
import sys
import os
import pytest
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"test": "data"}


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path
'''
            
            if not self.dry_run:
                with open(conftest_path, 'w', encoding='utf-8') as f:
                    f.write(conftest_content)
            count += 1
        
        return count
    
    def fix_deprecated_patterns(self, issues: Dict[str, List[str]]) -> int:
        """Fix deprecated test patterns."""
        count = 0
        test_files = list(self.base_dir.glob("tests/**/*.py"))
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix deprecated patterns
                content = self._fix_deprecated_test_patterns(content)
                
                if content != original_content:
                    if not self.dry_run:
                        with open(test_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                    count += 1
                    
            except Exception as e:
                self.errors.append(f"Error fixing patterns in {test_file}: {e}")
        
        return count
    
    def _fix_common_imports(self, content: str) -> str:
        """Fix common import issues."""
        # Fix chemml imports
        content = re.sub(
            r'from chemml\.([a-zA-Z_]+) import',
            r'from chemml.\1 import',
            content
        )
        
        # Fix relative imports
        content = re.sub(
            r'from \.\.([a-zA-Z_]+) import',
            r'from chemml.\1 import',
            content
        )
        
        return content
    
    def _fix_relative_imports(self, content: str, test_file: Path) -> str:
        """Fix relative import paths."""
        # Convert relative imports to absolute
        content = re.sub(
            r'from \.([a-zA-Z_]+) import',
            r'from chemml.\1 import',
            content
        )
        
        return content
    
    def _add_missing_imports(self, content: str) -> str:
        """Add commonly missing imports."""
        missing_imports = []
        
        # Check for usage without imports
        if 'np.' in content and 'import numpy' not in content:
            missing_imports.append('import numpy as np')
        
        if 'pd.' in content and 'import pandas' not in content:
            missing_imports.append('import pandas as pd')
        
        if 'plt.' in content and 'import matplotlib' not in content:
            missing_imports.append('import matplotlib.pyplot as plt')
        
        if missing_imports:
            import_block = '\n'.join(missing_imports) + '\n\n'
            # Add after existing imports
            lines = content.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_end = i + 1
            
            lines.insert(import_end, import_block)
            content = '\n'.join(lines)
        
        return content
    
    def _add_import_after_existing(self, content: str, import_line: str) -> str:
        """Add import after existing imports."""
        lines = content.split('\n')
        import_end = 0
        
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_end = i + 1
        
        lines.insert(import_end, import_line)
        return '\n'.join(lines)
    
    def _fix_deprecated_test_patterns(self, content: str) -> str:
        """Fix deprecated test patterns."""
        # Fix deprecated assert patterns
        content = re.sub(r'self\.assertEquals\(', 'self.assertEqual(', content)
        content = re.sub(r'self\.assertNotEquals\(', 'self.assertNotEqual(', content)
        
        # Fix deprecated unittest patterns
        content = re.sub(r'unittest\.TestCase', 'unittest.TestCase', content)
        
        return content
    
    def validate_fixes(self):
        """Validate that fixes resolved collection issues."""
        print(f"\n🔍 Validating fixes...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--collect-only", "-q",
                str(self.base_dir / "tests")
            ], capture_output=True, text=True, cwd=self.base_dir)
            
            error_count = len([line for line in result.stderr.split('\n') 
                             if 'error' in line.lower() or 'failed' in line.lower()])
            
            if error_count == 0:
                print("   ✅ All collection issues resolved!")
            else:
                print(f"   ⚠️  {error_count} collection issues remain")
                
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
    
    def generate_report(self):
        """Generate improvement report."""
        print(f"\n🎉 Test Collection Fix Summary")
        print("=" * 35)
        print(f"   ✅ Total fixes applied: {self.fixed_count}")
        
        if self.errors:
            print(f"   ⚠️  Errors encountered: {len(self.errors)}")
            for error in self.errors[:3]:
                print(f"      • {error}")
            if len(self.errors) > 3:
                print(f"      • ... and {len(self.errors) - 3} more")
        
        # Save detailed report
        report_path = self.base_dir / "reports" / "test_collection_fix_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": str(subprocess.run(["date"], capture_output=True, text=True).stdout.strip()),
            "total_fixes": self.fixed_count,
            "errors": self.errors,
            "dry_run": self.dry_run
        }
        
        if not self.dry_run:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"   📄 Detailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='ChemML Test Collection Fixer')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview changes without applying them')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.parent
    fixer = TestCollectionFixer(base_dir, dry_run=args.dry_run, verbose=args.verbose)
    fixer.fix_all_collection_issues()


if __name__ == "__main__":
    main()
