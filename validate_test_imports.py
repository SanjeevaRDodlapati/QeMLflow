#!/usr/bin/env python3
"""
Validate Test Import Fixes
==========================

This script validates what fix_test_imports.py would do without making changes.
It checks that all target modules exist before applying fixes.
"""

import os
import re
import sys
from pathlib import Path


def validate_module_exists(module_path: str) -> bool:
    """Check if a module exists in the current codebase."""
    # Convert module path to file path
    parts = module_path.split('.')
    
    # Try different possible locations
    possible_paths = [
        Path("src") / "/".join(parts) / "__init__.py",
        Path("src") / f"{'/'.join(parts)}.py",
        Path("/".join(parts)) / "__init__.py", 
        Path(f"{'/'.join(parts)}.py"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return True
            
    # Try importing directly
    try:
        sys.path.insert(0, 'src')
        __import__(module_path)
        return True
    except ImportError:
        pass
    
    return False


def analyze_imports_in_file(filepath: Path) -> dict:
    """Analyze imports in a single test file."""
    analysis = {
        'file': str(filepath),
        'current_imports': [],
        'proposed_fixes': [],
        'missing_modules': [],
        'safe_fixes': []
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, OSError) as e:
        analysis['error'] = str(e)
        return analysis
    
    # Define import path mappings (same as fix_test_imports.py)
    import_fixes = [
        (r'from utils\.(\w+)', r'from qemlflow.core.utils.\1'),
        (r'from src\.utils\.(\w+)', r'from qemlflow.core.utils.\1'),
        (r'from src\.data_processing\.(\w+)', r'from qemlflow.core.preprocessing.\1'),
        (r'from src\.models\.(\w+)', r'from qemlflow.models.\1'),
        (r'from src\.drug_design\.(\w+)', r'from qemlflow.research.drug_discovery.\1'),
        (r'from src\.qemlflow\.(\w+)', r'from qemlflow.\1'),
        (r'import utils\.(\w+)', r'import qemlflow.core.utils.\1'),
        (r'import src\.utils\.(\w+)', r'import qemlflow.core.utils.\1'),
        (r'import src\.data_processing\.(\w+)', r'import qemlflow.core.preprocessing.\1'),
        (r'import src\.models\.(\w+)', r'import qemlflow.models.\1'),
        (r'import src\.drug_design\.(\w+)', r'import qemlflow.research.drug_discovery.\1'),
    ]
    
    # Find current problematic imports
    for old_pattern, new_pattern in import_fixes:
        matches = re.findall(old_pattern, content)
        for match in matches:
            old_import = old_pattern.replace(r'(\w+)', match).replace('\\', '')
            new_import = new_pattern.replace(r'\1', match)
            
            analysis['current_imports'].append(old_import)
            analysis['proposed_fixes'].append((old_import, new_import))
            
            # Check if target module exists
            target_module = new_import.replace('from ', '').replace('import ', '').split(' ')[0]
            if validate_module_exists(target_module):
                analysis['safe_fixes'].append((old_import, new_import))
            else:
                analysis['missing_modules'].append(target_module)
    
    return analysis


def main():
    """Main validation function."""
    print("🔍 VALIDATING TEST IMPORT FIXES")
    print("=" * 50)
    
    test_dir = Path("tests")
    all_analyses = []
    total_files = 0
    files_with_issues = 0
    safe_fixes_count = 0
    unsafe_fixes_count = 0
    
    # Process all Python test files
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                total_files += 1
                filepath = Path(root) / file
                analysis = analyze_imports_in_file(filepath)
                
                if analysis.get('current_imports'):
                    files_with_issues += 1
                    all_analyses.append(analysis)
                    safe_fixes_count += len(analysis['safe_fixes'])
                    unsafe_fixes_count += len(analysis['missing_modules'])
    
    # Report results
    print("📊 ANALYSIS RESULTS:")
    print(f"  • Total test files: {total_files}")
    print(f"  • Files with import issues: {files_with_issues}")
    print(f"  • Safe fixes available: {safe_fixes_count}")
    print(f"  • Unsafe fixes (missing targets): {unsafe_fixes_count}")
    
    if unsafe_fixes_count > 0:
        print("\n⚠️  MISSING MODULES DETECTED:")
        missing_modules = set()
        for analysis in all_analyses:
            missing_modules.update(analysis['missing_modules'])
        
        for module in sorted(missing_modules):
            print(f"  ❌ {module}")
        
        print(f"\n❌ UNSAFE TO PROCEED - {unsafe_fixes_count} fixes would break imports")
        return False
    
    print(f"\n✅ SAFE TO PROCEED - All {safe_fixes_count} proposed fixes target existing modules")
    
    # Show sample of what would be fixed
    if all_analyses:
        print("\n📝 SAMPLE FIXES (first 5 files):")
        for analysis in all_analyses[:5]:
            print(f"\n  📁 {analysis['file']}")
            for old_import, new_import in analysis['safe_fixes'][:3]:
                print(f"    • {old_import} → {new_import}")
            if len(analysis['safe_fixes']) > 3:
                print(f"    ... and {len(analysis['safe_fixes']) - 3} more fixes")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
