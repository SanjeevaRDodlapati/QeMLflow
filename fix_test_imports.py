#!/usr/bin/env python3
"""
Fix Test Import Paths After Cleanup
===================================

This script fixes the broken import paths in tests after the aggressive cleanup.
The cleanup moved modules but test imports weren't updated.
"""

import os
import re
from pathlib import Path


def fix_import_paths():
    """Fix all broken import paths in test files."""
    test_dir = Path("tests")
    fixed_files = []
    
    # Define import path mappings
    import_fixes = [
        # Old utils imports → new qemlflow.core.utils imports
        (r'from utils\.(\w+)', r'from qemlflow.core.utils.\1'),
        
        # Old src. imports → new qemlflow imports  
        (r'from src\.utils\.(\w+)', r'from qemlflow.core.utils.\1'),
        (r'from src\.data_processing\.(\w+)', r'from qemlflow.core.preprocessing.\1'),
        (r'from src\.drug_design\.(\w+)', r'from qemlflow.research.drug_discovery.\1'),
        (r'from src\.qemlflow\.(\w+)', r'from qemlflow.\1'),
        
        # Import statements without from
        (r'import utils\.(\w+)', r'import qemlflow.core.utils.\1'),
        (r'import src\.utils\.(\w+)', r'import qemlflow.core.utils.\1'),
        (r'import src\.data_processing\.(\w+)', r'import qemlflow.core.preprocessing.\1'),
        (r'import src\.drug_design\.(\w+)', r'import qemlflow.research.drug_discovery.\1'),
        
        # Specific module mappings based on what exists
        (r'from data_processing\.(\w+)', r'from qemlflow.core.preprocessing.\1'),
        (r'import data_processing\.(\w+)', r'import qemlflow.core.preprocessing.\1'),
        
        # Specific patch statement fixes for test mocks
        (r'patch\("utils\.molecular_utils\.', r'patch("qemlflow.core.utils.molecular_utils.'),
        (r'patch\("utils\.(\w+)\.', r'patch("qemlflow.core.utils.\1.'),
        (r'patch\("src\.utils\.(\w+)\.', r'patch("qemlflow.core.utils.\1.'),
        (r'patch\("src\.data_processing\.(\w+)\.', r'patch("qemlflow.core.preprocessing.\1.'),
        (r'patch\("data_processing\.(\w+)\.', r'patch("qemlflow.core.preprocessing.\1.'),
    ]
    
    # Process all Python test files
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply import fixes
                    for old_pattern, new_pattern in import_fixes:
                        content = re.sub(old_pattern, new_pattern, content)
                    
                    # Write back if changed
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_files.append(str(filepath))
                        print(f"✅ Fixed imports in: {filepath}")
                
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {e}")
    
    return fixed_files


def validate_import_fixes_dry_run():
    """Validate what the import fixes would do without making changes."""
    test_dir = Path("tests")
    potential_fixes = []
    
    # Define import path mappings
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
    
    # Process all Python test files (dry run)
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Check what would change
                    for old_pattern, new_pattern in import_fixes:
                        matches = re.findall(old_pattern, content)
                        if matches:
                            potential_fixes.append({
                                'file': str(filepath),
                                'pattern': old_pattern,
                                'replacement': new_pattern,
                                'matches': matches
                            })
                
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {e}")
    
    return potential_fixes


def main():
    """Main function to fix import paths."""
    print("🔧 VALIDATING TEST IMPORT PATH FIXES")
    print("=" * 50)
    print("First performing dry-run validation...")
    
    # Step 1: Dry run validation
    potential_fixes = validate_import_fixes_dry_run()
    
    if potential_fixes:
        print(f"\\n📊 DRY RUN RESULTS: {len(potential_fixes)} potential fixes found")
        print("\\nSample fixes that would be applied:")
        for fix in potential_fixes[:5]:  # Show first 5
            print(f"  📁 {fix['file']}")
            print(f"     Pattern: {fix['pattern']}")
            print(f"     → Would become: {fix['replacement']}")
            print(f"     Matches: {fix['matches']}")
            print()
        
        if len(potential_fixes) > 5:
            print(f"  ... and {len(potential_fixes) - 5} more potential fixes")
        
        # Step 2: Validate that target modules exist
        print("\\n🔍 VALIDATING TARGET MODULES...")
        validation_passed = validate_target_modules()
        
        if validation_passed:
            print("\\n✅ All target modules exist. Safe to proceed with fixes.")
            
            response = input("\\nApply these fixes? (yes/no): ").lower().strip()
            if response == 'yes':
                print("\\n🔧 APPLYING FIXES...")
                fixed_files = fix_import_paths()
                
                if fixed_files:
                    print(f"\\n🎉 FIXED {len(fixed_files)} test files")
                else:
                    print("\\n✅ No fixes applied")
            else:
                print("\\n⏩ Fixes cancelled by user")
        else:
            print("\\n❌ Target module validation failed. NOT applying fixes.")
    else:
        print("\\n✅ No import path fixes needed")


def validate_target_modules():
    """Validate that target modules exist before applying fixes."""
    import sys
    sys.path.insert(0, 'src')
    
    target_modules = [
        'qemlflow.core.utils.molecular_utils',
        'qemlflow.core.preprocessing.feature_extraction',
        'qemlflow.research.drug_discovery.properties',
        'qemlflow.research.drug_discovery.admet',
        'qemlflow.core.utils.io_utils',
    ]
    
    all_valid = True
    for module in target_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            all_valid = False
    
    return all_valid


if __name__ == "__main__":
    main()
