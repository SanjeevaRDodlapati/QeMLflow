#!/usr/bin/env python3
"""
Minimal Test Import Fixer
=========================

This script fixes only the import patterns we know will work,
and skips tests that use missing modules.
"""

import os
import re
from pathlib import Path


def apply_safe_fixes():
    """Apply only the safe import fixes we've validated."""
    test_dir = Path("tests")
    fixed_files = []
    skipped_files = []
    
    # Only fix imports we know exist
    safe_fixes = [
        # Pattern 1: utils.molecular_utils → qemlflow.core.utils.molecular_utils
        ('from utils.molecular_utils', 'from qemlflow.core.utils.molecular_utils'),
        ('import utils.molecular_utils', 'import qemlflow.core.utils.molecular_utils'),
        
        # Pattern 2: utils.data_utils → qemlflow.core.utils.data_utils  
        ('from utils.data_utils', 'from qemlflow.core.utils.data_utils'),
        ('import utils.data_utils', 'import qemlflow.core.utils.data_utils'),
        
        # Pattern 3: utils.feature_utils → qemlflow.core.utils.feature_utils
        ('from utils.feature_utils', 'from qemlflow.core.utils.feature_utils'),
        ('import utils.feature_utils', 'import qemlflow.core.utils.feature_utils'),
        
        # Pattern 4: src.data_processing → qemlflow.core.preprocessing
        ('from src.data_processing.preprocessing', 'from qemlflow.core.preprocessing.preprocessing'),
        ('from src.data_processing.feature_extraction', 'from qemlflow.core.preprocessing.feature_extraction'),
        
        # Pattern 5: src.drug_design → qemlflow.research.drug_discovery
        ('from src.drug_design.', 'from qemlflow.research.drug_discovery.'),
        ('from src.drug_design import', 'from qemlflow.research.drug_discovery import'),
        
        # Pattern 6: src.qemlflow → qemlflow
        ('from src.qemlflow.', 'from qemlflow.'),
        
        # Pattern 7: Mock/patch paths for testing
        ("'src.qemlflow.observability.dashboard.PLOTLY_AVAILABLE'", "'qemlflow.observability.dashboard.PLOTLY_AVAILABLE'"),
        ('"src.qemlflow.observability.dashboard.PLOTLY_AVAILABLE"', '"qemlflow.observability.dashboard.PLOTLY_AVAILABLE"'),
        
        # Pattern 8: Other src.qemlflow patches in tests
        ("'src.qemlflow.", "'qemlflow."),
        ('"src.qemlflow.', '"qemlflow.'),
    ]
    
    # Process all Python test files
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Skip files that import missing modules
                    if 'from src.models.' in content:
                        print(f"⏭️  Skipping {filepath} - uses missing model modules")
                        skipped_files.append(str(filepath))
                        continue
                    
                    original_content = content
                    
                    # Apply safe fixes
                    for old_import, new_import in safe_fixes:
                        if old_import in content:
                            content = content.replace(old_import, new_import)
                    
                    # Write back if changed
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_files.append(str(filepath))
                        print(f"✅ Fixed imports in: {filepath}")
                
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {e}")
    
    return fixed_files, skipped_files


def main():
    """Main function."""
    print("🔧 MINIMAL TEST IMPORT FIXER")
    print("=" * 40)
    print("Applying only validated, safe import fixes...")
    
    fixed_files, skipped_files = apply_safe_fixes()
    
    print(f"\n📊 SUMMARY:")
    print(f"  • Files with safe fixes: {len(fixed_files)}")
    print(f"  • Files skipped (missing modules): {len(skipped_files)}")
    
    if fixed_files:
        print(f"\n✅ Fixed files:")
        for file in fixed_files[:5]:  # Show first 5
            print(f"  - {file}")
        if len(fixed_files) > 5:
            print(f"  ... and {len(fixed_files) - 5} more")
    
    if skipped_files:
        print(f"\n⏭️  Skipped files (need manual review):")
        for file in skipped_files[:3]:  # Show first 3
            print(f"  - {file}")
        if len(skipped_files) > 3:
            print(f"  ... and {len(skipped_files) - 3} more")
    
    return True


if __name__ == "__main__":
    main()
