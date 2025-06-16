#!/usr/bin/env python3
"""
Quick fix for unterminated docstrings in ChemML codebase.
Automatically detects and fixes missing opening triple quotes.
"""

import os
import re
from pathlib import Path


def fix_unterminated_docstrings(file_path: Path) -> bool:
    """Fix unterminated docstrings in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to match lines that look like they should start a docstring
        # but are missing the opening """
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Check if line looks like a docstring start but missing opening """
            if i == 0 and not line.strip().startswith('"""') and not line.strip().startswith('#'):
                # First line that's not a comment, check if it ends with """
                next_few_lines = lines[i:min(i+10, len(lines))]
                for j, next_line in enumerate(next_few_lines):
                    if '"""' in next_line and not next_line.strip().startswith('"""'):
                        # Found closing """, add opening
                        lines[i] = '"""\n' + lines[i]
                        break
        
        # Join lines back
        new_content = '\n'.join(lines)
        
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix all unterminated docstrings in the codebase."""
    base_dir = Path(__file__).parent.parent.parent
    src_dir = base_dir / "src"
    
    fixed_files = []
    
    # Find all Python files
    for py_file in src_dir.rglob("*.py"):
        if fix_unterminated_docstrings(py_file):
            fixed_files.append(py_file)
            print(f"✅ Fixed: {py_file.relative_to(base_dir)}")
    
    print(f"\n🎉 Fixed {len(fixed_files)} files with unterminated docstrings")
    
    if fixed_files:
        print("\nFixed files:")
        for file_path in fixed_files:
            print(f"  • {file_path.relative_to(base_dir)}")


if __name__ == "__main__":
    main()
