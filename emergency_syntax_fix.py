#!/usr/bin/env python3
"""
Emergency fix for escaped newline syntax errors
This is the root cause of GitHub Actions workflow failures
"""

import os
import re
from pathlib import Path

def fix_escaped_newlines(file_path):
    """Fix escaped newline syntax errors in Python files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix the specific pattern we keep seeing
        # Pattern: from typing import Something\nfrom typing import Other\n"""
        pattern = r'from typing import [^"]*\\n[^"]*"""'
        content = re.sub(pattern, '"""', content)
        
        # Also fix any standalone escaped newlines at the start of files
        if content.startswith('from typing import') and '\\n' in content[:100]:
            lines = content.split('\n')
            if lines[0].count('\\n') > 0:
                # Extract just the docstring
                if '"""' in content:
                    docstring_start = content.find('"""')
                    content = content[docstring_start:]
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return False

def main():
    """Fix all Python files with escaped newline issues."""
    src_dir = Path('src')
    fixed_files = []
    
    print("🚨 EMERGENCY FIX: Escaped Newline Syntax Errors")
    print("=" * 50)
    
    for py_file in src_dir.rglob('*.py'):
        if fix_escaped_newlines(py_file):
            fixed_files.append(str(py_file))
            print(f"✅ Fixed: {py_file}")
    
    print(f"\n🎯 Fixed {len(fixed_files)} files with syntax errors")
    print("This should resolve the GitHub Actions workflow failures!")

if __name__ == "__main__":
    main()
