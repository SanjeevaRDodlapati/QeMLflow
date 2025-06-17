#!/usr/bin/env python3
"""
Fix escaped newline syntax issues in import statements
"""

import os
import re
import sys

def fix_import_syntax(file_path):
    """Fix escaped newline characters in import statements"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to find problematic import statements with \n
        pattern = r'from typing import[^"]*\\n[^"]*"""'
        
        # Replace with just the docstring
        fixed_content = re.sub(pattern, '"""', content)
        
        # Also fix any remaining \n sequences in the first few lines
        lines = fixed_content.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            if i < 10 and '\\n' in line and 'from typing import' in line:
                # Skip this line as it's malformed
                continue
            fixed_lines.append(line)
        
        fixed_content = '\n'.join(fixed_lines)
        
        if content != fixed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all Python files with import syntax issues"""
    fixed_count = 0
    
    # Get files with the issue
    import subprocess
    result = subprocess.run([
        'find', 'src/', '-name', '*.py', '-exec', 
        'grep', '-l', r'from typing import.*\\n', '{}', ';'
    ], capture_output=True, text=True, cwd='.')
    
    if result.returncode == 0:
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f.strip()]
        
        for file_path in files:
            if fix_import_syntax(file_path):
                fixed_count += 1
    
    print(f"Fixed {fixed_count} files with import syntax issues")
    
    # Test basic import
    try:
        sys.path.insert(0, 'src')
        import qemlflow
        print("✅ QeMLflow import test successful after fixes")
        return True
    except Exception as e:
        print(f"❌ QeMLflow import still has issues: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
