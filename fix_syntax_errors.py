#!/usr/bin/env python3
"""
Fix syntax errors in ChemML codebase - specifically unmatched brackets and indentation issues.
"""

import os
import re
import subprocess
import sys

def get_e999_files():
    """Get files with E999 syntax errors."""
    try:
        result = subprocess.run(
            ["flake8", "src/", "--select=E999"],
            capture_output=True,
            text=True
        )
        
        files_with_errors = set()
        for line in result.stdout.strip().split('\n'):
            if line and ':' in line:
                file_path = line.split(':')[0]
                files_with_errors.add(file_path)
        
        return list(files_with_errors)
    except Exception as e:
        print(f"Error running flake8: {e}")
        return []

def fix_unmatched_bracket_in_all(file_path):
    """Fix commented __all__ declarations that are missing opening brackets."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Pattern to match #__all__ = [ followed by items and ending with ]
        pattern = r'#__all__ = \[\s*\n((?:\s*"[^"]+",?\s*\n)+)\s*\]'
        
        def fix_all_declaration(match):
            items = match.group(1)
            return f'__all__ = [\n{items}]'
        
        new_content = re.sub(pattern, fix_all_declaration, content)
        
        if new_content != content:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Fixed __all__ declaration in {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def fix_indentation_errors(file_path):
    """Fix simple indentation errors by checking common patterns."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        fixed = False
        new_lines = []
        
        for i, line in enumerate(lines):
            # Check for lines that start with unexpected indentation in __init__ files
            if ('__init__.py' in file_path and 
                line.strip() and 
                line.startswith('    ') and 
                i > 0 and 
                not lines[i-1].strip().endswith((':',  '[', '(', '\\'))):
                
                # Likely an unexpected indent - try to dedent
                dedented_line = line.lstrip()
                new_lines.append(dedented_line)
                fixed = True
                print(f"Fixed indentation in {file_path}:{i+1}")
            else:
                new_lines.append(line)
        
        if fixed:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing indentation in {file_path}: {e}")
        return False

def fix_incomplete_if_blocks(file_path):
    """Fix incomplete if blocks that are missing indented content."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Pattern: if statement followed by return at same indentation level
        pattern = r'(\s*)if[^:]*:\s*\n(\s*)return\s+'
        
        def fix_if_block(match):
            if_indent = match.group(1)
            return_indent = match.group(2)
            
            # If return is at same or less indentation than if, add proper indentation
            if len(return_indent) <= len(if_indent):
                return match.group(0).replace(
                    f'{return_indent}return',
                    f'{if_indent}    return'
                )
            return match.group(0)
        
        new_content = re.sub(pattern, fix_if_block, content)
        
        if new_content != content:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Fixed if block indentation in {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing if blocks in {file_path}: {e}")
        return False

def main():
    """Main function to fix syntax errors."""
    print("🔧 Fixing syntax errors in ChemML codebase...")
    
    error_files = get_e999_files()
    if not error_files:
        print("✅ No E999 syntax errors found!")
        return
    
    print(f"Found {len(error_files)} files with syntax errors")
    
    fixed_count = 0
    for file_path in error_files:
        print(f"\n🔍 Processing: {file_path}")
        
        if fix_unmatched_bracket_in_all(file_path):
            fixed_count += 1
        
        if fix_indentation_errors(file_path):
            fixed_count += 1
        
        if fix_incomplete_if_blocks(file_path):
            fixed_count += 1
    
    print(f"\n✅ Fixed syntax errors in {fixed_count} operations")
    
    # Re-check for remaining errors
    print("\n🔍 Checking for remaining E999 errors...")
    remaining_files = get_e999_files()
    if remaining_files:
        print(f"⚠️ {len(remaining_files)} files still have syntax errors:")
        for file_path in remaining_files[:10]:  # Show first 10
            print(f"  - {file_path}")
        if len(remaining_files) > 10:
            print(f"  ... and {len(remaining_files) - 10} more")
    else:
        print("✅ All E999 syntax errors have been fixed!")

if __name__ == "__main__":
    main()
