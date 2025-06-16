#!/usr/bin/env python3
"""
Quick and targeted syntax error fixer for ChemML.
Focuses on the most common syntax issues identified in the analysis.
"""

import ast
import re
import subprocess
from pathlib import Path


def get_syntax_error_files():
    """Get list of files with syntax errors."""
    try:
        result = subprocess.run(
            ['flake8', 'src/', '--select=E999', '--format=%(path)s'],
            capture_output=True,
            text=True
        )
        
        files = []
        for line in result.stdout.strip().split('\n'):
            if line.startswith('src/') and line not in files:
                files.append(line)
        
        return files
    except Exception as e:
        print(f"Error getting syntax error files: {e}")
        return []


def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Other error: {e}"


def fix_unterminated_docstrings(file_path):
    """Fix unterminated triple-quoted docstrings."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Find unclosed triple quotes at the end of docstrings
        # This pattern looks for function/class definitions followed by unclosed docstrings
        patterns_to_fix = [
            # Match function/class with unclosed docstring
            (r'(def\s+\w+.*?:\s*\n\s*"""[^"]*?)(\n\s*)(def\s|\nclass\s|\nif\s|\n[a-zA-Z_])', r'\1"""\2\3'),
            (r'(class\s+\w+.*?:\s*\n\s*"""[^"]*?)(\n\s*)(def\s|\nclass\s|\nif\s|\n[a-zA-Z_])', r'\1"""\2\3'),
        ]
        
        for pattern, replacement in patterns_to_fix:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        # Pattern 2: Fix orphaned opening triple quotes at start of file
        if content.startswith('Sample data generators') or content.startswith('Integration tests'):
            content = '"""\n' + content
        
        # Pattern 3: Add missing closing quotes at end of file if needed
        triple_quote_count = content.count('"""')
        if triple_quote_count % 2 == 1:  # Odd number means one is unclosed
            content = content.rstrip() + '\n"""\n'
        
        # Pattern 4: Fix specific invalid character issues
        content = content.replace('²', '**2')  # Replace superscript 2
        content = content.replace('🎯', 'target')  # Replace emoji
        content = content.replace('🎓', 'graduation')  # Replace emoji
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def fix_invalid_syntax_patterns(file_path):
    """Fix common invalid syntax patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_lines = lines[:]
        modified = False
        
        for i, line in enumerate(lines):
            # Fix invalid decimal literals (like 1.2.3)
            if re.search(r'\d+\.\d+\.\d+', line):
                lines[i] = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1_\2_\3', line)
                modified = True
            
            # Fix lines that start with text but should be docstrings
            if (i == 0 and not line.strip().startswith('"""') and not line.strip().startswith('#') 
                and not line.strip().startswith('import') and not line.strip().startswith('from')
                and line.strip() and not line.strip().startswith('__')):
                lines[i] = '"""\n' + line
                modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        return False
        
    except Exception as e:
        print(f"Error fixing syntax patterns in {file_path}: {e}")
        return False


def quick_syntax_fix():
    """Perform quick syntax fixes on all problematic files."""
    print("🔧 QUICK SYNTAX ERROR FIXER")
    print("=" * 50)
    print()
    
    # Get files with syntax errors
    syntax_error_files = get_syntax_error_files()
    
    if not syntax_error_files:
        print("✅ No syntax errors found!")
        return
    
    print(f"Found {len(syntax_error_files)} files with syntax errors")
    print()
    
    fixed_count = 0
    failed_count = 0
    
    for file_path in syntax_error_files:
        print(f"Fixing: {file_path}")
        
        # Check current syntax
        is_valid, error = check_python_syntax(file_path)
        if is_valid:
            print("  ✅ Already valid (may have been fixed)")
            continue
        
        print(f"  ❌ Error: {error[:100] if error else 'Unknown'}...")
        
        # Try to fix
        fixed1 = fix_unterminated_docstrings(file_path)
        fixed2 = fix_invalid_syntax_patterns(file_path)
        
        # Check if fixed
        is_valid_after, error_after = check_python_syntax(file_path)
        
        if is_valid_after:
            print("  ✅ FIXED!")
            fixed_count += 1
        else:
            print(f"  ❌ Still broken: {error_after[:50] if error_after else 'Unknown'}...")
            failed_count += 1
        
        print()
    
    print("🎯 SUMMARY:")
    print(f"  • Files processed: {len(syntax_error_files)}")
    print(f"  • Successfully fixed: {fixed_count}")
    print(f"  • Still need manual fix: {failed_count}")
    print(f"  • Success rate: {fixed_count/len(syntax_error_files)*100:.1f}%")
    
    if failed_count > 0:
        print()
        print("📝 Files that still need manual attention:")
        for file_path in syntax_error_files:
            is_valid, _ = check_python_syntax(file_path)
            if not is_valid:
                print(f"  • {file_path}")


if __name__ == "__main__":
    quick_syntax_fix()
