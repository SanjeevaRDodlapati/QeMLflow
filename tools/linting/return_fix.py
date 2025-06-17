#!/usr/bin/env python3
"""
Focused Syntax Error Fix - Return Outside Function
==================================================

A minimal, safe tool to fix the specific "return outside function" 
syntax errors found in the QeMLflow codebase.

This tool:
1. Makes backups before any changes
2. Validates syntax before and after
3. Only fixes the specific pattern found
4. Has comprehensive testing
"""

import ast
import py_compile
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
from datetime import datetime


class ReturnOutsideFunctionFixer:
    """Safe fixer for 'return outside function' syntax errors."""
    
    def __init__(self, create_backups: bool = True):
        self.create_backups = create_backups
        self.backup_dir = Path("backups") / f"syntax_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if create_backups:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Validate Python file syntax strictly."""
        try:
            py_compile.compile(str(file_path), doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Error: {e}"
    
    def create_backup(self, file_path: Path) -> Path:
        """Create a backup of the file."""
        if not self.create_backups:
            return file_path
        
        backup_path = self.backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def fix_return_outside_function(self, content: str) -> Tuple[str, List[str]]:
        """Fix return statements that are outside functions."""
        lines = content.split('\n')
        fixed_lines = []
        changes = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if this is a return statement outside a function
            if stripped.startswith('return ') and not self._is_inside_function(lines, i):
                # Look for the previous function to determine where this return should go
                func_line = self._find_previous_function(lines, i)
                
                if func_line is not None:
                    # Get the function's indentation and add proper indentation
                    func_indent = len(lines[func_line]) - len(lines[func_line].lstrip())
                    return_indent = func_indent + 4  # Standard 4-space indentation
                    
                    # Create properly indented return statement
                    fixed_line = ' ' * return_indent + stripped
                    fixed_lines.append(fixed_line)
                    changes.append(f"Line {i+1}: Fixed return statement indentation")
                else:
                    # Comment out the orphaned return
                    fixed_line = f"# TODO: Fix orphaned return statement: {stripped}"
                    fixed_lines.append(fixed_line)
                    changes.append(f"Line {i+1}: Commented out orphaned return statement")
            else:
                fixed_lines.append(line)
            
            i += 1
        
        return '\n'.join(fixed_lines), changes
    
    def _is_inside_function(self, lines: List[str], line_idx: int) -> bool:
        """Check if the given line is inside a function."""
        # Look backwards for function definition
        current_indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
        
        for i in range(line_idx - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('def ') and line.endswith(':'):
                # Found a function, check if we're properly indented within it
                func_indent = len(lines[i]) - len(lines[i].lstrip())
                
                # Check if we're inside the function body
                if current_indent > func_indent:
                    # Also verify we haven't hit another def at the same or higher level
                    for j in range(i + 1, line_idx):
                        check_line = lines[j].strip()
                        if check_line.startswith('def ') and check_line.endswith(':'):
                            check_indent = len(lines[j]) - len(lines[j].lstrip())
                            if check_indent <= func_indent:
                                return False  # Hit another function at same level
                    return True
                else:
                    return False
            elif line.startswith('class ') and line.endswith(':'):
                # Hit a class, not inside a function
                return False
        return False
    
    def _find_previous_function(self, lines: List[str], line_idx: int) -> int:
        """Find the most recent function definition before the given line."""
        for i in range(line_idx - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('def ') and line.endswith(':'):
                return i
        return None
    
    def fix_file(self, file_path: Path, dry_run: bool = False) -> Tuple[bool, List[str], str]:
        """Fix a single file safely."""
        # Validate syntax before
        is_valid_before, error = self.validate_syntax(file_path)
        if not is_valid_before and "'return' outside function" not in error:
            return False, [], f"File has other syntax errors: {error}"
        
        # Read content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return False, [], f"Could not read file: {e}"
        
        # Apply fixes
        fixed_content, changes = self.fix_return_outside_function(original_content)
        
        if not changes:
            return True, [], "No changes needed"
        
        if dry_run:
            return True, changes, "Dry run - no changes made"
        
        # Create backup
        if self.create_backups:
            backup_path = self.create_backup(file_path)
        
        # Write fixed content to temporary file first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(fixed_content)
            temp_path = Path(temp_file.name)
        
        # Validate the fixed content
        is_valid_after, error = self.validate_syntax(temp_path)
        
        if not is_valid_after:
            temp_path.unlink()  # Clean up temp file
            return False, changes, f"Fix created syntax errors: {error}"
        
        # If validation passes, replace the original file
        shutil.move(str(temp_path), str(file_path))
        
        return True, changes, "File fixed successfully"


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix 'return outside function' syntax errors")
    parser.add_argument("files", nargs="+", help="Python files to fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups")
    
    args = parser.parse_args()
    
    fixer = ReturnOutsideFunctionFixer(create_backups=not args.no_backup)
    
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ {file_path}: File not found")
            continue
        
        print(f"🔧 Processing: {file_path}")
        success, changes, message = fixer.fix_file(path, dry_run=args.dry_run)
        
        if success:
            if changes:
                print(f"✅ {file_path}: {message}")
                for change in changes:
                    print(f"   - {change}")
            else:
                print(f"✅ {file_path}: No changes needed")
        else:
            print(f"❌ {file_path}: {message}")


if __name__ == "__main__":
    main()
