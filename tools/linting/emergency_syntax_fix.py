#!/usr/bin/env python3
"""
Emergency Syntax Fix for ChemML
===============================

A minimal emergency fix to restore basic syntax validity so that 
linting tools can run again. This fixes the immediate "return outside function"
errors by simply commenting them out.

After this emergency fix, the comprehensive linting tools can run properly
and provide a real assessment of the codebase.
"""

import shutil
from pathlib import Path
from datetime import datetime


def emergency_syntax_fix(file_path: Path) -> bool:
    """Apply emergency fix to make file syntactically valid."""
    
    # Create backup
    backup_dir = Path("backups") / f"emergency_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        changes = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # If this is a return statement at the start of the line (outside function)
            if stripped.startswith('return ') and len(line) - len(line.lstrip()) == 0:
                # Comment it out for now
                fixed_lines.append(f"# EMERGENCY_FIX: {line}")
                changes += 1
                print(f"   Line {i+1}: Commented out orphaned return")
            else:
                fixed_lines.append(line)
        
        if changes > 0:
            with open(file_path, 'w') as f:
                f.writelines(fixed_lines)
            print(f"✅ Applied {changes} emergency fixes")
            return True
        else:
            print("ℹ️  No emergency fixes needed")
            return True
            
    except Exception as e:
        # Restore from backup
        shutil.copy2(backup_path, file_path)
        print(f"❌ Emergency fix failed: {e}")
        print(f"🔄 Restored from backup")
        return False


def main():
    """Apply emergency fix to the known problematic file."""
    file_path = Path("/Users/sanjeev/Downloads/Repos/ChemML/src/chemml/integrations/adapters/__init__.py")
    
    print("🚨 Emergency Syntax Fix for ChemML")
    print("==================================")
    print(f"🔧 Processing: {file_path}")
    
    if emergency_syntax_fix(file_path):
        print("\n✅ Emergency fix completed!")
        print("🔍 Now you can run the comprehensive linting tools to get accurate results.")
        print("⚠️  Remember: This is a temporary fix. Proper code structure needs to be restored.")
    else:
        print("\n❌ Emergency fix failed!")


if __name__ == "__main__":
    main()
