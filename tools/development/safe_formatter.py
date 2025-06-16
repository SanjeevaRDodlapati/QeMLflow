#!/usr/bin/env python3
"""
Simple Code Formatter for ChemML

Applies basic formatting improvements without breaking syntax.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def safe_format_file(file_path: Path) -> bool:
    """Safely format a single Python file."""
    try:
        # Check syntax first
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, str(file_path), 'exec')
        
        # Apply black formatting
        result = subprocess.run([
            sys.executable, "-m", "black",
            "--line-length", "88",
            "--quiet",
            str(file_path)
        ], capture_output=True)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"⚠️ Skipping {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Safe code formatter')
    parser.add_argument('files', nargs='*', help='Files to format')
    parser.add_argument('--directory', help='Format all Python files in directory')
    
    args = parser.parse_args()
    
    files_to_format = []
    
    if args.directory:
        directory = Path(args.directory)
        files_to_format.extend(directory.glob("**/*.py"))
    
    if args.files:
        files_to_format.extend([Path(f) for f in args.files])
    
    if not files_to_format:
        print("No files specified")
        return
    
    print(f"🎨 Formatting {len(files_to_format)} files...")
    
    success_count = 0
    for file_path in files_to_format:
        if safe_format_file(file_path):
            success_count += 1
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    print(f"🎉 Successfully formatted {success_count}/{len(files_to_format)} files")


if __name__ == "__main__":
    main()
