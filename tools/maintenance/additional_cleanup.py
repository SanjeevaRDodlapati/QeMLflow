#!/usr/bin/env python3
"""
Additional Repository Cleanup Script for QeMLflow

This script removes remaining legacy migration scripts, debug files,
and other artifacts that weren't handled in the initial cleanup.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def log_message(message: str, level: str = "INFO"):
    """Log messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def remove_file_safely(path: Path, description: str):
    """Remove a file safely with logging."""
    if path.exists() and path.is_file():
        try:
            path.unlink()
            log_message(f"Removed {description}: {path}")
            return True
        except Exception as e:
            log_message(f"Error removing {description} {path}: {e}", "ERROR")
            return False
    else:
        log_message(f"{description} not found: {path}")
        return True


def main():
    """Main additional cleanup function."""
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)

    log_message("Starting additional repository cleanup...")

    removed_files = []

    # Legacy migration and debug scripts to remove
    legacy_files = [
        "debug_rollback.sh",
        "scripts/rename_to_qemlflow.sh",
        "scripts/test_rename_script.sh",
        "scripts/utilities/rename_to_qemlflow.py",
        "scripts/migration/migrate_to_hybrid_architecture.py",
    ]

    log_message("Removing legacy migration and debug scripts...")
    for file_path in legacy_files:
        path = repo_root / file_path
        if remove_file_safely(path, f"Legacy script"):
            removed_files.append(str(path))

    # Check for any remaining .bak files
    log_message("Checking for any remaining backup files...")
    for bak_file in repo_root.rglob("*.bak"):
        if remove_file_safely(bak_file, "Backup file"):
            removed_files.append(str(bak_file))

    # Check for any debug directories
    debug_dirs = ["debug_test", "temp_test", "migration_test"]
    for debug_dir in debug_dirs:
        debug_path = repo_root / debug_dir
        if debug_path.exists() and debug_path.is_dir():
            try:
                shutil.rmtree(debug_path)
                log_message(f"Removed debug directory: {debug_path}")
                removed_files.append(str(debug_path))
            except Exception as e:
                log_message(
                    f"Error removing debug directory {debug_path}: {e}", "ERROR"
                )

    # Summary
    log_message("=== Additional Cleanup Summary ===")
    log_message(f"Files/directories removed: {len(removed_files)}")
    for item in removed_files:
        log_message(f"  - {item}")

    if removed_files:
        log_message("Additional cleanup completed successfully!")
        log_message(
            "NEXT STEP: Review and commit the changes with 'git add . && git commit'"
        )
    else:
        log_message("No additional cleanup needed - repository is already clean!")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during cleanup: {e}")
        sys.exit(1)
