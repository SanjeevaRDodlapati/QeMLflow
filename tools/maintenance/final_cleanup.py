#!/usr/bin/env python3
"""
Final Repository Cleanup Script for QeMLflow Migration

This script performs final cleanup tasks to remove all temporary files,
test environments, and unnecessary artifacts from the migration process.
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def log_message(message: str, level: str = "INFO"):
    """Log messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def remove_directory_safely(path: Path, description: str):
    """Remove a directory safely with logging."""
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
            log_message(f"Removed {description}: {path}")
            return True
        except Exception as e:
            log_message(f"Error removing {description} {path}: {e}", "ERROR")
            return False
    else:
        log_message(f"{description} not found: {path}")
        return True


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
    """Main cleanup function."""
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)

    log_message("Starting final repository cleanup...")

    cleanup_summary = {
        "timestamp": datetime.now().isoformat(),
        "removed_directories": [],
        "removed_files": [],
        "errors": [],
        "size_before_mb": 0,
        "size_after_mb": 0,
    }

    # Calculate initial size
    try:
        initial_size = sum(
            f.stat().st_size for f in Path(".").rglob("*") if f.is_file()
        )
        cleanup_summary["size_before_mb"] = initial_size / (1024 * 1024)
    except Exception as e:
        log_message(f"Error calculating initial size: {e}", "ERROR")

    # 1. Remove test environment directory
    test_env_path = repo_root / "test_rename_environment"
    if remove_directory_safely(test_env_path, "Test environment directory"):
        cleanup_summary["removed_directories"].append(str(test_env_path))

    # 2. Remove old backup directories in backups/
    backups_path = repo_root / "backups"
    if backups_path.exists():
        for item in backups_path.iterdir():
            if item.is_dir() and (
                "lint" in item.name.lower() or "robust" in item.name.lower()
            ):
                if remove_directory_safely(item, f"Old backup directory"):
                    cleanup_summary["removed_directories"].append(str(item))

    # 3. Remove site directory (can be regenerated)
    site_path = repo_root / "site"
    if remove_directory_safely(site_path, "Generated site directory"):
        cleanup_summary["removed_directories"].append(str(site_path))

    # 4. Remove any remaining .temp directories
    for temp_dir in repo_root.rglob(".temp"):
        if temp_dir.is_dir():
            if remove_directory_safely(temp_dir, "Temporary directory"):
                cleanup_summary["removed_directories"].append(str(temp_dir))

    # 5. Remove Python cache files
    cache_removed = 0
    for pycache in repo_root.rglob("__pycache__"):
        if pycache.is_dir():
            if remove_directory_safely(pycache, "Python cache directory"):
                cache_removed += 1
                cleanup_summary["removed_directories"].append(str(pycache))

    for pyc_file in repo_root.rglob("*.pyc"):
        if remove_file_safely(pyc_file, "Python compiled file"):
            cleanup_summary["removed_files"].append(str(pyc_file))

    # 6. Remove any .DS_Store files (macOS)
    for ds_store in repo_root.rglob(".DS_Store"):
        if remove_file_safely(ds_store, "macOS .DS_Store file"):
            cleanup_summary["removed_files"].append(str(ds_store))

    # 7. Remove any temporary files
    temp_patterns = ["*.tmp", "*.temp", "*.bak"]
    for pattern in temp_patterns:
        for temp_file in repo_root.rglob(pattern):
            if temp_file.is_file():
                if remove_file_safely(temp_file, f"Temporary file ({pattern})"):
                    cleanup_summary["removed_files"].append(str(temp_file))

    # 8. Clean up any empty directories (except .git and important dirs)
    important_dirs = {".git", "src", "docs", "tests", "tools", "examples", "notebooks"}
    empty_dirs_removed = []

    for root, dirs, files in os.walk(repo_root, topdown=False):
        root_path = Path(root)
        # Skip if it's an important directory or under .git
        if any(part in important_dirs for part in root_path.parts):
            continue
        if ".git" in root_path.parts:
            continue

        try:
            if not files and not dirs and root_path != repo_root:
                root_path.rmdir()
                empty_dirs_removed.append(str(root_path))
                log_message(f"Removed empty directory: {root_path}")
        except OSError:
            pass  # Directory not empty or permission issue

    cleanup_summary["removed_directories"].extend(empty_dirs_removed)

    # Calculate final size
    try:
        final_size = sum(f.stat().st_size for f in Path(".").rglob("*") if f.is_file())
        cleanup_summary["size_after_mb"] = final_size / (1024 * 1024)
        size_saved = (
            cleanup_summary["size_before_mb"] - cleanup_summary["size_after_mb"]
        )
        log_message(f"Size before cleanup: {cleanup_summary['size_before_mb']:.2f} MB")
        log_message(f"Size after cleanup: {cleanup_summary['size_after_mb']:.2f} MB")
        log_message(f"Space saved: {size_saved:.2f} MB")
    except Exception as e:
        log_message(f"Error calculating final size: {e}", "ERROR")

    # Save cleanup summary
    summary_file = (
        repo_root / "reports" / "migration_validation" / "final_cleanup_summary.json"
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(summary_file, "w") as f:
            json.dump(cleanup_summary, f, indent=2)
        log_message(f"Cleanup summary saved to: {summary_file}")
    except Exception as e:
        log_message(f"Error saving cleanup summary: {e}", "ERROR")

    # Summary statistics
    log_message("=== Final Cleanup Summary ===")
    log_message(f"Directories removed: {len(cleanup_summary['removed_directories'])}")
    log_message(f"Files removed: {len(cleanup_summary['removed_files'])}")
    log_message(f"Python cache directories removed: {cache_removed}")
    log_message(f"Empty directories removed: {len(empty_dirs_removed)}")

    if cleanup_summary.get("size_before_mb") and cleanup_summary.get("size_after_mb"):
        size_saved = (
            cleanup_summary["size_before_mb"] - cleanup_summary["size_after_mb"]
        )
        log_message(f"Total space saved: {size_saved:.2f} MB")

    log_message("Final cleanup completed successfully!")

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
