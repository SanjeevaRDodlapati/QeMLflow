#!/usr/bin/env python3
"""
Safe Renaming Script: QeMLflow -> QeMLflow

This script safely renames all instances of QeMLflow to QeMLflow throughout the codebase.
It includes comprehensive backup, validation, and rollback capabilities.

Usage:
    python safe_rename_to_qemlflow.py --dry-run      # Preview changes
    python safe_rename_to_qemlflow.py --backup-only  # Create backup only
    python safe_rename_to_qemlflow.py --execute      # Execute renaming
    python safe_rename_to_qemlflow.py --rollback BACKUP_DIR  # Rollback changes
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


class QeMLflowRenamer:
    def __init__(self):
        self.backup_dir = None
        self.changes_log = []
        self.errors = []

        # Define replacement patterns
        self.replacements = {
            # Exact matches (case sensitive)
            "qemlflow": "qemlflow",
            "QeMLflow": "QeMLflow",
            "QEMLFLOW": "QEMLFLOW",
            "qeMLflow": "qeMLflow",  # Handle camelCase variations
            # Package and module names
            "src/qemlflow": "src/qemlflow",
            "qemlflow/": "qemlflow/",
            "qemlflow.": "qemlflow.",
            "from qemlflow": "from qemlflow",
            "import qemlflow": "import qemlflow",
            # Configuration and metadata
            '"qemlflow"': '"qemlflow"',
            "'qemlflow'": "'qemlflow'",
            '"QeMLflow"': '"QeMLflow"',
            "'QeMLflow'": "'QeMLflow'",
            # Documentation and comments
            "# QeMLflow": "# QeMLflow",
            "## QeMLflow": "## QeMLflow",
            "### QeMLflow": "### QeMLflow",
        }

        # Files to process (by extension)
        self.processable_extensions = {
            ".py",
            ".md",
            ".txt",
            ".rst",
            ".yaml",
            ".yml",
            ".toml",
            ".cfg",
            ".ini",
            ".json",
            ".sh",
            ".bat",
            ".html",
            ".xml",
            ".js",
            ".css",
        }

        # Files and directories to skip
        self.skip_patterns = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            ".DS_Store",
            "Thumbs.db",
            "*.pyc",
            "*.pyo",
            "*.egg-info",
            "dist",
            "build",
            ".tox",
        }

        # Binary file patterns to skip
        self.binary_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".7z",
            ".exe",
            ".dmg",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".wav",
            ".flac",
        }

    def should_skip_path(self, path):
        """Check if a path should be skipped"""
        path_str = str(path)

        # Skip if it's a backup directory we created
        if "qemlflow_backup_" in path_str or "qemlflow_backup_" in path_str:
            return True

        # Check skip patterns
        for pattern in self.skip_patterns:
            if pattern in path_str or path.name == pattern:
                return True

        # Skip binary files
        if path.suffix.lower() in self.binary_extensions:
            return True

        return False

    def is_text_file(self, file_path):
        """Check if a file is a text file"""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:  # Null bytes indicate binary
                    return False
            return True
        except:
            return False

    def create_backup(self):
        """Create a complete backup of the current state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = f"qemlflow_backup_{timestamp}"

        print(f"📦 Creating backup in {self.backup_dir}...")

        try:
            # Create backup directory
            os.makedirs(self.backup_dir, exist_ok=True)

            # Copy entire project (excluding certain directories)
            for item in os.listdir("."):
                if not self.should_skip_path(Path(item)):
                    src = Path(item)
                    dst = Path(self.backup_dir) / item

                    if src.is_dir():
                        shutil.copytree(
                            src, dst, ignore=shutil.ignore_patterns(*self.skip_patterns)
                        )
                    else:
                        shutil.copy2(src, dst)

            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "original_dir": os.getcwd(),
                "replacements": self.replacements,
                "purpose": "QeMLflow to QeMLflow renaming backup",
            }

            with open(f"{self.backup_dir}/backup_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Backup created successfully: {self.backup_dir}")
            return True

        except Exception as e:
            print(f"❌ Backup failed: {e}")
            self.errors.append(f"Backup creation failed: {e}")
            return False

    def find_files_to_process(self):
        """Find all files that need processing"""
        files_to_process = []

        for root, dirs, files in os.walk("."):
            # Skip backup directories and other excluded dirs
            dirs[:] = [d for d in dirs if not self.should_skip_path(Path(root) / d)]

            for file in files:
                file_path = Path(root) / file

                if (
                    not self.should_skip_path(file_path)
                    and file_path.suffix in self.processable_extensions
                    and self.is_text_file(file_path)
                ):
                    files_to_process.append(file_path)

        return files_to_process

    def preview_changes(self, file_path):
        """Preview what changes would be made to a file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            modified_content = content
            changes_made = []

            for old, new in self.replacements.items():
                if old in modified_content:
                    count = modified_content.count(old)
                    modified_content = modified_content.replace(old, new)
                    changes_made.append(f"{old} -> {new} ({count} times)")

            return changes_made, modified_content if changes_made else None

        except Exception as e:
            self.errors.append(f"Error previewing {file_path}: {e}")
            return [], None

    def process_file(self, file_path, dry_run=False):
        """Process a single file"""
        try:
            changes, modified_content = self.preview_changes(file_path)

            if changes:
                change_info = {
                    "file": str(file_path),
                    "changes": changes,
                    "timestamp": datetime.now().isoformat(),
                }

                if not dry_run and modified_content:
                    # Write the modified content
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(modified_content)
                    change_info["status"] = "applied"
                else:
                    change_info["status"] = "preview" if dry_run else "no_changes"

                self.changes_log.append(change_info)
                return True

            return False

        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            self.errors.append(error_msg)
            print(f"⚠️  {error_msg}")
            return False

    def rename_directories(self, dry_run=False):
        """Rename directories (specifically src/qemlflow -> src/qemlflow)"""
        directories_to_rename = []

        # Find directories that need renaming
        for root, dirs, files in os.walk(".", topdown=False):
            for dir_name in dirs:
                if "qemlflow" in dir_name.lower():
                    old_path = Path(root) / dir_name
                    new_name = dir_name
                    for old, new in self.replacements.items():
                        new_name = new_name.replace(old, new)
                    new_path = Path(root) / new_name

                    if old_path != new_path:
                        directories_to_rename.append((old_path, new_path))

        # Rename directories
        for old_path, new_path in directories_to_rename:
            try:
                if dry_run:
                    print(f"📁 Would rename: {old_path} -> {new_path}")
                else:
                    print(f"📁 Renaming: {old_path} -> {new_path}")
                    shutil.move(str(old_path), str(new_path))

                self.changes_log.append(
                    {
                        "type": "directory_rename",
                        "old_path": str(old_path),
                        "new_path": str(new_path),
                        "status": "preview" if dry_run else "applied",
                    }
                )

            except Exception as e:
                error_msg = f"Error renaming directory {old_path}: {e}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")

    def dry_run(self):
        """Perform a dry run to preview all changes"""
        print("🔍 DRY RUN: Previewing all changes...")
        print("=" * 60)

        # Find files to process
        files_to_process = self.find_files_to_process()
        print(f"📄 Found {len(files_to_process)} files to process")

        # Preview directory renames
        print("\n📁 Directory renames:")
        self.rename_directories(dry_run=True)

        # Preview file changes
        print(f"\n📝 File content changes:")
        files_with_changes = 0

        for file_path in files_to_process:
            changes, _ = self.preview_changes(file_path)
            if changes:
                files_with_changes += 1
                print(f"\n📄 {file_path}:")
                for change in changes:
                    print(f"   • {change}")

        print(f"\n📊 SUMMARY:")
        print(f"   • Files with changes: {files_with_changes}/{len(files_to_process)}")
        print(
            f"   • Directory renames: {len([c for c in self.changes_log if c.get('type') == 'directory_rename'])}"
        )
        print(
            f"   • Total changes: {sum(len(c.get('changes', [])) for c in self.changes_log)}"
        )

        if self.errors:
            print(f"   • Errors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"     ⚠️  {error}")

    def execute_rename(self):
        """Execute the full renaming process"""
        print("🚀 EXECUTING RENAME: QeMLflow -> QeMLflow")
        print("=" * 60)

        # Create backup first
        if not self.create_backup():
            print("❌ Cannot proceed without backup!")
            return False

        try:
            # Find files to process
            files_to_process = self.find_files_to_process()
            print(f"📄 Processing {len(files_to_process)} files...")

            # Process files
            processed_files = 0
            for i, file_path in enumerate(files_to_process, 1):
                if self.process_file(file_path, dry_run=False):
                    processed_files += 1

                # Progress indicator
                if i % 10 == 0 or i == len(files_to_process):
                    print(f"   Progress: {i}/{len(files_to_process)} files processed")

            # Rename directories
            print("\n📁 Renaming directories...")
            self.rename_directories(dry_run=False)

            # Save changes log
            log_file = f"{self.backup_dir}/changes_log.json"
            with open(log_file, "w") as f:
                json.dump(
                    {
                        "changes": self.changes_log,
                        "errors": self.errors,
                        "summary": {
                            "files_processed": processed_files,
                            "total_files": len(files_to_process),
                            "directories_renamed": len(
                                [
                                    c
                                    for c in self.changes_log
                                    if c.get("type") == "directory_rename"
                                ]
                            ),
                        },
                    },
                    f,
                    indent=2,
                )

            print(f"\n✅ RENAMING COMPLETED!")
            print(f"   • Files processed: {processed_files}/{len(files_to_process)}")
            print(f"   • Backup created: {self.backup_dir}")
            print(f"   • Changes log: {log_file}")

            if self.errors:
                print(f"   ⚠️  Errors encountered: {len(self.errors)}")
                for error in self.errors:
                    print(f"     • {error}")
                print(f"   📝 Check {log_file} for details")

            return True

        except Exception as e:
            print(f"❌ Renaming failed: {e}")
            print(
                f"🔄 You can rollback using: python {sys.argv[0]} --rollback {self.backup_dir}"
            )
            return False

    def rollback(self, backup_dir):
        """Rollback changes from a backup"""
        print(f"🔄 ROLLING BACK from backup: {backup_dir}")
        print("=" * 60)

        if not os.path.exists(backup_dir):
            print(f"❌ Backup directory not found: {backup_dir}")
            return False

        try:
            # Load backup metadata
            metadata_file = f"{backup_dir}/backup_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                print(f"📋 Backup created: {metadata.get('timestamp', 'unknown')}")

            # Create a backup of current state before rollback
            current_backup = (
                f"pre_rollback_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print(f"📦 Creating backup of current state: {current_backup}")
            os.makedirs(current_backup, exist_ok=True)

            # Remove current files/dirs that exist in backup (except backup dirs)
            items_to_restore = os.listdir(backup_dir)
            items_to_restore.remove("backup_metadata.json")  # Skip metadata file
            if "changes_log.json" in items_to_restore:
                items_to_restore.remove("changes_log.json")  # Skip log file

            for item in items_to_restore:
                current_path = Path(item)
                if current_path.exists() and not self.should_skip_path(current_path):
                    # Backup current version
                    backup_path = Path(current_backup) / item
                    if current_path.is_dir():
                        shutil.copytree(current_path, backup_path)
                        shutil.rmtree(current_path)
                    else:
                        shutil.copy2(current_path, backup_path)
                        os.remove(current_path)

            # Restore from backup
            for item in items_to_restore:
                source_path = Path(backup_dir) / item
                target_path = Path(item)

                if source_path.is_dir():
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copy2(source_path, target_path)

            print(f"✅ ROLLBACK COMPLETED!")
            print(f"   • Restored from: {backup_dir}")
            print(f"   • Current state backed up to: {current_backup}")
            return True

        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Safe QeMLflow to QeMLflow renaming script"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying them"
    )
    parser.add_argument(
        "--backup-only", action="store_true", help="Create backup only, do not rename"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute the renaming process"
    )
    parser.add_argument(
        "--rollback", type=str, help="Rollback from specified backup directory"
    )

    args = parser.parse_args()

    renamer = QeMLflowRenamer()

    # Check arguments
    if args.rollback:
        success = renamer.rollback(args.rollback)
        sys.exit(0 if success else 1)
    elif args.dry_run:
        renamer.dry_run()
    elif args.backup_only:
        success = renamer.create_backup()
        sys.exit(0 if success else 1)
    elif args.execute:
        success = renamer.execute_rename()
        sys.exit(0 if success else 1)
    else:
        # No arguments provided, show help and run dry-run by default
        parser.print_help()
        print("\n" + "=" * 60)
        print("Running dry-run by default...")
        print("=" * 60)
        renamer.dry_run()


if __name__ == "__main__":
    main()
