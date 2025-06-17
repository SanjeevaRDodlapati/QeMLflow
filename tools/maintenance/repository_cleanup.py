#!/usr/bin/env python3
"""
QeMLflow Repository Cleanup Tool
===============================

Cleans up and organizes the repository after the comprehensive migration and testing.
This tool will:
1. Remove redundant files
2. Consolidate test outputs
3. Archive migration artifacts
4. Clean up temporary files
5. Organize documentation

Author: Repository Cleanup System
Date: June 17, 2025
"""

import os
import sys
import shutil
import json
from pathlib import Path
from typing import List, Dict, Set
import subprocess

class RepositoryCleanup:
    """Clean up and organize the QeMLflow repository."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.cleanup_actions = []
        self.preserved_files = []
        
    def log_action(self, action: str, details: str):
        """Log a cleanup action."""
        action_msg = f"🧹 {action}: {details}"
        self.cleanup_actions.append(action_msg)
        print(action_msg)
    
    def analyze_current_state(self):
        """Analyze the current repository state."""
        print("📊 Analyzing repository state...")
        
        # Count files by type
        file_counts = {}
        total_size = 0
        
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                file_counts[suffix] = file_counts.get(suffix, 0) + 1
                try:
                    total_size += file_path.stat().st_size
                except (OSError, PermissionError):
                    pass
        
        print(f"Total files: {sum(file_counts.values())}")
        print(f"Repository size: {total_size / (1024*1024):.1f} MB")
        
        # Show top file types
        sorted_types = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        print("Top file types:")
        for ext, count in sorted_types[:10]:
            print(f"  {ext or '(no extension)'}: {count}")
    
    def consolidate_validation_outputs(self):
        """Consolidate all validation and test outputs."""
        print("\n🗂️  Consolidating validation outputs...")
        
        # Create consolidated reports directory
        reports_dir = self.root_path / "reports" / "migration_validation"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all test/validation outputs
        validation_files = []
        
        # Look for validation files
        validation_patterns = [
            "**/migration_validation_report.json",
            "**/final_migration_report.json", 
            "**/ml_pipeline_test_report.json",
            "**/chemistry_ml_test_report.json",
            "**/*test_outputs*",
            "**/validation/*.png",
            "**/validation/*.json"
        ]
        
        for pattern in validation_patterns:
            for file_path in self.root_path.glob(pattern):
                if file_path.is_file():
                    validation_files.append(file_path)
        
        # Consolidate files
        consolidated_count = 0
        for file_path in validation_files:
            if 'migration_validation' not in str(file_path):  # Don't move files already in target
                relative_path = file_path.relative_to(self.root_path)
                new_path = reports_dir / file_path.name
                
                # Handle naming conflicts
                counter = 1
                while new_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    new_path = reports_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                try:
                    shutil.copy2(file_path, new_path)
                    consolidated_count += 1
                    self.log_action("CONSOLIDATE", f"{relative_path} -> {new_path.relative_to(self.root_path)}")
                except Exception as e:
                    print(f"⚠️  Could not consolidate {file_path}: {e}")
        
        self.log_action("CONSOLIDATE", f"Moved {consolidated_count} validation files to reports/migration_validation/")
    
    def archive_migration_tools(self):
        """Archive migration-specific tools."""
        print("\n📦 Archiving migration tools...")
        
        # Create migration archive
        archive_dir = self.root_path / "tools" / "archived" / "migration_2025_06_17"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Migration tools to archive
        migration_tools = [
            "tools/migration/safe_rename_to_qemlflow.py",
            "tools/migration/test_rename_script.py",
            "tools/validation/migration_fixer.py",
            "tools/validation/comprehensive_migration_test.py"
        ]
        
        archived_count = 0
        for tool_path_str in migration_tools:
            tool_path = self.root_path / tool_path_str
            if tool_path.exists():
                archive_file = archive_dir / tool_path.name
                try:
                    shutil.copy2(tool_path, archive_file)
                    archived_count += 1
                    self.log_action("ARCHIVE", f"{tool_path_str} -> {archive_file.relative_to(self.root_path)}")
                except Exception as e:
                    print(f"⚠️  Could not archive {tool_path}: {e}")
        
        self.log_action("ARCHIVE", f"Archived {archived_count} migration tools")
    
    def clean_temporary_files(self):
        """Remove temporary and cache files."""
        print("\n🗑️  Cleaning temporary files...")
        
        # Patterns for temporary files
        temp_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo", 
            "**/.pytest_cache",
            "**/node_modules",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.tmp",
            "**/*.temp",
            "**/.*cache*"
        ]
        
        cleaned_count = 0
        cleaned_size = 0
        
        for pattern in temp_patterns:
            for item in self.root_path.glob(pattern):
                try:
                    if item.is_file():
                        size = item.stat().st_size
                        item.unlink()
                        cleaned_count += 1
                        cleaned_size += size
                        self.log_action("DELETE", f"Temp file: {item.relative_to(self.root_path)}")
                    elif item.is_dir():
                        # Calculate directory size
                        dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        shutil.rmtree(item)
                        cleaned_count += 1
                        cleaned_size += dir_size
                        self.log_action("DELETE", f"Temp directory: {item.relative_to(self.root_path)}")
                except Exception as e:
                    print(f"⚠️  Could not clean {item}: {e}")
        
        self.log_action("CLEAN", f"Removed {cleaned_count} temp items, freed {cleaned_size/(1024*1024):.1f} MB")
    
    def remove_redundant_validation_scripts(self):
        """Remove redundant validation scripts after consolidation."""
        print("\n🔄 Removing redundant validation scripts...")
        
        # Scripts to remove after consolidation
        redundant_scripts = [
            "tools/validation/ml_pipeline_tests.py",
            "tools/validation/chemistry_ml_tests.py",
            "tools/validation/quick_migration_check.py"
        ]
        
        removed_count = 0
        for script_path_str in redundant_scripts:
            script_path = self.root_path / script_path_str
            if script_path.exists():
                try:
                    script_path.unlink()
                    removed_count += 1
                    self.log_action("REMOVE", f"Redundant script: {script_path_str}")
                except Exception as e:
                    print(f"⚠️  Could not remove {script_path}: {e}")
        
        # Remove ml_test_outputs directory if empty or move to reports
        ml_outputs_dir = self.root_path / "tools" / "validation" / "ml_test_outputs"
        if ml_outputs_dir.exists():
            try:
                # Move to reports if has content, otherwise remove
                files = list(ml_outputs_dir.rglob('*'))
                if files:
                    target_dir = self.root_path / "reports" / "migration_validation" / "ml_outputs"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for file_path in files:
                        if file_path.is_file():
                            shutil.move(str(file_path), str(target_dir / file_path.name))
                    self.log_action("MOVE", f"ML outputs moved to reports/migration_validation/ml_outputs/")
                
                # Remove the empty directory
                if ml_outputs_dir.exists():
                    shutil.rmtree(ml_outputs_dir)
                    self.log_action("REMOVE", "Empty ml_test_outputs directory")
                    
            except Exception as e:
                print(f"⚠️  Could not process ml_test_outputs: {e}")
        
        self.log_action("REMOVE", f"Removed {removed_count} redundant validation scripts")
    
    def organize_documentation(self):
        """Organize and consolidate documentation."""
        print("\n📚 Organizing documentation...")
        
        # Create final documentation structure
        final_docs = self.root_path / "docs" / "migration"
        final_docs.mkdir(parents=True, exist_ok=True)
        
        # Key documentation files to preserve
        key_docs = [
            "reports/final/qemlflow_renaming_implementation_plan.md",
            "reports/final/chemml_renaming_analysis.md", 
            "reports/final/qemlflow_script_validation_report.md"
        ]
        
        organized_count = 0
        for doc_path_str in key_docs:
            doc_path = self.root_path / doc_path_str
            if doc_path.exists():
                target_path = final_docs / doc_path.name
                try:
                    shutil.copy2(doc_path, target_path)
                    organized_count += 1
                    self.log_action("ORGANIZE", f"Doc: {doc_path_str} -> {target_path.relative_to(self.root_path)}")
                except Exception as e:
                    print(f"⚠️  Could not organize {doc_path}: {e}")
        
        self.log_action("ORGANIZE", f"Organized {organized_count} key documentation files")
    
    def create_cleanup_summary(self):
        """Create a summary of cleanup actions."""
        print("\n📋 Creating cleanup summary...")
        
        summary = {
            "cleanup_date": "2025-06-17",
            "cleanup_purpose": "Post-migration repository organization",
            "actions_taken": self.cleanup_actions,
            "preserved_files": self.preserved_files,
            "repository_structure": {
                "migration_reports": "reports/migration_validation/",
                "archived_tools": "tools/archived/migration_2025_06_17/",
                "key_documentation": "docs/migration/",
                "main_codebase": "src/qemlflow/"
            },
            "recommendations": [
                "Regular cleanup of test outputs",
                "Archive migration tools after major updates",
                "Maintain clean separation between core code and validation",
                "Use reports/ directory for all validation outputs"
            ]
        }
        
        summary_file = self.root_path / "reports" / "migration_validation" / "cleanup_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log_action("CREATE", f"Cleanup summary: {summary_file.relative_to(self.root_path)}")
    
    def update_gitignore(self):
        """Update .gitignore for cleaner repository."""
        print("\n🔧 Updating .gitignore...")
        
        gitignore_path = self.root_path / ".gitignore"
        
        # Additional entries for cleaner repo
        new_entries = [
            "",
            "# Post-migration cleanup",
            "tools/validation/ml_test_outputs/",
            "tools/validation/*_report.json",
            "**/*.tmp",
            "**/*.temp",
            "reports/temp/",
            ""
        ]
        
        try:
            # Read existing content
            if gitignore_path.exists():
                with open(gitignore_path, 'r') as f:
                    content = f.read()
            else:
                content = ""
            
            # Add new entries if not present
            for entry in new_entries:
                if entry.strip() and entry not in content:
                    content += entry + '\n'
            
            # Write back
            with open(gitignore_path, 'w') as f:
                f.write(content)
            
            self.log_action("UPDATE", ".gitignore with cleanup patterns")
            
        except Exception as e:
            print(f"⚠️  Could not update .gitignore: {e}")
    
    def run_full_cleanup(self):
        """Run the complete cleanup process."""
        print("🚀 Starting QeMLflow Repository Cleanup")
        print("=" * 60)
        
        # Analyze current state
        self.analyze_current_state()
        
        # Run cleanup steps
        self.consolidate_validation_outputs()
        self.archive_migration_tools()
        self.organize_documentation()
        self.remove_redundant_validation_scripts()
        self.clean_temporary_files()
        self.update_gitignore()
        self.create_cleanup_summary()
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 CLEANUP SUMMARY")
        print("=" * 60)
        print(f"Total actions performed: {len(self.cleanup_actions)}")
        
        print("\n🗂️  New Organization:")
        print("  📋 Migration Reports: reports/migration_validation/")
        print("  📦 Archived Tools: tools/archived/migration_2025_06_17/") 
        print("  📚 Key Docs: docs/migration/")
        print("  💼 Core Code: src/qemlflow/")
        
        print("\n✅ Repository cleanup complete!")
        print("🎯 QeMLflow is now organized and ready for continued development")
        
        # Git status
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  cwd=self.root_path, capture_output=True, text=True)
            if result.returncode == 0:
                changed_files = [line for line in result.stdout.strip().split('\n') if line.strip()]
                print(f"\n📝 Git status: {len(changed_files)} files changed")
            else:
                print("⚠️  Could not check git status")
        except Exception:
            print("⚠️  Git status check failed")

def main():
    """Main entry point."""
    # Get root directory
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent
    
    cleanup = RepositoryCleanup(str(root_dir))
    cleanup.run_full_cleanup()

if __name__ == "__main__":
    main()
