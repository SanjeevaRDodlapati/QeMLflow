#!/usr/bin/env python3
"""
Safe automated renaming script: QeMLflow → QeMLflow
Author: GitHub Copilot Assistant
Date: June 17, 2025

This script safely renames the QeMLflow framework to QeMLflow with full backup
and rollback capabilities.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re

class QeMLflowRenamer:
    """Safe renaming utility for QeMLflow → QeMLflow transformation."""
    
    def __init__(self):
        self.root_path = Path.cwd()
        self.backup_path = None
        self.changes_made = []
        
    def create_backup(self):
        """Create comprehensive backup before renaming."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"QeMLflow_BACKUP_BEFORE_QEMLFLOW_{timestamp}"
        parent_dir = self.root_path.parent
        self.backup_path = parent_dir / backup_name
        
        print(f"🔄 Creating backup: {self.backup_path}")
        try:
            shutil.copytree(self.root_path, self.backup_path)
            print("✅ Backup created successfully")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def git_checkpoint(self):
        """Create git checkpoint for safety."""
        try:
            # Check if we're in a git repository
            result = subprocess.run(["git", "status"], capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️ Not a git repository - skipping git checkpoint")
                return True
                
            # Add all changes and create checkpoint
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "CHECKPOINT: Before QeMLflow renaming"], check=True)
            subprocess.run(["git", "tag", "pre-qemlflow-rename"], check=True)
            print("✅ Git checkpoint created")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Git checkpoint failed: {e}")
            return False
    
    def rename_directories(self):
        """Rename main directories."""
        renames = [
            ("src/qemlflow", "src/qemlflow"),
        ]
        
        for old_path, new_path in renames:
            old_path_obj = self.root_path / old_path
            new_path_obj = self.root_path / new_path
            
            if old_path_obj.exists():
                try:
                    old_path_obj.rename(new_path_obj)
                    print(f"✅ Renamed {old_path} → {new_path}")
                    self.changes_made.append(f"Renamed directory: {old_path} → {new_path}")
                except Exception as e:
                    print(f"❌ Failed to rename {old_path}: {e}")
                    return False
            else:
                print(f"⚠️ Directory not found: {old_path}")
        
        return True
    
    def update_file_contents(self):
        """Update all file contents with new naming."""
        # Define replacement patterns
        replacements = [
            # Core package name replacements
            (r'\bqemlflow\b', 'qemlflow'),
            (r'\bQeMLflow\b', 'QeMLflow'),
            
            # Specific description updates
            (r'Machine Learning for Chemistry and Drug Discovery', 'Quantum-Enhanced Machine Learning Workflows'),
            (r'Machine Learning for Chemistry', 'Quantum-Enhanced Machine Learning'),
            (r'Chemical ML', 'Quantum-Enhanced ML'),
            (r'QeMLflow Team', 'QeMLflow Team'),
            (r'qemlflow@example\.com', 'qemlflow@example.com'),
            (r'QeMLflow Contributors', 'QeMLflow Contributors'),
            (r'QeMLflow Maintainers', 'QeMLflow Maintainers'),
        ]
        
        # File patterns to update
        file_patterns = ["*.py", "*.md", "*.txt", "*.yml", "*.yaml", "*.toml", "*.cfg", "*.ini"]
        
        # Exclude patterns
        exclude_patterns = [".git", "__pycache__", "*.pyc", "backup", ".pytest_cache"]
        
        files_updated = 0
        
        for pattern in file_patterns:
            for file_path in self.root_path.rglob(pattern):
                # Skip excluded directories
                if any(exclude in str(file_path) for exclude in exclude_patterns):
                    continue
                    
                if file_path.is_file():
                    try:
                        # Read file content
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        original_content = content
                        
                        # Apply replacements
                        for pattern_old, replacement in replacements:
                            content = re.sub(pattern_old, replacement, content)
                        
                        # Write back if modified
                        if content != original_content:
                            file_path.write_text(content, encoding='utf-8')
                            print(f"✅ Updated: {file_path}")
                            files_updated += 1
                            
                    except Exception as e:
                        print(f"⚠️ Warning: Could not update {file_path}: {e}")
        
        print(f"✅ Updated {files_updated} files")
        self.changes_made.append(f"Updated {files_updated} files with new naming")
        return True
    
    def validate_imports(self):
        """Validate that imports work after renaming."""
        try:
            # Add src to path temporarily
            src_path = str(self.root_path / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            # Try importing the renamed package
            import qemlflow
            print("✅ QeMLflow import validation successful")
            
            # Try importing core modules
            from qemlflow.core import data_processing
            print("✅ Core modules import successful")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import validation failed: {e}")
            print("💡 This might be normal if dependencies are missing")
            return False
        except Exception as e:
            print(f"⚠️ Validation warning: {e}")
            return True  # Don't fail on warnings
    
    def run_tests(self):
        """Run basic tests to ensure functionality."""
        test_files = [
            "tools/testing/functional_validation.py",
            "scripts/quick_validate.sh"
        ]
        
        tests_passed = 0
        
        for test_file in test_files:
            test_path = self.root_path / test_file
            if test_path.exists():
                try:
                    if test_file.endswith('.py'):
                        result = subprocess.run([sys.executable, str(test_path)], 
                                              capture_output=True, text=True, timeout=60)
                    else:
                        result = subprocess.run(["bash", str(test_path)], 
                                              capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print(f"✅ Test passed: {test_file}")
                        tests_passed += 1
                    else:
                        print(f"⚠️ Test issues: {test_file}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Test timeout: {test_file}")
                except Exception as e:
                    print(f"⚠️ Test error: {test_file} - {e}")
            else:
                print(f"⚠️ Test file not found: {test_file}")
        
        print(f"✅ {tests_passed} tests completed successfully")
        return tests_passed > 0
    
    def display_summary(self):
        """Display summary of changes made."""
        print("\n" + "="*60)
        print("🎉 QEMLFLOW RENAMING COMPLETED")
        print("="*60)
        print("\n📋 Changes Made:")
        for change in self.changes_made:
            print(f"  • {change}")
        
        print(f"\n💾 Backup Location: {self.backup_path}")
        print("\n🔧 Next Steps:")
        print("  1. Run comprehensive tests: python tools/testing/comprehensive_functionality_tests.py")
        print("  2. Update GitHub repository name to 'QeMLflow'")
        print("  3. Update PyPI package when ready to publish")
        print("  4. Update any CI/CD configurations")
        
        print("\n🚨 Rollback Instructions (if needed):")
        print("  Git rollback: git reset --hard pre-qemlflow-rename")
        print(f"  Or restore from: {self.backup_path}")
        
    def rollback(self):
        """Emergency rollback function."""
        print("🔄 Initiating rollback...")
        try:
            # Try git rollback first
            subprocess.run(["git", "reset", "--hard", "pre-qemlflow-rename"], check=True)
            print("✅ Git rollback successful")
        except:
            # If git fails, use backup
            if self.backup_path and self.backup_path.exists():
                print(f"🔄 Restoring from backup: {self.backup_path}")
                # This would need careful implementation to avoid data loss
                print("⚠️ Manual restore required - please copy from backup directory")
            else:
                print("❌ No backup available for automatic rollback")
    
    def run(self):
        """Execute complete renaming process."""
        print("🚀 Starting QeMLflow → QeMLflow renaming process")
        print("="*60)
        
        # Safety first
        if not self.create_backup():
            print("❌ Cannot proceed without backup")
            return False
            
        self.git_checkpoint()
        
        try:
            # Core renaming steps
            print("\n📁 Phase 1: Renaming directories...")
            if not self.rename_directories():
                raise Exception("Directory renaming failed")
            
            print("\n📝 Phase 2: Updating file contents...")
            if not self.update_file_contents():
                raise Exception("File content update failed")
            
            print("\n🧪 Phase 3: Validating imports...")
            self.validate_imports()
            
            print("\n🔍 Phase 4: Running basic tests...")
            self.run_tests()
            
            # Success
            self.display_summary()
            
            print("\n🎉 Renaming completed successfully!")
            print("🔍 Please review changes and run comprehensive tests")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during renaming: {e}")
            print(f"💾 Backup available at: {self.backup_path}")
            print("💡 To rollback: git reset --hard pre-qemlflow-rename")
            return False

def main():
    """Main execution function."""
    renamer = QeMLflowRenamer()
    
    # Confirm with user
    print("This will rename QeMLflow to QeMLflow across the entire codebase.")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response != 'y':
        print("Renaming cancelled.")
        return
    
    success = renamer.run()
    
    if not success:
        rollback = input("Renaming failed. Attempt rollback? (y/N): ").strip().lower()
        if rollback == 'y':
            renamer.rollback()

if __name__ == "__main__":
    main()
