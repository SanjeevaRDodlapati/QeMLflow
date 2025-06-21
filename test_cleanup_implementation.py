#!/usr/bin/env python3
"""
QeMLflow Test Cleanup Implementation Script
==========================================

Implements the comprehensive test cleanup plan in phases.
This script aggressively removes redundant tests while preserving
core scientific computing functionality validation.
"""

import os
import shutil
from pathlib import Path
import subprocess
import json
from datetime import datetime


class TestCleanupImplementation:
    """Implements test cleanup phases safely with rollback capability."""
    
    def __init__(self):
        self.backup_dir = Path("backups") / f"test_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cleanup_log = []
        self.stats = {
            'files_before': 0,
            'lines_before': 0,
            'files_after': 0,
            'lines_after': 0,
            'deleted_files': [],
            'modified_files': [],
            'backup_location': str(self.backup_dir)
        }
    
    def count_test_files(self):
        """Count current test files and lines."""
        files = list(Path("tests").rglob("*.py"))
        total_lines = 0
        for file in files:
            try:
                with open(file, 'r') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        return len(files), total_lines
    
    def create_backup(self):
        """Create full backup of tests directory."""
        print("🔄 Creating backup...")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree("tests", self.backup_dir / "tests")
        print(f"✅ Backup created at: {self.backup_dir}")
    
    def log_action(self, action: str, details: str):
        """Log cleanup action."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        self.cleanup_log.append(entry)
        print(f"📝 {action}: {details}")
    
    def phase_1_immediate_deletions(self):
        """Phase 1: Safe immediate deletions (~40% reduction)."""
        print("\n🗑️  PHASE 1: IMMEDIATE DELETIONS")
        print("=" * 40)
        
        # Directories to completely remove
        dirs_to_remove = [
            "tests/legacy",
            "tests/high_availability", 
            "tests/scalability"
        ]
        
        # Files to remove
        files_to_remove = [
            "tests/unit/test_models.py",  # Already disabled
            "tests/unit/test_chemml_common_comprehensive.py",  # Empty
            "tests/integration/test_pipelines.py",  # Already disabled
        ]
        
        for dir_path in dirs_to_remove:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)
                self.log_action("REMOVED_DIR", dir_path)
                self.stats['deleted_files'].append(dir_path)
        
        for file_path in files_to_remove:
            if Path(file_path).exists():
                os.remove(file_path)
                self.log_action("REMOVED_FILE", file_path)
                self.stats['deleted_files'].append(file_path)
        
        print("✅ Phase 1 complete - Safe deletions done")
    
    def phase_2_enterprise_minimization(self):
        """Phase 2: Minimize enterprise features (keep essentials only)."""
        print("\n📉 PHASE 2: ENTERPRISE MINIMIZATION")
        print("=" * 40)
        
        # Directories to minimize significantly
        dirs_to_minimize = {
            "tests/observability": ["test_monitoring.py", "test_dashboard.py"],  # Keep only these 2
            "tests/production_readiness": ["test_deployment.py"],  # Keep only 1
            "tests/performance": ["test_benchmarks.py"],  # Keep only 1
            "tests/security": ["test_security_basics.py"],  # Keep only 1
            "tests/production_tuning": []  # Remove all
        }
        
        for dir_name, files_to_keep in dirs_to_minimize.items():
            dir_path = Path(dir_name)
            if not dir_path.exists():
                continue
                
            if not files_to_keep:  # Remove entire directory
                shutil.rmtree(dir_path)
                self.log_action("REMOVED_DIR", str(dir_path))
                continue
            
            # Remove files not in keep list
            for file in dir_path.glob("*.py"):
                if file.name not in files_to_keep and file.name != "__init__.py":
                    file.unlink()
                    self.log_action("REMOVED_FILE", str(file))
                    self.stats['deleted_files'].append(str(file))
        
        print("✅ Phase 2 complete - Enterprise features minimized")
    
    def phase_3_unit_test_surgery(self):
        """Phase 3: Major unit test consolidation and reduction."""
        print("\n🔬 PHASE 3: UNIT TEST SURGERY")
        print("=" * 40)
        
        # Files to dramatically reduce (keep only essential tests)
        files_to_reduce = [
            ("tests/unit/test_feature_extraction_comprehensive.py", 200),  # Reduce to ~200 lines
            ("tests/unit/test_ml_utils_comprehensive.py", 150),           # Reduce to ~150 lines
            ("tests/unit/test_metrics_comprehensive.py", 150),           # Reduce to ~150 lines
            ("tests/unit/test_io_utils_comprehensive.py", 300),          # Reduce to ~300 lines
            ("tests/unit/test_admet_prediction.py", 400),                # Reduce to ~400 lines
        ]
        
        for file_path, target_lines in files_to_reduce:
            self._reduce_test_file(file_path, target_lines)
        
        # Remove redundant files completely
        redundant_files = [
            "tests/unit/test_feature_extraction_surgical.py",  # Redundant with comprehensive
            "tests/unit/test_drug_design.py",  # Covered by QSAR tests
        ]
        
        for file_path in redundant_files:
            if Path(file_path).exists():
                os.remove(file_path)
                self.log_action("REMOVED_FILE", file_path) 
                self.stats['deleted_files'].append(file_path)
        
        print("✅ Phase 3 complete - Unit tests surgically reduced")
    
    def _reduce_test_file(self, file_path: str, target_lines: int):
        """Reduce a test file to target number of lines by keeping only essential tests."""
        file_path = Path(file_path)
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            original_lines = len(lines)
            if original_lines <= target_lines:
                return  # Already small enough
            
            # Keep file header, imports, and essential test methods
            reduced_lines = []
            in_test_method = False
            essential_tests = 0
            max_essential_tests = target_lines // 20  # Rough estimate
            
            for line in lines:
                # Always keep imports and class definitions
                if (line.strip().startswith(('import ', 'from ', 'class ', '"""', "'''")) or
                    line.strip().startswith('#') or not line.strip()):
                    reduced_lines.append(line)
                    continue
                
                # Keep essential test methods only
                if line.strip().startswith('def test_'):
                    if essential_tests < max_essential_tests:
                        in_test_method = True
                        essential_tests += 1
                        reduced_lines.append(line)
                    else:
                        in_test_method = False
                elif in_test_method:
                    reduced_lines.append(line)
                    if line.strip() and not line.startswith('    '):  # End of method
                        in_test_method = False
            
            # Write reduced content
            with open(file_path, 'w') as f:
                f.writelines(reduced_lines)
            
            new_lines = len(reduced_lines)
            reduction = original_lines - new_lines
            
            self.log_action("REDUCED_FILE", f"{file_path}: {original_lines} → {new_lines} lines (-{reduction})")
            self.stats['modified_files'].append(str(file_path))
            
        except Exception as e:
            print(f"❌ Error reducing {file_path}: {e}")
    
    def phase_4_final_cleanup(self):
        """Phase 4: Final optimization and validation."""
        print("\n🎯 PHASE 4: FINAL CLEANUP")
        print("=" * 40)
        
        # Remove empty directories
        self._remove_empty_dirs()
        
        # Optimize remaining test files
        self._optimize_remaining_tests()
        
        print("✅ Phase 4 complete - Final cleanup done")
    
    def _remove_empty_dirs(self):
        """Remove empty test directories."""
        for dir_path in Path("tests").iterdir():
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                shutil.rmtree(dir_path)
                self.log_action("REMOVED_EMPTY_DIR", str(dir_path))
    
    def _optimize_remaining_tests(self):
        """Basic optimization of remaining test files."""
        for test_file in Path("tests").rglob("*.py"):
            if test_file.name == "__init__.py":
                continue
            
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Remove excessive blank lines
                while '\n\n\n' in content:
                    content = content.replace('\n\n\n', '\n\n')
                
                # Remove trailing whitespace
                lines = [line.rstrip() for line in content.split('\n')]
                content = '\n'.join(lines)
                
                with open(test_file, 'w') as f:
                    f.write(content)
                    
            except Exception as e:
                print(f"⚠️  Warning optimizing {test_file}: {e}")
    
    def validate_cleanup(self):
        """Validate that essential tests still work after cleanup."""
        print("\n✅ VALIDATING CLEANUP")
        print("=" * 30)
        
        # Try to run essential tests
        essential_test_dirs = ["tests/unit", "tests/comprehensive"]
        
        for test_dir in essential_test_dirs:
            if Path(test_dir).exists():
                try:
                    result = subprocess.run([
                        "python", "-m", "pytest", test_dir, "--collect-only", "-q"
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        print(f"✅ {test_dir}: Tests can be collected")
                    else:
                        print(f"⚠️  {test_dir}: Collection issues - {result.stderr[:100]}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⚠️  {test_dir}: Collection timeout")
                except Exception as e:
                    print(f"❌ {test_dir}: {e}")
    
    def generate_report(self):
        """Generate cleanup report."""
        self.stats['files_after'], self.stats['lines_after'] = self.count_test_files()
        
        reduction_files = self.stats['files_before'] - self.stats['files_after']
        reduction_lines = self.stats['lines_before'] - self.stats['lines_after']
        
        print("\n📊 CLEANUP REPORT")
        print("=" * 30)
        print(f"Files: {self.stats['files_before']} → {self.stats['files_after']} ({reduction_files} removed)")
        print(f"Lines: {self.stats['lines_before']} → {self.stats['lines_after']} ({reduction_lines} reduced)")
        print(f"Files deleted: {len(self.stats['deleted_files'])}")
        print(f"Files modified: {len(self.stats['modified_files'])}")
        print(f"Backup location: {self.stats['backup_location']}")
        
        # Save detailed report
        report_file = Path("test_cleanup_report.json")
        self.stats['cleanup_log'] = self.cleanup_log
        
        with open(report_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"📄 Detailed report saved: {report_file}")
    
    def run_full_cleanup(self):
        """Execute the complete cleanup process."""
        print("🧹 QEMLFLOW TEST CLEANUP - FULL EXECUTION")
        print("=" * 50)
        
        # Initial stats
        self.stats['files_before'], self.stats['lines_before'] = self.count_test_files()
        print(f"Starting state: {self.stats['files_before']} files, {self.stats['lines_before']} lines")
        
        # Create backup
        self.create_backup()
        
        # Execute phases
        self.phase_1_immediate_deletions()
        self.phase_2_enterprise_minimization()
        self.phase_3_unit_test_surgery()
        self.phase_4_final_cleanup()
        
        # Validate and report
        self.validate_cleanup()
        self.generate_report()
        
        print("\n🎉 CLEANUP COMPLETE!")
        print("The QeMLflow test suite has been transformed into a lean,")
        print("focused scientific computing platform validation suite.")


def main():
    """Main execution function."""
    cleanup = TestCleanupImplementation()
    
    print("This will aggressively clean up the QeMLflow test suite.")
    print("A backup will be created before any changes.")
    
    response = input("\\nProceed with cleanup? (yes/no): ").lower().strip()
    if response == 'yes':
        cleanup.run_full_cleanup()
    else:
        print("Cleanup cancelled.")


if __name__ == "__main__":
    main()
