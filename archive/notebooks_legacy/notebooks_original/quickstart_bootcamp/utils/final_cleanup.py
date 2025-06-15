#!/usr/bin/env python3
"""
ChemML Bootcamp Final Cleanup Script
====================================

This script performs final cleanup of redundant files and validates
the optimized bootcamp structure.

Usage:
    python final_cleanup.py --cleanup-all
    python final_cleanup.py --validate-only
    python final_cleanup.py --backup-redundant
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


class BootcampFinalCleanup:
    """Final cleanup and validation for optimized bootcamp."""

    def __init__(self, bootcamp_dir: str):
        self.bootcamp_dir = Path(bootcamp_dir)
        self.docs_dir = self.bootcamp_dir.parent / "docs"
        self.backup_dir = self.bootcamp_dir / "cleanup_backup"

        # Files to keep (essential optimized structure)
        self.essential_files = {
            "docs": {
                "GET_STARTED.md",
                "LEARNING_PATHS.md",
                "REFERENCE.md",
                "README.md",
            },
            "bootcamp": {
                # Core notebooks
                "day_01_ml_cheminformatics_project.ipynb",
                "day_02_deep_learning_molecules_project.ipynb",
                "day_03_molecular_docking_project.ipynb",
                "day_04_quantum_chemistry_project.ipynb",
                # Modular notebooks
                "day_05_module_1_foundations.ipynb",
                "day_05_module_2_advanced.ipynb",
                "day_05_module_3_production.ipynb",
                "day_06_module_1_quantum_foundations.ipynb",
                "day_06_module_2_vqe_algorithms.ipynb",
                "day_06_module_3_quantum_production.ipynb",
                "day_07_module_1_integration.ipynb",
                "day_07_module_2_multimodal_workflows.ipynb",
                "day_07_module_3_deployment.ipynb",
                # Original notebooks (kept for reference)
                "day_05_quantum_ml_project.ipynb",
                "day_06_quantum_computing_project.ipynb",
                "day_07_integration_project.ipynb",
                # Core structure
                "README.md",
                "OPTIMIZATION_MIGRATION_GUIDE.md",
                # Validation tools
                "integration_test_suite.py",
                "ux_validation_framework.py",
                "performance_optimizer.py",
                "final_cleanup.py",
            },
            "assessment": {
                "simple_progress_tracker.py",
                "daily_checkpoints.md",
                "completion_badges.py",
            },
        }

        # Files to remove (redundant implementation/planning files)
        self.redundant_files = {
            "docs": {
                "documentation_assessment_and_plan.md",
                "documentation_integration_guide.md",
                "documentation_organization_summary.md",
                "validation_testing_framework.md",
            },
            "bootcamp": {
                "ASSESSMENT_INTEGRATION_GUIDE.md",
                "BOOTCAMP_REVIEW_ASSESSMENT.md",
                "DAY_2_INTEGRATION_COMPLETION.md",
                "IMPLEMENTATION_PLAN.md",
                "IMPLEMENTATION_SUMMARY.md",
                "REMAINING_DAYS_INTEGRATION_PLAN.md",
                "day_05_modularization_plan.md",
                "OPTIMIZATION_IMPLEMENTATION_PLAN.md",
                "STEP_2_SIMPLIFIED_ASSESSMENT.md",
                "STEP_3_STREAMLINED_DOCS.md",
                "STEP_4_MULTI_PACE_TRACKS.md",
                "STEP_5_NOTEBOOK_MODULARIZATION.md",
            },
        }

        # Directories to remove (fragmented content)
        self.redundant_dirs = {
            "docs": {
                "getting_started",
                "planning",
                "reference",
                "resources",
                "roadmaps",
            }
        }

    def cleanup_all(self, backup: bool = True) -> Dict[str, any]:
        """Perform complete cleanup with optional backup."""
        print("🧹 Starting ChemML Bootcamp Final Cleanup")
        print("=" * 50)

        cleanup_results = {
            "files_removed": [],
            "dirs_removed": [],
            "files_backed_up": [],
            "validation_results": {},
            "timestamp": datetime.now().isoformat(),
        }

        if backup:
            print("📦 Creating backup of redundant files...")
            self._create_backup(cleanup_results)

        print("🗑️  Removing redundant files...")
        self._remove_redundant_files(cleanup_results)

        print("📁 Removing redundant directories...")
        self._remove_redundant_dirs(cleanup_results)

        print("✅ Validating optimized structure...")
        validation_results = self._validate_structure()
        cleanup_results["validation_results"] = validation_results

        print("📊 Generating cleanup report...")
        self._generate_cleanup_report(cleanup_results)

        return cleanup_results

    def _create_backup(self, results: Dict):
        """Create backup of files before removal."""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)

        # Backup redundant docs files
        for filename in self.redundant_files["docs"]:
            file_path = self.docs_dir / filename
            if file_path.exists():
                backup_path = self.backup_dir / "docs" / filename
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                results["files_backed_up"].append(str(file_path))

        # Backup redundant bootcamp files
        for filename in self.redundant_files["bootcamp"]:
            file_path = self.bootcamp_dir / filename
            if file_path.exists():
                backup_path = self.backup_dir / "bootcamp" / filename
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                results["files_backed_up"].append(str(file_path))

        # Backup redundant directories
        for dirname in self.redundant_dirs["docs"]:
            dir_path = self.docs_dir / dirname
            if dir_path.exists():
                backup_path = self.backup_dir / "docs" / dirname
                shutil.copytree(dir_path, backup_path, dirs_exist_ok=True)
                results["files_backed_up"].append(str(dir_path))

        print(
            f"  📦 Backed up {len(results['files_backed_up'])} items to {self.backup_dir}"
        )

    def _remove_redundant_files(self, results: Dict):
        """Remove redundant files."""
        # Remove redundant docs files
        for filename in self.redundant_files["docs"]:
            file_path = self.docs_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    results["files_removed"].append(str(file_path))
                    print(f"  ✅ Removed: {file_path}")
                except Exception as e:
                    print(f"  ❌ Error removing {file_path}: {e}")

        # Remove redundant bootcamp files
        for filename in self.redundant_files["bootcamp"]:
            file_path = self.bootcamp_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    results["files_removed"].append(str(file_path))
                    print(f"  ✅ Removed: {file_path}")
                except Exception as e:
                    print(f"  ❌ Error removing {file_path}: {e}")

    def _remove_redundant_dirs(self, results: Dict):
        """Remove redundant directories."""
        for dirname in self.redundant_dirs["docs"]:
            dir_path = self.docs_dir / dirname
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    results["dirs_removed"].append(str(dir_path))
                    print(f"  ✅ Removed directory: {dir_path}")
                except Exception as e:
                    print(f"  ❌ Error removing directory {dir_path}: {e}")

    def _validate_structure(self) -> Dict[str, any]:
        """Validate the optimized structure."""
        validation = {
            "essential_files_present": {},
            "redundant_files_absent": {},
            "structure_score": 0,
            "issues": [],
        }

        # Check essential files are present
        total_essential = 0
        present_essential = 0

        # Check docs
        for filename in self.essential_files["docs"]:
            file_path = self.docs_dir / filename
            total_essential += 1
            if file_path.exists():
                present_essential += 1
                validation["essential_files_present"][f"docs/{filename}"] = True
            else:
                validation["essential_files_present"][f"docs/{filename}"] = False
                validation["issues"].append(f"Missing essential file: docs/{filename}")

        # Check bootcamp files
        for filename in self.essential_files["bootcamp"]:
            file_path = self.bootcamp_dir / filename
            total_essential += 1
            if file_path.exists():
                present_essential += 1
                validation["essential_files_present"][f"bootcamp/{filename}"] = True
            else:
                validation["essential_files_present"][f"bootcamp/{filename}"] = False
                validation["issues"].append(
                    f"Missing essential file: bootcamp/{filename}"
                )

        # Check assessment files
        assessment_dir = self.bootcamp_dir / "assessment"
        for filename in self.essential_files["assessment"]:
            file_path = assessment_dir / filename
            total_essential += 1
            if file_path.exists():
                present_essential += 1
                validation["essential_files_present"][f"assessment/{filename}"] = True
            else:
                validation["essential_files_present"][f"assessment/{filename}"] = False
                validation["issues"].append(
                    f"Missing essential file: assessment/{filename}"
                )

        # Check redundant files are absent
        total_redundant = 0
        absent_redundant = 0

        for filename in self.redundant_files["docs"]:
            file_path = self.docs_dir / filename
            total_redundant += 1
            if not file_path.exists():
                absent_redundant += 1
                validation["redundant_files_absent"][f"docs/{filename}"] = True
            else:
                validation["redundant_files_absent"][f"docs/{filename}"] = False
                validation["issues"].append(
                    f"Redundant file still present: docs/{filename}"
                )

        for filename in self.redundant_files["bootcamp"]:
            file_path = self.bootcamp_dir / filename
            total_redundant += 1
            if not file_path.exists():
                absent_redundant += 1
                validation["redundant_files_absent"][f"bootcamp/{filename}"] = True
            else:
                validation["redundant_files_absent"][f"bootcamp/{filename}"] = False
                validation["issues"].append(
                    f"Redundant file still present: bootcamp/{filename}"
                )

        # Calculate structure score
        essential_score = (
            present_essential / total_essential if total_essential > 0 else 0
        )
        cleanup_score = absent_redundant / total_redundant if total_redundant > 0 else 1
        validation["structure_score"] = (essential_score + cleanup_score) / 2

        print(f"  📊 Essential files: {present_essential}/{total_essential}")
        print(f"  🧹 Redundant files removed: {absent_redundant}/{total_redundant}")
        print(f"  🎯 Structure score: {validation['structure_score']:.1%}")

        return validation

    def _generate_cleanup_report(self, results: Dict):
        """Generate comprehensive cleanup report."""
        print("\n" + "=" * 50)
        print("🧹 CLEANUP REPORT")
        print("=" * 50)

        print(f"\n📊 Cleanup Summary:")
        print(f"  🗑️  Files removed: {len(results['files_removed'])}")
        print(f"  📁 Directories removed: {len(results['dirs_removed'])}")
        print(f"  📦 Items backed up: {len(results['files_backed_up'])}")

        validation = results["validation_results"]
        structure_score = validation["structure_score"]

        print(f"\n✅ Structure Validation:")
        print(f"  🎯 Structure score: {structure_score:.1%}")

        if structure_score >= 0.95:
            status = "🟢 EXCELLENT - Cleanup successful"
        elif structure_score >= 0.9:
            status = "🟡 GOOD - Minor issues remain"
        else:
            status = "🔴 ISSUES - Manual cleanup needed"

        print(f"  📈 Status: {status}")

        if validation["issues"]:
            print(f"\n⚠️  Issues Found:")
            for issue in validation["issues"][:5]:  # Show first 5 issues
                print(f"    • {issue}")

            if len(validation["issues"]) > 5:
                print(f"    ... and {len(validation['issues']) - 5} more issues")

        # Final structure overview
        print(f"\n📁 Optimized Structure:")
        print(
            f"  📚 Core docs: {sum(1 for k, v in validation['essential_files_present'].items() if k.startswith('docs/') and v)}/4"
        )
        print(
            f"  📓 Notebooks: {sum(1 for k, v in validation['essential_files_present'].items() if k.startswith('bootcamp/') and '.ipynb' in k and v)}"
        )
        print(
            f"  📊 Assessment: {sum(1 for k, v in validation['essential_files_present'].items() if k.startswith('assessment/') and v)}/3"
        )

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.bootcamp_dir / f"cleanup_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Detailed report saved to: {report_file}")

        if self.backup_dir.exists() and len(results["files_backed_up"]) > 0:
            print(f"💼 Backup available at: {self.backup_dir}")

        return results


def main():
    """Main entry point for final cleanup."""
    import argparse

    parser = argparse.ArgumentParser(description="ChemML Bootcamp Final Cleanup")
    parser.add_argument(
        "--cleanup-all", action="store_true", help="Perform complete cleanup"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate structure only"
    )
    parser.add_argument(
        "--backup-redundant", action="store_true", help="Backup redundant files only"
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument(
        "--bootcamp-dir", default=".", help="Path to bootcamp directory"
    )

    args = parser.parse_args()

    cleaner = BootcampFinalCleanup(args.bootcamp_dir)

    if args.validate_only:
        print("✅ Validating bootcamp structure...")
        validation = cleaner._validate_structure()
        print(f"Structure score: {validation['structure_score']:.1%}")
        return 0 if validation["structure_score"] >= 0.9 else 1

    elif args.backup_redundant:
        print("📦 Creating backup only...")
        results = {"files_backed_up": []}
        cleaner._create_backup(results)
        return 0

    else:  # Default to cleanup-all
        backup = not args.no_backup
        results = cleaner.cleanup_all(backup=backup)

        # Exit with error if cleanup had issues
        structure_score = results["validation_results"]["structure_score"]
        return 0 if structure_score >= 0.9 else 1


if __name__ == "__main__":
    exit(main())
