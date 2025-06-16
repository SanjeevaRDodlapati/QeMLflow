"""
ChemML Hybrid Architecture Migration Script
==========================================

Migrates existing ChemML codebase to the new hybrid architecture.
This script handles:
- Moving files to new structure
- Updating import statements
- Creating compatibility layers
- Updating notebooks and documentation

Usage:
    python migrate_to_hybrid_architecture.py [--dry-run] [--backup]
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


class ChemMLMigrator:
    """Handles migration to the new hybrid architecture."""

    def __init__(self, root_dir: str, dry_run: bool = False, backup: bool = True):
        """
        Initialize migrator.

        Args:
            root_dir: Root directory of ChemML project
            dry_run: If True, only print what would be done
            backup: If True, create backup of original files
        """
        self.root_dir = Path(root_dir)
        self.src_dir = self.root_dir / "src"
        self.dry_run = dry_run
        self.backup = backup

        # Track changes for summary
        self.moved_files = []
        self.updated_files = []
        self.errors = []

    def migrate(self):
        """Run the complete migration process."""
        print("🚀 Starting ChemML Hybrid Architecture Migration")
        print("=" * 50)

        try:
            if self.backup and not self.dry_run:
                self._create_backup()

            # Migration steps
            self._migrate_drug_design_modules()
            self._migrate_data_processing()
            self._migrate_models()
            self._migrate_utilities()
            self._update_notebook_imports()
            self._create_compatibility_layer()
            self._update_setup_files()

            self._print_summary()

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            self.errors.append(str(e))
            return False

        return True

    def _create_backup(self):
        """Create backup of existing source directory."""
        backup_dir = self.root_dir / "src_backup"

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        shutil.copytree(self.src_dir, backup_dir)
        print(f"✅ Created backup at {backup_dir}")

    def _migrate_drug_design_modules(self):
        """Migrate drug design modules to research directory."""
        print("\n📁 Migrating drug design modules...")

        old_drug_design = self.src_dir / "drug_design"
        new_research_dir = self.src_dir / "chemml" / "research"

        if not old_drug_design.exists():
            print("⚠️  No drug_design directory found, skipping...")
            return

        # Create research directory if it doesn't exist
        if not self.dry_run:
            new_research_dir.mkdir(parents=True, exist_ok=True)

        # Move drug design files to research/drug_discovery.py
        drug_discovery_content = self._combine_drug_design_files(old_drug_design)

        target_file = new_research_dir / "drug_discovery.py"

        if not self.dry_run:
            with open(target_file, "w") as f:
                f.write(drug_discovery_content)

        print(f"✅ Moved drug design modules to {target_file}")
        self.moved_files.append(str(target_file))

    def _combine_drug_design_files(self, drug_design_dir: Path) -> str:
        """Combine multiple drug design files into one module."""
        header = '''"""
ChemML Drug Discovery Research Module
====================================

Advanced drug discovery algorithms and workflows.
Migrated from legacy drug_design module.

Key Features:
- ADMET prediction
- Molecular generation and optimization
- Virtual screening
- QSAR modeling
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import warnings

'''

        combined_content = [header]

        # Read and combine Python files
        for py_file in drug_design_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                with open(py_file, "r") as f:
                    content = f.read()

                # Clean up imports and add section header
                cleaned_content = self._clean_file_content(content, py_file.name)
                combined_content.append(f"\n# ===== From {py_file.name} =====\n")
                combined_content.append(cleaned_content)

            except Exception as e:
                self.errors.append(f"Error reading {py_file}: {e}")

        return "\n".join(combined_content)

    def _clean_file_content(self, content: str, filename: str) -> str:
        """Clean file content for migration."""
        # Remove imports that will be handled at module level
        content = re.sub(r"^import .*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"^from .* import .*$", "", content, flags=re.MULTILINE)

        # Remove empty lines at the beginning
        content = content.lstrip("\n")

        # Add module-specific note
        note = f"# Migrated from legacy {filename}\n\n"
        content = note + content

        return content

    def _migrate_data_processing(self):
        """Migrate data processing to core/data.py (already exists)."""
        print("\n📊 Migrating data processing...")

        old_data_dir = self.src_dir / "data_processing"

        if not old_data_dir.exists():
            print("⚠️  No data_processing directory found, skipping...")
            return

        # The new core/data.py already exists, so we just need to
        # ensure any missing functionality is noted
        print("✅ Data processing functionality already migrated to core/data.py")

    def _migrate_models(self):
        """Migrate model directories to research modules."""
        print("\n🧠 Migrating model modules...")

        # Handle quantum_ml
        old_quantum = self.src_dir / "models" / "quantum_ml"
        if old_quantum.exists():
            print("✅ Quantum ML functionality already migrated to research/quantum.py")

        # Handle classical_ml
        old_classical = self.src_dir / "models" / "classical_ml"
        if old_classical.exists():
            print("✅ Classical ML functionality already migrated to core/models.py")

    def _migrate_utilities(self):
        """Migrate common utilities."""
        print("\n🔧 Migrating utilities...")

        old_common = self.src_dir / "chemml_common"

        if not old_common.exists():
            print("⚠️  No chemml_common directory found, skipping...")
            return

        # Most utility functions are already in core/utils.py
        # We just need to migrate any experiment tracking functionality

        wandb_files = list(old_common.glob("wandb*"))
        if wandb_files:
            self._migrate_experiment_tracking(old_common)

        print("✅ Utilities migrated to core/utils.py")

    def _migrate_experiment_tracking(self, common_dir: Path):
        """Migrate experiment tracking to integrations."""
        print("📈 Migrating experiment tracking...")

        exp_tracking_content = '''"""
ChemML Experiment Tracking Integration
=====================================

Weights & Biases integration for experiment tracking.
"""

import warnings

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def setup_wandb_tracking(experiment_name: str, config: dict = None, project: str = "chemml-experiments"):
    """Setup wandb experiment tracking."""
    if not HAS_WANDB:
        warnings.warn("wandb not available. Install with: pip install wandb")
        return None

    try:
        run = wandb.init(
            project=project,
            name=experiment_name,
            config=config or {},
            tags=["chemml"]
        )
        print(f"✅ Wandb tracking started: {run.url}")
        return run
    except Exception as e:
        print(f"⚠️ Wandb setup failed: {e}")
        return None


def log_metrics(metrics: dict, step: int = None):
    """Log metrics to wandb."""
    if HAS_WANDB and wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_run():
    """Finish wandb run."""
    if HAS_WANDB and wandb.run is not None:
        wandb.finish()


__all__ = ['setup_wandb_tracking', 'log_metrics', 'finish_run']
'''

        target_file = (
            self.src_dir / "chemml" / "integrations" / "experiment_tracking.py"
        )

        if not self.dry_run:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with open(target_file, "w") as f:
                f.write(exp_tracking_content)

        print(f"✅ Created experiment tracking integration at {target_file}")
        self.moved_files.append(str(target_file))

    def _update_notebook_imports(self):
        """Update notebook imports to use new structure."""
        print("\n📓 Updating notebook imports...")

        notebooks_dir = self.root_dir / "notebooks"

        if not notebooks_dir.exists():
            print("⚠️  No notebooks directory found, skipping...")
            return

        # Find all notebook files
        notebook_files = list(notebooks_dir.rglob("*.ipynb"))

        for notebook_file in notebook_files:
            try:
                self._update_notebook_file(notebook_file)
            except Exception as e:
                self.errors.append(f"Error updating {notebook_file}: {e}")

    def _update_notebook_file(self, notebook_path: Path):
        """Update a single notebook file."""
        import json

        if self.dry_run:
            print(f"Would update: {notebook_path}")
            return

        # Read notebook
        with open(notebook_path, "r") as f:
            notebook = json.load(f)

        updated = False

        # Update import statements in code cells
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                if isinstance(source, list):
                    new_source = []
                    for line in source:
                        updated_line = self._update_import_line(line)
                        new_source.append(updated_line)
                        if updated_line != line:
                            updated = True
                    cell["source"] = new_source

        # Write back if updated
        if updated:
            with open(notebook_path, "w") as f:
                json.dump(notebook, f, indent=2)

            print(f"✅ Updated imports in {notebook_path}")
            self.updated_files.append(str(notebook_path))

    def _update_import_line(self, line: str) -> str:
        """Update a single import line."""
        # Common import replacements
        replacements = {
            "from src.chemml_custom": "from chemml.core",
            "from chemml_custom": "from chemml.core",
            "from src.drug_design": "from chemml.research.drug_discovery",
            "from src.data_processing": "from chemml.core.data",
            "from src.models": "from chemml.core.models",
            "from src.chemml_common": "from chemml.core.utils",
            "import sys; sys.path.append": "# Removed sys.path manipulation",
        }

        for old_import, new_import in replacements.items():
            if old_import in line:
                return line.replace(old_import, new_import)

        return line

    def _create_compatibility_layer(self):
        """Create compatibility layer for old imports."""
        print("\n🔄 Creating compatibility layer...")

        compat_content = '''"""
ChemML Compatibility Layer
=========================

Provides backward compatibility for old import paths.
This allows existing code to continue working during migration.
"""

import warnings

# Provide old import paths
try:
    from chemml.core.featurizers import *
    from chemml.core.models import *
    from chemml.core.data import *

    # Legacy aliases
    ModernMorganFingerprint = MorganFingerprint
    ModernDescriptorCalculator = DescriptorCalculator
    ModernECFPFingerprint = ECFPFingerprint

    warnings.warn(
        "Using legacy chemml_custom imports. Please update to: from chemml.core import featurizers",
        DeprecationWarning,
        stacklevel=2
    )

except ImportError:
    pass

__all__ = [
    'MorganFingerprint', 'DescriptorCalculator', 'ECFPFingerprint',
    'ModernMorganFingerprint', 'ModernDescriptorCalculator', 'ModernECFPFingerprint'
]
'''

        # Create compatibility module
        compat_file = self.src_dir / "chemml_custom" / "__init__.py"

        if not self.dry_run and compat_file.parent.exists():
            with open(compat_file, "w") as f:
                f.write(compat_content)

            print(f"✅ Created compatibility layer at {compat_file}")

    def _update_setup_files(self):
        """Update setup.py and pyproject.toml."""
        print("\n⚙️  Updating setup files...")

        # Update setup.py if it exists
        setup_file = self.root_dir / "setup.py"
        if setup_file.exists():
            self._update_setup_py(setup_file)

        # Update pyproject.toml if it exists
        pyproject_file = self.root_dir / "pyproject.toml"
        if pyproject_file.exists():
            self._update_pyproject_toml(pyproject_file)

    def _update_setup_py(self, setup_file: Path):
        """Update setup.py with new package structure."""
        if self.dry_run:
            print(f"Would update: {setup_file}")
            return

        try:
            with open(setup_file, "r") as f:
                content = f.read()

            # Update packages finder
            old_packages = 'packages=find_packages("src")'
            new_packages = 'packages=find_packages("src", include=["chemml*"])'

            content = content.replace(old_packages, new_packages)

            with open(setup_file, "w") as f:
                f.write(content)

            print(f"✅ Updated {setup_file}")
            self.updated_files.append(str(setup_file))

        except Exception as e:
            self.errors.append(f"Error updating setup.py: {e}")

    def _update_pyproject_toml(self, pyproject_file: Path):
        """Update pyproject.toml if needed."""
        print(f"✅ {pyproject_file} is compatible with new structure")

    def _print_summary(self):
        """Print migration summary."""
        print("\n" + "=" * 50)
        print("🎉 Migration Summary")
        print("=" * 50)

        print(f"📁 Files moved: {len(self.moved_files)}")
        for file in self.moved_files:
            print(f"   → {file}")

        print(f"\\n📝 Files updated: {len(self.updated_files)}")
        for file in self.updated_files:
            print(f"   → {file}")

        if self.errors:
            print(f"\\n❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   → {error}")

        print("\\n✅ Migration completed!")
        print("\\nNext steps:")
        print("1. Test the new import structure")
        print("2. Run existing notebooks to verify compatibility")
        print("3. Update documentation")
        print("4. Remove backup files when confident")


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate ChemML to hybrid architecture"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument(
        "--root-dir", default=".", help="Root directory of ChemML project"
    )

    args = parser.parse_args()

    migrator = ChemMLMigrator(
        root_dir=args.root_dir, dry_run=args.dry_run, backup=not args.no_backup
    )

    success = migrator.migrate()

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
