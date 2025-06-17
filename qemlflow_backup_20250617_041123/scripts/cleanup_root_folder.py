#!/usr/bin/env python3
"""
Root Folder Cleanup and Organization
===================================

Reorganizes the cluttered root folder by moving documentation, status files,
and other items to appropriate subdirectories while maintaining core project files.

Strategy:
1. Keep essential project files in root
2. Move documentation to docs/
3. Organize status/history files
4. Clean up temporary/generated files
5. Create clear folder structure
"""

import shutil
from datetime import datetime
from pathlib import Path


class RootFolderCleaner:
    """Comprehensive root folder cleanup utility."""

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.backup_created = False

        # Define what should stay in root
        self.keep_in_root = {
            # Essential project files
            "README.md",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            "requirements-core.txt",
            "Makefile",
            "Dockerfile",
            "docker-compose.yml",
            # Configuration files
            ".gitignore",
            ".config/",
            # Essential directories
            "src/",
            "tests/",
            "docs/",
            "config/",
            "examples/",
            "notebooks/",
            "scripts/",
            "tools/",
            # Git and CI
            ".git/",
            ".github/",
            ".pytest_cache/",
            # Data and generated content (keep for now)
            "data/",
            "logs/",
            "reports/",
            "site/",
            "build/",
            "dist/",
            "chemml_env/",
            "archive/",
            "assessments/",
            "boltz_cache/",
        }

        # Define organization structure
        self.move_patterns = {
            "docs/project-status/": [
                "*_STATUS_*.md",
                "*_COMPLETE.md",
                "*_ASSESSMENT.md",
                "PROJECT_*.md",
                "PRODUCTION_*.md",
                "CODEBASE_*.md",
                "FOLDER_*.md",
            ],
            "docs/development/": [
                "DEVELOPMENT*.md",
                "NEXT_STEPS*.md",
                "CONTRIBUTING.md",
            ],
            "docs/archive/": ["FINAL_*.md"],
        }

    def analyze_current_state(self):
        """Analyze current root folder state."""
        print("🔍 Analyzing Root Folder State")
        print("=" * 40)

        all_items = list(self.root.iterdir())

        # Categorize items
        should_stay = []
        should_move = []
        md_files = []

        for item in all_items:
            if item.name in self.keep_in_root:
                should_stay.append(item)
            elif item.suffix == ".md" and item.name != "README.md":
                md_files.append(item)
                should_move.append(item)
            elif not item.name.startswith(".") and item.is_file():
                should_move.append(item)

        print("📊 Analysis Results:")
        print(f"  • Total items: {len(all_items)}")
        print(f"  • Should stay in root: {len(should_stay)}")
        print(f"  • Should be moved: {len(should_move)}")
        print(f"  • Markdown files to organize: {len(md_files)}")

        print("\n📄 Markdown files to move:")
        for md_file in md_files:
            print(f"  • {md_file.name}")

        return should_stay, should_move, md_files

    def create_backup(self):
        """Create backup of current state."""
        if self.backup_created:
            return

        print("\n💾 Creating Backup")
        print("-" * 20)

        backup_dir = (
            self.root
            / "archive"
            / f"root_cleanup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup markdown files and other items to be moved
        for item in self.root.iterdir():
            if (item.suffix == ".md" and item.name != "README.md") or (
                item.is_file()
                and item.name not in self.keep_in_root
                and not item.name.startswith(".")
            ):
                try:
                    shutil.copy2(item, backup_dir / item.name)
                except Exception as e:
                    print(f"⚠️ Backup warning for {item.name}: {e}")

        print(f"✅ Backup created: {backup_dir}")
        self.backup_created = True

    def organize_files(self):
        """Organize files according to patterns."""
        print("\n📁 Organizing Files")
        print("-" * 20)

        moves_made = 0

        for target_dir, patterns in self.move_patterns.items():
            target_path = self.root / target_dir
            target_path.mkdir(parents=True, exist_ok=True)

            for pattern in patterns:
                matching_files = list(self.root.glob(pattern))

                for file_path in matching_files:
                    if file_path.is_file() and file_path.name not in self.keep_in_root:
                        try:
                            destination = target_path / file_path.name
                            shutil.move(str(file_path), str(destination))
                            print(f"  📄 {file_path.name} → {target_dir}")
                            moves_made += 1
                        except Exception as e:
                            print(f"⚠️ Error moving {file_path.name}: {e}")

        return moves_made

    def create_organization_index(self):
        """Create an index of the new organization."""
        print("\n📋 Creating Organization Index")
        print("-" * 30)

        index_content = f"""# ChemML Project Organization

**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 Root Directory Structure

### Essential Project Files (Root Level)
- `README.md` - Main project documentation
- `pyproject.toml` - Python project configuration
- `setup.py` - Python package setup
- `requirements.txt` - Python dependencies
- `Makefile` - Build automation
- `Dockerfile` - Container configuration
- `.config/mkdocs.yml` - Documentation configuration

### Core Directories
- `src/` - Source code
- `tests/` - Test suite
- `docs/` - Documentation (including reorganized project files)
- `config/` - Configuration files
- `examples/` - Usage examples
- `notebooks/` - Jupyter notebooks
- `scripts/` - Utility scripts
- `tools/` - Development tools

### Documentation Organization

#### Project Status (`docs/project-status/`)
- Project status reports
- Completion assessments
- Production readiness documentation

#### Development (`docs/development/`)
- Development history and guides
- Contributing guidelines
- Next steps and planning

#### Archive (`docs/archive/`)
- Final reports and historical documentation

### Data and Generated Content
- `data/` - Data files and cache
- `logs/` - Log files
- `reports/` - Generated reports
- `archive/` - Archived content and backups

## 🧹 Cleanup Benefits

1. **Cleaner Root**: Only essential project files remain
2. **Better Organization**: Documentation is categorized
3. **Easier Navigation**: Related files are grouped together
4. **Maintained History**: All files preserved in organized structure
5. **Professional Appearance**: Clean, focused project layout

## 📍 File Locations

All moved files can be found in their new organized locations:
- Status files: `docs/project-status/`
- Development docs: `docs/development/`
- Archive files: `docs/archive/`

Original files are backed up in `archive/root_cleanup_backup_*/`
"""

        # Save index
        index_path = self.root / "docs" / "PROJECT_ORGANIZATION.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)

        print(f"✅ Organization index created: {index_path}")

    def cleanup_root(self):
        """Perform complete root folder cleanup."""
        print("🧹 ChemML Root Folder Cleanup")
        print("=" * 35)

        # Analyze current state
        should_stay, should_move, md_files = self.analyze_current_state()

        if not should_move:
            print("✅ Root folder is already clean!")
            return

        # Create backup
        self.create_backup()

        # Organize files
        moves_made = self.organize_files()

        # Create organization index
        self.create_organization_index()

        # Final summary
        print("\n🎯 Cleanup Summary")
        print("-" * 18)
        print(f"✅ Files moved: {moves_made}")
        print("✅ Backup created: Yes")
        print("✅ Organization index: Created")
        print("✅ Root folder: Cleaned and organized")

        # Show final root state
        self.show_final_state()

    def show_final_state(self):
        """Show final root directory state."""
        print("\n📁 Final Root Directory Contents")
        print("-" * 35)

        root_items = sorted(
            [
                item
                for item in self.root.iterdir()
                if not item.name.startswith(".") and item.name != "archive"
            ]
        )

        for item in root_items:
            if item.is_dir():
                print(f"📁 {item.name}/")
            else:
                print(f"📄 {item.name}")

        print("\n🎉 Root folder is now clean and organized!")


def main():
    """Run root folder cleanup."""
    cleaner = RootFolderCleaner()
    cleaner.cleanup_root()


if __name__ == "__main__":
    main()
