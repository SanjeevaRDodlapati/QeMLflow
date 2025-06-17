#!/usr/bin/env python3
"""
QeMLflow File Classification and Protection Tool

This tool implements the file classification system and applies appropriate
protection mechanisms based on the file's importance to the framework.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml
import stat
from datetime import datetime

class QeMLflowFileClassifier:
    """Classifies and protects QeMLflow repository files."""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.config_file = self.repo_root / ".qemlflow-protection.yaml"
        self.critical_files_registry = self.repo_root / "CRITICAL_FILES.md"
        
        # File classification patterns
        self.core_patterns = [
            "src/qemlflow/__init__.py",
            "src/qemlflow/core/**/*.py",
            "src/qemlflow/config/**/*.py",
            "src/qemlflow/utils/__init__.py",
            "src/qemlflow/utils/logging.py",
            "setup.py",
            "pyproject.toml", 
            "requirements-core.txt",
            ".gitignore"
        ]
        
        self.middle_patterns = [
            "src/qemlflow/research/**/*.py",
            "src/qemlflow/integrations/**/*.py",
            "src/qemlflow/enterprise/**/*.py",
            "src/qemlflow/advanced/**/*.py",
            "src/qemlflow/tutorials/**/*.py",
            "tests/**/*.py",
            "scripts/**/*.py",
            "tools/**/*.py",
            "requirements.txt",
            "docker-compose.yml",
            "Dockerfile",
            "Makefile",
            ".config/**/*",
            "conftest.py"
        ]
        
        self.outer_patterns = [
            "docs/**/*",
            "examples/**/*", 
            "notebooks/**/*",
            "reports/**/*",
            "data/**/*",
            "*.md",
            "qemlflow_backup_*/**/*",
            "backups/**/*",
            ".archive/**/*"
        ]
        
    def classify_file(self, filepath: Path) -> str:
        """Classify a file into protection layers."""
        rel_path = filepath.relative_to(self.repo_root)
        str_path = str(rel_path)
        
        # Check core patterns first (highest priority)
        for pattern in self.core_patterns:
            if self._matches_pattern(str_path, pattern):
                return "core"
                
        # Check middle patterns
        for pattern in self.middle_patterns:
            if self._matches_pattern(str_path, pattern):
                return "middle"
                
        # Check outer patterns  
        for pattern in self.outer_patterns:
            if self._matches_pattern(str_path, pattern):
                return "outer"
                
        # Default to middle if no pattern matches
        return "middle"
    
    def _matches_pattern(self, filepath: str, pattern: str) -> bool:
        """Check if a file path matches a pattern."""
        import fnmatch
        
        # Handle directory patterns ending with /**/*
        if pattern.endswith("/**/*"):
            dir_pattern = pattern[:-5]  # Remove /**/*
            if fnmatch.fnmatch(filepath, dir_pattern) or filepath.startswith(dir_pattern + "/"):
                return True
                
        # Handle recursive patterns with **/
        if "**/" in pattern:
            # Convert ** to * for basic matching
            simple_pattern = pattern.replace("**/", "*")
            if fnmatch.fnmatch(filepath, simple_pattern):
                return True
                
        # Direct pattern matching
        return fnmatch.fnmatch(filepath, pattern)
    
    def analyze_repository(self) -> Dict[str, List[str]]:
        """Analyze the entire repository and classify all files."""
        classification = {
            "core": [],
            "middle": [],
            "outer": []
        }
        
        for root, dirs, files in os.walk(self.repo_root):
            # Skip hidden directories except .config
            dirs[:] = [d for d in dirs if not d.startswith('.') or d in ['.config', '.github']]
            
            for file in files:
                filepath = Path(root) / file
                
                # Skip hidden files except important ones
                if file.startswith('.') and file not in ['.gitignore', '.flake8']:
                    continue
                    
                try:
                    classification_level = self.classify_file(filepath)
                    rel_path = str(filepath.relative_to(self.repo_root))
                    classification[classification_level].append(rel_path)
                except Exception as e:
                    print(f"Warning: Could not classify {filepath}: {e}")
                    
        return classification
    
    def apply_permissions(self, classification: Dict[str, List[str]], dry_run: bool = True):
        """Apply appropriate permissions based on classification."""
        permission_map = {
            "core": 0o444,    # Read-only for all
            "middle": 0o644,  # Owner write, others read  
            "outer": 0o664    # Group write allowed
        }
        
        changes_made = []
        
        for level, files in classification.items():
            target_permissions = permission_map[level]
            
            for file_path in files:
                full_path = self.repo_root / file_path
                
                if not full_path.exists():
                    continue
                    
                current_permissions = full_path.stat().st_mode & 0o777
                
                if current_permissions != target_permissions:
                    if not dry_run:
                        try:
                            full_path.chmod(target_permissions)
                            changes_made.append(f"Changed {file_path}: {oct(current_permissions)} -> {oct(target_permissions)}")
                        except Exception as e:
                            print(f"Error changing permissions for {file_path}: {e}")
                    else:
                        changes_made.append(f"Would change {file_path}: {oct(current_permissions)} -> {oct(target_permissions)}")
                        
        return changes_made
    
    def generate_critical_files_registry(self, classification: Dict[str, List[str]]):
        """Generate the CRITICAL_FILES.md registry."""
        content = """# 🔴 CRITICAL FILES REGISTRY

## Core Framework Files (Require 2+ Reviewer Approval)

### Framework Entry Points
"""
        
        # Add core files with descriptions
        core_files = sorted(classification["core"])
        
        # Group files by category
        categories = {
            "Framework Entry Points": [],
            "Core Modules": [],
            "Configuration Files": [],
            "Build and Setup": []
        }
        
        for file_path in core_files:
            if "__init__.py" in file_path and "core" in file_path:
                categories["Framework Entry Points"].append(file_path)
            elif file_path.startswith("src/qemlflow/core/"):
                categories["Core Modules"].append(file_path)
            elif file_path in ["setup.py", "pyproject.toml", "requirements-core.txt"]:
                categories["Build and Setup"].append(file_path)
            elif file_path.startswith("src/qemlflow/config/"):
                categories["Configuration Files"].append(file_path)
            else:
                categories["Core Modules"].append(file_path)
        
        for category, files in categories.items():
            if files:
                content += f"\n### {category}\n"
                for file_path in files:
                    content += f"- `{file_path}` - {self._get_file_description(file_path)}\n"
        
        content += f"""

## Review Requirements for Core Files
- **2+ reviewer approval** required for all core file changes
- **Comprehensive testing** must pass before merge
- **Rollback plan** must be documented
- **Impact assessment** must be completed
- **Breaking change analysis** required

## Protection Status
- Total Core Files: {len(core_files)}
- Total Middle Layer Files: {len(classification['middle'])}
- Total Outer Layer Files: {len(classification['outer'])}
- Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Emergency Contact
- Core Maintainer: QeMLflow Development Team
- Emergency Contact: qemlflow-emergency@example.com

## Quick Commands
```bash
# Check file classification
python tools/maintenance/file_classifier.py --classify <file>

# Apply protection
python tools/maintenance/file_classifier.py --protect

# Audit permissions
python tools/maintenance/file_classifier.py --audit
```
"""
        
        with open(self.critical_files_registry, 'w') as f:
            f.write(content)
            
        print(f"✅ Generated critical files registry: {self.critical_files_registry}")
    
    def _get_file_description(self, file_path: str) -> str:
        """Get a description for a file based on its path and purpose."""
        descriptions = {
            "setup.py": "Package installation configuration",
            "pyproject.toml": "Project configuration and metadata", 
            "requirements-core.txt": "Essential dependencies",
            ".gitignore": "Version control exclusion rules",
            "src/qemlflow/__init__.py": "Framework entry point and API exports",
            "src/qemlflow/core/__init__.py": "Core module API exports",
            "src/qemlflow/core/data.py": "Data handling and I/O operations",
            "src/qemlflow/core/models.py": "Base model classes and interfaces",
            "src/qemlflow/core/featurizers.py": "Molecular featurization framework",
            "src/qemlflow/core/evaluation.py": "Model evaluation and metrics",
            "src/qemlflow/core/utils.py": "Core utility functions",
            "src/qemlflow/core/exceptions.py": "Custom exception classes",
            "src/qemlflow/config/__init__.py": "Configuration management",
            "src/qemlflow/utils/logging.py": "Logging configuration"
        }
        
        return descriptions.get(file_path, "Core framework component")
    
    def create_protection_config(self):
        """Create the protection configuration file."""
        config = {
            'protection_levels': {
                'core': {
                    'permissions': '444',
                    'require_review': True,
                    'min_reviewers': 2,
                    'require_tests': True,
                    'description': 'Critical files - maximum protection'
                },
                'middle': {
                    'permissions': '644', 
                    'require_review': True,
                    'min_reviewers': 1,
                    'require_tests': True,
                    'description': 'Important files - moderate protection'
                },
                'outer': {
                    'permissions': '664',
                    'require_review': False,
                    'min_reviewers': 0, 
                    'require_tests': False,
                    'description': 'Flexible files - minimal protection'
                }
            },
            'monitoring': {
                'enabled': True,
                'alerts': True,
                'backup_on_change': True,
                'log_all_changes': True
            },
            'emergency': {
                'bypass_protection': False,
                'emergency_contacts': [
                    'qemlflow-emergency@example.com'
                ],
                'escalation_timeout': 3600
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        print(f"✅ Created protection configuration: {self.config_file}")
    
    def audit_permissions(self) -> Dict[str, List[str]]:
        """Audit current file permissions against expected values."""
        classification = self.analyze_repository()
        
        permission_map = {
            "core": 0o444,
            "middle": 0o644,
            "outer": 0o664
        }
        
        issues = {
            "incorrect_permissions": [],
            "missing_files": [],
            "unclassified_files": []
        }
        
        for level, files in classification.items():
            expected_permissions = permission_map[level]
            
            for file_path in files:
                full_path = self.repo_root / file_path
                
                if not full_path.exists():
                    issues["missing_files"].append(file_path)
                    continue
                    
                current_permissions = full_path.stat().st_mode & 0o777
                
                if current_permissions != expected_permissions:
                    issues["incorrect_permissions"].append({
                        "file": file_path,
                        "level": level,
                        "current": oct(current_permissions),
                        "expected": oct(expected_permissions)
                    })
        
        return issues
        
    def main(self):
        """Main execution function."""
        import argparse
        
        parser = argparse.ArgumentParser(description="QeMLflow File Classification and Protection Tool")
        parser.add_argument("--analyze", action="store_true", help="Analyze repository and show classification")
        parser.add_argument("--protect", action="store_true", help="Apply protection permissions")
        parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
        parser.add_argument("--audit", action="store_true", help="Audit current permissions")
        parser.add_argument("--setup", action="store_true", help="Setup protection system")
        parser.add_argument("--classify", type=str, help="Classify a specific file")
        
        args = parser.parse_args()
        
        if args.setup:
            print("🛡️  Setting up QeMLflow protection system...")
            classification = self.analyze_repository()
            self.create_protection_config()
            self.generate_critical_files_registry(classification)
            print("✅ Protection system setup complete!")
            
        elif args.analyze:
            print("📊 Analyzing repository file classification...")
            classification = self.analyze_repository()
            
            for level, files in classification.items():
                print(f"\n🔵 {level.upper()} Layer ({len(files)} files):")
                for file_path in sorted(files)[:10]:  # Show first 10
                    print(f"  - {file_path}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more files")
                    
        elif args.protect:
            print("🛡️  Applying file protection...")
            classification = self.analyze_repository()
            changes = self.apply_permissions(classification, dry_run=args.dry_run)
            
            if changes:
                print(f"\n📝 {'Would make' if args.dry_run else 'Made'} {len(changes)} permission changes:")
                for change in changes[:20]:  # Show first 20
                    print(f"  {change}")
                if len(changes) > 20:
                    print(f"  ... and {len(changes) - 20} more changes")
            else:
                print("✅ All file permissions are correct!")
                
        elif args.audit:
            print("🔍 Auditing file permissions...")
            issues = self.audit_permissions()
            
            if issues["incorrect_permissions"]:
                print(f"\n⚠️  Found {len(issues['incorrect_permissions'])} permission issues:")
                for issue in issues["incorrect_permissions"][:10]:
                    print(f"  {issue['file']}: {issue['current']} -> {issue['expected']} ({issue['level']})")
                    
            if issues["missing_files"]:
                print(f"\n❌ Found {len(issues['missing_files'])} missing files:")
                for file_path in issues["missing_files"][:5]:
                    print(f"  {file_path}")
                    
            if not any(issues.values()):
                print("✅ All file permissions are correct!")
                
        elif args.classify:
            file_path = Path(args.classify)
            if file_path.exists():
                classification = self.classify_file(file_path)
                print(f"📁 {args.classify} -> {classification.upper()} layer")
            else:
                print(f"❌ File not found: {args.classify}")
                
        else:
            parser.print_help()

if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent.parent
    classifier = QeMLflowFileClassifier(str(repo_root))
    classifier.main()
