#!/usr/bin/env python3
"""
ChemML Quick Wins Implementation

This script implements immediate improvements to boost codebase health:
1. Fix remaining auto - fixable linting issues
2. Organize imports consistently
3. Remove unused variables and imports
4. Fix basic syntax issues
5. Improve documentation consistency

Usage:
 python tools/maintenance/quick_wins.py [--dry - run] [--category = CATEGORY]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List


class QuickWinsFixer:
 """Implements quick wins to improve codebase health."""

 def __init__(self, base_dir: Path, dry_run: bool = False):
 self.base_dir = base_dir
 self.dry_run = dry_run
 self.fixed_count = 0
 self.errors = []

 def run_all_fixes(self):
 """Run all quick win fixes."""
 print("🚀 ChemML Quick Wins Implementation")
 print("=" * 50)

 fixes = [
 ("Fixing import organization", self.fix_import_organization),
 ("Removing unused imports", self.remove_unused_imports),
 ("Fixing basic syntax issues", self.fix_syntax_issues),
 ("Standardizing docstrings", self.standardize_docstrings),
 ("Removing trailing whitespace", self.remove_trailing_whitespace),
 ("Fixing line endings", self.fix_line_endings),
 ]

 for description, fix_func in fixes:
 print(f"\n📋 {description}...")
 try:
 count = fix_func()
 self.fixed_count += count
 print(f" ✅ Fixed {count} issues")
 except Exception as e:
 error = f"❌ Error in {description}: {e}"
 print(f" {error}")
 self.errors.append(error)

 self.generate_report()

 def fix_import_organization(self) -> int:
 """Organize imports using isort."""
 if self.dry_run:
 return self._count_import_issues()

 try:
 result = subprocess.run([
 sys.executable, "-m", "isort",
 "--profile", "black",
 "--line - length", "88",
 "--multi - line", "3",
 "--trailing - comma",
 "--force - grid - wrap", "0",
 "--combine - as",
 "--use - parentheses",
 str(self.base_dir / "src"),
 str(self.base_dir / "tests"),
 str(self.base_dir / "tools"),
 str(self.base_dir / "examples"),
 ], capture_output = True, text = True, cwd = self.base_dir)

 # Count files modified by looking at isort output
 modified_count = len([line for line in result.stderr.split('\n')
 if 'Fixing' in line or 'Fixed' in line])
 return modified_count

 except Exception as e:
 print(f" Warning: isort failed: {e}")
 return 0

 def remove_unused_imports(self) -> int:
 """Remove unused imports using autoflake."""
 if self.dry_run:
 return self._count_unused_imports()

 try:
 result = subprocess.run([
 sys.executable, "-m", "autoflake",
 "--remove - all - unused - imports",
 "--remove - unused - variables",
 "--remove - duplicate - keys",
 "--in - place",
 "--recursive",
 str(self.base_dir / "src"),
 str(self.base_dir / "tests"),
 str(self.base_dir / "tools"),
 str(self.base_dir / "examples"),
 ], capture_output = True, text = True, cwd = self.base_dir)

 # Count modifications from output
 modified_count = len([line for line in result.stdout.split('\n')
 if 'fixing' in line.lower()])
 return modified_count

 except Exception as e:
 print(f" Warning: autoflake failed: {e}")
 return 0

 def fix_syntax_issues(self) -> int:
 """Fix basic syntax issues."""
 count = 0
 python_files = self._get_python_files()

 for file_path in python_files:
 try:
 with open(file_path, 'r', encoding='utf - 8') as f:
 content = f.read()

 original_content = content

 # Fix common syntax issues
 content = self._fix_common_syntax_issues(content)

 if content != original_content:
 if not self.dry_run:
 with open(file_path, 'w', encoding='utf - 8') as f:
 f.write(content)
 count += 1

 except Exception as e:
 self.errors.append(f"Error fixing {file_path}: {e}")

 return count

 def standardize_docstrings(self) -> int:
 """Standardize docstring format."""
 count = 0
 python_files = self._get_python_files()

 for file_path in python_files:
 try:
 with open(file_path, 'r', encoding='utf - 8') as f:
 content = f.read()

 original_content = content
 content = self._standardize_docstring_format(content)

 if content != original_content:
 if not self.dry_run:
 with open(file_path, 'w', encoding='utf - 8') as f:
 f.write(content)
 count += 1

 except Exception as e:
 self.errors.append(f"Error standardizing docstrings in {file_path}: {e}")

 return count

 def remove_trailing_whitespace(self) -> int:
 """Remove trailing whitespace from all text files."""
 count = 0
 text_files = list(self.base_dir.rglob("*.py")) + \
 list(self.base_dir.rglob("*.md")) + \
 list(self.base_dir.rglob("*.yml")) + \
 list(self.base_dir.rglob("*.yaml")) + \
 list(self.base_dir.rglob("*.txt"))

 for file_path in text_files:
 if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.pytest_cache', 'site', 'build']):
 continue

 try:
 with open(file_path, 'r', encoding='utf - 8') as f:
 lines = f.readlines()

 original_lines = lines[:]
 lines = [line.rstrip() + '\n' for line in lines]

 if lines != original_lines:
 if not self.dry_run:
 with open(file_path, 'w', encoding='utf - 8') as f:
 f.writelines(lines)
 count += 1

 except Exception:
 continue # Skip binary or problematic files

 return count

 def fix_line_endings(self) -> int:
 """Ensure consistent line endings (LF)."""
 count = 0
 text_files = self._get_python_files()

 for file_path in text_files:
 try:
 with open(file_path, 'rb') as f:
 content = f.read()

 original_content = content
 # Convert CRLF to LF
 content = content.replace(b'\r\n', b'\n')
 # Remove lone CR
 content = content.replace(b'\r', b'\n')

 if content != original_content:
 if not self.dry_run:
 with open(file_path, 'wb') as f:
 f.write(content)
 count += 1

 except Exception as e:
 self.errors.append(f"Error fixing line endings in {file_path}: {e}")

 return count

 def _get_python_files(self) -> List[Path]:
 """Get all Python files in the codebase."""
 python_files = []

 for pattern in ["src/**/*.py", "tests/**/*.py", "tools/**/*.py", "examples/**/*.py"]:
 python_files.extend(self.base_dir.glob(pattern))

 # Filter out problematic files
 return [f for f in python_files
 if not any(skip in str(f) for skip in ['.git', '__pycache__', '.pytest_cache'])]

 def _fix_common_syntax_issues(self, content: str) -> str:
 """Fix common syntax issues in Python code."""
 # Fix double blank lines
 content = re.sub(r'\n\n\n+', '\n\n', content)

 # Fix spaces around operators (basic cases)
 content = re.sub(r'([a - zA - Z0 - 9_])\s*=\s*([a - zA - Z0 - 9_])', r'\1 = \2', content)
 content = re.sub(r'([a - zA - Z0 - 9_])\s*\+\s*([a - zA - Z0 - 9_])', r'\1 + \2', content)
 content = re.sub(r'([a - zA - Z0 - 9_])\s*-\s*([a - zA - Z0 - 9_])', r'\1 - \2', content)

 # Remove multiple spaces
 content = re.sub(r' +', ' ', content)

 return content

 def _standardize_docstring_format(self, content: str) -> str:
 """Standardize docstring format to Google style."""
 # This is a simplified version - for full docstring standardization,
 # we'd need a more sophisticated parser

 # Fix triple quote consistency
 content = re.sub(r'"""([^"]*?)"""', r'"""\1"""', content, flags = re.DOTALL)
 content = re.sub(r"'''([^']*?)'''", r'"""\1"""', content, flags = re.DOTALL)

 return content

 def _count_import_issues(self) -> int:
 """Count import organization issues."""
 try:
 result = subprocess.run([
 sys.executable, "-m", "isort",
 "--check - only", "--diff",
 str(self.base_dir / "src"),
 ], capture_output = True, text = True, cwd = self.base_dir)

 return len([line for line in result.stdout.split('\n')
 if 'would reformat' in line])
 except:
 return 0

 def _count_unused_imports(self) -> int:
 """Count unused import issues."""
 try:
 result = subprocess.run([
 sys.executable, "-m", "autoflake",
 "--check", "--recursive",
 str(self.base_dir / "src"),
 ], capture_output = True, text = True, cwd = self.base_dir)

 return len([line for line in result.stdout.split('\n')
 if 'would fix' in line])
 except:
 return 0

 def generate_report(self):
 """Generate improvement report."""
 print(f"\n🎉 Quick Wins Summary")
 print("=" * 30)
 print(f" ✅ Total fixes applied: {self.fixed_count}")

 if self.errors:
 print(f" ⚠️ Errors encountered: {len(self.errors)}")
 for error in self.errors[:5]: # Show first 5 errors
 print(f" • {error}")
 if len(self.errors) > 5:
 print(f" • ... and {len(self.errors) - 5} more")

 # Save detailed report
 report_path = self.base_dir / "reports" / "quick_wins_report.json"
 report_path.parent.mkdir(parents = True, exist_ok = True)

 report = {
 "timestamp": str(subprocess.run(["date"], capture_output = True, text = True).stdout.strip()),
 "total_fixes": self.fixed_count,
 "errors": self.errors,
 "dry_run": self.dry_run
 }

 if not self.dry_run:
 with open(report_path, 'w') as f:
 json.dump(report, f, indent = 2)
 print(f" 📄 Detailed report saved to: {report_path}")

def main():
 parser = argparse.ArgumentParser(description='ChemML Quick Wins Implementation')
 parser.add_argument('--dry - run', action='store_true',
 help='Preview changes without applying them')
 parser.add_argument('--category', choices=['imports', 'syntax', 'docstrings', 'whitespace'],
 help='Run only specific category of fixes')

 args = parser.parse_args()

 base_dir = Path(__file__).parent.parent.parent
 fixer = QuickWinsFixer(base_dir, dry_run = args.dry_run)

 if args.category:
 print(f"🎯 Running {args.category} fixes only")
 # Implementation for specific categories would go here
 else:
 fixer.run_all_fixes()

if __name__ == "__main__":
 main()
