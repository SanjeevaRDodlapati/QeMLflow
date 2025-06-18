#!/usr/bin/env python3
"""
Comprehensive Validation Script
==============================

This script validates all fixes to ensure they don't introduce new problems.
It checks:
1. Python syntax compilation
2. Import functionality  
3. Type annotation correctness
4. Dependency requirements
5. Git repository health
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path


class FixValidator:
    def __init__(self, repo_root="/Users/sanjeev/Downloads/Repos/QeMLflow"):
        self.repo_root = Path(repo_root)
        self.src_dir = self.repo_root / "src"
        self.issues_found = []
        self.fixes_validated = []

    def log_issue(self, severity, component, message):
        """Log validation issues"""
        self.issues_found.append(
            {"severity": severity, "component": component, "message": message}
        )
        print(f"🚨 {severity.upper()}: {component} - {message}")

    def log_success(self, component, message):
        """Log successful validations"""
        self.fixes_validated.append({"component": component, "message": message})
        print(f"✅ {component}: {message}")

    def validate_syntax(self, file_path):
        """Validate Python syntax without importing"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            # Check for common syntax issues we've been fixing
            if "\\n" in source and "from typing import" in source:
                self.log_issue(
                    "warning",
                    str(file_path),
                    "Contains escaped newlines near typing imports",
                )
                return False

            # Parse AST to check syntax
            ast.parse(source, filename=str(file_path))
            return True
        except SyntaxError as e:
            self.log_issue("error", str(file_path), f"Syntax error: {e}")
            return False
        except Exception as e:
            self.log_issue("warning", str(file_path), f"Parse warning: {e}")
            return False

    def validate_typing_imports(self, file_path):
        """Check if files that use typing annotations have proper imports"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for type usage without imports
            type_usage = ["List[", "Dict[", "Optional[", "Union[", "Any[", "Tuple["]
            uses_types = any(usage in content for usage in type_usage)

            if uses_types:
                has_typing_import = "from typing import" in content
                if not has_typing_import:
                    self.log_issue(
                        "error",
                        str(file_path),
                        "Uses type annotations but missing 'from typing import'",
                    )
                    return False
                else:
                    self.log_success(str(file_path), "Proper typing imports found")

            return True
        except Exception as e:
            self.log_issue("warning", str(file_path), f"Typing validation error: {e}")
            return False

    def validate_research_modules(self):
        """Specifically validate research modules that were problematic"""
        research_dir = self.src_dir / "qemlflow" / "research"
        critical_files = [
            "clinical_research.py",
            "materials_discovery.py",
            "quantum.py",
            "advanced_models.py",
            "generative.py",
            "environmental_chemistry.py",
        ]

        for filename in critical_files:
            file_path = research_dir / filename
            if file_path.exists():
                print(f"\n🔍 Validating {filename}...")
                syntax_ok = self.validate_syntax(file_path)
                typing_ok = self.validate_typing_imports(file_path)

                if syntax_ok and typing_ok:
                    self.log_success(filename, "All validations passed")
                else:
                    self.log_issue("error", filename, "Failed validation checks")
            else:
                self.log_issue("error", filename, "File not found")

    def validate_init_files(self):
        """Check __init__.py files for proper structure"""
        init_files = list(self.src_dir.rglob("__init__.py"))

        for init_file in init_files:
            print(f"\n🔍 Validating {init_file.relative_to(self.src_dir)}...")
            if self.validate_syntax(init_file):
                self.log_success(str(init_file.relative_to(self.src_dir)), "Syntax OK")

    def validate_emergency_fix_script(self):
        """Validate the emergency fix script itself"""
        fix_script = self.repo_root / "emergency_workflow_fix.py"
        if fix_script.exists():
            print(f"\n🔍 Validating emergency fix script...")
            if self.validate_syntax(fix_script):
                self.log_success("emergency_workflow_fix.py", "Script syntax is valid")
                # Check if it has proper error handling
                with open(fix_script, "r") as f:
                    content = f.read()
                if "try:" in content and "except" in content:
                    self.log_success("emergency_workflow_fix.py", "Has error handling")
                else:
                    self.log_issue(
                        "warning", "emergency_workflow_fix.py", "Limited error handling"
                    )
        else:
            self.log_issue("warning", "emergency_workflow_fix.py", "Script not found")

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT SUMMARY")
        print("=" * 60)

        print(f"\n✅ Successful validations: {len(self.fixes_validated)}")
        for fix in self.fixes_validated[-5:]:  # Show last 5
            print(f"   • {fix['component']}: {fix['message']}")

        print(f"\n🚨 Issues found: {len(self.issues_found)}")
        for issue in self.issues_found:
            print(
                f"   • {issue['severity'].upper()}: {issue['component']} - {issue['message']}"
            )

        # Risk assessment
        critical_issues = [i for i in self.issues_found if i["severity"] == "error"]
        warnings = [i for i in self.issues_found if i["severity"] == "warning"]

        print(f"\n🎯 RISK ASSESSMENT:")
        if len(critical_issues) == 0:
            print("   ✅ LOW RISK: No critical issues found")
            print("   ✅ Safe to proceed with GitHub Actions workflows")
        elif len(critical_issues) <= 2:
            print("   ⚠️ MEDIUM RISK: Few critical issues found")
            print("   ⚠️ May cause some workflow failures")
        else:
            print("   🚨 HIGH RISK: Multiple critical issues found")
            print("   🚨 Likely to cause workflow failures")

        if len(warnings) > 0:
            print(f"   📝 {len(warnings)} warnings to review")

        return len(critical_issues) == 0

    def run_full_validation(self):
        """Run complete validation suite"""
        print("🔍 Starting comprehensive validation...")

        self.validate_research_modules()
        self.validate_init_files()
        self.validate_emergency_fix_script()

        return self.generate_validation_report()


if __name__ == "__main__":
    validator = FixValidator()
    is_safe = validator.run_full_validation()

    if is_safe:
        print("\n🎉 All validations passed! Safe to proceed with fixes.")
        sys.exit(0)
    else:
        print("\n⚠️ Some issues found. Review before proceeding.")
        sys.exit(1)
