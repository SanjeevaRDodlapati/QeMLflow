#!/usr/bin/env python3
"""
VALIDATED Comprehensive Validation Script
========================================

This script safely validates all fixes to ensure they don't introduce new problems.
Corrected version with proper error handling and logic.
"""

import ast
import os
import sys
from pathlib import Path


class SafeFixValidator:
    def __init__(self, repo_root=None):
        # Auto-detect repository root instead of hard-coding
        if repo_root is None:
            current_dir = Path.cwd()
            # Look for repository markers
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                    repo_root = parent
                    break
            else:
                repo_root = current_dir
        
        self.repo_root = Path(repo_root)
        self.src_dir = self.repo_root / "src"
        self.issues_found = []
        self.fixes_validated = []
    
    def log_issue(self, severity, component, message):
        """Safely log validation issues"""
        issue = {
            'severity': severity,
            'component': str(component), 
            'message': str(message)
        }
        self.issues_found.append(issue)
        print(f"🚨 {severity.upper()}: {Path(component).name} - {message}")
    
    def log_success(self, component, message):
        """Safely log successful validations"""
        fix = {
            'component': str(component),
            'message': str(message)
        }
        self.fixes_validated.append(fix)
        print(f"✅ {Path(component).name}: {message}")
    
    def safe_read_file(self, file_path):
        """Safely read file with proper error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except FileNotFoundError:
            self.log_issue("error", file_path, "File not found")
            return None
        except PermissionError:
            self.log_issue("error", file_path, "Permission denied")
            return None
        except Exception as e:
            self.log_issue("warning", file_path, f"Read error: {e}")
            return None
    
    def validate_syntax(self, file_path):
        """Validate Python syntax safely"""
        content = self.safe_read_file(file_path)
        if content is None:
            return False
            
        try:
            # Check for specific problematic patterns we've been fixing
            problematic_patterns = [
                ("from typing import.*?\\\\n", "Escaped newlines in typing imports"),
                ("\\\\n.*?\"\"\"", "Escaped newlines before docstrings")
            ]
            
            import re
            for pattern, description in problematic_patterns:
                if re.search(pattern, content):
                    self.log_issue("error", file_path, f"Found issue: {description}")
                    return False
            
            # Parse AST to validate syntax
            ast.parse(content, filename=str(file_path))
            return True
            
        except SyntaxError as e:
            self.log_issue("error", file_path, f"Syntax error at line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            self.log_issue("warning", file_path, f"Parse warning: {e}")
            return False
    
    def validate_typing_imports(self, file_path):
        """Check typing imports are correct and complete"""
        content = self.safe_read_file(file_path)
        if content is None:
            return False
            
        try:
            # Corrected type usage patterns
            type_patterns = [
                r'\bList\[',
                r'\bDict\[', 
                r'\bOptional\[',
                r'\bUnion\[',
                r'\bTuple\[',
                r'\bAny\b',  # Any doesn't use brackets
                r'\bCallable\['
            ]
            
            import re
            uses_types = any(re.search(pattern, content) for pattern in type_patterns)
            
            if uses_types:
                # Check for proper typing import
                typing_import_patterns = [
                    r'from typing import.*?(List|Dict|Optional|Union|Any|Tuple|Callable)',
                    r'import typing'
                ]
                
                has_typing_import = any(re.search(pattern, content) for pattern in typing_import_patterns)
                
                if not has_typing_import:
                    self.log_issue("error", file_path, "Uses type annotations but missing typing imports")
                    return False
                else:
                    self.log_success(file_path, "Proper typing imports found")
            
            return True
            
        except Exception as e:
            self.log_issue("warning", file_path, f"Typing validation error: {e}")
            return False
    
    def validate_critical_files(self):
        """Validate files that are known to cause workflow failures"""
        if not self.src_dir.exists():
            self.log_issue("error", "src directory", "Source directory not found")
            return False
            
        critical_paths = [
            "qemlflow/__init__.py",
            "qemlflow/research/__init__.py", 
            "qemlflow/research/clinical_research.py",
            "qemlflow/research/materials_discovery.py",
            "qemlflow/research/quantum.py"
        ]
        
        all_valid = True
        for rel_path in critical_paths:
            file_path = self.src_dir / rel_path
            if file_path.exists():
                print(f"\n🔍 Validating {rel_path}...")
                syntax_ok = self.validate_syntax(file_path)
                typing_ok = self.validate_typing_imports(file_path)
                
                if syntax_ok and typing_ok:
                    self.log_success(rel_path, "All validations passed")
                else:
                    all_valid = False
            else:
                self.log_issue("warning", rel_path, "File not found")
        
        return all_valid
    
    def test_basic_imports(self):
        """Test if basic imports work (safe approach)"""
        try:
            # Test if we can at least compile the main init file
            main_init = self.src_dir / "qemlflow" / "__init__.py"
            if main_init.exists():
                content = self.safe_read_file(main_init)
                if content:
                    compile(content, str(main_init), 'exec')
                    self.log_success("Import test", "Main __init__.py compiles successfully")
                    return True
        except Exception as e:
            self.log_issue("error", "Import test", f"Compilation failed: {e}")
        
        return False
    
    def generate_safe_report(self):
        """Generate validation report with risk assessment"""
        print("\n" + "="*60)
        print("📊 SAFE VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✅ Successful validations: {len(self.fixes_validated)}")
        for fix in self.fixes_validated[-3:]:  # Show last 3 to avoid spam
            print(f"   • {fix['message']}")
        
        critical_issues = [i for i in self.issues_found if i['severity'] == 'error']
        warnings = [i for i in self.issues_found if i['severity'] == 'warning']
        
        print(f"\n🚨 Critical issues: {len(critical_issues)}")
        for issue in critical_issues[-3:]:  # Show last 3 critical issues
            print(f"   • {issue['component']}: {issue['message']}")
        
        print(f"\n⚠️ Warnings: {len(warnings)}")
        
        # Safe risk assessment
        is_safe = len(critical_issues) == 0
        print(f"\n🎯 SAFETY ASSESSMENT:")
        if is_safe:
            print("   ✅ SAFE: No critical issues found")
            print("   ✅ Workflow fixes should work correctly")
        else:
            print("   🚨 UNSAFE: Critical issues need fixing")
            print("   🚨 May introduce new workflow failures")
        
        return is_safe
    
    def run_safe_validation(self):
        """Run validation with maximum safety"""
        print("🔍 Starting SAFE comprehensive validation...")
        print(f"📂 Repository root: {self.repo_root}")
        print(f"📂 Source directory: {self.src_dir}")
        
        try:
            files_valid = self.validate_critical_files()
            imports_work = self.test_basic_imports()
            
            overall_safe = files_valid and imports_work
            return self.generate_safe_report() and overall_safe
            
        except Exception as e:
            self.log_issue("error", "Validation process", f"Unexpected error: {e}")
            return False


if __name__ == "__main__":
    try:
        validator = SafeFixValidator()
        is_safe = validator.run_safe_validation()
        
        if is_safe:
            print("\n🎉 VALIDATION PASSED! Safe to proceed with workflow fixes.")
            sys.exit(0)
        else:
            print("\n⚠️ VALIDATION FAILED! Review issues before proceeding.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n🚨 VALIDATION SCRIPT ERROR: {e}")
        print("⚠️ Cannot guarantee safety. Manual review required.")
        sys.exit(1)
