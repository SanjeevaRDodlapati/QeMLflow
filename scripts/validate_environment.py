#!/usr/bin/env python3
"""
QeMLflow Environment Validation Script

Validates that the development environment is properly set up.
"""

import sys
import time
import subprocess
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)


def print_status(item, status, details=""):
    """Print formatted status."""
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {item}: {details}")


def check_python_version():
    """Check Python version compatibility."""
    print_header("Python Version Check")
    
    version = sys.version_info
    print_status("Python Version", True, f"{version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print_status("Version Compatibility", True, "Compatible with QeMLflow")
        return True
    else:
        print_status("Version Compatibility", False, "Requires Python 3.8+")
        return False


def check_virtual_environment():
    """Check if we're in a virtual environment."""
    print_header("Virtual Environment Check")
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    print_status("Virtual Environment", in_venv, 
                "Active" if in_venv else "Not active - consider using 'source venv/bin/activate'")
    
    if in_venv:
        print_status("Python Path", True, sys.executable)
    
    return in_venv


def check_core_imports():
    """Check if core dependencies can be imported."""
    print_header("Core Dependencies Check")
    
    core_modules = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'), 
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    success_count = 0
    
    for module, name in core_modules:
        try:
            __import__(module)
            print_status(name, True, "Available")
            success_count += 1
        except ImportError:
            print_status(name, False, "Not installed")
    
    return success_count == len(core_modules)


def check_qemlflow_import():
    """Check if QeMLflow can be imported."""
    print_header("QeMLflow Import Test")
    
    try:
        # Time the import
        start_time = time.time()
        import qemlflow
        import_time = time.time() - start_time
        
        print_status("QeMLflow Import", True, f"Success in {import_time:.2f}s")
        
        # Check if import time is reasonable (< 5s per philosophy)
        fast_import = import_time < 5.0
        print_status("Import Performance", fast_import, 
                    f"{'Fast' if fast_import else 'Slow'} ({import_time:.2f}s)")
        
        return True
        
    except ImportError as e:
        print_status("QeMLflow Import", False, f"Failed: {e}")
        return False


def check_development_tools():
    """Check if development tools are available."""
    print_header("Development Tools Check")
    
    tools = [
        ('pytest', 'PyTest'),
        ('black', 'Black Formatter'),
        ('isort', 'Import Sorter'),
        ('flake8', 'Flake8 Linter'),
        ('mypy', 'MyPy Type Checker'),
    ]
    
    success_count = 0
    
    for tool, name in tools:
        try:
            result = subprocess.run([tool, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print_status(name, True, "Available")
                success_count += 1
            else:
                print_status(name, False, "Not working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_status(name, False, "Not installed")
    
    return success_count >= len(tools) - 1  # Allow one tool to be missing


def check_project_structure():
    """Check if project structure is correct."""
    print_header("Project Structure Check")
    
    required_paths = [
        ('src/qemlflow', 'Source Code'),
        ('tests', 'Test Suite'),
        ('docs', 'Documentation'),
        ('requirements.txt', 'Requirements File'),
        ('pyproject.toml', 'Project Configuration'),
        ('Makefile', 'Build System'),
    ]
    
    success_count = 0
    
    for path, name in required_paths:
        path_obj = Path(path)
        exists = path_obj.exists()
        print_status(name, exists, f"{'Found' if exists else 'Missing'}: {path}")
        if exists:
            success_count += 1
    
    return success_count == len(required_paths)


def run_quick_test():
    """Run a quick test to ensure basic functionality."""
    print_header("Quick Functionality Test")
    
    try:
        # Try to run a simple test
        result = subprocess.run(['python', '-m', 'pytest', '--version'],
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print_status("Test Framework", True, "PyTest working")
            
            # Try to run actual tests if they exist
            if Path('tests').exists():
                result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-x', '--tb=no', '-q'],
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print_status("Test Suite", True, "Basic tests passing")
                    return True
                else:
                    print_status("Test Suite", False, "Some tests failing")
                    return False
            else:
                print_status("Test Suite", False, "No tests directory found")
                return False
        else:
            print_status("Test Framework", False, "PyTest not working")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Quick Test", False, "Timeout - tests taking too long")
        return False
    except Exception as e:
        print_status("Quick Test", False, f"Error: {e}")
        return False


def main():
    """Main validation function."""
    print("🌍 QeMLflow Environment Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Core Dependencies", check_core_imports),
        ("QeMLflow Import", check_qemlflow_import),
        ("Development Tools", check_development_tools),
        ("Project Structure", check_project_structure),
        ("Quick Tests", run_quick_test),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_status(check_name, False, f"Error during check: {e}")
            results.append((check_name, False))
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        print_status(check_name, result, "PASS" if result else "FAIL")
    
    print(f"\n📊 Overall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 Environment validation PASSED! You're ready to develop with QeMLflow.")
        return 0
    elif passed >= total * 0.7:
        print("\n⚠️  Environment validation mostly passed, but some issues need attention.")
        print("   Check the failed items above and fix them for optimal development experience.")
        return 1
    else:
        print("\n❌ Environment validation FAILED! Please fix the issues above before proceeding.")
        print("   Consider running 'make setup-dev' to set up the environment properly.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
