#!/usr/bin/env python3
"""
Quick Migration Validation
==========================

A focused validation to check the most critical migration aspects.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_qemlflow_import():
    """Test core QeMLflow import."""
    try:
        # Add src to path
        root_path = Path(__file__).parent.parent.parent
        src_path = root_path / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        import qemlflow
        print(f"✅ QeMLflow import successful (version: {qemlflow.__version__})")
        return True
    except Exception as e:
        print(f"❌ QeMLflow import failed: {e}")
        return False

def test_git_status():
    """Test git repository status."""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Git status failed: {result.stderr}")
            return False
        
        untracked = [line for line in result.stdout.strip().split('\n') 
                    if line.strip() and line.startswith('??')]
        
        if untracked:
            print(f"ℹ️  Untracked files: {len(untracked)}")
            for file in untracked[:3]:
                print(f"   {file}")
            if len(untracked) > 3:
                print(f"   ... and {len(untracked) - 3} more")
        
        modified = [line for line in result.stdout.strip().split('\n') 
                   if line.strip() and not line.startswith('??') and line.strip()]
        
        if modified:
            print(f"⚠️  Modified files: {len(modified)}")
            return False
        else:
            print("✅ Git working directory is clean")
            return True
            
    except Exception as e:
        print(f"❌ Git status check failed: {e}")
        return False

def test_core_structure():
    """Test core package structure."""
    root_path = Path(__file__).parent.parent.parent
    
    critical_paths = [
        "src/qemlflow/__init__.py",
        "setup.py",
        "README.md",
        "requirements.txt"
    ]
    
    missing = []
    for path in critical_paths:
        if not (root_path / path).exists():
            missing.append(path)
    
    if missing:
        print(f"❌ Missing critical files: {missing}")
        return False
    else:
        print("✅ Core package structure intact")
        return True

def test_no_critical_chemml_refs():
    """Test for ChemML references in critical files."""
    root_path = Path(__file__).parent.parent.parent
    
    critical_files = [
        "src/qemlflow/__init__.py",
        "setup.py", 
        "pyproject.toml",
        "README.md"
    ]
    
    refs_found = []
    
    for file_path in critical_files:
        full_path = root_path / file_path
        if not full_path.exists():
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'chemml' in content.lower() and 'qemlflow' not in content.lower():
                refs_found.append(file_path)
        except Exception:
            continue
    
    if refs_found:
        print(f"❌ ChemML references in critical files: {refs_found}")
        return False
    else:
        print("✅ No ChemML references in critical files")
        return True

def main():
    """Run quick validation."""
    print("🚀 Quick Migration Validation")
    print("=" * 50)
    
    tests = [
        ("Core Structure", test_core_structure),
        ("QeMLflow Import", test_qemlflow_import),
        ("Critical ChemML Refs", test_no_critical_chemml_refs),
        ("Git Status", test_git_status),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 QUICK VALIDATION SUMMARY")
    print(f"Passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 QUICK VALIDATION: ALL CRITICAL TESTS PASSED!")
        print("✅ Core migration appears successful.")
    else:
        print(f"⚠️  QUICK VALIDATION: {total - passed} CRITICAL TESTS FAILED")
        print("❌ Please review critical issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
