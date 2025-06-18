#!/usr/bin/env python3
"""
Terminal Diagnostic and Fix Script
=================================

This script diagnoses and attempts to fix terminal responsiveness issues.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def diagnose_terminal_issues():
    """Diagnose potential terminal problems"""
    print("🔧 TERMINAL DIAGNOSTIC REPORT")
    print("="*40)
    
    # Check current working directory
    try:
        cwd = os.getcwd()
        print(f"✅ Current directory: {cwd}")
    except Exception as e:
        print(f"❌ Directory issue: {e}")
    
    # Check environment variables
    important_vars = ['PATH', 'SHELL', 'TERM', 'PWD']
    for var in important_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"📝 {var}: {value[:100]}...")
    
    # Check if we can execute basic commands
    basic_commands = [
        ('python', ['python', '--version']),
        ('git', ['git', '--version']),
        ('ls', ['ls', '-la']),
        ('pwd', ['pwd'])
    ]
    
    print("\n🧪 Testing basic commands:")
    for name, cmd in basic_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()[:50]}...")
            else:
                print(f"⚠️ {name}: returned code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {name}: timed out")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    # Check process info
    print(f"\n📊 Process info:")
    print(f"   PID: {os.getpid()}")
    print(f"   Python executable: {sys.executable}")
    print(f"   Python version: {sys.version}")
    

def suggest_terminal_fixes():
    """Suggest fixes for common terminal issues"""
    print("\n🛠️ SUGGESTED TERMINAL FIXES:")
    print("="*40)
    
    fixes = [
        "1. Try using absolute paths instead of relative paths",
        "2. Check if any background processes are hanging",
        "3. Verify file permissions on repository directory", 
        "4. Clear terminal environment variables if corrupted",
        "5. Use file operations instead of shell commands when possible",
        "6. Add explicit timeouts to subprocess calls",
        "7. Check disk space and memory availability"
    ]
    
    for fix in fixes:
        print(f"   {fix}")


def test_workarounds():
    """Test alternative approaches that might work better"""
    print("\n🔄 TESTING WORKAROUNDS:")
    print("="*40)
    
    # Test file operations
    try:
        test_file = Path("/tmp/terminal_test.txt")
        test_file.write_text("Terminal test")
        content = test_file.read_text()
        test_file.unlink()
        print("✅ File operations: Working")
    except Exception as e:
        print(f"❌ File operations: {e}")
    
    # Test Python subprocess with timeout
    try:
        result = subprocess.run(['echo', 'test'], capture_output=True, text=True, timeout=2)
        print(f"✅ Subprocess with timeout: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Subprocess with timeout: {e}")
    
    # Test direct Python execution
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('print("Direct Python execution works")')
            temp_script = f.name
        
        result = subprocess.run([sys.executable, temp_script], capture_output=True, text=True, timeout=5)
        os.unlink(temp_script)
        print(f"✅ Direct Python execution: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Direct Python execution: {e}")


def create_robust_command_runner():
    """Create a more robust command execution function"""
    print("\n🔨 CREATING ROBUST COMMAND RUNNER")
    print("="*40)
    
    runner_code = '''
def robust_run_command(cmd, timeout=30, use_shell=True):
    """More robust command execution with multiple fallback strategies"""
    import subprocess
    import sys
    import os
    
    strategies = [
        # Strategy 1: Normal execution
        lambda: subprocess.run(cmd, shell=use_shell, capture_output=True, text=True, timeout=timeout),
        
        # Strategy 2: With explicit PATH
        lambda: subprocess.run(cmd, shell=use_shell, capture_output=True, text=True, 
                              timeout=timeout, env=dict(os.environ, PATH=os.environ.get('PATH', ''))),
        
        # Strategy 3: With cwd explicitly set
        lambda: subprocess.run(cmd, shell=use_shell, capture_output=True, text=True,
                              timeout=timeout, cwd=os.getcwd()),
                              
        # Strategy 4: Direct execution without shell
        lambda: subprocess.run(cmd.split() if isinstance(cmd, str) else cmd, 
                              capture_output=True, text=True, timeout=timeout)
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            result = strategy()
            print(f"✅ Strategy {i} worked: {result.stdout[:50]}...")
            return result
        except Exception as e:
            print(f"❌ Strategy {i} failed: {e}")
    
    print("🚨 All strategies failed")
    return None
'''
    
    # Save to file for future use
    runner_file = Path("robust_command_runner.py")
    runner_file.write_text(runner_code)
    print(f"✅ Saved robust command runner to: {runner_file}")


if __name__ == "__main__":
    diagnose_terminal_issues()
    suggest_terminal_fixes()
    test_workarounds()
    create_robust_command_runner()
    
    print("\n🎯 TERMINAL DIAGNOSTIC COMPLETE")
    print("="*40)
    print("Review the output above to identify terminal issues.")
    print("Use the suggested fixes and workarounds as needed.")
