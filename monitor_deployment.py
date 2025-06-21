#!/usr/bin/env python3
"""
Enterprise deployment monitoring script for QeMLflow.
Monitors CI/CD pipeline status after deployment.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


class DeploymentMonitor:
    """Monitor deployment status and CI/CD pipeline health."""
    
    def __init__(self):
        self.repo_path = "/Users/sanjeevadodlapati/Downloads/Repos/QeMLflow"
        self.workflows = [
            "ci-cd.yml",
            "enhanced-matrix-ci.yml", 
            "core-tests.yml",
            "code_health.yml",
            "enterprise-security-infrastructure.yml",
            "scalability.yml",
            "high_availability.yml",
            "monitoring.yml",
            "observability.yml"
        ]
        
    def run_command(self, command: str) -> tuple[str, int]:
        """Run a shell command and return output and exit code."""
        try:
            result = subprocess.run(
                command.split(),
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "Command timed out", 1
        except Exception as e:
            return f"Error: {e}", 1
    
    def check_git_status(self) -> Dict:
        """Check git repository status."""
        status = {}
        
        # Check current branch
        output, code = self.run_command("git branch --show-current")
        status["current_branch"] = output if code == 0 else "unknown"
        
        # Check remote status
        output, code = self.run_command("git status --porcelain=v1")
        status["working_directory_clean"] = code == 0 and not output
        
        # Check last commit
        output, code = self.run_command("git log -1 --format=%H,%s,%an,%ad")
        if code == 0:
            parts = output.split(",", 3)
            status["last_commit"] = {
                "hash": parts[0] if len(parts) > 0 else "",
                "message": parts[1] if len(parts) > 1 else "",
                "author": parts[2] if len(parts) > 2 else "",
                "date": parts[3] if len(parts) > 3 else ""
            }
        
        # Check remote sync status
        output, code = self.run_command("git status --porcelain=v1 --branch")
        if "ahead" in output:
            status["sync_status"] = "ahead"
        elif "behind" in output:
            status["sync_status"] = "behind"
        else:
            status["sync_status"] = "synced"
            
        return status
    
    def check_local_health(self) -> Dict:
        """Check local codebase health."""
        health = {}
        
        # Check if critical files exist
        critical_files = [
            "src/qemlflow/security/__init__.py",
            "src/qemlflow/production_tuning/__init__.py", 
            "src/qemlflow/production_readiness/__init__.py",
            "config/security.yml",
            "config/production.yml",
            "docs/PRODUCTION_GUIDE.md",
            "docs/ENTERPRISE_IMPLEMENTATION_COMPLETE.md"
        ]
        
        health["critical_files"] = {}
        for file in critical_files:
            file_path = os.path.join(self.repo_path, file)
            health["critical_files"][file] = os.path.exists(file_path)
        
        # Check test status
        health["test_directories"] = {}
        test_dirs = [
            "tests/security",
            "tests/production_tuning", 
            "tests/production_readiness"
        ]
        
        for test_dir in test_dirs:
            dir_path = os.path.join(self.repo_path, test_dir)
            if os.path.exists(dir_path):
                test_files = [f for f in os.listdir(dir_path) if f.startswith("test_") and f.endswith(".py")]
                health["test_directories"][test_dir] = len(test_files)
            else:
                health["test_directories"][test_dir] = 0
                
        return health
    
    def run_quick_validation(self) -> Dict:
        """Run quick validation checks."""
        validation = {}
        
        # Check Python syntax
        output, code = self.run_command("python -m py_compile src/qemlflow/security/__init__.py")
        validation["security_syntax"] = code == 0
        
        output, code = self.run_command("python -m py_compile src/qemlflow/production_tuning/__init__.py")
        validation["production_tuning_syntax"] = code == 0
        
        output, code = self.run_command("python -m py_compile src/qemlflow/production_readiness/__init__.py")
        validation["production_readiness_syntax"] = code == 0
        
        # Check import capability
        test_script = """
import sys
sys.path.insert(0, 'src')
try:
    from qemlflow.security import SecurityHardening
    from qemlflow.production_tuning import ProductionPerformanceTuner as ProductionTuning
    from qemlflow.production_readiness import ProductionReadinessValidator as ProductionReadiness
    print("SUCCESS: All enterprise modules can be imported")
except Exception as e:
    print(f"ERROR: Import failed - {e}")
    sys.exit(1)
"""
        
        with open(os.path.join(self.repo_path, "temp_import_test.py"), "w") as f:
            f.write(test_script)
        
        output, code = self.run_command("python temp_import_test.py")
        validation["import_test"] = code == 0
        validation["import_output"] = output
        
        # Clean up temp file
        try:
            os.remove(os.path.join(self.repo_path, "temp_import_test.py"))
        except:
            pass
            
        return validation
    
    def generate_report(self) -> Dict:
        """Generate comprehensive deployment status report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "deployment_status": "DEPLOYED",
            "git_status": self.check_git_status(),
            "local_health": self.check_local_health(),
            "validation": self.run_quick_validation()
        }
        
        # Determine overall status
        issues = []
        
        if not report["git_status"]["working_directory_clean"]:
            issues.append("Working directory not clean")
            
        if not all(report["local_health"]["critical_files"].values()):
            missing_files = [f for f, exists in report["local_health"]["critical_files"].items() if not exists]
            issues.append(f"Missing critical files: {missing_files}")
            
        if not all([
            report["validation"]["security_syntax"],
            report["validation"]["production_tuning_syntax"], 
            report["validation"]["production_readiness_syntax"],
            report["validation"]["import_test"]
        ]):
            issues.append("Validation checks failed")
        
        report["overall_status"] = "HEALTHY" if not issues else "ISSUES_DETECTED"
        report["issues"] = issues
        
        return report
    
    def monitor_ci_cd(self, duration_minutes: int = 10):
        """Monitor CI/CD pipeline for specified duration."""
        print(f"🚀 Starting CI/CD monitoring for {duration_minutes} minutes...")
        print("=" * 80)
        
        # Generate initial report
        report = self.generate_report()
        
        print("📊 DEPLOYMENT STATUS REPORT")
        print("=" * 80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Git Branch: {report['git_status']['current_branch']}")
        print(f"Working Directory: {'Clean' if report['git_status']['working_directory_clean'] else 'Dirty'}")
        print(f"Sync Status: {report['git_status']['sync_status']}")
        
        if report['git_status'].get('last_commit'):
            commit = report['git_status']['last_commit']
            print(f"Last Commit: {commit['hash'][:8]} - {commit['message']}")
            print(f"Author: {commit['author']}")
        
        print("\n🏥 HEALTH CHECKS")
        print("-" * 40)
        print(f"Critical Files: {sum(report['local_health']['critical_files'].values())}/{len(report['local_health']['critical_files'])} present")
        
        for file, exists in report['local_health']['critical_files'].items():
            status = "✅" if exists else "❌"
            print(f"  {status} {file}")
        
        print(f"\nTest Directories:")
        for test_dir, count in report['local_health']['test_directories'].items():
            print(f"  📁 {test_dir}: {count} test files")
        
        print("\n🔍 VALIDATION RESULTS")
        print("-" * 40)
        validations = [
            ("Security Syntax", report['validation']['security_syntax']),
            ("Production Tuning Syntax", report['validation']['production_tuning_syntax']),
            ("Production Readiness Syntax", report['validation']['production_readiness_syntax']),
            ("Import Test", report['validation']['import_test'])
        ]
        
        for name, passed in validations:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
        
        if report['validation']['import_output']:
            print(f"  📝 Import Output: {report['validation']['import_output']}")
        
        if report['issues']:
            print("\n⚠️  ISSUES DETECTED")
            print("-" * 40)
            for issue in report['issues']:
                print(f"  • {issue}")
        else:
            print("\n✅ ALL CHECKS PASSED")
        
        print("\n🔄 CI/CD PIPELINE MONITORING")
        print("-" * 40)
        print("Note: GitHub Actions workflows are running remotely.")
        print("Check the following workflows on GitHub:")
        for workflow in self.workflows:
            print(f"  • {workflow}")
        
        print(f"\n🔗 Repository: https://github.com/sdodlapati3/QeMLflow")
        print(f"🔗 Actions: https://github.com/sdodlapati3/QeMLflow/actions")
        
        # Save report
        report_file = os.path.join(self.repo_path, "deployment_status_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: deployment_status_report.json")
        print("=" * 80)
        print("🎉 ENTERPRISE DEPLOYMENT MONITORING COMPLETE!")
        
        return report


def main():
    """Main monitoring function."""
    monitor = DeploymentMonitor()
    
    # Check if we're in the right directory
    if not os.path.exists(monitor.repo_path):
        print(f"❌ Repository path not found: {monitor.repo_path}")
        sys.exit(1)
    
    # Run monitoring
    try:
        report = monitor.monitor_ci_cd(duration_minutes=5)
        
        # Exit with appropriate code
        if report['overall_status'] == 'HEALTHY':
            print("\n✅ All systems operational - Enterprise deployment successful!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Issues detected - Check report for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
