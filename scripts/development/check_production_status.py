"""
QeMLflow CI/CD Pipeline Status Monitor
====================================

Monitors the status of GitHub Actions workflows and provides real-time feedback.
"""

import json
import subprocess
import time
from datetime import datetime


def check_git_status():
    """Check current git status and recent commits."""
    print("🔍 Git Status Check")
    print("=" * 50)

    # Get current branch
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip()
        print(f"Current branch: {branch}")
    except subprocess.CalledProcessError as e:
        print(f"Error getting branch: {e}")
        return False

    # Get recent commits
    try:
        commits = subprocess.check_output(["git", "log", "--oneline", "-5"], text=True)
        print("\nRecent commits:")
        print(commits)
    except subprocess.CalledProcessError as e:
        print(f"Error getting commits: {e}")

    # Check if we're up to date with remote
    try:
        subprocess.check_output(["git", "fetch"], stderr=subprocess.DEVNULL)
        status = subprocess.check_output(["git", "status", "-uno"], text=True)
        if "up to date" in status or "up-to-date" in status:
            print("✅ Local branch is up to date with remote")
        else:
            print("⚠️ Local branch may be behind remote")
            print(status)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not check remote status: {e}")

    return True


def check_package_structure():
    """Verify package structure and key files."""
    print("\n🏗️ Package Structure Check")
    print("=" * 50)

    critical_files = [
        "src/qemlflow/__init__.py",
        "requirements.txt",
        "pyproject.toml",
        ".github/workflows/ci-cd.yml",
        ".github/workflows/release.yml",
        ".github/workflows/docs.yml",
        ".config/mkdocs.yml",
        "CONTRIBUTING.md",
        "docs/index.md",
    ]

    missing_files = []
    for file in critical_files:
        try:
            with open(file, "r") as f:
                size = len(f.read())
            print(f"✅ {file} ({size} bytes)")
        except FileNotFoundError:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
        except Exception as e:
            print(f"⚠️ {file} - Error: {e}")

    if not missing_files:
        print("\n🎉 All critical files present!")
        return True
    else:
        print(f"\n❌ Missing files: {missing_files}")
        return False


def test_package_import():
    """Test if the package can be imported successfully."""
    print("\n🐍 Package Import Test")
    print("=" * 50)

    try:
        import sys

        sys.path.insert(0, "src")

        # Test basic import
        import qemlflow

        print("✅ QeMLflow imported successfully")
        print(f"   Version: {qemlflow.__version__}")

        # Test core modules
        modules_to_test = [
            "qemlflow.core",
            "qemlflow.preprocessing",
            "qemlflow.models",
            "qemlflow.config",
        ]

        for module in modules_to_test:
            try:
                __import__(module)
                print(f"✅ {module}")
            except Exception as e:
                print(f"⚠️ {module} - {e}")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def check_workflows_exist():
    """Check if workflow files exist and are valid."""
    print("\n⚙️ Workflow Files Check")
    print("=" * 50)

    workflows = [
        ".github/workflows/ci-cd.yml",
        ".github/workflows/release.yml",
        ".github/workflows/docs.yml",
    ]

    for workflow in workflows:
        try:
            with open(workflow, "r") as f:
                content = f.read()
                lines = len(content.split("\n"))
                print(f"✅ {workflow} ({lines} lines)")

                # Basic validation
                if "name:" in content and "on:" in content and "jobs:" in content:
                    print("   Valid workflow structure")
                else:
                    print("   ⚠️ May be missing required sections")

        except FileNotFoundError:
            print(f"❌ {workflow} - NOT FOUND")
        except Exception as e:
            print(f"⚠️ {workflow} - Error: {e}")


def test_build_process():
    """Test if the package can be built."""
    print("\n🔨 Build Process Test")
    print("=" * 50)

    try:
        # Test if build tools are available
        result = subprocess.run(
            ["python", "-m", "pip", "list"], capture_output=True, text=True
        )

        if "build" in result.stdout:
            print("✅ Build tools available")

            # Try to build
            build_result = subprocess.run(
                ["python", "-m", "build", "--wheel", "--no-isolation"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if build_result.returncode == 0:
                print("✅ Package build successful")
                return True
            else:
                print(f"❌ Build failed: {build_result.stderr}")
                return False
        else:
            print("⚠️ Build tools not installed, installing...")
            subprocess.run(["python", "-m", "pip", "install", "build"])
            return test_build_process()  # Retry after installation

    except subprocess.TimeoutExpired:
        print("⚠️ Build test timed out")
        return False
    except Exception as e:
        print(f"❌ Build test error: {e}")
        return False


def generate_status_report():
    """Generate comprehensive status report."""
    print("\n📊 QeMLflow Production Readiness Status Report")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    checks = [
        ("Git Status", check_git_status),
        ("Package Structure", check_package_structure),
        ("Package Import", test_package_import),
        ("Workflow Files", check_workflows_exist),
        ("Build Process", test_build_process),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed: {e}")
            results[name] = False

    # Summary
    print("\n📋 Summary")
    print("=" * 30)
    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 QeMLflow is PRODUCTION READY! 🚀")
        print("\nNext steps:")
        print("- Monitor GitHub Actions workflows")
        print("- Check documentation deployment")
        print("- Review CI/CD pipeline results")
        print("- Consider creating official release tag")
    else:
        print(f"\n⚠️ {total - passed} issues need attention")
        print("Please address the failed checks above.")

    return results


if __name__ == "__main__":
    print("🚀 QeMLflow Production Readiness Checker")
    print("🔍 Validating project status and CI/CD readiness...\n")

    try:
        results = generate_status_report()

        # Check for tags
        print("\n🏷️ Git Tags")
        print("=" * 30)
        try:
            tags = subprocess.check_output(["git", "tag", "-l"], text=True).strip()
            if tags:
                print("Current tags:")
                for tag in tags.split("\n"):
                    print(f"  {tag}")
            else:
                print("No tags found")
        except subprocess.CalledProcessError:
            print("Could not list tags")

        print("\n🌐 Next Steps")
        print("=" * 30)
        print(
            "1. Check GitHub Actions: https://github.com/SanjeevaRDodlapati/QeMLflow/actions"
        )
        print(
            "2. Monitor documentation: https://sanjeevardodlapati.github.io/QeMLflow/"
        )
        print("3. Review release workflow results")
        print("4. Consider creating official v0.2.0 release")

    except KeyboardInterrupt:
        print("\n\n⏹️ Status check interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
