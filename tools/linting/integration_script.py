#!/usr/bin/env python3
"""
Integration Script for Robust Multi-Layer Linting
=================================================

This script demonstrates how to integrate the robust linting framework
into different development workflows (CI/CD, pre-commit, IDE integration).
"""

import os
import subprocess
import sys
from pathlib import Path


def run_robust_linting_for_ci():
    """Run robust linting for CI/CD pipeline."""
    print("🚀 Running CI/CD Robust Linting Check...")

    # Run the robust linter
    result = subprocess.run(
        [sys.executable, "tools/linting/robust_multi_linter.py", "--scan-project"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("❌ Robust linting failed")
        print(result.stderr)
        return False

    # Parse output to check for high-confidence issues
    output = result.stdout
    if "Strong Consensus (3+ tools): 0" not in output:
        print("⚠️ High-confidence issues found - review required")
        print(output)
        return False

    print("✅ No high-confidence issues found")
    return True


def run_robust_linting_for_precommit(file_paths):
    """Run robust linting for pre-commit hook."""
    print(f"🔍 Running pre-commit robust linting on {len(file_paths)} files...")

    if not file_paths:
        return True

    # Run robust linter on specific files
    cmd = [sys.executable, "tools/linting/robust_multi_linter.py"] + file_paths
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)

    # Allow commit if no strong consensus issues
    return "Strong Consensus (3+ tools): 0" in result.stdout


def setup_git_hooks():
    """Set up git hooks for robust linting."""
    git_dir = Path(".git")
    if not git_dir.exists():
        print("❌ Not a git repository")
        return False

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    # Create pre-commit hook
    pre_commit_hook = hooks_dir / "pre-commit"
    hook_content = f"""#!/bin/bash
# Robust multi-tool linting pre-commit hook

# Get list of staged Python files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\\.py$')

if [ -z "$FILES" ]; then
    echo "No Python files to check"
    exit 0
fi

echo "🔍 Running robust linting on staged files..."
python {Path.cwd()}/tools/linting/robust_multi_linter.py $FILES

# Check if there are high-confidence issues
if [[ $? -ne 0 ]]; then
    echo "❌ High-confidence linting issues found. Commit blocked."
    echo "Please fix the issues or use 'git commit --no-verify' to skip."
    exit 1
fi

echo "✅ Robust linting passed"
exit 0
"""

    with open(pre_commit_hook, "w") as f:
        f.write(hook_content)

    # Make executable
    os.chmod(pre_commit_hook, 0o755)

    print("✅ Git pre-commit hook installed")
    return True


def create_vscode_tasks():
    """Create VS Code tasks for robust linting."""
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)

    tasks_file = vscode_dir / "tasks.json"

    tasks_config = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Robust Lint: Current File",
                "type": "shell",
                "command": "python",
                "args": ["tools/linting/robust_multi_linter.py", "${file}"],
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared",
                },
                "problemMatcher": [],
                "detail": "Run robust multi-tool linting on current file",
            },
            {
                "label": "Robust Lint: Full Project",
                "type": "shell",
                "command": "python",
                "args": ["tools/linting/robust_multi_linter.py", "--scan-project"],
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared",
                },
                "problemMatcher": [],
                "detail": "Run robust multi-tool linting on entire project",
            },
            {
                "label": "Robust Lint: High Confidence Only",
                "type": "shell",
                "command": "python",
                "args": ["tools/linting/robust_multi_linter.py", "--scan-project"],
                "group": "build",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared",
                },
                "problemMatcher": [],
                "detail": "Show only high-confidence consensus issues",
            },
        ],
    }

    import json

    with open(tasks_file, "w") as f:
        json.dump(tasks_config, f, indent=2)

    print("✅ VS Code tasks created")
    return True


def main():
    """Main integration setup."""
    print("🔧 Setting up Robust Multi-Layer Linting Integration")
    print("=" * 60)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "ci":
            success = run_robust_linting_for_ci()
            sys.exit(0 if success else 1)

        elif command == "precommit":
            file_paths = sys.argv[2:] if len(sys.argv) > 2 else []
            success = run_robust_linting_for_precommit(file_paths)
            sys.exit(0 if success else 1)

        elif command == "setup-hooks":
            setup_git_hooks()

        elif command == "setup-vscode":
            create_vscode_tasks()

        elif command == "setup-all":
            setup_git_hooks()
            create_vscode_tasks()
            print("\n🎉 Robust linting integration setup complete!")
            print("\nAvailable commands:")
            print("  python integration_script.py ci                 # Run for CI/CD")
            print(
                "  python integration_script.py precommit [files] # Run for pre-commit"
            )
            print(
                "  python integration_script.py setup-hooks       # Install git hooks"
            )
            print(
                "  python integration_script.py setup-vscode      # Create VS Code tasks"
            )

    else:
        print("Usage: python integration_script.py <command>")
        print("\nCommands:")
        print("  ci                 - Run robust linting for CI/CD")
        print("  precommit [files]  - Run robust linting for pre-commit")
        print("  setup-hooks        - Install git pre-commit hook")
        print("  setup-vscode       - Create VS Code tasks")
        print("  setup-all          - Setup everything")


if __name__ == "__main__":
    main()
