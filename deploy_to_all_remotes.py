#!/usr/bin/env python3
"""
QeMLflow Multi-Remote Deployment Tool (Python Version)

A Python-based deployment tool that automatically switches GitHub accounts
and pushes to multiple remotes. Provides better cross-platform compatibility
and more detailed error handling than the bash version.

Usage:
    python deploy_to_all_remotes.py [options] [commit_message]
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional, Union


class GitHubDeployer:
    """Multi-remote GitHub deployment tool with automatic account switching."""
    
    def __init__(self):
        # Configuration: (remote_name, github_username)
        self.remotes = [
            ("origin", "SanjeevaRDodlapati"),
            ("sdodlapa", "sdodlapa"),
            ("sdodlapati3", "sdodlapati3")
        ]
        
        # Colors for console output
        self.colors = {
            'RED': '\033[0;31m',
            'GREEN': '\033[0;32m',
            'YELLOW': '\033[1;33m',
            'BLUE': '\033[0;34m',
            'NC': '\033[0m'  # No Color
        }
    
    def log(self, level: str, message: str) -> None:
        """Log messages with color coding."""
        color = self.colors.get(level.upper(), self.colors['NC'])
        print(f"{color}[{level.upper()}]{self.colors['NC']} {message}")
    
    def run_command(self, cmd: List[str], capture_output: bool = True, 
                    check: bool = True) -> Optional[subprocess.CompletedProcess]:
        """Run a shell command and return the result."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError:
            return None
    
    def check_git_repository(self) -> bool:
        """Check if current directory is a git repository."""
        try:
            self.run_command(["git", "rev-parse", "--git-dir"])
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_git_status(self) -> str:
        """Get current git status (clean or uncommitted)."""
        result = self.run_command(["git", "status", "--porcelain"])
        if result is None:
            return "unknown"
        return "clean" if not result.stdout.strip() else "uncommitted"
    
    def get_current_branch(self) -> str:
        """Get current git branch."""
        result = self.run_command(["git", "branch", "--show-current"])
        if result is None:
            return "unknown"
        return result.stdout.strip()
    
    def check_gh_auth(self, username: str) -> bool:
        """Check if GitHub CLI is authenticated for a specific user."""
        try:
            self.run_command(["gh", "auth", "status", "--user", username])
            return True
        except subprocess.CalledProcessError:
            return False
    
    def switch_github_account(self, username: str) -> bool:
        """Switch to a different GitHub account."""
        self.log("info", f"Switching to GitHub account: {username}")
        
        try:
            self.run_command(["gh", "auth", "switch", "--user", username])
            self.log("green", f"Successfully switched to: {username}")
            return True
        except subprocess.CalledProcessError:
            self.log("red", f"Failed to switch to GitHub account: {username}")
            return False
    
    def check_remote_exists(self, remote: str) -> bool:
        """Check if a git remote exists."""
        try:
            self.run_command(["git", "remote", "get-url", remote])
            return True
        except subprocess.CalledProcessError:
            return False
    
    def push_to_remote(self, remote: str, branch: str = "main") -> bool:
        """Push to a specific remote."""
        self.log("info", f"Pushing to remote: {remote} (branch: {branch})")
        
        if not self.check_remote_exists(remote):
            self.log("red", f"Remote '{remote}' not found in git configuration")
            return False
        
        try:
            self.run_command(["git", "push", remote, branch])
            self.log("green", f"Successfully pushed to: {remote}")
            return True
        except subprocess.CalledProcessError:
            self.log("red", f"Failed to push to: {remote}")
            return False
    
    def commit_changes(self, message: str) -> bool:
        """Commit changes with the provided message."""
        if not message:
            return True
            
        self.log("info", f"Committing changes with message: {message}")
        
        # Add all changes
        try:
            self.run_command(["git", "add", "-A"])
        except subprocess.CalledProcessError:
            self.log("red", "Failed to add changes")
            return False
        
        # Check if there are changes to commit
        try:
            result = self.run_command(["git", "diff", "--staged", "--quiet"], check=False)
            if result.returncode == 0:
                self.log("yellow", "No changes to commit")
                return True
        except subprocess.CalledProcessError:
            pass
        
        # Commit changes
        try:
            self.run_command(["git", "commit", "-m", message])
            self.log("green", "Changes committed successfully")
            return True
        except subprocess.CalledProcessError:
            self.log("red", "Failed to commit changes")
            return False
    
    def create_deployment_summary(self, deployments: List[Dict]) -> str:
        """Create a deployment summary file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get git information
        try:
            commit_hash = self.run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
            commit_message = self.run_command(["git", "log", "-1", "--pretty=%B"]).stdout.strip()
            current_branch = self.get_current_branch()
        except subprocess.CalledProcessError:
            commit_hash = "unknown"
            commit_message = "unknown"
            current_branch = "unknown"
        
        # Ensure deployment directory exists
        os.makedirs("docs/deployment", exist_ok=True)
        
        summary_file = f"docs/deployment/deployment_{date_suffix}.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"""# Multi-Remote Deployment Summary

**Date:** {timestamp}  
**Commit:** {commit_hash}  
**Branch:** {current_branch}  

## Commit Message
```
{commit_message}
```

## Deployment Status

| Remote | Account | Status | Notes |
|--------|---------|--------|-------|
""")
            
            for deployment in deployments:
                status_icon = "✅" if deployment['success'] else "❌"
                f.write(f"| {deployment['remote']} | {deployment['user']} | {status_icon} {deployment['status']} | {deployment['notes']} |\n")
            
            successful = sum(1 for d in deployments if d['success'])
            failed = len(deployments) - successful
            
            f.write(f"""

## Summary
- **Successful:** {successful} remotes
- **Failed:** {failed} remotes

*Generated by deploy_to_all_remotes.py*
""")
        
        return summary_file
    
    def test_authentication(self) -> bool:
        """Test authentication for all configured accounts."""
        self.log("info", "Testing authentication for all configured accounts...")
        
        all_authenticated = True
        for remote, username in self.remotes:
            if self.check_gh_auth(username):
                self.log("green", f"{username}: ✅ Authenticated")
            else:
                self.log("red", f"{username}: ❌ Not authenticated")
                all_authenticated = False
        
        if all_authenticated:
            self.log("green", "All accounts are properly authenticated")
        else:
            self.log("red", "Some accounts need authentication. Run: gh auth login --user <username>")
        
        return all_authenticated
    
    def show_status(self) -> None:
        """Show current deployment configuration and status."""
        self.log("info", "QeMLflow Multi-Remote Deployment Configuration")
        print()
        print(f"Current directory: {os.getcwd()}")
        
        try:
            origin_url = self.run_command(["git", "remote", "get-url", "origin"]).stdout.strip()
        except subprocess.CalledProcessError:
            origin_url = "Not configured"
        
        print(f"Git repository: {origin_url}")
        print(f"Current branch: {self.get_current_branch()}")
        print(f"Git status: {self.get_git_status()}")
        print()
        print("Configured remotes:")
        
        for remote, username in self.remotes:
            try:
                remote_url = self.run_command(["git", "remote", "get-url", remote]).stdout.strip()
            except subprocess.CalledProcessError:
                remote_url = "Not configured"
            print(f"  {remote} ({username}): {remote_url}")
        
        print()
    
    def deploy(self, commit_message: Optional[str] = None, branch: str = "main") -> bool:
        """Main deployment function."""
        self.log("info", "Starting QeMLflow multi-remote deployment")
        self.log("info", f"Current directory: {os.getcwd()}")
        self.log("info", f"Git status: {self.get_git_status()}")
        
        # Ensure we're in a git repository
        if not self.check_git_repository():
            self.log("red", "Not in a git repository")
            return False
        
        # Commit changes if message provided
        if commit_message:
            if not self.commit_changes(commit_message):
                self.log("red", "Failed to commit changes, aborting deployment")
                return False
        
        # Check git status after potential commit
        if self.get_git_status() != "clean":
            self.log("yellow", "Working directory has uncommitted changes")
            self.log("info", "These changes will not be pushed to remotes")
        
        deployments = []
        successful_deployments = []
        failed_deployments = []
        
        self.log("info", f"Deploying to {len(self.remotes)} remotes...")
        
        # Process each remote
        for remote_name, github_user in self.remotes:
            self.log("info", f"Processing remote: {remote_name} (user: {github_user})")
            
            deployment = {
                'remote': remote_name,
                'user': github_user,
                'success': False,
                'status': 'Failed',
                'notes': ''
            }
            
            # Check authentication
            if not self.check_gh_auth(github_user):
                self.log("red", f"Skipping {remote_name} due to authentication issues")
                deployment['notes'] = "Authentication failed"
                failed_deployments.append(remote_name)
                deployments.append(deployment)
                continue
            
            # Switch GitHub account
            if not self.switch_github_account(github_user):
                self.log("red", f"Skipping {remote_name} due to account switch failure")
                deployment['notes'] = "Account switch failed"
                failed_deployments.append(remote_name)
                deployments.append(deployment)
                continue
            
            # Push to remote
            if self.push_to_remote(remote_name, branch):
                deployment['success'] = True
                deployment['status'] = 'Success'
                deployment['notes'] = 'Deployed successfully'
                successful_deployments.append(remote_name)
            else:
                deployment['notes'] = 'Push failed'
                failed_deployments.append(remote_name)
            
            deployments.append(deployment)
        
        # Create deployment summary
        summary_file = self.create_deployment_summary(deployments)
        
        # Final report
        print()
        self.log("info", "Deployment Summary:")
        if successful_deployments:
            self.log("green", f"Successfully deployed to: {', '.join(successful_deployments)}")
        else:
            self.log("yellow", "No successful deployments")
        
        if failed_deployments:
            self.log("red", f"Failed deployments: {', '.join(failed_deployments)}")
            self.log("info", f"Deployment summary saved to: {summary_file}")
            return False
        else:
            self.log("green", "All deployments completed successfully!")
            self.log("info", f"Deployment summary saved to: {summary_file}")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="QeMLflow Multi-Remote Deployment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_to_all_remotes.py "feat: add new feature"
  python deploy_to_all_remotes.py --branch develop "fix: critical bug"
  python deploy_to_all_remotes.py --test
  python deploy_to_all_remotes.py --status
        """
    )
    
    parser.add_argument(
        "commit_message",
        nargs="?",
        help="Commit message (optional)"
    )
    
    parser.add_argument(
        "-b", "--branch",
        default="main",
        help="Branch to push (default: main)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test authentication for all accounts"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show deployment configuration and status"
    )
    
    args = parser.parse_args()
    
    deployer = GitHubDeployer()
    
    if args.test:
        success = deployer.test_authentication()
        sys.exit(0 if success else 1)
    
    if args.status:
        deployer.show_status()
        deployer.test_authentication()
        return
    
    # Main deployment
    success = deployer.deploy(args.commit_message, args.branch)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
