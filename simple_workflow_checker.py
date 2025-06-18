#!/usr/bin/env python3
"""
Simple GitHub Actions Status Checker
===================================

Quick status check for GitHub Actions workflows with robust repository detection.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


def get_repo_info() -> Tuple[str, str]:
    """Get repository owner and name from various sources."""
    
    # Try different methods to get repo info
    methods = [
        # Method 1: Git remote URL
        lambda: _get_repo_from_git_remote(),
        
        # Method 2: GitHub CLI
        lambda: _get_repo_from_gh_cli(),
        
        # Method 3: Manual fallback
        lambda: ("SanjeevaRDodlapati", "QeMLflow")  # Default for this project
    ]
    
    for method in methods:
        try:
            owner, name = method()
            if owner and name:
                print(f"✅ Detected repository: {owner}/{name}")
                return owner, name
        except Exception as e:
            print(f"⚠️  Method failed: {e}")
            continue
    
    raise ValueError("Could not determine repository information")


def _get_repo_from_git_remote() -> Tuple[str, str]:
    """Extract repo info from git remote."""
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True
    )
    
    remote_url = result.stdout.strip()
    print(f"Git remote URL: {remote_url}")
    
    # Handle various URL formats
    patterns = [
        r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$",  # Both SSH and HTTPS
        r"github\.com-\w+:([^/]+)/([^/]+?)(?:\.git)?$",  # SSH with custom host
    ]
    
    for pattern in patterns:
        match = re.search(pattern, remote_url)
        if match:
            return match.group(1), match.group(2)
    
    raise ValueError(f"Could not parse repository from URL: {remote_url}")


def _get_repo_from_gh_cli() -> Tuple[str, str]:
    """Get repo info from GitHub CLI."""
    result = subprocess.run(
        ["gh", "repo", "view", "--json", "owner,name"],
        capture_output=True,
        text=True,
        check=True
    )
    
    data = json.loads(result.stdout)
    return data["owner"]["login"], data["name"]


def check_workflow_status(owner: str, repo: str) -> Dict:
    """Check GitHub Actions workflow status."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    
    try:
        print(f"🔍 Fetching workflow runs from GitHub API...")
        response = requests.get(api_url, params={"per_page": 10}, timeout=30)
        
        if response.status_code == 403:
            print("⚠️  API rate limit exceeded - using limited data")
            return {"error": "rate_limit", "message": "GitHub API rate limit exceeded"}
        
        response.raise_for_status()
        data = response.json()
        
        runs = data.get("workflow_runs", [])
        if not runs:
            return {"error": "no_runs", "message": "No workflow runs found"}
        
        # Analyze recent runs
        status_counts = {"success": 0, "failure": 0, "cancelled": 0, "in_progress": 0}
        recent_runs = []
        
        for run in runs[:10]:  # Analyze last 10 runs
            conclusion = run.get("conclusion") or run.get("status", "unknown")
            status_counts[conclusion] = status_counts.get(conclusion, 0) + 1
            
            recent_runs.append({
                "id": run["id"],
                "name": run["name"],
                "status": conclusion,
                "created_at": run["created_at"],
                "head_sha": run["head_sha"][:8],
                "html_url": run["html_url"]
            })
        
        return {
            "success": True,
            "total_runs": len(runs),
            "status_counts": status_counts,
            "recent_runs": recent_runs,
            "latest_run": runs[0] if runs else None
        }
        
    except requests.RequestException as e:
        return {"error": "api_error", "message": f"API request failed: {e}"}


def print_workflow_summary(status_data: Dict):
    """Print a summary of workflow status."""
    print("\n" + "="*70)
    print("           GITHUB ACTIONS WORKFLOW STATUS")
    print("="*70)
    
    if "error" in status_data:
        print(f"❌ Error: {status_data['message']}")
        return
    
    # Overall statistics
    counts = status_data["status_counts"]
    total = sum(counts.values())
    
    print(f"📊 RECENT WORKFLOW SUMMARY (Last {total} runs):")
    print(f"   ✅ Success: {counts.get('success', 0)}")
    print(f"   ❌ Failure: {counts.get('failure', 0)}")
    print(f"   ⏸️  Cancelled: {counts.get('cancelled', 0)}")
    print(f"   🔄 In Progress: {counts.get('in_progress', 0)}")
    
    # Calculate health percentage
    if total > 0:
        success_rate = (counts.get('success', 0) / total) * 100
        health_emoji = "🟢" if success_rate >= 80 else "🟡" if success_rate >= 60 else "🔴"
        print(f"   {health_emoji} Success Rate: {success_rate:.1f}%")
    
    # Latest run details
    latest = status_data.get("latest_run")
    if latest:
        status_emoji = {"success": "✅", "failure": "❌", "cancelled": "⏸️"}.get(latest.get("conclusion"), "🔄")
        print(f"\n🕒 LATEST RUN:")
        print(f"   {status_emoji} {latest['name']}")
        print(f"   📅 {latest['created_at']}")
        print(f"   🔗 {latest['html_url']}")
    
    # Recent runs list
    print(f"\n📋 RECENT RUNS:")
    for run in status_data["recent_runs"][:5]:
        status_emoji = {"success": "✅", "failure": "❌", "cancelled": "⏸️"}.get(run["status"], "🔄")
        print(f"   {status_emoji} {run['name']} ({run['head_sha']}) - {run['created_at'][:10]}")
    
    print("="*70)


def main():
    """Main function."""
    print("🚀 GitHub Actions Workflow Status Checker")
    print("="*50)
    
    try:
        # Get repository information
        owner, repo = get_repo_info()
        
        # Check workflow status
        status_data = check_workflow_status(owner, repo)
        
        # Print summary
        print_workflow_summary(status_data)
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "repository": f"{owner}/{repo}",
            "status_data": status_data
        }
        
        with open("workflow_status.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: workflow_status.json")
        
        # Determine exit code based on health
        if status_data.get("success"):
            failure_count = status_data["status_counts"].get("failure", 0)
            if failure_count > 2:
                print(f"\n⚠️  Multiple recent failures detected ({failure_count})")
                sys.exit(1)
            else:
                print(f"\n✅ Workflow status looks good!")
                sys.exit(0)
        else:
            print(f"\n❌ Unable to determine workflow health")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
