#!/usr/bin/env python3
"""
GitHub Actions Workflow Monitor & Diagnostic System
==================================================

This script provides comprehensive monitoring and diagnostics for GitHub Actions workflows,
including failure analysis, root cause identification, and automated remediation suggestions.

Features:
- Real-time workflow status monitoring
- Detailed failure analysis with log parsing
- Root cause identification 
- Automated fix suggestions
- Historical trend analysis
- Robust error detection and reporting

Usage:
    python github_actions_monitor.py [--repo REPO] [--check-logs] [--analyze-failures]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import requests
except ImportError:
    print("⚠️  Installing required dependency: requests")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


class GitHubActionsMonitor:
    """Comprehensive GitHub Actions monitoring and diagnostic system."""

    def __init__(self, repo_path: str = ".", github_token: Optional[str] = None):
        self.repo_path = Path(repo_path).resolve()
        self.github_token = github_token
        
        # Extract repo info from git remote
        self.repo_owner, self.repo_name = self._get_repo_info()
        
        # API configuration
        self.api_base = "https://api.github.com"
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "repository": f"{self.repo_owner}/{self.repo_name}",
            "workflow_status": {},
            "failure_analysis": {},
            "root_causes": [],
            "recommendations": [],
            "health_score": 0,
        }

    def _get_repo_info(self) -> Tuple[str, str]:
        """Extract repository owner and name from git remote."""
        try:
            # Get remote origin URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            remote_url = result.stdout.strip()
            
            # Parse GitHub URL (supports both HTTPS and SSH)
            if "github.com" in remote_url:
                # Extract owner/repo from URL
                if remote_url.startswith("git@"):
                    # SSH format: git@github.com:owner/repo.git
                    match = re.search(r"github\.com:([^/]+)/([^/]+?)(?:\.git)?$", remote_url)
                else:
                    # HTTPS format: https://github.com/owner/repo.git
                    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$", remote_url)
                
                if match:
                    return match.group(1), match.group(2)
            
            raise ValueError(f"Could not parse GitHub repository from URL: {remote_url}")
            
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Could not get git remote URL: {e}")

    def get_workflow_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent workflow runs from GitHub Actions."""
        url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
        params = {"per_page": limit}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get("workflow_runs", [])
            
        except requests.RequestException as e:
            print(f"❌ Error fetching workflow runs: {e}")
            return []

    def get_workflow_jobs(self, run_id: int) -> List[Dict[str, Any]]:
        """Fetch jobs for a specific workflow run."""
        url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/actions/runs/{run_id}/jobs"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get("jobs", [])
            
        except requests.RequestException as e:
            print(f"❌ Error fetching jobs for run {run_id}: {e}")
            return []

    def get_job_logs(self, job_id: int) -> str:
        """Fetch logs for a specific job."""
        url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/actions/jobs/{job_id}/logs"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                return f"Could not fetch logs (HTTP {response.status_code})"
                
        except requests.RequestException as e:
            return f"Error fetching logs: {e}"

    def analyze_failure_logs(self, logs: str) -> Dict[str, Any]:
        """Analyze failure logs to identify root causes."""
        analysis = {
            "error_types": [],
            "missing_imports": [],
            "syntax_errors": [],
            "dependency_issues": [],
            "test_failures": [],
            "root_cause": "unknown",
            "suggestions": []
        }
        
        # Common error patterns
        patterns = {
            "import_error": [
                r"ImportError: No module named ['\"]([^'\"]+)['\"]",
                r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
                r"cannot import name ['\"]([^'\"]+)['\"]"
            ],
            "syntax_error": [
                r"SyntaxError: (.+)",
                r"IndentationError: (.+)",
                r"TabError: (.+)"
            ],
            "type_error": [
                r"TypeError: (.+)",
                r"NameError: name ['\"]([^'\"]+)['\"] is not defined"
            ],
            "dependency_error": [
                r"pip.*ERROR: (.+)",
                r"Could not find a version that satisfies the requirement (.+)",
                r"No matching distribution found for (.+)"
            ],
            "test_failure": [
                r"FAILED (.+) - (.+)",
                r"AssertionError: (.+)",
                r"pytest.*FAILED"
            ]
        }
        
        for error_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, logs, re.IGNORECASE | re.MULTILINE)
                if matches:
                    analysis["error_types"].append(error_type)
                    
                    if error_type == "import_error":
                        analysis["missing_imports"].extend(matches)
                    elif error_type == "syntax_error":
                        analysis["syntax_errors"].extend(matches)
                    elif error_type == "dependency_error":
                        analysis["dependency_issues"].extend(matches)
                    elif error_type == "test_failure":
                        analysis["test_failures"].extend(matches)
        
        # Determine primary root cause
        if "import_error" in analysis["error_types"]:
            analysis["root_cause"] = "missing_imports"
            analysis["suggestions"].append("Add missing import statements")
            analysis["suggestions"].append("Check typing module imports")
        elif "syntax_error" in analysis["error_types"]:
            analysis["root_cause"] = "syntax_errors"
            analysis["suggestions"].append("Fix syntax errors in Python files")
            analysis["suggestions"].append("Check for proper indentation")
        elif "dependency_error" in analysis["error_types"]:
            analysis["root_cause"] = "dependency_issues"
            analysis["suggestions"].append("Update requirements.txt")
            analysis["suggestions"].append("Fix package version conflicts")
        elif "test_failure" in analysis["error_types"]:
            analysis["root_cause"] = "test_failures"
            analysis["suggestions"].append("Fix failing test cases")
            analysis["suggestions"].append("Update test assertions")
        
        return analysis

    def check_current_status(self) -> Dict[str, Any]:
        """Check current workflow status and analyze recent failures."""
        print(f"🔍 Checking GitHub Actions status for {self.repo_owner}/{self.repo_name}")
        
        runs = self.get_workflow_runs(20)
        if not runs:
            return {"status": "error", "message": "Could not fetch workflow runs"}
        
        status_summary = {
            "total_runs": len(runs),
            "recent_status": {},
            "failure_count": 0,
            "success_count": 0,
            "latest_run": None,
        }
        
        # Analyze recent runs
        for run in runs:
            conclusion = run.get("conclusion", "unknown")
            status_summary["recent_status"][run["id"]] = {
                "status": conclusion,
                "created_at": run["created_at"],
                "workflow_name": run["name"],
                "head_sha": run["head_sha"][:8]
            }
            
            if conclusion == "failure":
                status_summary["failure_count"] += 1
            elif conclusion == "success":
                status_summary["success_count"] += 1
        
        if runs:
            status_summary["latest_run"] = runs[0]
        
        self.results["workflow_status"] = status_summary
        return status_summary

    def analyze_recent_failures(self, max_failures: int = 5) -> List[Dict[str, Any]]:
        """Analyze recent workflow failures in detail."""
        print("🔍 Analyzing recent workflow failures...")
        
        runs = self.get_workflow_runs(20)
        failed_runs = [run for run in runs if run.get("conclusion") == "failure"][:max_failures]
        
        failure_analyses = []
        
        for run in failed_runs:
            print(f"📋 Analyzing failed run: {run['name']} ({run['id']})")
            
            analysis = {
                "run_id": run["id"],
                "workflow_name": run["name"],
                "created_at": run["created_at"],
                "head_sha": run["head_sha"][:8],
                "jobs": [],
                "combined_analysis": {}
            }
            
            # Get jobs for this run
            jobs = self.get_workflow_jobs(run["id"])
            
            all_logs = ""
            for job in jobs:
                if job.get("conclusion") == "failure":
                    job_analysis = {
                        "job_name": job["name"],
                        "job_id": job["id"],
                        "logs": "",
                        "analysis": {}
                    }
                    
                    print(f"  📝 Fetching logs for failed job: {job['name']}")
                    logs = self.get_job_logs(job["id"])
                    job_analysis["logs"] = logs[:5000]  # Limit log size
                    job_analysis["analysis"] = self.analyze_failure_logs(logs)
                    
                    analysis["jobs"].append(job_analysis)
                    all_logs += logs + "\n"
            
            # Combined analysis of all failed jobs
            analysis["combined_analysis"] = self.analyze_failure_logs(all_logs)
            failure_analyses.append(analysis)
        
        self.results["failure_analysis"] = failure_analyses
        return failure_analyses

    def generate_recommendations(self) -> List[str]:
        """Generate automated recommendations based on analysis."""
        recommendations = []
        
        # Analyze failure patterns
        if "failure_analysis" in self.results:
            all_root_causes = []
            all_suggestions = []
            
            for failure in self.results["failure_analysis"]:
                combined = failure.get("combined_analysis", {})
                if combined.get("root_cause"):
                    all_root_causes.append(combined["root_cause"])
                all_suggestions.extend(combined.get("suggestions", []))
            
            # Most common root cause
            if all_root_causes:
                from collections import Counter
                most_common = Counter(all_root_causes).most_common(1)[0][0]
                
                if most_common == "missing_imports":
                    recommendations.append("🎯 PRIORITY: Add missing typing imports (List, Dict, Optional, Union, Any)")
                    recommendations.append("🔧 Run: python -c \"from typing import List, Dict, Optional, Union, Any\"")
                elif most_common == "syntax_errors":
                    recommendations.append("🎯 PRIORITY: Fix Python syntax errors")
                    recommendations.append("🔧 Run: python -m py_compile <file> to check syntax")
                elif most_common == "dependency_issues":
                    recommendations.append("🎯 PRIORITY: Fix dependency installation issues")
                    recommendations.append("🔧 Update requirements.txt and package versions")
        
        # General recommendations
        recommendations.extend([
            "📊 Monitor this dashboard regularly for early issue detection",
            "🔄 Set up automated monitoring with scheduled runs",
            "📝 Keep detailed logs of all fixes for future reference"
        ])
        
        self.results["recommendations"] = recommendations
        return recommendations

    def calculate_health_score(self) -> int:
        """Calculate overall repository health score (0-100)."""
        score = 100
        
        if "workflow_status" in self.results:
            status = self.results["workflow_status"]
            total = status.get("total_runs", 1)
            failures = status.get("failure_count", 0)
            
            if total > 0:
                failure_rate = failures / total
                score -= int(failure_rate * 50)  # Deduct up to 50 points for failures
        
        if "failure_analysis" in self.results:
            failure_count = len(self.results["failure_analysis"])
            score -= min(failure_count * 10, 30)  # Deduct up to 30 points for recent failures
        
        self.results["health_score"] = max(score, 0)
        return score

    def print_summary(self):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print("           GITHUB ACTIONS WORKFLOW ANALYSIS SUMMARY")
        print("="*80)
        
        # Repository info
        print(f"📁 Repository: {self.repo_owner}/{self.repo_name}")
        print(f"📅 Analysis Time: {self.results['timestamp']}")
        
        # Workflow status
        if "workflow_status" in self.results:
            status = self.results["workflow_status"]
            print(f"\n📊 WORKFLOW STATUS:")
            print(f"   Total Recent Runs: {status.get('total_runs', 0)}")
            print(f"   ✅ Successful: {status.get('success_count', 0)}")
            print(f"   ❌ Failed: {status.get('failure_count', 0)}")
            
            if status.get("latest_run"):
                latest = status["latest_run"]
                print(f"   🕒 Latest: {latest['name']} - {latest.get('conclusion', 'unknown')}")
        
        # Health score
        health = self.calculate_health_score()
        health_emoji = "🟢" if health >= 80 else "🟡" if health >= 60 else "🔴"
        print(f"\n{health_emoji} HEALTH SCORE: {health}/100")
        
        # Failure analysis
        if "failure_analysis" in self.results and self.results["failure_analysis"]:
            print(f"\n❌ RECENT FAILURES ANALYZED: {len(self.results['failure_analysis'])}")
            
            for i, failure in enumerate(self.results["failure_analysis"][:3], 1):
                combined = failure.get("combined_analysis", {})
                root_cause = combined.get("root_cause", "unknown")
                print(f"   {i}. {failure['workflow_name']} - Root Cause: {root_cause}")
        
        # Recommendations
        if "recommendations" in self.results:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.results["recommendations"][:5]:
                print(f"   {rec}")
        
        print("\n" + "="*80)

    def save_results(self, output_file: str = "github_actions_analysis.json"):
        """Save analysis results to JSON file."""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"💾 Analysis results saved to: {output_path.resolve()}")


def main():
    """Main function to run GitHub Actions monitoring and analysis."""
    parser = argparse.ArgumentParser(description="Monitor and analyze GitHub Actions workflows")
    parser.add_argument("--repo", default=".", help="Repository path (default: current directory)")
    parser.add_argument("--token", help="GitHub token for API access (optional)")
    parser.add_argument("--check-logs", action="store_true", help="Analyze failure logs in detail")
    parser.add_argument("--save-results", help="Save results to JSON file")
    parser.add_argument("--max-failures", type=int, default=5, help="Maximum number of failures to analyze")
    
    args = parser.parse_args()
    
    try:
        # Initialize monitor
        monitor = GitHubActionsMonitor(args.repo, args.token)
        
        # Check current status
        monitor.check_current_status()
        
        # Analyze failures if requested
        if args.check_logs:
            monitor.analyze_recent_failures(args.max_failures)
        
        # Generate recommendations
        monitor.generate_recommendations()
        
        # Print summary
        monitor.print_summary()
        
        # Save results if requested
        if args.save_results:
            monitor.save_results(args.save_results)
        
        # Return appropriate exit code
        health_score = monitor.calculate_health_score()
        if health_score < 50:
            print(f"\n⚠️  Low health score ({health_score}/100) - immediate attention required!")
            sys.exit(1)
        elif health_score < 80:
            print(f"\n⚠️  Moderate health score ({health_score}/100) - monitor closely")
            sys.exit(0)
        else:
            print(f"\n✅ Good health score ({health_score}/100)")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Error running GitHub Actions monitor: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
