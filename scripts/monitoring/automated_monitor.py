"""
ChemML Automated Monitoring System
==================================

Comprehensive monitoring system that automatically checks:
1. GitHub Actions workflow status
2. Documentation site availability and health
3. Release status and deployment
4. Repository health metrics

This script can be run manually or scheduled as a cron job.
"""

import json
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class ChemMLMonitor:
    """Automated monitoring system for ChemML project."""

    def __init__(
        self, repo_owner: str = "SanjeevaRDodlapati", repo_name: str = "ChemML"
    ):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_api_base = "https://api.github.com"
        self.docs_url = f"https://{repo_owner.lower()}.github.io/{repo_name}/"
        self.repo_url = f"https://github.com/{repo_owner}/{repo_name}"

        # Status tracking
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "workflows": {},
            "documentation": {},
            "releases": {},
            "repository": {},
            "issues": [],
        }

    def log_issue(self, severity: str, component: str, message: str):
        """Log an issue found during monitoring."""
        self.results["issues"].append(
            {
                "severity": severity,
                "component": component,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def check_github_api_rate_limit(self) -> bool:
        """Check if we can make GitHub API calls."""
        try:
            response = requests.get(f"{self.github_api_base}/rate_limit", timeout=10)
            if response.status_code == 200:
                data = response.json()
                remaining = data["rate"]["remaining"]
                if remaining < 10:
                    self.log_issue(
                        "warning",
                        "github_api",
                        f"Low API rate limit: {remaining} requests remaining",
                    )
                    return False
                return True
        except Exception as e:
            self.log_issue("error", "github_api", f"Cannot check rate limit: {e}")
            return False
        return True

    def check_workflows(self) -> Dict:
        """Check GitHub Actions workflow status."""
        workflow_status = {
            "overall": "unknown",
            "workflows": [],
            "recent_runs": [],
            "last_updated": datetime.now().isoformat(),
        }

        if not self.check_github_api_rate_limit():
            workflow_status["overall"] = "error"
            return workflow_status

        try:
            # Get workflow runs
            url = f"{self.github_api_base}/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
            params = {"per_page": 20, "status": "completed"}

            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                self.log_issue(
                    "error",
                    "workflows",
                    f"Cannot fetch workflows: HTTP {response.status_code}",
                )
                workflow_status["overall"] = "error"
                return workflow_status

            data = response.json()
            runs = data.get("workflow_runs", [])

            if not runs:
                self.log_issue("warning", "workflows", "No workflow runs found")
                workflow_status["overall"] = "warning"
                return workflow_status

            # Analyze recent runs
            success_count = 0
            failure_count = 0
            recent_runs = []

            for run in runs[:10]:  # Check last 10 runs
                run_info = {
                    "name": run.get("name", "Unknown"),
                    "status": run.get("conclusion", run.get("status", "unknown")),
                    "created_at": run.get("created_at", ""),
                    "html_url": run.get("html_url", ""),
                    "head_branch": run.get("head_branch", ""),
                    "event": run.get("event", ""),
                }
                recent_runs.append(run_info)

                if run.get("conclusion") == "success":
                    success_count += 1
                elif run.get("conclusion") in ["failure", "cancelled", "timed_out"]:
                    failure_count += 1

            workflow_status["recent_runs"] = recent_runs

            # Determine overall status
            if failure_count == 0 and success_count > 0:
                workflow_status["overall"] = "success"
            elif failure_count > success_count:
                workflow_status["overall"] = "failure"
                self.log_issue(
                    "error",
                    "workflows",
                    f"{failure_count} failed runs out of last {len(recent_runs)}",
                )
            else:
                workflow_status["overall"] = "warning"
                self.log_issue("warning", "workflows", "Mixed workflow results")

            # Get specific workflows
            workflows_url = f"{self.github_api_base}/repos/{self.repo_owner}/{self.repo_name}/actions/workflows"
            workflows_response = requests.get(workflows_url, timeout=10)

            if workflows_response.status_code == 200:
                workflows_data = workflows_response.json()
                for workflow in workflows_data.get("workflows", []):
                    workflow_status["workflows"].append(
                        {
                            "name": workflow.get("name", ""),
                            "state": workflow.get("state", ""),
                            "path": workflow.get("path", ""),
                            "html_url": workflow.get("html_url", ""),
                        }
                    )

        except Exception as e:
            self.log_issue("error", "workflows", f"Error checking workflows: {e}")
            workflow_status["overall"] = "error"

        return workflow_status

    def check_documentation_site(self) -> Dict:
        """Check documentation site availability and health."""
        doc_status = {
            "overall": "unknown",
            "url": self.docs_url,
            "response_time": None,
            "status_code": None,
            "content_checks": {},
            "last_updated": datetime.now().isoformat(),
        }

        try:
            start_time = time.time()
            response = requests.get(self.docs_url, timeout=15)
            response_time = time.time() - start_time

            doc_status["response_time"] = round(response_time, 2)
            doc_status["status_code"] = response.status_code

            if response.status_code == 200:
                content = response.text.lower()

                # Check for expected content
                content_checks = {
                    "title_present": "chemml" in content,
                    "navigation_present": "getting started" in content
                    or "quick start" in content,
                    "content_loaded": len(content) > 1000,
                    "no_404_error": "404" not in content and "not found" not in content,
                    "material_theme": "material" in content or "mkdocs" in content,
                }

                doc_status["content_checks"] = content_checks

                # Determine overall status
                if all(content_checks.values()):
                    doc_status["overall"] = "success"
                elif (
                    content_checks["content_loaded"] and content_checks["no_404_error"]
                ):
                    doc_status["overall"] = "warning"
                    self.log_issue(
                        "warning", "documentation", "Some content checks failed"
                    )
                else:
                    doc_status["overall"] = "failure"
                    self.log_issue(
                        "error", "documentation", "Major content issues detected"
                    )

                if response_time > 5.0:
                    self.log_issue(
                        "warning",
                        "documentation",
                        f"Slow response time: {response_time:.2f}s",
                    )

            elif response.status_code == 404:
                doc_status["overall"] = "failure"
                self.log_issue(
                    "error",
                    "documentation",
                    "Documentation site returns 404 - not deployed",
                )

            else:
                doc_status["overall"] = "failure"
                self.log_issue(
                    "error",
                    "documentation",
                    f"Documentation site returns HTTP {response.status_code}",
                )

        except requests.exceptions.Timeout:
            doc_status["overall"] = "failure"
            self.log_issue("error", "documentation", "Documentation site timeout")

        except requests.exceptions.ConnectionError:
            doc_status["overall"] = "failure"
            self.log_issue(
                "error", "documentation", "Cannot connect to documentation site"
            )

        except Exception as e:
            doc_status["overall"] = "failure"
            self.log_issue(
                "error", "documentation", f"Error checking documentation: {e}"
            )

        return doc_status

    def check_releases(self) -> Dict:
        """Check release status and latest release info."""
        release_status = {
            "overall": "unknown",
            "latest_release": None,
            "release_count": 0,
            "tags": [],
            "last_updated": datetime.now().isoformat(),
        }

        try:
            # Check releases via API
            url = f"{self.github_api_base}/repos/{self.repo_owner}/{self.repo_name}/releases"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                releases = response.json()
                release_status["release_count"] = len(releases)

                if releases:
                    latest = releases[0]
                    release_status["latest_release"] = {
                        "tag_name": latest.get("tag_name", ""),
                        "name": latest.get("name", ""),
                        "published_at": latest.get("published_at", ""),
                        "prerelease": latest.get("prerelease", False),
                        "draft": latest.get("draft", False),
                        "html_url": latest.get("html_url", ""),
                    }
                    release_status["overall"] = "success"
                else:
                    release_status["overall"] = "warning"
                    self.log_issue("warning", "releases", "No releases found")

            # Check tags
            tags_url = (
                f"{self.github_api_base}/repos/{self.repo_owner}/{self.repo_name}/tags"
            )
            tags_response = requests.get(tags_url, timeout=10)

            if tags_response.status_code == 200:
                tags = tags_response.json()
                release_status["tags"] = [tag.get("name", "") for tag in tags[:10]]

        except Exception as e:
            self.log_issue("error", "releases", f"Error checking releases: {e}")
            release_status["overall"] = "error"

        return release_status

    def check_repository_health(self) -> Dict:
        """Check overall repository health metrics."""
        repo_status = {
            "overall": "unknown",
            "info": {},
            "last_commit": {},
            "issues_count": 0,
            "last_updated": datetime.now().isoformat(),
        }

        try:
            # Get repository info
            url = f"{self.github_api_base}/repos/{self.repo_owner}/{self.repo_name}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                repo_data = response.json()
                repo_status["info"] = {
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "open_issues": repo_data.get("open_issues_count", 0),
                    "size": repo_data.get("size", 0),
                    "language": repo_data.get("language", ""),
                    "updated_at": repo_data.get("updated_at", ""),
                    "pushed_at": repo_data.get("pushed_at", ""),
                }
                repo_status["issues_count"] = repo_data.get("open_issues_count", 0)

                # Check last commit
                commits_url = f"{self.github_api_base}/repos/{self.repo_owner}/{self.repo_name}/commits"
                commits_response = requests.get(
                    commits_url, params={"per_page": 1}, timeout=10
                )

                if commits_response.status_code == 200:
                    commits = commits_response.json()
                    if commits:
                        last_commit = commits[0]
                        repo_status["last_commit"] = {
                            "sha": last_commit.get("sha", "")[:8],
                            "message": last_commit.get("commit", {}).get("message", "")[
                                :100
                            ],
                            "author": last_commit.get("commit", {})
                            .get("author", {})
                            .get("name", ""),
                            "date": last_commit.get("commit", {})
                            .get("author", {})
                            .get("date", ""),
                        }

                repo_status["overall"] = "success"

        except Exception as e:
            self.log_issue("error", "repository", f"Error checking repository: {e}")
            repo_status["overall"] = "error"

        return repo_status

    def run_comprehensive_check(self) -> Dict:
        """Run all monitoring checks and return comprehensive results."""
        print("🔍 Starting ChemML Comprehensive Monitoring...")
        print("=" * 60)

        # Run all checks
        print("📊 Checking GitHub Actions workflows...")
        self.results["workflows"] = self.check_workflows()

        print("📚 Checking documentation site...")
        self.results["documentation"] = self.check_documentation_site()

        print("🏷️ Checking releases...")
        self.results["releases"] = self.check_releases()

        print("🔧 Checking repository health...")
        self.results["repository"] = self.check_repository_health()

        # Determine overall status
        component_statuses = [
            self.results["workflows"]["overall"],
            self.results["documentation"]["overall"],
            self.results["releases"]["overall"],
            self.results["repository"]["overall"],
        ]

        if all(status == "success" for status in component_statuses):
            self.results["overall_status"] = "success"
        elif any(status == "error" for status in component_statuses):
            self.results["overall_status"] = "error"
        else:
            self.results["overall_status"] = "warning"

        return self.results

    def print_results(self):
        """Print monitoring results in a human-readable format."""
        print("\n📋 ChemML Monitoring Results")
        print("=" * 60)
        print(f"Timestamp: {self.results['timestamp']}")
        print(
            f"Overall Status: {self.get_status_emoji(self.results['overall_status'])} {self.results['overall_status'].upper()}"
        )

        # Workflows
        print(
            f"\n⚙️ GitHub Actions Workflows: {self.get_status_emoji(self.results['workflows']['overall'])}"
        )
        if self.results["workflows"]["recent_runs"]:
            print("   Recent runs:")
            for run in self.results["workflows"]["recent_runs"][:3]:
                status_emoji = self.get_status_emoji(run["status"])
                print(f"   • {status_emoji} {run['name']} - {run['status']}")

        # Documentation
        print(
            f"\n📚 Documentation Site: {self.get_status_emoji(self.results['documentation']['overall'])}"
        )
        print(f"   URL: {self.results['documentation']['url']}")
        if self.results["documentation"]["status_code"]:
            print(f"   Status: HTTP {self.results['documentation']['status_code']}")
        if self.results["documentation"]["response_time"]:
            print(
                f"   Response time: {self.results['documentation']['response_time']}s"
            )

        # Releases
        print(
            f"\n🏷️ Releases: {self.get_status_emoji(self.results['releases']['overall'])}"
        )
        print(f"   Total releases: {self.results['releases']['release_count']}")
        if self.results["releases"]["latest_release"]:
            latest = self.results["releases"]["latest_release"]
            print(f"   Latest: {latest['tag_name']} - {latest['name']}")

        # Repository
        print(
            f"\n🔧 Repository Health: {self.get_status_emoji(self.results['repository']['overall'])}"
        )
        if self.results["repository"]["info"]:
            info = self.results["repository"]["info"]
            print(
                f"   Stars: {info['stars']} | Forks: {info['forks']} | Open Issues: {info['open_issues']}"
            )

        # Issues
        if self.results["issues"]:
            print(f"\n⚠️ Issues Found ({len(self.results['issues'])}):")
            for issue in self.results["issues"][-5:]:  # Show last 5 issues
                severity_emoji = "🔴" if issue["severity"] == "error" else "🟡"
                print(f"   {severity_emoji} [{issue['component']}] {issue['message']}")

        print("\n🌐 Quick Access Links:")
        print(f"   • GitHub Actions: {self.repo_url}/actions")
        print(f"   • Documentation: {self.docs_url}")
        print(f"   • Releases: {self.repo_url}/releases")

    def get_status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        emoji_map = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "failure": "❌",
            "unknown": "❓",
        }
        return emoji_map.get(status.lower(), "❓")

    def save_results(self, filepath: str = "monitoring_results.json"):
        """Save results to JSON file."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"\n💾 Results saved to: {filepath}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")


def main():
    """Main monitoring function."""
    monitor = ChemMLMonitor()

    try:
        # Run comprehensive check
        results = monitor.run_comprehensive_check()

        # Print results
        monitor.print_results()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitor.save_results(f"logs/monitoring_{timestamp}.json")

        # Exit with appropriate code
        if results["overall_status"] == "success":
            print("\n🎉 All systems operational!")
            sys.exit(0)
        elif results["overall_status"] == "warning":
            print("\n⚠️ Some issues detected, but system functional")
            sys.exit(1)
        else:
            print("\n🚨 Critical issues detected!")
            sys.exit(2)

    except KeyboardInterrupt:
        print("\n\n⏹️ Monitoring interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
