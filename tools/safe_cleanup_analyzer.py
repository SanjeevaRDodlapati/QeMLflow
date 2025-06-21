#!/usr/bin/env python3
"""
Safe QeMLflow Codebase Cleanup - Manual Review Version
=====================================================

This script identifies cleanup candidates but requires manual review
before any deletions. Follows safety-first principles.
"""

import os
from pathlib import Path
from typing import List, Dict
import json


class SafeCleanupAnalyzer:
    """Analyze codebase for cleanup opportunities without automatic deletion."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.analysis = {
            "temporary_files": [],
            "empty_directories": [],
            "duplicate_docs": [],
            "cache_artifacts": [],
            "safe_to_remove": [],
            "needs_review": []
        }
    
    def analyze_cleanup_opportunities(self):
        """Analyze what could be cleaned up safely."""
        print("🔍 Analyzing codebase for cleanup opportunities...")
        print("=" * 60)
        
        self._analyze_temporary_files()
        self._analyze_empty_directories()
        self._analyze_documentation()
        self._analyze_cache_artifacts()
        
        self._generate_analysis_report()
        
    def _analyze_temporary_files(self):
        """Identify temporary files safely."""
        print("📄 Analyzing temporary files...")
        
        # Only look for clearly temporary files
        safe_temp_patterns = [
            "temp_*.py",
            "*.tmp",
            ".DS_Store",
            "deployment_status_report.json"  # Our monitoring script generates this
        ]
        
        for pattern in safe_temp_patterns:
            for file in self.repo_path.glob(pattern):
                if file.is_file():
                    self.analysis["temporary_files"].append({
                        "path": str(file),
                        "size": file.stat().st_size,
                        "reason": f"Matches pattern: {pattern}"
                    })
    
    def _analyze_empty_directories(self):
        """Identify truly empty directories."""
        print("📁 Analyzing directories...")
        
        # Only check clearly temporary directories
        check_dirs = ["cache", "logs", "metrics_data"]
        
        for dir_name in check_dirs:
            dir_path = self.repo_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                files = list(dir_path.rglob("*"))
                files = [f for f in files if f.is_file()]  # Only count files
                
                if not files:
                    self.analysis["empty_directories"].append({
                        "path": str(dir_path),
                        "reason": "Empty directory"
                    })
                else:
                    self.analysis["needs_review"].append({
                        "path": str(dir_path),
                        "file_count": len(files),
                        "reason": "Directory contains files - manual review needed"
                    })
    
    def _analyze_documentation(self):
        """Identify potentially redundant documentation."""
        print("📚 Analyzing documentation...")
        
        # Look for clearly redundant docs
        redundant_patterns = [
            "CLEANUP_*.md",  
            "PHASE_*_COMPLETION_*.md",
            "*_DIAGNOSTIC_*.md"
        ]
        
        for pattern in redundant_patterns:
            for file in self.repo_path.glob(pattern):
                if file.is_file():
                    self.analysis["duplicate_docs"].append({
                        "path": str(file),
                        "size": file.stat().st_size,
                        "reason": f"Potentially redundant: {pattern}"
                    })
    
    def _analyze_cache_artifacts(self):
        """Identify cache and build artifacts."""
        print("🗂️  Analyzing cache artifacts...")
        
        cache_dirs = [".mypy_cache", ".pytest_cache", "__pycache__"]
        
        for dir_name in cache_dirs:
            for dir_path in self.repo_path.rglob(dir_name):
                if dir_path.is_dir():
                    file_count = len(list(dir_path.rglob("*")))
                    self.analysis["cache_artifacts"].append({
                        "path": str(dir_path),
                        "file_count": file_count,
                        "reason": "Cache/build artifact directory"
                    })
    
    def _generate_analysis_report(self):
        """Generate detailed analysis report."""
        report_path = self.repo_path / "CLEANUP_ANALYSIS.json"
        
        # Add summary
        self.analysis["summary"] = {
            "total_temporary_files": len(self.analysis["temporary_files"]),
            "total_empty_directories": len(self.analysis["empty_directories"]),
            "total_duplicate_docs": len(self.analysis["duplicate_docs"]),
            "total_cache_artifacts": len(self.analysis["cache_artifacts"]),
            "items_needing_review": len(self.analysis["needs_review"])
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis, f, indent=2)
        
        print(f"\n📊 CLEANUP ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Temporary files found: {self.analysis['summary']['total_temporary_files']}")
        print(f"Empty directories: {self.analysis['summary']['total_empty_directories']}")
        print(f"Duplicate docs: {self.analysis['summary']['total_duplicate_docs']}")
        print(f"Cache artifacts: {self.analysis['summary']['total_cache_artifacts']}")
        print(f"Items needing review: {self.analysis['summary']['items_needing_review']}")
        
        print(f"\n📄 Detailed analysis saved to: {report_path}")
        print("\n⚠️  IMPORTANT: Review the analysis before making any changes!")


def main():
    """Run safe cleanup analysis."""
    analyzer = SafeCleanupAnalyzer()
    
    try:
        analyzer.analyze_cleanup_opportunities()
        print("\n✅ Safe analysis completed!")
        print("💡 Review CLEANUP_ANALYSIS.json before proceeding with any cleanup")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
