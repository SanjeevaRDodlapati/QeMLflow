# 🧹 QeMLflow Codebase Cleanup Plan

## 🎯 **CLEANUP OBJECTIVES:**
1. Remove redundant/temporary files created during CI/CD fixes
2. Organize files into appropriate directories
3. Merge related configuration files
4. Streamline project structure
5. Maintain clean, professional codebase

## 📁 **FILES TO DELETE (Redundant/Temporary):**

### **Status Reports & Analysis Files:**
- ❌ `CICD_IMPROVEMENT_REPORT.md`
- ❌ `COMMIT_MESSAGE.txt`
- ❌ `COMPREHENSIVE_CI_CD_DIAGNOSTIC_REPORT.md`
- ❌ `CURRENT_STATUS_REPORT.md`
- ❌ `DEPLOYMENT_STATUS.txt`
- ❌ `FINAL_DEPLOYMENT_INSTRUCTIONS.md`
- ❌ `FINAL_STATUS_REPORT.md`
- ❌ `MANUAL_DEPLOYMENT_REQUIRED.md`
- ❌ `PRIORITY_EXECUTION_GUIDE.txt`
- ❌ `STATUS_REPORT.md`
- ❌ `WORKFLOW_FAILURE_RESOLUTION.md`
- ❌ `WORKFLOW_RESOLUTION_COMPLETE.md`
- ❌ `WORKFLOW_VALIDATION.md`

### **JSON Status/Analysis Files:**
- ❌ `current_detailed_status.json`
- ❌ `current_failures.json`
- ❌ `current_status_after_fixes.json`
- ❌ `current_test_fix_status.json`
- ❌ `detailed_workflow_failures.json`
- ❌ `final_config_fix_status.json`
- ❌ `final_status_check.json`
- ❌ `latest_detailed_status.json`
- ❌ `latest_failure_analysis.json`
- ❌ `latest_failure_details.json`
- ❌ `latest_run_status.json`
- ❌ `latest_status_check.json`
- ❌ `maintenance_report.json`
- ❌ `test_fix_status.json`
- ❌ `workflow_analysis.json`
- ❌ `workflow_analysis_results.json`
- ❌ `workflow_status.json`

### **Temporary Python Scripts:**
- ❌ `automated_maintenance.py`
- ❌ `commit_workflow_fixes.py`
- ❌ `comprehensive_validation.py`
- ❌ `continuous_monitor.py`
- ❌ `critical_import_test.py`
- ❌ `debug_status_check.py`
- ❌ `emergency_syntax_fix.py`
- ❌ `emergency_workflow_fix.py`
- ❌ `fix_import_syntax.py`
- ❌ `github_actions_monitor.py`
- ❌ `quick_syntax_check.py`
- ❌ `quick_test.py`
- ❌ `safe_git_commit.py`
- ❌ `safe_validation.py`
- ❌ `simple_workflow_checker.py`
- ❌ `terminal_diagnostic.py`
- ❌ `test_critical_error_strategy.py`
- ❌ `validate_commit_script.py`
- ❌ `workflow_progress_tracker.py`

### **Temporary Shell Scripts:**
- ❌ `demonstrate_workflow_fix.sh`
- ❌ `deploy_fixes.sh`
- ❌ `deploy_github_actions_fix.sh`
- ❌ `emergency_commit_fix.sh`
- ❌ `fix_naming_consistency.sh`
- ❌ `safe_commit.sh`
- ❌ `validate_naming_fix.sh`

### **Redundant Directories:**
- ❌ `qemlflow_backup_20250617_041123/` (old backup)
- ❌ `qemlflow_env/` (virtual environment - should be in .gitignore)
- ❌ `cache/` (temporary cache)
- ❌ `plots/` (if empty or redundant)
- ❌ `logs/` (if just temporary logs)

## 📂 **FILES TO REORGANIZE:**

### **Move to `scripts/` directory:**
- 🔄 Keep any useful scripts in `scripts/`
- 🔄 Organize by purpose (setup/, monitoring/, utilities/)

### **Configuration Consolidation:**
- 🔄 Keep `.flake8`, `mypy.ini`, `pytest.ini` (needed for tools)
- 🔄 Evaluate if `.pre-commit-config.yaml` is actively used
- 🔄 Consolidate into `pyproject.toml` where possible

### **Documentation Organization:**
- 🔄 Move all user-facing docs to `docs/`
- 🔄 Keep only `README.md` and `CRITICAL_FILES.md` in root
- 🔄 Archive old documentation in `docs/archive/`

## 🎯 **FINAL STRUCTURE TARGET:**

```
QeMLflow/
├── README.md                 # Main project readme
├── CRITICAL_FILES.md         # Keep for reference
├── pyproject.toml           # Main config
├── requirements*.txt        # Dependencies
├── Makefile                 # Build automation
├── Dockerfile              # Container config
├── docker-compose.yml      # Multi-container
├── mkdocs.yml              # Documentation
├── .gitignore              # Git ignore rules
├── .github/                # GitHub Actions (clean)
├── src/qemlflow/           # Source code
├── tests/                  # Test suite
├── docs/                   # Documentation
├── examples/               # Example code
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts (organized)
├── tools/                  # Development tools (organized)
├── data/                   # Data directories
├── reports/                # Analysis reports (organized)
└── backups/                # Clean backups only
```

## ✅ **CLEANUP EXECUTION PHASES:**

### **Phase 1: Remove Temporary Files**
- Delete all temporary status/analysis files
- Remove temporary Python/shell scripts
- Clean up redundant directories

### **Phase 2: Reorganize Structure**
- Move scripts to appropriate locations
- Consolidate configuration files
- Organize documentation

### **Phase 3: Final Validation**
- Ensure all workflows still pass
- Verify no important files were deleted
- Update .gitignore for future cleanliness

## 🛡️ **SAFETY MEASURES:**
1. Create backup before major cleanup
2. Test workflows after each cleanup phase
3. Keep CLEANUP_PLAN.md for reference during cleanup
4. Validate that all critical functionality remains intact
