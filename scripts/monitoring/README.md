# QeMLflow Automated Monitoring System

This directory contains a comprehensive automated monitoring system for the QeMLflow project that automatically checks if workflows are successful and the documentation site is live and working.

## 🔍 Monitoring Components

### 1. Comprehensive Monitoring (`automated_monitor.py`)
The main monitoring system that provides:

- **GitHub Actions Workflow Status**: Checks recent workflow runs, success/failure rates
- **Documentation Site Health**: Tests site availability, response time, content validation
- **Release Status**: Monitors release deployment and version tags
- **Repository Health**: Tracks stars, forks, issues, commit activity
- **Issue Logging**: Records problems with severity levels and timestamps
- **JSON Reports**: Saves detailed results for historical tracking

**Usage:**
```bash
python scripts/monitoring/automated_monitor.py
```

**Exit Codes:**
- `0`: All systems operational
- `1`: Warnings detected but functional
- `2`: Critical issues detected

### 2. Status Dashboard (`status_dashboard.py`)
A user-friendly dashboard that provides:

- **Quick Status Checks**: Validates project structure and key files
- **Visual Status Reports**: Uses emojis and formatting for easy reading
- **Component Breakdown**: Shows status of workflows, docs, releases, repository
- **Action Recommendations**: Suggests next steps to resolve issues
- **Useful Links**: Quick access to GitHub Actions, documentation, etc.

**Usage:**
```bash
python scripts/monitoring/status_dashboard.py
```

### 3. Quick Status Checker (`../development/quick_status_check.py`)
A fast local checker that validates:

- **Workflow Files**: Ensures GitHub Actions workflows exist
- **Documentation Setup**: Checks MkDocs configuration and files
- **Git Status**: Validates repository state and remote configuration
- **Python Environment**: Confirms virtual environment and dependencies

**Usage:**
```bash
python scripts/development/quick_status_check.py
```

## 🤖 Automated Workflows

### 1. Scheduled Monitoring (`.github/workflows/monitoring.yml`)
- **Schedule**: Runs every 6 hours automatically
- **Manual Trigger**: Can be triggered via workflow_dispatch
- **GitHub Issues**: Automatically creates issues on critical failures
- **Artifacts**: Uploads monitoring results for review
- **Notifications**: Provides status updates in workflow logs

### 2. Simple Test Workflow (`.github/workflows/simple-test.yml`)
- **Basic Validation**: Tests fundamental GitHub Actions functionality
- **Environment Check**: Validates Python, checkout, and project structure
- **Debugging**: Helps identify basic workflow issues
- **Quick Feedback**: Provides fast validation of changes

## 📊 Monitoring Features

### Real-time Checks
- ✅ **GitHub Actions workflows** (success/failure rates, recent runs)
- ✅ **Documentation site deployment** (HTTP status, response time, content)
- ✅ **Release management** (latest releases, version tags)
- ✅ **Repository health** (activity, issues, commits)

### Automated Alerts
- 🚨 **Critical failure detection** (all workflows failing)
- ⚠️ **Warning identification** (slow response times, missing content)
- 📧 **GitHub issue creation** (automatic alerts on problems)
- 📈 **Historical tracking** (JSON logs for trend analysis)

### User-Friendly Reports
- 🎯 **Actionable recommendations** (specific steps to fix issues)
- 🔗 **Quick access links** (direct links to relevant GitHub pages)
- 📋 **Status summaries** (clear success/warning/error indicators)
- 🕐 **Timestamp tracking** (when issues were detected)

## 🛠️ Usage Examples

### Daily Monitoring
```bash
# Run comprehensive monitoring
python scripts/monitoring/automated_monitor.py

# View user-friendly dashboard
python scripts/monitoring/status_dashboard.py

# Quick local validation
python scripts/development/quick_status_check.py
```

### CI/CD Integration
```bash
# In deployment scripts
python scripts/monitoring/automated_monitor.py
if [ $? -eq 0 ]; then
    echo "All systems operational - proceeding with deployment"
else
    echo "Issues detected - halting deployment"
    exit 1
fi
```

### Debugging Workflow Issues
```bash
# Check if basic workflows work
# (triggers simple-test.yml workflow)
git push origin main

# Monitor results
python scripts/monitoring/automated_monitor.py

# Get detailed status
python scripts/monitoring/status_dashboard.py
```

## 📁 File Structure

```
scripts/monitoring/
├── automated_monitor.py      # Main monitoring system
├── status_dashboard.py       # User-friendly dashboard
└── README.md                # This documentation

scripts/development/
└── quick_status_check.py     # Fast local validation

.github/workflows/
├── monitoring.yml            # Automated monitoring workflow
├── simple-test.yml          # Basic test workflow
├── ci-cd.yml               # Main CI/CD pipeline
├── docs.yml                # Documentation deployment
└── release.yml             # Release automation

logs/
└── monitoring_*.json        # Historical monitoring results
```

## 🎯 Quick Start

1. **Run Initial Check:**
   ```bash
   python scripts/monitoring/status_dashboard.py
   ```

2. **Fix Any Critical Issues:**
   - Check GitHub Actions tab for workflow failures
   - Verify documentation builds locally: `mkdocs serve`
   - Ensure repository permissions for GitHub Pages

3. **Enable Automated Monitoring:**
   - The monitoring workflow runs automatically every 6 hours
   - Check `.github/workflows/monitoring.yml` for configuration
   - Monitor results in GitHub Actions artifacts

4. **Regular Maintenance:**
   ```bash
   # Weekly status check
   python scripts/monitoring/automated_monitor.py

   # Before major changes
   python scripts/development/quick_status_check.py
   ```

## 🔧 Troubleshooting

### Common Issues

1. **Workflows Failing:**
   - Check GitHub Actions tab for error details
   - Run simple test workflow to validate basic functionality
   - Review dependency installations in CI/CD workflows

2. **Documentation Site 404:**
   - Verify GitHub Pages is enabled in repository settings
   - Check if documentation workflow completed successfully
   - Test local build: `mkdocs build --clean`

3. **Monitoring Script Errors:**
   - Ensure internet connectivity for API calls
   - Check GitHub API rate limits
   - Verify repository name and owner in script configuration

### Debug Commands
```bash
# Test documentation build
mkdocs serve

# Validate workflow syntax
yamllint .github/workflows/*.yml

# Check Python environment
python scripts/development/quick_status_check.py

# Force workflow run
# (Push to main branch or use GitHub web interface)
```

## 🚀 Advanced Features

- **API Integration**: Connects to GitHub REST API for real-time data
- **Configurable Thresholds**: Customize warning/error conditions
- **Historical Analysis**: JSON logs enable trend monitoring
- **Multi-environment Support**: Can monitor different branches/environments
- **Extensible Architecture**: Easy to add new monitoring checks

## 📈 Metrics Tracked

- Workflow success/failure rates over time
- Documentation site response times and availability
- Repository activity and health indicators
- Issue detection and resolution timeframes
- System reliability and uptime statistics
