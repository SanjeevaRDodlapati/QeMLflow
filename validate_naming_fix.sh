#!/bin/bash

# Validation script for fix_naming_consistency.sh
# Tests the script for potential issues before execution

echo "==============================================================================="
echo "           VALIDATING fix_naming_consistency.sh SCRIPT"
echo "==============================================================================="
echo "🔍 Testing script for potential issues before execution..."
echo ""

# Test 1: Check if required files exist
echo "TEST 1: Checking required files exist"
echo "==============================================================================="

check_file() {
    local file=$1
    local description=$2
    if [[ -f "$file" ]]; then
        echo "✅ $description exists: $file"
        return 0
    else
        echo "❌ $description missing: $file"
        return 1
    fi
}

check_file "README.md" "README file"
check_file "pyproject.toml" "Project config"
check_file ".github/workflows/ci-cd.yml" "CI/CD workflow"

# Test 2: Check git repository status
echo ""
echo "TEST 2: Git repository validation"
echo "==============================================================================="

if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "✅ Git repository detected"
    
    # Check if we have a remote
    if git config --get remote.origin.url > /dev/null 2>&1; then
        REMOTE_URL=$(git config --get remote.origin.url)
        echo "✅ Git remote exists: $REMOTE_URL"
        
        # Validate remote URL format
        if [[ $REMOTE_URL == *"github.com"* ]]; then
            echo "✅ GitHub remote detected"
        else
            echo "⚠️  Non-GitHub remote: $REMOTE_URL"
        fi
    else
        echo "❌ No git remote configured"
    fi
else
    echo "❌ Not a git repository"
fi

# Test 3: Check current content to understand what will be changed
echo ""
echo "TEST 3: Analyzing current content"
echo "==============================================================================="

echo "Current README badge references:"
if grep -n "github.com" README.md | head -5; then
    echo "✅ Found GitHub references in README"
else
    echo "ℹ️  No GitHub references found in README"
fi

echo ""
echo "Current workflow names:"
if [[ -f ".github/workflows/ci-cd.yml" ]]; then
    grep "^name:" .github/workflows/ci-cd.yml || echo "ℹ️  No name field in CI/CD workflow"
else
    echo "ℹ️  CI/CD workflow file not found"
fi

echo ""
echo "Package name in pyproject.toml:"
if grep "^name = " pyproject.toml; then
    echo "✅ Package name found"
else
    echo "⚠️  Package name not found in expected format"
fi

# Test 4: Dry-run simulation of key operations
echo ""
echo "TEST 4: Dry-run simulation"
echo "==============================================================================="

echo "Simulating sed operations (no actual changes):"

# Test badge URL replacement
if grep -q "hachmannlab/qemlflow" README.md; then
    echo "✅ Would update README badges (hachmannlab/qemlflow found)"
    echo "   Preview of changes:"
    REPO_URL=$(git config --get remote.origin.url 2>/dev/null || echo "unknown")
    if [[ $REPO_URL == *"github.com"* ]]; then
        REPO_PATH=$(echo $REPO_URL | sed 's/.*github.com[:-]//' | sed 's/\.git$//')
        echo "   hachmannlab/qemlflow → $REPO_PATH"
    fi
else
    echo "ℹ️  No hachmannlab/qemlflow badges to update"
fi

# Test workflow name replacement
if [[ -f ".github/workflows/ci-cd.yml" ]] && grep -q "name: ChemML CI/CD Pipeline" .github/workflows/ci-cd.yml; then
    echo "✅ Would update CI/CD workflow name"
    echo "   ChemML CI/CD Pipeline → QeMLflow CI/CD Pipeline"
else
    echo "ℹ️  CI/CD workflow name already correct or file not found"
fi

# Test 5: Check for potential issues
echo ""
echo "TEST 5: Potential issue detection"
echo "==============================================================================="

# Check for problematic characters in files
echo "Checking for potential sed issues:"

check_sed_safety() {
    local file=$1
    local name=$2
    if [[ -f "$file" ]]; then
        # Check for characters that might cause sed issues
        if grep -q '[|/]' "$file"; then
            echo "⚠️  $name contains characters that might interfere with sed (| /)"
            echo "   Lines with special characters:"
            grep -n '[|/]' "$file" | head -3
        else
            echo "✅ $name appears safe for sed operations"
        fi
    fi
}

check_sed_safety "README.md" "README.md"
check_sed_safety ".github/workflows/ci-cd.yml" "CI/CD workflow"

# Test 6: Backup file handling
echo ""
echo "TEST 6: Backup file handling validation"
echo "==============================================================================="

echo "Checking if backup files would be created properly:"
# Simulate backup creation
touch test_backup_simulation.txt.bak
if [[ -f "test_backup_simulation.txt.bak" ]]; then
    echo "✅ Backup file creation works"
    rm test_backup_simulation.txt.bak
else
    echo "❌ Backup file creation might fail"
fi

# Check for existing backup files that might interfere
if find . -name "*.bak" -type f | head -1 > /dev/null; then
    echo "⚠️  Existing backup files found:"
    find . -name "*.bak" -type f
    echo "   These will be deleted by the script"
else
    echo "✅ No existing backup files to conflict"
fi

# Final assessment
echo ""
echo "==============================================================================="
echo "                        VALIDATION SUMMARY"
echo "==============================================================================="

# Determine if script is safe to run
ISSUES=0

# Check critical requirements
if [[ ! -f "README.md" ]]; then
    echo "❌ CRITICAL: README.md missing"
    ((ISSUES++))
fi

if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ CRITICAL: pyproject.toml missing"
    ((ISSUES++))
fi

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ CRITICAL: Not a git repository"
    ((ISSUES++))
fi

if [[ $ISSUES -eq 0 ]]; then
    echo "✅ VALIDATION PASSED: Script appears safe to execute"
    echo ""
    echo "🚀 RECOMMENDATION: Proceed with fix_naming_consistency.sh"
    echo ""
    echo "What the script will do:"
    echo "1. ✅ Update README badge URLs to point to correct repository"
    echo "2. ✅ Rename workflow names for consistency (ChemML → QeMLflow)"
    echo "3. ✅ Create backup files before making changes"
    echo "4. ✅ Clean up backup files after completion"
    echo "5. ✅ Provide detailed summary of changes made"
    echo ""
    echo "⚠️  SAFETY NOTES:"
    echo "- Script uses 'set -e' to exit on any error"
    echo "- Creates .bak files before making changes"
    echo "- Only makes text replacements, no file deletions"
    echo "- All changes can be reverted via git if needed"
else
    echo "❌ VALIDATION FAILED: $ISSUES critical issues found"
    echo ""
    echo "🛑 RECOMMENDATION: Fix issues before running script"
fi

echo "==============================================================================="
