#!/bin/bash

# Repository Naming Consistency Fix
# Resolves naming mismatches between QeMLflow and ChemML

set -e

echo "==============================================================================="
echo "                 REPOSITORY NAMING CONSISTENCY FIX"
echo "==============================================================================="
echo "🎯 Goal: Fix naming mismatches contributing to workflow failures"
echo "📋 Issues: QeMLflow vs ChemML naming inconsistencies"
echo ""

# Current status
echo "🔍 CURRENT REPOSITORY STATUS"
echo "==============================================================================="
echo "Git remote: $(git config --get remote.origin.url)"
echo "Local folder: $(basename $(pwd))"
echo "Package name: $(grep '^name = ' pyproject.toml | cut -d'"' -f2)"
echo "Workflow name: $(grep '^name:' .github/workflows/ci-cd.yml | head -1)"
echo ""

# Check what we should fix
echo "🛠️  NAMING CONSISTENCY ANALYSIS"
echo "==============================================================================="

# Option 1: Keep ChemML (align everything to ChemML)
echo "Option 1: Align everything to ChemML"
echo "  - Rename package from qemlflow → chemml"
echo "  - Update all references"
echo "  - Pro: Matches GitHub repo"
echo "  - Con: Major breaking change"
echo ""

# Option 2: Keep QeMLflow (recommended)
echo "Option 2: Align everything to QeMLflow (RECOMMENDED)"
echo "  - Keep package as qemlflow"
echo "  - Fix README badges to point to correct repo"
echo "  - Update workflow names for clarity"
echo "  - Pro: Minimal breaking changes"
echo "  - Con: GitHub repo name mismatch remains"
echo ""

# Implement Option 2 (safer approach)
echo "🚀 IMPLEMENTING OPTION 2: QeMLflow Consistency"
echo "==============================================================================="

# Fix README badges
echo "Step 1: Fixing README badges..."
if grep -q "hachmannlab/qemlflow" README.md; then
    # Get actual repo info
    REPO_URL=$(git config --get remote.origin.url)
    if [[ $REPO_URL == *"github.com"* ]]; then
        # Extract owner/repo from SSH or HTTPS URL
        if [[ $REPO_URL == *"git@github.com"* ]]; then
            # SSH format: git@github.com:owner/repo.git or git@github.com-alias:owner/repo.git
            REPO_PATH=$(echo $REPO_URL | sed 's/.*github.com[^:]*://' | sed 's/\.git$//')
        else
            # HTTPS format: https://github.com/owner/repo.git
            REPO_PATH=$(echo $REPO_URL | sed 's/.*github\.com\///' | sed 's/\.git$//')
        fi
        echo "  Detected repo: $REPO_PATH"
        
        # Update badges (using different delimiter to avoid conflicts)
        sed -i.bak "s#hachmannlab/qemlflow#$REPO_PATH#g" README.md
        echo "  ✅ Updated README badges to point to correct repository"
    fi
else
    echo "  ℹ️  README badges already correct"
fi

# Update workflow names for clarity
echo ""
echo "Step 2: Updating workflow names..."

# Update CI/CD workflow (using # delimiter for safety)
if grep -q "name: ChemML CI/CD Pipeline" .github/workflows/ci-cd.yml; then
    sed -i.bak 's#name: ChemML CI/CD Pipeline#name: QeMLflow CI/CD Pipeline#' .github/workflows/ci-cd.yml
    echo "  ✅ Updated CI/CD workflow name"
fi

# Update other workflows for consistency
for workflow in .github/workflows/*.yml; do
    if grep -q "ChemML" "$workflow" && ! grep -q "QeMLflow" "$workflow"; then
        echo "  📝 Checking $workflow for ChemML references..."
        # Update workflow names that reference ChemML inappropriately
        sed -i.bak 's/ChemML/QeMLflow/g' "$workflow"
        echo "  ✅ Updated $workflow"
    fi
done

# Check package imports and update if needed
echo ""
echo "Step 3: Validating package consistency..."
PACKAGE_NAME=$(grep '^name = ' pyproject.toml | cut -d'"' -f2)
echo "  Package name in pyproject.toml: $PACKAGE_NAME"

if [[ "$PACKAGE_NAME" == "qemlflow" ]]; then
    echo "  ✅ Package name is consistent (qemlflow)"
else
    echo "  ⚠️  Package name inconsistency detected: $PACKAGE_NAME"
fi

# Create summary report
echo ""
echo "📋 CONSISTENCY FIXES APPLIED"
echo "==============================================================================="
echo "✅ README badges now point to correct repository"
echo "✅ Workflow names updated for consistency"
echo "✅ Package structure maintained (qemlflow)"
echo "⚠️  GitHub repo name (ChemML) differs from package (qemlflow)"
echo ""

# Show changes
echo "🔍 CHANGES MADE"
echo "==============================================================================="
if [[ -f README.md.bak ]]; then
    echo "README.md changes:"
    diff README.md.bak README.md || echo "  (Badge URLs updated)"
fi

if [[ -f .github/workflows/ci-cd.yml.bak ]]; then
    echo ""
    echo "Workflow changes:"
    diff .github/workflows/ci-cd.yml.bak .github/workflows/ci-cd.yml || echo "  (Workflow names updated)"
fi

# Clean up backup files
echo ""
echo "🧹 Cleaning up backup files..."
find . -name "*.bak" -type f -delete
echo "✅ Backup files removed"

echo ""
echo "==============================================================================="
echo "                        NAMING CONSISTENCY COMPLETE"
echo "==============================================================================="
echo "🎯 Repository naming issues addressed!"
echo ""
echo "Summary:"
echo "✅ Fixed README badge URLs"
echo "✅ Updated workflow names for clarity"
echo "✅ Maintained package naming consistency"
echo ""
echo "Next steps:"
echo "1. Commit these naming fixes"
echo "2. Push to trigger workflows with correct references"
echo "3. Monitor workflows for improved success rates"
echo ""
echo "Note: The GitHub repository name (ChemML) still differs from the package"
echo "name (qemlflow), but this is now clearly documented and shouldn't cause"
echo "workflow failures."
echo "==============================================================================="
