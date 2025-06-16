#!/bin/bash
# ChemML Quick Validation Script (5 minutes)
# Purpose: Fast validation for immediate feedback after changes

set -e  # Exit on any error

echo "🚀 ChemML Quick Validation Starting..."
echo "⏱️  Expected time: ~5 minutes"
echo "============================================"

# Start timer
start_time=$(date +%s)

# Step 1: Core imports test
echo "📦 Testing core imports..."
python -c "
import chemml
from chemml.core import ChemMLPipeline
from chemml.core.models import Model, BaseModel
print('✅ Core imports successful')
" || (echo "❌ Core imports failed" && exit 1)

# Step 2: Basic integration test
echo "🔗 Testing basic integration..."
timeout 120 python examples/quickstart/basic_integration.py > /dev/null 2>&1 || (echo "❌ Basic integration failed" && exit 1)
echo "✅ Basic integration successful"

# Step 3: Quick comprehensive tests (max 3 failures)
echo "🧪 Running comprehensive tests (max 3 failures)..."
pytest tests/comprehensive/ -x --tb=short --maxfail=3 --quiet || (echo "❌ Comprehensive tests failed" && exit 1)
echo "✅ Comprehensive tests passed"

# Step 4: Quick health check
echo "🏥 Quick health assessment..."
python tools/linting/health_tracker.py --report | grep -E "(Health Score|Total Issues)" || true

# Calculate and display execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "🎉 Quick Validation COMPLETED!"
echo "⏱️  Execution time: ${minutes}m ${seconds}s"
echo "============================================"
echo "✅ All critical systems functional"
echo "💡 Run 'scripts/medium_validate.sh' for deeper validation"
