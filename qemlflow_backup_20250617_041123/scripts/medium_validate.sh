#!/bin/bash
# ChemML Medium Validation Script (15 minutes)  
# Purpose: Comprehensive validation for regular development cycles

set -e  # Exit on any error

echo "🔍 ChemML Medium Validation Starting..."
echo "⏱️  Expected time: ~15 minutes"
echo "============================================"

# Start timer
start_time=$(date +%s)

# Step 1: Comprehensive linting (quiet mode)
echo "🔧 Running comprehensive linting..."
python tools/linting/comprehensive_linter.py --quiet || (echo "⚠️  Linting issues found (non-blocking)" && true)

# Step 2: Unit tests with failure limit
echo "🧪 Running unit tests (max 5 failures)..."
pytest tests/unit/ -x --maxfail=5 --tb=short || (echo "❌ Unit tests failed" && exit 1)
echo "✅ Unit tests passed"

# Step 3: Core functionality verification
echo "📦 Verifying core ChemML functionality..."
python -c "
import chemml
from chemml.core import ChemMLPipeline
from chemml.core.models import create_rf_model, create_linear_model
from chemml.integrations import ExternalModelManager

# Test pipeline creation
pipeline = ChemMLPipeline()
print('✅ ChemMLPipeline created successfully')

# Test model creation  
rf_model = create_rf_model()
linear_model = create_linear_model()
print('✅ Model creation successful')

# Test integration manager
manager = ExternalModelManager()
print('✅ Integration manager successful')

print('✅ Core functionality verified')
" || (echo "❌ Core functionality failed" && exit 1)

# Step 4: Integration example validation
echo "🎯 Testing integration examples..."
python examples/integrations/framework/comprehensive_enhanced_demo.py > /dev/null 2>&1 || (echo "⚠️  Integration demo issues (non-blocking)" && true)

# Step 5: Health and security check
echo "🛡️  Security and health assessment..."
python tools/linting/health_tracker.py --report | head -15

# Step 6: Test collection validation
echo "🔍 Validating test collection..."
test_count=$(pytest --collect-only --quiet 2>/dev/null | grep -E "test session starts" -A 10 | grep "collected" | awk '{print $1}' || echo "0")
if [ "$test_count" -gt 200 ]; then
    echo "✅ Test collection: $test_count tests found"
else
    echo "⚠️  Test collection: Only $test_count tests found"
fi

# Calculate and display execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "🎉 Medium Validation COMPLETED!"
echo "⏱️  Execution time: ${minutes}m ${seconds}s"
echo "============================================"
echo "✅ Development environment validated"
echo "💡 Run 'scripts/full_validate.sh' for complete validation"
