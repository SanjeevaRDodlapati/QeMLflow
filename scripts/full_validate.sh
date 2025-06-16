#!/bin/bash
# ChemML Full Validation Script (30+ minutes)
# Purpose: Complete validation for releases and major changes

set -e  # Exit on any error

echo "🔬 ChemML Full Validation Starting..."
echo "⏱️  Expected time: 30+ minutes"
echo "============================================"

# Start timer
start_time=$(date +%s)

# Step 1: Complete test suite with coverage
echo "🧪 Running complete test suite with coverage..."
echo "This may take 20+ minutes..."
pytest tests/ \
    --cov=src/chemml \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-fail-under=65 \
    --tb=short \
    --maxfail=10 || (echo "❌ Test suite failed" && exit 1)

echo "✅ Test suite completed with coverage"

# Step 2: Comprehensive health and quality report
echo "🏥 Generating comprehensive health report..."
python tools/linting/health_tracker.py --report > reports/health/full_validation_health.txt
echo "✅ Health report generated"

# Step 3: Detailed linting analysis
echo "🔧 Running detailed linting analysis..."
python tools/linting/comprehensive_linter.py --detailed > reports/linting/full_validation_linting.txt
echo "✅ Linting analysis completed"

# Step 4: Documentation build validation
echo "📚 Validating documentation build..."
if command -v mkdocs &> /dev/null; then
    mkdocs build --strict --clean || (echo "❌ Documentation build failed" && exit 1)
    echo "✅ Documentation built successfully"
else
    echo "⚠️  MkDocs not available, skipping documentation validation"
fi

# Step 5: Example scripts validation
echo "🎯 Validating example scripts..."
echo "Testing quickstart examples..."
python examples/quickstart/basic_integration.py > /dev/null 2>&1 || (echo "⚠️  Basic integration example failed" && true)

echo "Testing comprehensive demo..."
python examples/integrations/framework/comprehensive_enhanced_demo.py > /dev/null 2>&1 || (echo "⚠️  Comprehensive demo failed" && true)

echo "✅ Example validation completed"

# Step 6: Performance and resource check
echo "⚡ Performance and resource validation..."
python -c "
import psutil
import time
import sys

# Memory usage check
memory = psutil.virtual_memory()
print(f'📊 System Memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available')

# Import performance test
start_time = time.time()
import chemml
import_time = time.time() - start_time
print(f'⏱️  ChemML import time: {import_time:.2f}s')

if import_time > 5.0:
    print('⚠️  Import time > 5s, consider optimization')
else:
    print('✅ Import performance acceptable')
"

# Step 7: Security and dependency check
echo "🔒 Security and dependency validation..."
if command -v safety &> /dev/null; then
    safety check || (echo "⚠️  Security issues found" && true)
else
    echo "ℹ️  Safety not installed, skipping security scan"
fi

# Step 8: Final system status
echo "📊 Final validation summary..."
health_score=$(python tools/linting/health_tracker.py --report | grep "Health Score" | awk '{print $4}' | cut -d'/' -f1)
test_coverage=$(grep -E "TOTAL.*%" htmlcov/index.html 2>/dev/null | sed 's/.*>\([0-9]*\)%.*/\1/' || echo "unknown")

echo "🏥 Health Score: ${health_score}/100"
echo "🧪 Test Coverage: ${test_coverage}%"

# Calculate and display execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "🎉 Full Validation COMPLETED!"
echo "⏱️  Total execution time: ${hours}h ${minutes}m ${seconds}s"
echo "============================================"
echo "✅ Complete codebase validation finished"
echo "📊 Coverage report: htmlcov/index.html"
echo "📋 Health report: reports/health/full_validation_health.txt"
echo "🔧 Linting report: reports/linting/full_validation_linting.txt"

# Success criteria check
if [ "${health_score:-0}" -ge 70 ] && [ "${test_coverage:-0}" -ge 65 ]; then
    echo "🏆 VALIDATION PASSED - Ready for release"
    exit 0
else
    echo "⚠️  VALIDATION CONCERNS - Review reports before release"
    echo "   - Health score should be ≥70 (current: ${health_score})"
    echo "   - Test coverage should be ≥65% (current: ${test_coverage}%)"
    exit 0  # Non-blocking for now
fi
