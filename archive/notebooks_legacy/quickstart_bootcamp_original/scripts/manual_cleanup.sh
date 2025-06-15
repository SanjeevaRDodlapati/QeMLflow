#!/bin/bash
# ChemML Bootcamp Manual Cleanup Script
# Remove redundant files to achieve clean integration

echo "🧹 Starting ChemML Bootcamp Manual Cleanup..."

# Navigate to bootcamp directory
cd /Users/sanjeevadodlapati/Downloads/Repos/ChemML/notebooks/quickstart_bootcamp

echo "📁 Removing redundant implementation files..."
rm -f ASSESSMENT_INTEGRATION_GUIDE.md
rm -f BOOTCAMP_REVIEW_ASSESSMENT.md
rm -f DAY_2_INTEGRATION_COMPLETION.md
rm -f IMPLEMENTATION_PLAN.md
rm -f IMPLEMENTATION_SUMMARY.md
rm -f OPTIMIZATION_IMPLEMENTATION_PLAN.md
rm -f REMAINING_DAYS_INTEGRATION_PLAN.md
rm -f STEP_2_SIMPLIFIED_ASSESSMENT.md
rm -f STEP_3_STREAMLINED_DOCS.md
rm -f STEP_4_MULTI_PACE_TRACKS.md
rm -f STEP_5_NOTEBOOK_MODULARIZATION.md
rm -f day_05_modularization_plan.md

echo "📊 Removing redundant assessment files..."
rm -f assessment_framework.py
rm -f multi_day_progress_tracker.py
rm -f test_assessment_integration.py
rm -f ASSESSMENT_INTEGRATION_REPORT.json
rm -f section2_assessment.json

echo "🔧 Removing development artifacts..."
rm -f analyze_notebooks.py

# Navigate to docs directory
cd /Users/sanjeevadodlapati/Downloads/Repos/ChemML/docs

echo "📚 Removing redundant documentation files..."
rm -f documentation_assessment_and_plan.md
rm -f documentation_integration_guide.md
rm -f documentation_organization_summary.md
rm -f validation_testing_framework.md

echo "📁 Removing redundant documentation directories..."
rm -rf getting_started/
rm -rf planning/
rm -rf reference/
rm -rf resources/
rm -rf roadmaps/

echo "✅ Cleanup complete!"
echo "📊 Remaining structure:"
echo "  Bootcamp: $(ls /Users/sanjeevadodlapati/Downloads/Repos/ChemML/notebooks/quickstart_bootcamp/*.ipynb | wc -l) notebooks"
echo "  Docs: $(ls /Users/sanjeevadodlapati/Downloads/Repos/ChemML/docs/*.md | wc -l) core documents"
