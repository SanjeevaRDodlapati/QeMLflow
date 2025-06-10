#!/bin/bash

# ChemML Bootcamp Maintenance Script
# This script performs routine cleanup and organization tasks

echo "🧹 ChemML Bootcamp Cleanup & Maintenance"
echo "========================================"

# Navigate to bootcamp directory
BOOTCAMP_DIR="/Users/sanjeevadodlapati/Downloads/Repos/ChemML/notebooks/quickstart_bootcamp"
cd "$BOOTCAMP_DIR" || exit

echo "📁 Current directory: $(pwd)"

# 1. Remove Python cache files
echo "🗑️  Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null

# 2. Remove temporary Jupyter files
echo "🗑️  Removing Jupyter temporary files..."
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null

# 3. Remove temporary assessment data (but preserve framework)
echo "🗑️  Cleaning temporary assessment data..."
find assessments/ -name "*.tmp" -delete 2>/dev/null
find assessments/ -name "temp_*" -delete 2>/dev/null

# 4. Clean up any stray log files
echo "🗑️  Removing log files..."
find . -name "*.log" -delete 2>/dev/null

# 5. Display current structure
echo ""
echo "📊 Current bootcamp structure:"
echo "├── $(ls -1 days/ | wc -l | tr -d ' ') day folders"
echo "├── $(ls -1 assessments/ | wc -l | tr -d ' ') assessment files"
echo "├── $(ls -1 docs/ | wc -l | tr -d ' ') documentation files"
echo "├── $(ls -1 scripts/ | wc -l | tr -d ' ') scripts"
echo "└── $(ls -1 utils/ | wc -l | tr -d ' ') utility files"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "💡 Next steps:"
echo "   1. Activate conda environment: source ../../chemml_env/bin/activate"
echo "   2. Start with Day 1: cd days/day_01 && jupyter lab"
echo "   3. If you encounter issues, check docs/SESSION_CLEANUP_GUIDE.md"
