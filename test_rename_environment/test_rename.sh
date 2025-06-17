#!/bin/bash

# Test version of the renaming script
set -e

echo "Starting test rename process..."

# Text replacements in files
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) -exec sed -i.bak 's/chemml/qemlflow/g' {} \;
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) -exec sed -i.bak 's/ChemML/QeMLflow/g' {} \;
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) -exec sed -i.bak 's/CHEMML/QEMLFLOW/g' {} \;

# Directory renaming (test with a copy first)
if [ -d "src/chemml" ]; then
    echo "Renaming src/chemml to src/qemlflow..."
    mv src/chemml src/qemlflow
fi

echo "Test rename process completed!"
