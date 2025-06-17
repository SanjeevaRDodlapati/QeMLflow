#!/bin/bash

# Quick rollback test debug
set -e

TEST_DIR="./debug_test"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "Creating test file..."
echo "QeMLflow test content" > test.py

echo "Creating backup and modifying..."
cp test.py test.py.bak
sed -i.bak2 's/QeMLflow/QeMLflow/g' test.py

echo "Content after modification:"
cat test.py

echo "Backup files:"
ls -la *.bak*

echo "Restoring from backup:"
cp test.py.bak test.py

echo "Content after restore:"
cat test.py

if grep -q "QeMLflow" test.py; then
    echo "✓ Rollback successful!"
else
    echo "✗ Rollback failed!"
fi

cd ..
rm -rf "$TEST_DIR"
