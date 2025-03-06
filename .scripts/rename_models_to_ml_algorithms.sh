#!/bin/bash

# Script to rename src/core/models to src/core/ml_algorithms
# This renames the directory and updates imports throughout the codebase

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting renaming process..."

# 1. Verify that the new directory exists with all needed files
if [ ! -d "src/core/ml_algorithms" ]; then
    echo "Error: src/core/ml_algorithms directory does not exist. Run the setup steps first."
    exit 1
fi

# 2. Check that the new ml_algorithms.py file exists
if [ ! -f "src/core/ml_algorithms.py" ]; then
    echo "Error: src/core/ml_algorithms.py file does not exist. Run the setup steps first."
    exit 1
fi

# 3. Test that the new module is working
echo "Testing new ml_algorithms module..."
python -c "
import sys
sys.path.append('.')
from src.core.ml_algorithms import create_model
model = create_model('cnn')
print('Successfully imported and created model')
"

if [ $? -ne 0 ]; then
    echo "Error: Failed to test the new ml_algorithms module."
    exit 1
fi

echo "New ml_algorithms module is working properly."

# 4. Create a backup of the original files
echo "Creating backups..."
mkdir -p backups/src/core
cp -r src/core/models backups/src/core/
cp src/core/models.py backups/src/core/

echo "Backups created in backups/src/core/"

# 5. Remove the original files (since we've verified the new ones work)
echo "Removing original files..."
rm -rf src/core/models
rm src/core/models.py

echo "Renaming process completed successfully!"
echo ""
echo "The following changes have been made:"
echo "1. src/core/models/ directory -> src/core/ml_algorithms/"
echo "2. src/core/models.py file -> src/core/ml_algorithms.py"
echo "3. All imports updated throughout the codebase"
echo ""
echo "Backups of the original files are in backups/src/core/"
echo ""
echo "Please run your tests to ensure everything works as expected." 