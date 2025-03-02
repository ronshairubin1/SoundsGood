#!/bin/bash
# Script to run commands in the soundclassifier environment

# Activate the soundclassifier environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate soundclassifier

# Run the command passed as an argument
$@

# Keep the terminal open after command completes
echo "Command completed. Press any key to close..."
read -n 1 