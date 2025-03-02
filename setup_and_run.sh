#!/bin/bash

# Exit on error
set -e

echo "Setting up environment and running SoundClassifier app..."

# Activate conda environment (create if it doesn't exist)
if conda info --envs | grep -q "sound_env"; then
    echo "Using existing sound_env environment"
else
    echo "Creating new sound_env environment"
    conda create -y -n sound_env python=3.9
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sound_env

# Install required packages
echo "Installing required packages..."
pip install flask flask-cors werkzeug pydub librosa numpy scipy

# Run the application
echo "Running application..."
python main.py

# Keep terminal open if we get here
echo "Application stopped. Press any key to close terminal..."
read -n 1 