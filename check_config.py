#!/usr/bin/env python3

import os
from config import Config

def check_config():
    print("Current configuration:")
    print(f"BASE_DIR: {Config.BASE_DIR}")
    print(f"SOUNDS_DIR: {Config.SOUNDS_DIR}")
    print(f"TRAINING_SOUNDS_DIR: {Config.TRAINING_SOUNDS_DIR}")
    print(f"SOUNDS_DIR is TRAINING_SOUNDS_DIR: {Config.SOUNDS_DIR == Config.TRAINING_SOUNDS_DIR}")
    
    # Create test file
    test_file = os.path.join(Config.SOUNDS_DIR, "test_config.txt")
    with open(test_file, "w") as f:
        f.write("Test file created by check_config.py")
    print(f"Test file created at: {test_file}")
    
    # Check if it exists in training_sounds location
    training_test_file = os.path.join(Config.TRAINING_SOUNDS_DIR, "test_config.txt")
    if os.path.exists(training_test_file):
        print("SUCCESS: File exists in TRAINING_SOUNDS_DIR as expected")
    else:
        print("ERROR: File does not exist in TRAINING_SOUNDS_DIR")
        
    # Clean up
    os.remove(test_file)
    print("Test file removed")

if __name__ == "__main__":
    check_config() 