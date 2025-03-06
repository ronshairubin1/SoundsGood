#!/usr/bin/env python3

import os
import re
import shutil

def update_config_file():
    """
    Update the config.py file to use the new models directory structure.
    Makes a backup of the original file before modifying.
    """
    config_file = "config.py"
    backup_file = "config.py.bak"
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found!")
        return False
    
    # Create backup
    shutil.copy2(config_file, backup_file)
    print(f"Created backup of {config_file} as {backup_file}")
    
    # Read the config file
    with open(config_file, "r") as f:
        content = f.read()
    
    # Pattern to match the MODELS_DIR line
    models_dir_pattern = r"(MODELS_DIR\s*=\s*os\.path\.join\(BASE_DIR,\s*[\"'])models([\"']\))"
    
    # Replace with new structure
    new_content = re.sub(
        models_dir_pattern,
        r"\1data/models\2",
        content
    )
    
    # Check if any changes were made
    if new_content == content:
        print("No changes needed in config.py")
        return True
    
    # Write the updated content back to the file
    with open(config_file, "w") as f:
        f.write(new_content)
    
    print(f"Updated {config_file} to use the new models directory structure")
    print("Old line: MODELS_DIR = os.path.join(BASE_DIR, 'models')")
    print("New line: MODELS_DIR = os.path.join(BASE_DIR, 'data/models')")
    return True

if __name__ == "__main__":
    update_config_file() 