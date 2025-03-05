#!/usr/bin/env python3
"""
Move Legacy Feature Extractors

This script identifies and moves legacy feature extractor implementations to a legacy folder.
It creates backup copies of the original files and updates references as needed.

Usage:
    python backend/move_legacy_extractors.py
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legacy_migration.log"),
        logging.StreamHandler()
    ]
)

# Define the legacy feature extractor files to be moved
LEGACY_FILES = [
    "src/ml/feature_extractor.py",  # AudioFeatureExtractor
    "src/ml/audio_processing.py",   # UnifiedFeatureExtractor
    "unified_feature_extractor.py", # ComprehensiveFeatureExtractor
    "advanced_feature_extractor.py" # AdvancedFeatureExtractor
]

# Define the legacy directory
LEGACY_DIR = "legacy/feature_extractors"

def create_legacy_folder():
    """
    Create the legacy folder structure
    """
    os.makedirs(LEGACY_DIR, exist_ok=True)
    logging.info(f"Created legacy directory: {LEGACY_DIR}")
    
    # Create a README for the legacy folder
    readme_path = os.path.join(LEGACY_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write("""# Legacy Feature Extractors

This directory contains legacy feature extractor implementations that have been 
consolidated into the unified FeatureExtractor class in `backend/features/extractor.py`.

These files are kept for reference and backward compatibility but should not be 
used in new code. A migration script is available at `backend/migrate_features.py`
to convert features extracted using these legacy implementations to the new format.

## Files

- `feature_extractor.py`: Contains the original AudioFeatureExtractor
- `audio_processing.py`: Contains the original UnifiedFeatureExtractor
- `unified_feature_extractor.py`: Contains the ComprehensiveFeatureExtractor
- `advanced_feature_extractor.py`: Contains the AdvancedFeatureExtractor

## Migration

To migrate features extracted with these legacy implementations, use:

```
python backend/migrate_features.py --input_dir <legacy_features_dir> --output_dir <new_features_dir>
```
""")
    logging.info(f"Created README at {readme_path}")

def move_file_to_legacy(file_path):
    """
    Move a file to the legacy folder
    
    Args:
        file_path: Path to the file to move
        
    Returns:
        Tuple of (success, destination_path)
    """
    try:
        if not os.path.exists(file_path):
            logging.warning(f"File does not exist: {file_path}")
            return False, None
        
        # Create backup first
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        logging.info(f"Created backup: {backup_path}")
        
        # Determine destination path
        dest_path = os.path.join(LEGACY_DIR, os.path.basename(file_path))
        
        # Move file
        shutil.copy2(file_path, dest_path)
        logging.info(f"Moved {file_path} to {dest_path}")
        
        # Add legacy notice to the file
        with open(dest_path, "r") as f:
            content = f.read()
        
        legacy_notice = f"""# LEGACY CODE - DO NOT USE IN NEW PROJECTS
# This file has been moved to the legacy folder and is maintained only for reference.
# Please use the unified FeatureExtractor in backend/features/extractor.py instead.
# 
# Original location: {file_path}
# 
"""
        
        with open(dest_path, "w") as f:
            f.write(legacy_notice + content)
            
        # Create an import forwarder in the original location
        with open(file_path, "w") as f:
            f.write(f"""# This file has been moved to the legacy folder.
# This import forwarder maintains backward compatibility.
# Please use the unified FeatureExtractor in backend/features/extractor.py for new code.

import warnings
import os

warnings.warn(
    "You are using a legacy feature extractor that has been moved to the legacy folder. "
    "Please use the unified FeatureExtractor in backend/features/extractor.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the legacy location
legacy_path = os.path.join("{LEGACY_DIR}", os.path.basename(__file__))
with open(legacy_path, "r") as f:
    exec(f.read())

# Also import the new extractor for convenience
try:
    from backend.features.extractor import FeatureExtractor
except ImportError:
    pass
""")
            
        logging.info(f"Created import forwarder at {file_path}")
        return True, dest_path
    
    except Exception as e:
        logging.error(f"Error moving {file_path} to legacy: {e}")
        return False, None

def process_all_files():
    """
    Process all legacy files
    
    Returns:
        Number of successfully moved files
    """
    create_legacy_folder()
    
    success_count = 0
    for file_path in LEGACY_FILES:
        success, _ = move_file_to_legacy(file_path)
        if success:
            success_count += 1
    
    logging.info(f"Moved {success_count} files to legacy folder")
    return success_count

def main():
    process_all_files()

if __name__ == "__main__":
    main() 