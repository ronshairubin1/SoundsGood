#!/usr/bin/env python3
"""
Update Feature Extractor Default Path

This script updates the default cache directory path in the FeatureExtractor class
to use the backend data structure.
"""

import os
import re
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extractor_update.log"),
        logging.StreamHandler()
    ]
)

def update_feature_extractor():
    """
    Update the default cache directory in the FeatureExtractor class.
    """
    feature_extractor_path = os.path.join('backend', 'features', 'extractor.py')
    
    if not os.path.exists(feature_extractor_path):
        logging.error("Feature extractor file not found: %s", feature_extractor_path)
        return False
    
    # Create a backup
    backup_path = feature_extractor_path + '.bak'
    shutil.copy2(feature_extractor_path, backup_path)
    logging.info("Created backup of feature extractor: %s", backup_path)
    
    # Read the file
    with open(feature_extractor_path, 'r') as f:
        content = f.read()
    
    # Find and replace the default cache_dir parameter
    # The pattern looks for: cache_dir="feature_cache" or cache_dir="backend/data/features/cache"
    pattern = r'(cache_dir=")([^"]+)(")'
    replacement = r'\1backend/data/features/cache\3'
    
    # Check if the pattern exists in the content
    if not re.search(pattern, content):
        logging.warning("Could not find cache_dir parameter in feature extractor")
        return False
    
    # Make the replacement
    new_content = re.sub(pattern, replacement, content)
    
    # Write the updated content back to the file
    with open(feature_extractor_path, 'w') as f:
        f.write(new_content)
    
    logging.info("Updated feature extractor default cache directory")
    return True

def main():
    logging.info("Starting feature extractor update process")
    
    success = update_feature_extractor()
    
    if success:
        logging.info("Feature extractor update completed successfully!")
    else:
        logging.error("Feature extractor update failed")

if __name__ == "__main__":
    main() 