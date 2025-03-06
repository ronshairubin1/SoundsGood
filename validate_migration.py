#!/usr/bin/env python3
"""
Migration Validation Script

This script validates that the migration to the backend data structure was successful
by checking key files and directories.
"""

import os
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("migration_validation.log"),
        logging.StreamHandler()
    ]
)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DATA_DIR = os.path.join(BASE_DIR, 'backend', 'data')
BACKEND_FEATURE_CACHE = os.path.join(BACKEND_DATA_DIR, 'features', 'cache')

def validate_directories():
    """Validate that key directories exist and contain files."""
    directories = [
        os.path.join(BACKEND_DATA_DIR, 'sounds', 'raw'),
        os.path.join(BACKEND_DATA_DIR, 'sounds', 'chopped'),
        os.path.join(BACKEND_DATA_DIR, 'sounds', 'augmented'),
        BACKEND_FEATURE_CACHE,
        os.path.join(BACKEND_DATA_DIR, 'models'),
        os.path.join(BACKEND_DATA_DIR, 'dictionaries')
    ]
    
    all_valid = True
    
    for directory in directories:
        if not os.path.exists(directory):
            logging.error("Directory does not exist: %s", directory)
            all_valid = False
            continue
            
        # Check if directory contains any files
        has_files = False
        for _, _, files in os.walk(directory):
            if files:
                has_files = True
                break
                
        if not has_files:
            logging.warning("Directory exists but contains no files: %s", directory)
    
    return all_valid

def validate_feature_cache():
    """Validate that the feature cache contains files and they are valid JSON."""
    if not os.path.exists(BACKEND_FEATURE_CACHE):
        logging.error("Feature cache directory does not exist: %s", BACKEND_FEATURE_CACHE)
        return False
    
    json_files = [f for f in os.listdir(BACKEND_FEATURE_CACHE) if f.endswith('.json')]
    
    if not json_files:
        logging.warning("Feature cache contains no JSON files")
        return False
    
    # Check a sample of JSON files to make sure they are valid
    sample_size = min(5, len(json_files))
    for i in range(sample_size):
        json_file = os.path.join(BACKEND_FEATURE_CACHE, json_files[i])
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json.load(f)
        except Exception as e:
            logging.error("Invalid JSON file: %s, error: %s", json_file, str(e))
            return False
    
    logging.info("Feature cache validation successful: %s", BACKEND_FEATURE_CACHE)
    return True

def validate_config():
    """Validate that the config.py file has been updated."""
    config_path = os.path.join(BASE_DIR, 'config.py')
    
    if not os.path.exists(config_path):
        logging.error("Config file does not exist: %s", config_path)
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for backend paths
    backend_references = [
        "'backend', 'data'",
        "backend/data"
    ]
    
    for ref in backend_references:
        if ref not in content:
            logging.warning("Config file does not contain backend reference: %s", ref)
            return False
    
    logging.info("Config validation successful")
    return True

def validate_feature_extractor():
    """Validate that the feature extractor has been updated."""
    extractor_path = os.path.join(BASE_DIR, 'backend', 'features', 'extractor.py')
    
    if not os.path.exists(extractor_path):
        logging.error("Feature extractor file does not exist: %s", extractor_path)
        return False
    
    with open(extractor_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for backend cache path
    if 'backend/data/features/cache' not in content:
        logging.warning("Feature extractor does not use backend cache path")
        return False
    
    logging.info("Feature extractor validation successful")
    return True

def main():
    """Main validation function."""
    logging.info("Starting migration validation")
    
    # Validate directories
    dirs_valid = validate_directories()
    if not dirs_valid:
        logging.warning("Directory validation failed")
    
    # Validate feature cache
    cache_valid = validate_feature_cache()
    if not cache_valid:
        logging.warning("Feature cache validation failed")
    
    # Validate config
    config_valid = validate_config()
    if not config_valid:
        logging.warning("Config validation failed")
    
    # Validate feature extractor
    extractor_valid = validate_feature_extractor()
    if not extractor_valid:
        logging.warning("Feature extractor validation failed")
    
    # Overall validation
    if dirs_valid and cache_valid and config_valid and extractor_valid:
        logging.info("Migration validation SUCCESSFUL! All checks passed.")
        print("\n✅ Migration validation SUCCESSFUL! All checks passed.\n")
        return True
    else:
        logging.error("Migration validation FAILED! Some checks did not pass.")
        print("\n❌ Migration validation FAILED! Some checks did not pass.\n")
        return False

if __name__ == "__main__":
    main() 