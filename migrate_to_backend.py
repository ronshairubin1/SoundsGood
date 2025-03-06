#!/usr/bin/env python3
"""
Migration Script for SoundClassifier

This script migrates data from the root data structure to the backend data structure.
It performs a safe migration by:
1. First copying files to new locations
2. Then validating that everything works
3. Finally creating backups of old data

IMPORTANT: Run this script from the project root directory.
"""

import os
import shutil
import json
import logging
import sys
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("migration.log"),
        logging.StreamHandler()
    ]
)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DATA_DIR = os.path.join(BASE_DIR, 'data')
BACKEND_DATA_DIR = os.path.join(BASE_DIR, 'backend', 'data')
ROOT_FEATURE_CACHE = os.path.join(BASE_DIR, 'feature_cache')
BACKEND_FEATURE_CACHE = os.path.join(BACKEND_DATA_DIR, 'features', 'cache')
LEGACY_DIR = os.path.join(BASE_DIR, 'legacy')

def ensure_directories_exist():
    """Create all necessary directories if they don't exist."""
    directories = [
        os.path.join(BACKEND_DATA_DIR, 'sounds', 'raw'),
        os.path.join(BACKEND_DATA_DIR, 'sounds', 'chopped'),
        os.path.join(BACKEND_DATA_DIR, 'sounds', 'augmented'),
        os.path.join(BACKEND_DATA_DIR, 'features', 'cache'),
        os.path.join(BACKEND_DATA_DIR, 'features', 'results'),
        os.path.join(BACKEND_DATA_DIR, 'features', 'unified'),
        os.path.join(BACKEND_DATA_DIR, 'features', 'enhanced'),
        os.path.join(BACKEND_DATA_DIR, 'features', 'plots'),
        os.path.join(BACKEND_DATA_DIR, 'features', 'model_specific'),
        os.path.join(BACKEND_DATA_DIR, 'models', 'cnn'),
        os.path.join(BACKEND_DATA_DIR, 'models', 'rf'),
        os.path.join(BACKEND_DATA_DIR, 'models', 'ensemble'),
        os.path.join(BACKEND_DATA_DIR, 'dictionaries'),
        os.path.join(LEGACY_DIR, 'data'),
        os.path.join(LEGACY_DIR, 'feature_cache'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def copy_sounds():
    """Copy sounds from root data structure to backend."""
    # Map of source directories to destination directories
    sound_dirs = {
        os.path.join(ROOT_DATA_DIR, 'sounds', 'raw_sounds'): 
            os.path.join(BACKEND_DATA_DIR, 'sounds', 'raw'),
        os.path.join(ROOT_DATA_DIR, 'sounds', 'training_sounds'): 
            os.path.join(BACKEND_DATA_DIR, 'sounds', 'chopped'),
        os.path.join(ROOT_DATA_DIR, 'sounds', 'augmented_sounds'): 
            os.path.join(BACKEND_DATA_DIR, 'sounds', 'augmented'),
    }
    
    for src, dst in sound_dirs.items():
        if os.path.exists(src):
            # For each subdirectory (class name)
            for class_name in os.listdir(src):
                src_class_dir = os.path.join(src, class_name)
                dst_class_dir = os.path.join(dst, class_name)
                
                # Skip if not a directory
                if not os.path.isdir(src_class_dir):
                    continue
                    
                # Create destination class directory
                os.makedirs(dst_class_dir, exist_ok=True)
                
                # Copy all files in the class directory
                for filename in os.listdir(src_class_dir):
                    src_file = os.path.join(src_class_dir, filename)
                    dst_file = os.path.join(dst_class_dir, filename)
                    
                    # Skip if not a file
                    if not os.path.isfile(src_file):
                        continue
                        
                    # Copy file if it doesn't exist or is newer
                    if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                        shutil.copy2(src_file, dst_file)
                        logging.info(f"Copied: {src_file} -> {dst_file}")
    
    # Copy sounds.json if it exists
    src_sounds_json = os.path.join(ROOT_DATA_DIR, 'sounds', 'sounds.json')
    dst_sounds_json = os.path.join(BACKEND_DATA_DIR, 'sounds', 'sounds.json')
    if os.path.exists(src_sounds_json):
        shutil.copy2(src_sounds_json, dst_sounds_json)
        logging.info(f"Copied: {src_sounds_json} -> {dst_sounds_json}")

def copy_feature_cache():
    """Copy feature cache files to the backend structure."""
    if os.path.exists(ROOT_FEATURE_CACHE):
        for filename in os.listdir(ROOT_FEATURE_CACHE):
            src_file = os.path.join(ROOT_FEATURE_CACHE, filename)
            dst_file = os.path.join(BACKEND_FEATURE_CACHE, filename)
            
            # Skip if not a file
            if not os.path.isfile(src_file):
                continue
                
            # Copy file if it doesn't exist or is newer
            if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                shutil.copy2(src_file, dst_file)
                logging.info(f"Copied: {src_file} -> {dst_file}")

def copy_models():
    """Copy models from root data structure to backend."""
    src_models_dir = os.path.join(ROOT_DATA_DIR, 'models')
    
    if os.path.exists(src_models_dir):
        # Copy CNN models
        src_cnn_dir = os.path.join(src_models_dir, 'cnn')
        dst_cnn_dir = os.path.join(BACKEND_DATA_DIR, 'models', 'cnn')
        if os.path.exists(src_cnn_dir):
            for filename in os.listdir(src_cnn_dir):
                src_file = os.path.join(src_cnn_dir, filename)
                dst_file = os.path.join(dst_cnn_dir, filename)
                if os.path.isfile(src_file):
                    if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                        shutil.copy2(src_file, dst_file)
                        logging.info(f"Copied: {src_file} -> {dst_file}")
        
        # Copy RF models
        src_rf_dir = os.path.join(src_models_dir, 'rf')
        dst_rf_dir = os.path.join(BACKEND_DATA_DIR, 'models', 'rf')
        if os.path.exists(src_rf_dir):
            for filename in os.listdir(src_rf_dir):
                src_file = os.path.join(src_rf_dir, filename)
                dst_file = os.path.join(dst_rf_dir, filename)
                if os.path.isfile(src_file):
                    if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                        shutil.copy2(src_file, dst_file)
                        logging.info(f"Copied: {src_file} -> {dst_file}")
        
        # Copy Ensemble models
        src_ensemble_dir = os.path.join(src_models_dir, 'ensemble')
        dst_ensemble_dir = os.path.join(BACKEND_DATA_DIR, 'models', 'ensemble')
        if os.path.exists(src_ensemble_dir):
            for filename in os.listdir(src_ensemble_dir):
                src_file = os.path.join(src_ensemble_dir, filename)
                dst_file = os.path.join(dst_ensemble_dir, filename)
                if os.path.isfile(src_file):
                    if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                        shutil.copy2(src_file, dst_file)
                        logging.info(f"Copied: {src_file} -> {dst_file}")
        
        # Copy models.json if it exists
        src_models_json = os.path.join(src_models_dir, 'models.json')
        dst_models_json = os.path.join(BACKEND_DATA_DIR, 'models', 'models.json')
        if os.path.exists(src_models_json):
            shutil.copy2(src_models_json, dst_models_json)
            logging.info(f"Copied: {src_models_json} -> {dst_models_json}")
        
        # Copy root .h5 files
        for filename in os.listdir(src_models_dir):
            if filename.endswith('.h5'):
                src_file = os.path.join(src_models_dir, filename)
                # Determine destination based on filename
                if 'cnn' in filename.lower():
                    dst_file = os.path.join(dst_cnn_dir, filename)
                elif 'rf' in filename.lower():
                    dst_file = os.path.join(dst_rf_dir, filename)
                elif 'ensemble' in filename.lower():
                    dst_file = os.path.join(dst_ensemble_dir, filename)
                else:
                    # Default to CNN if can't determine
                    dst_file = os.path.join(dst_cnn_dir, filename)
                
                if os.path.isfile(src_file):
                    if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                        shutil.copy2(src_file, dst_file)
                        logging.info(f"Copied: {src_file} -> {dst_file}")

def copy_dictionaries():
    """Copy dictionaries from root data structure to backend."""
    src_dict_dir = os.path.join(ROOT_DATA_DIR, 'dictionaries')
    dst_dict_dir = os.path.join(BACKEND_DATA_DIR, 'dictionaries')
    
    if os.path.exists(src_dict_dir):
        for filename in os.listdir(src_dict_dir):
            src_file = os.path.join(src_dict_dir, filename)
            dst_file = os.path.join(dst_dict_dir, filename)
            
            if os.path.isfile(src_file):
                if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                    shutil.copy2(src_file, dst_file)
                    logging.info(f"Copied: {src_file} -> {dst_file}")

def update_configuration():
    """Update config.py to use backend paths."""
    config_path = os.path.join(BASE_DIR, 'config.py')
    
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return False
    
    # Read the current config file
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Create a backup of the original config
    backup_path = os.path.join(BASE_DIR, 'config.py.bak')
    with open(backup_path, 'w') as f:
        f.write(config_content)
    logging.info(f"Created backup of config.py: {backup_path}")
    
    # Update paths to use backend structure
    # We'll use string replacement for safety rather than parsing and rewriting
    replacements = [
        ("RAW_SOUNDS_DIR = os.path.join(DATA_DIR, 'sounds', 'raw_sounds')", 
         "RAW_SOUNDS_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'sounds', 'raw')"),
        
        ("TRAINING_SOUNDS_DIR = os.path.join(DATA_DIR, 'sounds', 'training_sounds')", 
         "TRAINING_SOUNDS_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'sounds', 'chopped')"),
        
        ("MODELS_DIR = os.path.join(BASE_DIR, 'data/models')", 
         "MODELS_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'models')"),
        
        ("DICTIONARIES_DIR = os.path.join(DATA_DIR, 'dictionaries')", 
         "DICTIONARIES_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'dictionaries')"),
    ]
    
    new_config_content = config_content
    for old, new in replacements:
        new_config_content = new_config_content.replace(old, new)
    
    # Write the updated config
    with open(config_path, 'w') as f:
        f.write(new_config_content)
    
    logging.info("Updated config.py with backend paths")
    return True

def create_legacy_folders():
    """Move original data to legacy folders."""
    # Check if user wants to proceed
    proceed = input("Do you want to move original data to legacy folders? (y/n): ")
    if proceed.lower() != 'y':
        logging.info("Skipping legacy folder creation")
        return
    
    # Create timestamps for backup folders
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Move root data to legacy
    if os.path.exists(ROOT_DATA_DIR):
        legacy_data_dir = os.path.join(LEGACY_DIR, 'data_' + timestamp)
        try:
            shutil.move(ROOT_DATA_DIR, legacy_data_dir)
            logging.info(f"Moved: {ROOT_DATA_DIR} -> {legacy_data_dir}")
        except Exception as e:
            logging.error(f"Error moving data directory: {e}")
    
    # Move feature cache to legacy
    if os.path.exists(ROOT_FEATURE_CACHE):
        legacy_feature_cache = os.path.join(LEGACY_DIR, 'feature_cache_' + timestamp)
        try:
            shutil.move(ROOT_FEATURE_CACHE, legacy_feature_cache)
            logging.info(f"Moved: {ROOT_FEATURE_CACHE} -> {legacy_feature_cache}")
        except Exception as e:
            logging.error(f"Error moving feature cache: {e}")

def main():
    """Main migration function."""
    logging.info("Starting migration process")
    
    # Step 1: Ensure all necessary directories exist
    ensure_directories_exist()
    
    # Step 2: Copy all data to new locations
    logging.info("Copying sounds...")
    copy_sounds()
    
    logging.info("Copying feature cache...")
    copy_feature_cache()
    
    logging.info("Copying models...")
    copy_models()
    
    logging.info("Copying dictionaries...")
    copy_dictionaries()
    
    # Step 3: Update configuration to use backend paths
    logging.info("Updating configuration...")
    success = update_configuration()
    if not success:
        logging.error("Failed to update configuration. Migration incomplete.")
        return
    
    logging.info("Migration completed successfully!")
    logging.info("Please test the application to ensure everything is working correctly.")
    logging.info("If the application works correctly, you can create legacy folders for the old data.")
    
    # Step 4: Offer to move original data to legacy folders
    create_legacy_folders()

if __name__ == "__main__":
    main() 