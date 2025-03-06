#!/usr/bin/env python
"""
Configuration Consolidation Script

This script migrates all configuration data to a centralized location
while maintaining backward compatibility. It moves:

1. Dictionary data from config/dictionaries.json → data/dictionaries/dictionaries.json
2. Active dictionary from config/active_dictionary.json → data/dictionaries/dictionaries.json
3. Analysis files from config/analysis/* → data/analysis/*

After running this script, the application will maintain functionality while
using a single source of truth for all configuration data.
"""

import os
import json
import shutil
import logging
from datetime import datetime
from config import Config

# Set up logging
logging.basicConfig(
    filename=os.path.join(Config.LOGS_DIR, 'migrate_config.log'),
    level=logging.INFO, 
    format='%(levelname)s - %(message)s'
)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DICTIONARIES_DIR = os.path.join(DATA_DIR, 'dictionaries')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
LEGACY_DIR = os.path.join(BASE_DIR, 'legacy', 'config')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')

# Source files
CONFIG_DICTIONARIES_PATH = os.path.join(CONFIG_DIR, 'dictionaries.json')
ACTIVE_DICTIONARY_PATH = os.path.join(CONFIG_DIR, 'active_dictionary.json')
CONFIG_ANALYSIS_DIR = os.path.join(CONFIG_DIR, 'analysis')

# Destination files
CONSOLIDATED_DICTIONARIES_PATH = os.path.join(DICTIONARIES_DIR, 'dictionaries.json')

# Ensure directories exist
os.makedirs(DICTIONARIES_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(LEGACY_DIR, exist_ok=True)

def migrate_dictionaries():
    """Migrate dictionaries.json and active_dictionary.json to consolidated format"""
    logging.info("Migrating dictionary data...")
    
    # Initialize consolidated data structure
    consolidated_data = {
        "dictionaries": {},
        "active_dictionary": None
    }
    
    # Load existing consolidated data if it exists
    if os.path.exists(CONSOLIDATED_DICTIONARIES_PATH):
        try:
            with open(CONSOLIDATED_DICTIONARIES_PATH, 'r') as f:
                existing_data = json.load(f)
                consolidated_data = existing_data
                logging.info(f"Loaded existing consolidated data with {len(existing_data.get('dictionaries', {}))} dictionaries")
        except Exception as e:
            logging.error(f"Error loading existing consolidated data: {e}")
    
    # Load dictionaries from config
    if os.path.exists(CONFIG_DICTIONARIES_PATH):
        try:
            with open(CONFIG_DICTIONARIES_PATH, 'r') as f:
                config_dicts = json.load(f)
                
                # Handle different formats - might be a list or an object with 'dictionaries' key
                dictionaries = []
                if isinstance(config_dicts, dict) and 'dictionaries' in config_dicts:
                    dictionaries = config_dicts['dictionaries']
                elif isinstance(config_dicts, list):
                    dictionaries = config_dicts
                
                # Process each dictionary
                for dictionary in dictionaries:
                    dict_name = dictionary.get('name')
                    if not dict_name:
                        continue
                    
                    # Add to consolidated data
                    consolidated_data['dictionaries'][dict_name] = dictionary
                    logging.info(f"Migrated dictionary: {dict_name}")
            
            # Copy to legacy directory for backup
            shutil.copy2(CONFIG_DICTIONARIES_PATH, os.path.join(LEGACY_DIR, 'dictionaries.json'))
            logging.info(f"Backed up {CONFIG_DICTIONARIES_PATH} to legacy directory")
            
        except Exception as e:
            logging.error(f"Error migrating dictionaries.json: {e}")
    
    # Load active dictionary
    if os.path.exists(ACTIVE_DICTIONARY_PATH):
        try:
            with open(ACTIVE_DICTIONARY_PATH, 'r') as f:
                active_dict = json.load(f)
                active_name = active_dict.get('name')
                
                if active_name:
                    consolidated_data['active_dictionary'] = active_name
                    logging.info(f"Set active dictionary to: {active_name}")
            
            # Copy to legacy directory for backup
            shutil.copy2(ACTIVE_DICTIONARY_PATH, os.path.join(LEGACY_DIR, 'active_dictionary.json'))
            logging.info(f"Backed up {ACTIVE_DICTIONARY_PATH} to legacy directory")
            
        except Exception as e:
            logging.error(f"Error migrating active_dictionary.json: {e}")
    
    # Save consolidated data
    try:
        with open(CONSOLIDATED_DICTIONARIES_PATH, 'w') as f:
            json.dump(consolidated_data, f, indent=4)
        logging.info(f"Saved consolidated dictionary data to {CONSOLIDATED_DICTIONARIES_PATH}")
    except Exception as e:
        logging.error(f"Error saving consolidated dictionary data: {e}")
        return False
    
    return True

def migrate_analysis_files():
    """Migrate analysis files from config/analysis to data/analysis"""
    if not os.path.exists(CONFIG_ANALYSIS_DIR):
        logging.info("No analysis directory found to migrate")
        return True
    
    try:
        # Copy all files from config/analysis to data/analysis
        for filename in os.listdir(CONFIG_ANALYSIS_DIR):
            src_path = os.path.join(CONFIG_ANALYSIS_DIR, filename)
            dst_path = os.path.join(ANALYSIS_DIR, filename)
            
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                logging.info(f"Copied analysis file: {filename}")
        
        logging.info("Analysis files migration complete")
        return True
    except Exception as e:
        logging.error(f"Error migrating analysis files: {e}")
        return False

def update_config_pointer():
    """Create a pointer file in the old config location to help with migration"""
    pointer_content = """# This file was created by the configuration migration script
# The actual configuration files have been moved to more appropriate locations:
# - Dictionary data: data/dictionaries/dictionaries.json
# - Analysis data: data/analysis/
#
# This directory is kept for backward compatibility but should be phased out.
"""
    
    try:
        with open(os.path.join(CONFIG_DIR, 'README_MIGRATION.txt'), 'w') as f:
            f.write(pointer_content)
        logging.info("Created migration pointer file in config directory")
        return True
    except Exception as e:
        logging.error(f"Error creating migration pointer: {e}")
        return False

def main():
    """Run the full migration process"""
    logging.info("Starting configuration consolidation...")
    
    # Step 1: Migrate dictionaries
    if not migrate_dictionaries():
        logging.error("Dictionary migration failed")
        return False
    
    # Step 2: Migrate analysis files
    if not migrate_analysis_files():
        logging.error("Analysis file migration failed")
        return False
    
    # Step 3: Update config pointer
    if not update_config_pointer():
        logging.error("Config pointer update failed")
        return False
    
    logging.info("Configuration consolidation complete!")
    logging.info(f"All dictionary data is now in: {CONSOLIDATED_DICTIONARIES_PATH}")
    logging.info(f"All analysis data is now in: {ANALYSIS_DIR}")
    logging.info("Legacy copies of the files are preserved in the legacy directory")
    
    return True

if __name__ == "__main__":
    main() 