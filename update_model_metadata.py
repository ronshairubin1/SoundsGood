#!/usr/bin/env python3

import os
import json
import logging
import glob
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
DICTIONARIES_PATH = os.path.join(BASE_DIR, "data", "dictionaries", "dictionaries.json")

# Define dictionary name mappings for special cases
DICTIONARY_MAPPINGS = {
    "Two_words": "Two_words",
    "Michal_Base": "Michal Base",
    "Michal_Plus_Ooah": "Michal Plus Ooah"
}

def normalize_name(name):
    """Normalize a name by lowercasing and removing spaces, underscores, etc."""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def update_model_metadata():
    """
    Update all model metadata files with correct class names from the dictionaries.json file.
    """
    # Load dictionaries
    if not os.path.exists(DICTIONARIES_PATH):
        logging.error(f"Dictionaries file not found at {DICTIONARIES_PATH}")
        return False
    
    try:
        with open(DICTIONARIES_PATH, 'r') as f:
            dictionaries_data = json.load(f)
        
        dictionaries = dictionaries_data.get('dictionaries', {})
        logging.info(f"Loaded {len(dictionaries)} dictionaries")
        
        # Create normalized name lookups
        normalized_dicts = {}
        for dict_name, dict_data in dictionaries.items():
            normalized_dicts[normalize_name(dict_name)] = dict_data
        
        # Find all metadata files
        metadata_files = glob.glob(os.path.join(MODELS_DIR, "**", "*_metadata.json"), recursive=True)
        logging.info(f"Found {len(metadata_files)} metadata files")
        
        updated_count = 0
        
        for metadata_file in metadata_files:
            try:
                # Extract model name from metadata filename
                filename = os.path.basename(metadata_file)
                model_name_parts = filename.split('_')
                # Get the dictionary name part (usually before _cnn_ or similar)
                if len(model_name_parts) >= 3:
                    model_prefix = '_'.join(model_name_parts[:-2])  # Skip the timestamp and metadata suffix
                else:
                    model_prefix = model_name_parts[0]  # Just use the first part
                
                logging.info(f"Processing model: {model_prefix}")
                
                # Load current metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Try lookup through mappings
                dict_name = DICTIONARY_MAPPINGS.get(model_prefix)
                matching_dict = None
                
                if dict_name and dict_name in dictionaries:
                    matching_dict = dictionaries[dict_name]
                    logging.info(f"Found match through mapping: {model_prefix} -> {dict_name}")
                else:
                    # Try direct match
                    for dict_name, dict_data in dictionaries.items():
                        if normalize_name(dict_name) == normalize_name(model_prefix):
                            matching_dict = dict_data
                            logging.info(f"Found direct match: {model_prefix} -> {dict_name}")
                            break
                    
                    # If still no match, try partial matching
                    if not matching_dict:
                        normalized_prefix = normalize_name(model_prefix)
                        for norm_name, dict_data in normalized_dicts.items():
                            if normalized_prefix.startswith(norm_name) or norm_name.startswith(normalized_prefix):
                                matching_dict = dict_data
                                logging.info(f"Found partial match: {model_prefix} -> {norm_name}")
                                break
                
                if matching_dict:
                    # Update class names
                    if 'classes' in matching_dict:
                        # Use classes field if available
                        class_names = matching_dict['classes']
                    else:
                        # Otherwise use sounds field
                        class_names = matching_dict['sounds']
                    
                    metadata['class_names'] = class_names
                    metadata['num_classes'] = len(class_names)
                    
                    # Save updated metadata
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    logging.info(f"Updated metadata for {model_prefix} with {len(class_names)} classes")
                    updated_count += 1
                else:
                    logging.warning(f"No matching dictionary found for model {model_prefix}")
            
            except Exception as e:
                logging.error(f"Error updating metadata file {metadata_file}: {e}")
        
        logging.info(f"Updated {updated_count} of {len(metadata_files)} metadata files")
        return True
        
    except Exception as e:
        logging.error(f"Error updating metadata: {e}")
        return False

if __name__ == "__main__":
    print("Model Metadata Update Script")
    success = update_model_metadata()
    
    if success:
        print("\nSuccess! Model metadata has been updated with correct class names.")
        print("You should now be able to see sound classes for all models in the web interface.")
    else:
        print("\nError: Failed to update model metadata. See the logs above for details.")