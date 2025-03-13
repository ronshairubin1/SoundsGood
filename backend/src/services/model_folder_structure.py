"""
Model folder structure implementation for SoundClassifier.

This file creates functions to help migrate the model storage to the new structure.
The new structure will be:
backend/data/models/{model_name}_{model_type}_{dateandtime}/
/{model_name}_{model_type}_{dateandtime}{.h5 or _metadata.json}
"""

import os
import json
import logging
import shutil
from datetime import datetime
from glob import glob

def create_model_folder(base_dir, model_name, model_type):
    """
    Create a dedicated folder for a model using the new folder structure.
    
    Args:
        base_dir: The base directory for models
        model_name: The name of the model
        model_type: The type of model (cnn, rf, ensemble)
        
    Returns:
        tuple: (folder_path, model_path, metadata_path)
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    folder_name = f"{model_name}_{model_type}_{timestamp}"
    model_folder = os.path.join(base_dir, folder_name)
    os.makedirs(model_folder, exist_ok=True)
    
    # Determine file extension based on model type
    if model_type == 'cnn':
        extension = '.h5'
    else:  # rf or ensemble
        extension = '.joblib'
    
    model_path = os.path.join(model_folder, f"{model_name}{extension}")
    metadata_path = os.path.join(model_folder, f"{model_name}_metadata.json")
    
    return folder_name, model_folder, model_path, metadata_path

def update_models_json_to_new_structure(models_json_path):
    """
    Update the models.json file to use 'folder' instead of 'file_path' and 'file_path_to_metadata'.
    
    Args:
        models_json_path: Path to the models.json file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the current models.json file
        with open(models_json_path, 'r') as f:
            registry = json.load(f)
        
        # Create a backup of the original file
        backup_path = f"{models_json_path}.bak"
        shutil.copy2(models_json_path, backup_path)
        logging.info(f"Created backup of models.json at {backup_path}")
        
        # Update each model to use the 'folder' field instead of file paths
        if 'models' in registry:
            for model_type, models in registry['models'].items():
                for model_id, model_info in models.items():
                    # If the model already has 'folder', skip it
                    if 'folder' in model_info:
                        continue
                        
                    # Extract information needed to build the folder path
                    if 'file_path' in model_info and 'file_path_to_metadata' in model_info:
                        # Create the folder name based on the model ID
                        # (Assuming model_id follows the pattern: name_type_timestamp)
                        folder_name = model_id
                        
                        # Update the model_info with the folder field
                        model_info['folder'] = folder_name
                        
                        # Remove the old file path fields
                        if 'file_path' in model_info:
                            del model_info['file_path']
                        if 'file_path_to_metadata' in model_info:
                            del model_info['file_path_to_metadata']
        
        # Write the updated registry back to the file
        with open(models_json_path, 'w') as f:
            json.dump(registry, f, indent=2)
            
        logging.info(f"Updated models.json to use 'folder' instead of file paths")
        return True
        
    except Exception as e:
        logging.error(f"Error updating models.json: {str(e)}")
        return False

def migrate_existing_models_to_new_structure(base_dir):
    """
    Migrate existing models to the new folder structure.
    
    Args:
        base_dir: The base directory for models
        
    Returns:
        dict: Statistics about the migration
    """
    stats = {
        'models_migrated': 0,
        'errors': 0
    }
    
    try:
        # Check for models.json
        models_json_path = os.path.join(base_dir, 'models.json')
        if not os.path.exists(models_json_path):
            logging.error(f"models.json not found at {models_json_path}")
            return stats
        
        # Load models.json
        with open(models_json_path, 'r') as f:
            registry = json.load(f)
        
        # Iterate through each model type
        for model_type in ['cnn', 'rf', 'ensemble']:
            type_dir = os.path.join(base_dir, model_type)
            if not os.path.exists(type_dir):
                continue
                
            # Find all model files
            if model_type == 'cnn':
                model_files = glob(os.path.join(type_dir, '*.h5'))
            else:
                model_files = glob(os.path.join(type_dir, '*.joblib'))
                
            for model_file in model_files:
                try:
                    model_filename = os.path.basename(model_file)
                    model_name = os.path.splitext(model_filename)[0]
                    
                    # Check if metadata file exists
                    metadata_file = os.path.join(type_dir, f"{model_name}_metadata.json")
                    
                    # Create new folder structure
                    folder_name = f"{model_name}_{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    new_folder = os.path.join(base_dir, folder_name)
                    os.makedirs(new_folder, exist_ok=True)
                    
                    # Copy files to new location
                    shutil.copy2(model_file, os.path.join(new_folder, os.path.basename(model_file)))
                    
                    if os.path.exists(metadata_file):
                        shutil.copy2(metadata_file, os.path.join(new_folder, os.path.basename(metadata_file)))
                    
                    stats['models_migrated'] += 1
                    logging.info(f"Migrated model {model_name} to new folder structure at {new_folder}")
                    
                except Exception as e:
                    stats['errors'] += 1
                    logging.error(f"Error migrating model {model_file}: {str(e)}")
        
        # Update models.json to use the new structure
        update_models_json_to_new_structure(models_json_path)
        
        return stats
        
    except Exception as e:
        logging.error(f"Error during model migration: {str(e)}")
        stats['errors'] += 1
        return stats
