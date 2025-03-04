#!/usr/bin/env python3

import os
import json
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
MODEL_NAME = "Default_cnn_20250303172659"
MODEL_TYPE = "cnn"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
SOURCE_MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.h5")

# Target directory for the nested structure
TARGET_DIR = os.path.join(MODELS_DIR, MODEL_TYPE, MODEL_NAME)
TARGET_MODEL_PATH = os.path.join(TARGET_DIR, f"{MODEL_NAME}.h5")
TARGET_METADATA_PATH = os.path.join(TARGET_DIR, f"{MODEL_NAME}_metadata.json")

# Models registry path
MODELS_JSON_PATH = os.path.join(MODELS_DIR, "models.json")

def fix_model():
    # Check if source model exists
    if not os.path.exists(SOURCE_MODEL_PATH):
        logging.error(f"Source model not found at {SOURCE_MODEL_PATH}")
        return False
    
    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    logging.info(f"Created directory {TARGET_DIR}")
    
    # Copy model file to target location
    shutil.copy2(SOURCE_MODEL_PATH, TARGET_MODEL_PATH)
    logging.info(f"Copied model from {SOURCE_MODEL_PATH} to {TARGET_MODEL_PATH}")
    
    # Create metadata file
    metadata = {
        'class_names': [],  # This would ideally be populated from your data, leaving empty for now
        'input_shape': [80, 80, 1],  # Standard input shape for audio CNNs in your app
        'num_classes': 2,  # Default value, adjust if needed
        'architecture': "CNN",
        'created_at': datetime.now().isoformat(),
        'keras_version': "2.15.0",  # Using TF 2.15.0 
        'tensorflow_version': "2.15.0",
    }
    
    with open(TARGET_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Created metadata file at {TARGET_METADATA_PATH}")
    
    # Update models.json registry
    try:
        # Load existing models.json
        if os.path.exists(MODELS_JSON_PATH):
            with open(MODELS_JSON_PATH, 'r') as f:
                models_registry = json.load(f)
        else:
            # Create new registry if it doesn't exist
            models_registry = {"models": {"cnn": {}, "rf": {}, "ensemble": {}}}
        
        # Add our model to the registry
        if "models" not in models_registry:
            models_registry["models"] = {}
        if MODEL_TYPE not in models_registry["models"]:
            models_registry["models"][MODEL_TYPE] = {}
        
        # Register the model
        models_registry["models"][MODEL_TYPE][MODEL_NAME] = {
            "id": MODEL_NAME,
            "name": MODEL_NAME,
            "type": MODEL_TYPE,
            "file_path": f"{MODEL_TYPE}/{MODEL_NAME}/{MODEL_NAME}.h5"
        }
        
        # Save updated registry
        with open(MODELS_JSON_PATH, 'w') as f:
            json.dump(models_registry, f, indent=2)
        logging.info(f"Updated models registry in {MODELS_JSON_PATH}")
        
        return True
    except Exception as e:
        logging.error(f"Error updating models.json: {e}")
        return False

if __name__ == "__main__":
    print("Model Fix Script")
    success = fix_model()
    
    if success:
        print(f"\nSuccess! Model {MODEL_NAME} has been:")
        print(f"1. Copied to the correct nested structure at {TARGET_MODEL_PATH}")
        print(f"2. Given a metadata file at {TARGET_METADATA_PATH}")
        print(f"3. Registered in models.json")
        print("\nYou can now use this model in the application.")
    else:
        print("\nError: Failed to fix the model. See the logs above for details.") 