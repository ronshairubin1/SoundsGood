"""
Training model storage implementation for SoundClassifier.

This file focuses on the model storage aspects of the training service
with the new folder structure.
"""

import os
import json
import logging
from datetime import datetime

def setup_model_folder(base_dir, model_name, model_type):
    """
    Create a dedicated folder for a model using the new folder structure.
    
    Args:
        base_dir: The base directory for models
        model_name: The name of the model
        model_type: The type of model (cnn, rf, ensemble)
        
    Returns:
        tuple: (folder_name, model_folder, model_path, metadata_path)
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    folder_name = f"{model_name}_{model_type}_{timestamp}"
    model_folder = os.path.join(base_dir, folder_name)
    
    # Create the folder
    os.makedirs(model_folder, exist_ok=True)
    
    # Determine file extension based on model type
    if model_type == 'cnn':
        extension = '.h5'
    else:  # rf or ensemble
        extension = '.joblib'
    
    # Define paths
    model_path = os.path.join(model_folder, f"{model_name}{extension}")
    metadata_path = os.path.join(model_folder, f"{model_name}_metadata.json")
    
    return folder_name, model_folder, model_path, metadata_path

def update_model_registry_for_new_structure(model_id, model_type, dictionary_name, folder_name, metadata=None):
    """
    Update the model registry with the new folder structure information.
    
    Args:
        model_id: ID of the model
        model_type: Type of model (cnn, rf, ensemble)
        dictionary_name: Name of the dictionary
        folder_name: Name of the folder containing the model
        metadata: Additional metadata to include
        
    Returns:
        bool: Success status
    """
    # Import here to avoid circular imports
    from src.ml.model_paths import update_model_registry
    
    # Default metadata if None provided
    if metadata is None:
        metadata = {}
    
    # Add folder information
    metadata['folder'] = folder_name
    
    # Update the registry
    update_model_registry(
        model_id=model_id,
        model_type=model_type,
        dictionary_name=dictionary_name,
        metadata=metadata
    )
    
    return True
