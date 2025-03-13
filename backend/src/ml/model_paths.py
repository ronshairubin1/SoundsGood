"""Module for managing model paths and storage in the SoundClassifier application.

This module provides functions for:
1. Getting model directories and paths based on model type and dictionary name
2. Managing model metadata through JSON files
3. Creating and updating the model registry
4. Supporting the new directory structure for model files storage
"""

import os
import json
import logging
from glob import glob
from datetime import datetime
from backend.config import Config

logger = logging.getLogger(__name__)

def get_extension_for_model_type(model_type):
    """Get the file extension for a given model type"""
    model_type = model_type.lower()
    if model_type == 'cnn':
        return '.h5'
    elif model_type in ['rf', 'random_forest', 'ensemble']:
        return '.joblib'
    else:
        return None

# Define the models base dir in the backend structure
def get_models_base_dir():
    """Get the main models directory in the backend structure"""
    return os.path.join(Config.BASE_DIR, 'data', 'models')

def get_model_dir(dictionary_name, model_type):
    """Get the directory path for a model based on dictionary name and type"""
    # Use the backend/data/models directory structure
    base_dir = os.path.join(get_models_base_dir(), model_type.lower())
    model_dir = os.path.join(base_dir, dictionary_name)
    return model_dir

def create_model_folder(model_name, model_type, dictionary_name=None):
    """
    Create a model folder with the new structure format: {model_name}_{model_type}_{dateandtime}/
    
    Args:
        model_name: The base name of the model
        model_type: The type of model (cnn, rf, ensemble)
        dictionary_name: The optional dictionary name
        
    Returns:
        tuple: (folder_path, model_id)
    """
    # Generate a timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create model_id
    if dictionary_name:
        model_id = f"{dictionary_name}_{model_type}_{timestamp}"
    else:
        model_id = f"{model_name}_{model_type}_{timestamp}"
    
    # Define base directory in backend structure
    base_dir = get_models_base_dir()
    
    # Create model folder path
    folder_path = os.path.join(base_dir, model_id)
    
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)
    
    logger.info(f"Created model folder: {folder_path}")
    
    return folder_path, model_id

def get_cnn_model_path(model_id=None, dictionary_name=None, version=None):
    """
    Get the path to a CNN model file using the new directory structure:
    - If model_id is provided, check if it's a directory in models/ root
    
    Args:
        model_id: Optional model ID
        dictionary_name: Optional dictionary name
        version: Optional version string
        
    Returns:
        string: Path to the model file
    """
    # Base models directory
    base_models_dir = get_models_base_dir()
    
    # Check for model_id folder (new structure)
    if model_id:
        potential_folder = os.path.join(base_models_dir, model_id)
        if os.path.isdir(potential_folder):
            # Look for .h5 file in the folder
            h5_files = glob(os.path.join(potential_folder, '*.h5'))
            if h5_files:
                return h5_files[0]  # Return the first .h5 file found
    
    # If dictionary_name is provided, look for it
    if dictionary_name:
        # Check if there's a directory for this dictionary
        dict_dir = os.path.join(base_models_dir, 'cnn', dictionary_name)
        if os.path.exists(dict_dir):
            # Find the latest model file for this dictionary
            model_files = glob(os.path.join(dict_dir, '*.h5'))
            if model_files:
                # Sort by modification time and return the latest
                latest_file = sorted(model_files, key=os.path.getmtime)[-1]
                return latest_file
    
    # If version is specified, use that
    if version:
        version_path = os.path.join(base_models_dir, 'cnn', f"{dictionary_name}_{version}.h5")
        if os.path.exists(version_path):
            return version_path
    
    # Default to best model
    best_model_path = os.path.join(base_models_dir, 'best_cnn_model.h5')
    if os.path.exists(best_model_path):
        return best_model_path
    
    return None

def get_rf_model_path(model_id=None, dictionary_name=None):
    """
    Get the path to a Random Forest model file.
    Handles folder structure.
    
    Args:
        model_id: Optional model ID
        dictionary_name: Optional dictionary name
        
    Returns:
        string: Path to the model file
    """
    # Base models directory
    base_models_dir = get_models_base_dir()
    
    # Check for model_id folder (new structure)
    if model_id:
        potential_folder = os.path.join(base_models_dir, model_id)
        if os.path.isdir(potential_folder):
            # Look for .joblib file in the folder
            joblib_files = glob(os.path.join(potential_folder, '*.joblib'))
            if joblib_files:
                return joblib_files[0]  # Return the first .joblib file found
    
    # If dictionary_name is provided, look for it
    if dictionary_name:
        # Find all models for this dictionary
        model_pattern = os.path.join(base_models_dir, 'rf', f"{dictionary_name}_*.joblib")
        model_files = glob(model_pattern)
        if model_files:
            # Sort by modification time and return the latest
            latest_file = sorted(model_files, key=os.path.getmtime)[-1]
            return latest_file
    
    return None

def get_ensemble_model_path(dictionary_name=None):
    """
    Get the path to an ensemble model file.
    
    Args:
        dictionary_name: Optional dictionary name
        
    Returns:
        string: Path to the model file
    """
    # Base models directory
    base_dir = os.path.join(get_models_base_dir(), 'ensemble')
    
    # If dictionary name is specified, look for models for that dictionary
    if dictionary_name:
        # Try to find models for this dictionary
        model_pattern = os.path.join(base_dir, f"{dictionary_name}_*.joblib")
        model_files = glob(model_pattern)
        if model_files:
            # Sort by modification time and return the latest
            latest_file = sorted(model_files, key=os.path.getmtime)[-1]
            return latest_file
    
    # If no specific model found, check for any ensemble models
    model_files = glob(os.path.join(base_dir, '*.joblib'))
    if model_files:
        # Sort by modification time and return the latest
        latest_file = sorted(model_files, key=os.path.getmtime)[-1]
        return latest_file
    
    return None

def get_models_registry_path():
    """Get the path to the models registry JSON file"""
    # The models registry is always in the backend models directory
    return os.path.join(get_models_base_dir(), 'models.json')

def save_model_metadata(folder_path, metadata):
    """Save model metadata to a JSON file in the specified folder"""
    metadata_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return metadata_path

def load_model_metadata(model_id, model_type):
    """
    Load metadata for a model.
    Handles folder structure.
    
    Args:
        model_id: ID of the model
        model_type: Type of model (cnn, rf, ensemble)
        
    Returns:
        dict: Metadata dictionary or None if not found
    """
    # Base models directory
    base_models_dir = get_models_base_dir()
    
    # Check for model_id folder (new structure)
    potential_folder = os.path.join(base_models_dir, model_id)
    if os.path.isdir(potential_folder):
        # Look for metadata file in the folder
        metadata_files = glob(os.path.join(potential_folder, '*_metadata.json'))
        if metadata_files:
            try:
                with open(metadata_files[0], 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata from {metadata_files[0]}: {str(e)}")
    
    return None

def update_model_registry(model_id, model_type, dictionary_name=None, metadata=None, is_best=False, folder_path=None):
    """
    Add or update a model in the models.json registry.
    
    Supports both legacy file path structure and new folder-based structure:
    - For legacy: Stores file_path and file_path_to_metadata
    - For new structure: Stores a 'folder' field pointing to the dedicated model folder
    
    Args:
        model_id: ID of the model
        model_type: Type of model (cnn, rf, ensemble)
        dictionary_name: Optional dictionary name
        metadata: Optional metadata to include
        is_best: Whether this is the best model for its dictionary
        folder_path: Optional path to the model folder (for new structure)
        
    Returns:
        bool: Success status
    """
    registry_path = get_models_registry_path()
    
    # Create or load the registry
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r', encoding='utf-8') as file_handle:
                registry = json.load(file_handle)
        except (IOError, json.JSONDecodeError) as error:
            logger.error("Error loading model registry: %s", str(error))
            registry = {"models": {"cnn": {}, "rf": {}, "ensemble": {}},
                       "counts": {"cnn": 0, "rf": 0, "ensemble": 0, "total": 0}}
    else:
        registry = {"models": {"cnn": {}, "rf": {}, "ensemble": {}},
                   "counts": {"cnn": 0, "rf": 0, "ensemble": 0, "total": 0}}

    # Ensure all required structures exist
    if "models" not in registry:
        registry["models"] = {}
    if "counts" not in registry:
        registry["counts"] = {"cnn": 0, "rf": 0, "ensemble": 0, "total": 0}

    model_type = model_type.lower()
    if model_type not in registry["models"]:
        registry["models"][model_type] = {}

    # Extract current timestamp
    current_time = datetime.now().isoformat()

    # Extract dictionary name from model_id if not provided
    if not dictionary_name:
        parts = model_id.split('_')
        if len(parts) > 1:
            dictionary_name = parts[0]
    
    # Extract timestamp from model_id if available
    timestamp = None
    parts = model_id.split('_')
    if len(parts) >= 3:
        timestamp = parts[2]

    # Prepare model entry
    model_entry = {
        "id": model_id,
        "name": (f"{dictionary_name} {model_type.upper()} Model ({timestamp})"
                if dictionary_name and timestamp else model_id),
        "type": model_type,
        "dictionary": dictionary_name,
        "created_at": current_time
    }

    # Handle folder-based structure vs legacy path structure
    if folder_path:
        # For new folder structure: Use folder field
        # Store relative path from models directory
        base_models_dir = get_models_base_dir()
        rel_folder_path = os.path.relpath(folder_path, base_models_dir)
        model_entry["folder"] = rel_folder_path
    else:
        # For legacy structure: Use file_path fields
        file_ext = get_extension_for_model_type(model_type)
        if timestamp:
            path_template = "{type}/{dict}/{dict}_{time}_{type}.{ext}"
            model_entry["file_path"] = path_template.format(
                type=model_type.lower(),
                dict=dictionary_name,
                time=timestamp,
                ext=file_ext
            )
        else:
            # Fallback for models without the expected naming pattern
            model_entry["file_path"] = f"{model_type.lower()}/{dictionary_name}/{model_id}.{file_ext}"
        
        # Include metadata path for backward compatibility
        model_entry["file_path_to_metadata"] = f"{model_type.lower()}/{model_id}_metadata.json"

    # Add metadata fields if provided
    if metadata:
        for key, value in metadata.items():
            model_entry[key] = value
    # If metadata wasn't provided, try to load it from the file
    elif not metadata:
        loaded_metadata = load_model_metadata(model_id, model_type)
        if loaded_metadata:
            # Copy certain fields from metadata to the registry entry
            for key in ["class_names", "num_classes", "accuracy", "input_shape"]:
                if key in loaded_metadata:
                    model_entry[key] = loaded_metadata[key]

    # Add is_best marker if specified
    if is_best:
        model_entry["is_best"] = True

        # If this is the best model, update best_models list in registry
        if "best_models" not in registry:
            registry["best_models"] = {}

        if dictionary_name:
            registry["best_models"][dictionary_name] = model_id

    # Add/update the model entry
    registry["models"][model_type][model_id] = model_entry

    # Update counts
    counts = {model_type: 0 for model_type in registry["models"].keys()}
    for mtype, models in registry["models"].items():
        counts[mtype] = len(models)
    counts["total"] = sum(counts.values())
    registry["counts"] = counts

    # Save the updated registry
    try:
        with open(registry_path, 'w', encoding='utf-8') as file_handle:
            json.dump(registry, file_handle, indent=2)
        logger.info("Updated model registry with %s", model_id)
        return True
    except (IOError, TypeError, json.JSONDecodeError) as error:
        logger.error("Error updating model registry: %s", str(error))
        return False

def create_sample_model(model_type, dictionary_name="test"):
    """
    Create a sample model directory and placeholder files for testing purposes
    
    Args:
        model_type: Type of model (cnn, rf, ensemble)
        dictionary_name: Dictionary name for the model (default: "test")
        
    Returns:
        tuple: (model_id, model_path, metadata)
    """
    # Create unique model ID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_id = f"{dictionary_name}_{model_type}_{timestamp}"
    
    # Create model folder using new structure
    folder_path, _ = create_model_folder(model_id, model_type, dictionary_name)
    
    # Determine file extension
    extension = '.h5' if model_type == 'cnn' else '.joblib'
    
    # Create empty model file
    model_path = os.path.join(folder_path, f"{model_id}.{extension}")
    with open(model_path, 'wb') as f:
        f.write(b'PLACEHOLDER')
    
    # Create metadata file
    metadata_filename = f"{model_id}_metadata.json"
    metadata_path = os.path.join(folder_path, metadata_filename)
    
    # Sample metadata
    metadata = {
        "class_names": ["class1", "class2", "class3"],
        "num_classes": 3,
        "accuracy": 0.85,
        "input_shape": [128, 128, 1] if model_type == "cnn" else None,
        "created_at": datetime.now().isoformat(),
        "is_sample": True
    }
    
    # Write metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Created sample %s model: %s", model_type, model_id)
    return model_id, model_path, metadata

def synchronize_model_registry():
    """
    Scan the model folders and update models.json to match the actual files on disk.
    This ensures the registry is in sync with the filesystem.
    Focuses exclusively on the new folder-based structure.
    """
    logger.info("Starting model registry synchronization...")
    
    # Define the registry path
    registry_path = get_models_registry_path()
    logger.info(f"Registry path: {registry_path}")
    
    # Import OrderedDict to maintain key order in JSON
    from collections import OrderedDict
    
    # Create a new registry structure with counts at the top
    registry = OrderedDict([
        ("counts", {
            "cnn": 0,
            "rf": 0,
            "ensemble": 0,
            "total": 0
        }),
        ("models", {
            "cnn": {},
            "rf": {},
            "ensemble": {}
        }),
        ("best_models", {})
    ])
    
    # Load existing registry if it exists (to preserve any data not in the files)
    existing_registry = None
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                existing_registry = json.load(f)
            logger.info(f"Loaded existing registry with {len(existing_registry.get('models', {}).get('cnn', {}))} CNN models")
            # Copy best_models data if it exists
            if existing_registry and "best_models" in existing_registry:
                registry["best_models"] = existing_registry["best_models"]
        except Exception as e:
            logger.error(f"Error loading existing registry: {str(e)}")
    else:
        logger.warning(f"Registry file does not exist at {registry_path}, will create new one")
    
    # Base models directory
    base_models_dir = get_models_base_dir()
    logger.info(f"Scanning base models directory: {base_models_dir}")
    
    if not os.path.exists(base_models_dir):
        logger.warning(f"Models directory {base_models_dir} does not exist, creating it")
        os.makedirs(base_models_dir, exist_ok=True)
        
        # Write the empty registry to disk
        try:
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2)
            logger.info("Created new empty registry file")
        except Exception as e:
            logger.error(f"Error creating new registry: {str(e)}")
            
        return True  # Return early as there's nothing to scan
    
    # Log directory contents for debugging
    try:
        all_items = os.listdir(base_models_dir)
        dirs = [d for d in all_items if os.path.isdir(os.path.join(base_models_dir, d))]
        files = [f for f in all_items if os.path.isfile(os.path.join(base_models_dir, f))]
        logger.info(f"Found {len(dirs)} directories and {len(files)} files in {base_models_dir}")
        logger.info(f"Directories: {dirs}")
        logger.info(f"Files: {files}")
    except Exception as e:
        logger.error(f"Error listing directory contents: {str(e)}")
    
    # Find all model directories at the top level (new structure)
    # Look for directories with the pattern: {dictionary}_{modeltype}_{timestamp}
    try:
        model_dirs = [d for d in os.listdir(base_models_dir) 
                    if os.path.isdir(os.path.join(base_models_dir, d)) and '_' in d]
        
        logger.info(f"Found {len(model_dirs)} potential model directories: {model_dirs}")
        
        # If no model directories found with new pattern, scan recursively for model files
        if not model_dirs:
            logger.warning("No model directories found with new naming pattern, scanning for model files directly")
            
            # Scan for .h5 (CNN) and .joblib (RF/Ensemble) files
            for root, dirs, files in os.walk(base_models_dir):
                for file in files:
                    if file.endswith('.h5'):
                        logger.info(f"Found CNN model file: {os.path.join(root, file)}")
                        # Extract model name from filename
                        model_name = os.path.splitext(file)[0]
                        directory_name = os.path.basename(root)
                        
                        # Try to determine dictionary name and model type
                        if '_' in model_name:
                            parts = model_name.split('_')
                            if len(parts) >= 2:
                                dictionary_name = parts[0]
                                model_type = 'cnn'
                                model_id = model_name
                                
                                # Create model entry
                                model_entry = {
                                    "id": model_id,
                                    "name": f"{dictionary_name} {model_type.upper()} Model",
                                    "type": model_type,
                                    "dictionary": dictionary_name,
                                    "file_path": os.path.join(root, file),
                                    "created_at": datetime.now().isoformat()
                                }
                                
                                # Look for metadata file
                                metadata_path = os.path.join(root, f"{model_name}_metadata.json")
                                if os.path.exists(metadata_path):
                                    try:
                                        with open(metadata_path, 'r', encoding='utf-8') as f:
                                            metadata = json.load(f)
                                            
                                        # Copy important fields from metadata
                                        for key in ["class_names", "num_classes", "accuracy", "input_shape"]:
                                            if key in metadata:
                                                model_entry[key] = metadata[key]
                                    except Exception as e:
                                        logger.error(f"Error loading metadata for {model_id}: {str(e)}")
                                
                                # Add to registry
                                registry["models"]["cnn"][model_id] = model_entry
                    
                    elif file.endswith('.joblib'):
                        # Similar process for RF/Ensemble models
                        logger.info(f"Found RF/Ensemble model file: {os.path.join(root, file)}")
                        model_name = os.path.splitext(file)[0]
                        
                        # Determine if RF or Ensemble
                        model_type = 'rf'  # Default
                        if 'ensemble' in model_name.lower():
                            model_type = 'ensemble'
                        
                        # Try to determine dictionary name
                        if '_' in model_name:
                            parts = model_name.split('_')
                            if len(parts) >= 2:
                                dictionary_name = parts[0]
                                model_id = model_name
                                
                                # Create model entry
                                model_entry = {
                                    "id": model_id,
                                    "name": f"{dictionary_name} {model_type.upper()} Model",
                                    "type": model_type,
                                    "dictionary": dictionary_name,
                                    "file_path": os.path.join(root, file),
                                    "created_at": datetime.now().isoformat()
                                }
                                
                                # Look for metadata file
                                metadata_path = os.path.join(root, f"{model_name}_metadata.json")
                                if os.path.exists(metadata_path):
                                    try:
                                        with open(metadata_path, 'r', encoding='utf-8') as f:
                                            metadata = json.load(f)
                                            
                                        # Copy important fields from metadata
                                        for key in ["class_names", "num_classes", "accuracy"]:
                                            if key in metadata:
                                                model_entry[key] = metadata[key]
                                    except Exception as e:
                                        logger.error(f"Error loading metadata for {model_id}: {str(e)}")
                                
                                # Add to registry
                                registry["models"][model_type][model_id] = model_entry
                
    except Exception as e:
        logger.error(f"Error scanning for model directories: {str(e)}")
        model_dirs = []
    
    # Process model directories with the new naming pattern
    for model_dir_name in model_dirs:
        model_dir_path = os.path.join(base_models_dir, model_dir_name)
        
        # Parse model directory name to extract metadata
        parts = model_dir_name.split('_')
        if len(parts) < 3:
            logger.warning(f"Skipping directory with invalid naming format: {model_dir_name}")
            continue
            
        # Extract model details from directory name
        dictionary_name = parts[0]
        model_type = parts[1].lower()
        # timestamp is parts[2] and any remaining parts
        
        # Validate model type
        if model_type not in ["cnn", "rf", "ensemble"]:
            logger.warning(f"Unknown model type in directory: {model_type}")
            continue
        
        model_id = model_dir_name
        
        # Find model file in the directory
        if model_type == "cnn":
            model_files = glob(os.path.join(model_dir_path, '*.h5'))
        else:  # rf or ensemble
            model_files = glob(os.path.join(model_dir_path, '*.joblib'))
        
        if not model_files:
            logger.warning(f"No model file found in directory: {model_dir_path}")
            continue
            
        logger.info(f"Found model file in {model_dir_path}: {model_files[0]}")
            
        # Find metadata file in the directory
        metadata_files = glob(os.path.join(model_dir_path, '*_metadata.json'))
        
        # Load metadata if exists
        metadata = None
        if metadata_files:
            try:
                with open(metadata_files[0], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata for {model_id}")
            except Exception as e:
                logger.error(f"Error loading metadata for {model_id}: {str(e)}")
        
        # Create model entry
        model_entry = {
            "id": model_id,
            "name": f"{dictionary_name} {model_type.upper()} Model",
            "type": model_type,
            "dictionary": dictionary_name,
            "folder": os.path.basename(model_dir_path),  # Store relative folder name
            "file_path": model_files[0]  # Also store the actual file path
        }
        
        # If there's existing data for this model, preserve certain fields
        if (existing_registry and "models" in existing_registry and 
            model_type in existing_registry["models"] and 
            model_id in existing_registry["models"][model_type]):
            
            existing_entry = existing_registry["models"][model_type][model_id]
            
            # Preserve created_at if it exists
            if "created_at" in existing_entry:
                model_entry["created_at"] = existing_entry["created_at"]
            else:
                # Use file creation time as fallback
                try:
                    file_ctime = os.path.getctime(model_files[0])
                    model_entry["created_at"] = datetime.fromtimestamp(file_ctime).isoformat()
                except:
                    model_entry["created_at"] = datetime.now().isoformat()
            
            # Preserve is_best if it exists
            if "is_best" in existing_entry:
                model_entry["is_best"] = existing_entry["is_best"]
        else:
            # Use file creation time for created_at
            try:
                file_ctime = os.path.getctime(model_files[0])
                model_entry["created_at"] = datetime.fromtimestamp(file_ctime).isoformat()
            except:
                model_entry["created_at"] = datetime.now().isoformat()
        
        # Add metadata fields
        if metadata:
            # Copy important fields from metadata
            for key in ["class_names", "num_classes", "accuracy", "input_shape"]:
                if key in metadata:
                    model_entry[key] = metadata[key]
        
        # Add to registry
        registry["models"][model_type][model_id] = model_entry
        logger.info(f"Added {model_type} model {model_id} to registry")
    
    # Update counts at the top level of the registry (counts remain at the top)
    for model_type in registry["models"]:
        registry["counts"][model_type] = len(registry["models"][model_type])
    registry["counts"]["total"] = sum(registry["counts"][model_type] for model_type in ["cnn", "rf", "ensemble"])
    
    logger.info(f"Updated registry counts: CNN={registry['counts']['cnn']}, RF={registry['counts']['rf']}, Ensemble={registry['counts']['ensemble']}, Total={registry['counts']['total']}")
    
    # Check for best_cnn_model.h5 and identify what it links to
    best_model_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'best_cnn_model.h5')
    if os.path.exists(best_model_path) and os.path.islink(best_model_path):
        try:
            real_path = os.path.realpath(best_model_path)
            # Extract the model ID from the path 
            # Path would be: .../data/models/DictName_cnn_timestamp/DictName_cnn_timestamp.h5 (new structure)
            path_parts = real_path.split(os.path.sep)
            
            # Try to identify model_id based on directory name (new structure)
            if len(path_parts) >= 2:
                potential_model_id = path_parts[-2]  # Get the directory name
                if (potential_model_id in model_dirs and 
                    potential_model_id.split('_')[1].lower() == 'cnn'):
                    model_id = potential_model_id
                    
                    # Extract dictionary name
                    dict_name = model_id.split('_')[0]
                    
                    # Mark this model as the best for its dictionary
                    if model_id in registry["models"]["cnn"]:
                        registry["models"]["cnn"][model_id]["is_best"] = True
                        registry["best_models"][dict_name] = model_id
                        logger.info(f"Identified best_cnn_model.h5 as {model_id} for dictionary {dict_name}")
        except Exception as e:
            logger.error(f"Error processing best_cnn_model.h5: {str(e)}")
    
    # Save the updated registry
    try:
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Model registry synchronized successfully. Found {registry['counts']['total']} models.")
        
        # If no models found, create a backup of the registry for debugging
        if registry['counts']['total'] == 0:
            backup_path = f"{registry_path}.empty_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2)
            logger.warning(f"No models found. Created registry backup at {backup_path}")
            
        return True
    except Exception as e:
        logger.error(f"Error saving synchronized registry: {str(e)}")
        return False

def trigger_model_registry_update(model_type=None, model_id=None, action="update"):
    """Trigger an update to the model registry by creating or updating a trigger file"""
    trigger_dir = os.path.join(Config.BASE_DIR, 'data', 'triggers')
    os.makedirs(trigger_dir, exist_ok=True)
    
    trigger_file = os.path.join(trigger_dir, 'model_registry_update.json')
    
    trigger_data = {
        "action": action,
        "timestamp": datetime.now().isoformat(),
    }
    
    if model_type:
        trigger_data["model_type"] = model_type
    if model_id:
        trigger_data["model_id"] = model_id
        
    try:
        with open(trigger_file, 'w', encoding='utf-8') as f:
            json.dump(trigger_data, f, indent=2)
        logger.info("Created model registry update trigger file")
        return True
    except Exception as e:
        logger.error(f"Error creating trigger file: {str(e)}")
        return False

def get_model_counts_from_registry():
    """
    Get model counts directly from the models.json registry file.
    
    Returns:
        dict: A dictionary containing the counts for each model type and total count.
              Format: {'cnn': X, 'rf': Y, 'ensemble': Z, 'total': N}
              If the registry doesn't exist, returns default counts of 0.
    """
    registry_path = get_models_registry_path()
    default_counts = {"cnn": 0, "rf": 0, "ensemble": 0, "total": 0}
    
    if not os.path.exists(registry_path):
        logger.warning(f"Models registry not found at {registry_path}, returning default counts")
        return default_counts
    
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        # Check if counts are available in the registry
        if "counts" in registry:
            logger.info(f"Found counts in registry: {registry['counts']}")
            return registry["counts"]
        
        # If counts are not in registry, calculate them
        logger.warning("Counts not found in registry, calculating from model entries")
        counts = default_counts.copy()
        
        if "models" in registry:
            for model_type in ["cnn", "rf", "ensemble"]:
                if model_type in registry["models"]:
                    counts[model_type] = len(registry["models"][model_type])
            
            counts["total"] = sum(counts[model_type] for model_type in ["cnn", "rf", "ensemble"])
        
        return counts
        
    except Exception as e:
        logger.error(f"Error reading model counts from registry: {str(e)}")
        return default_counts

def count_model_files_from_registry(directory=None):
    """
    Count model files by synchronizing the registry and returning counts.
    This is a unified version of the count_model_files function that was duplicated
    in main.py and dashboard_api.py.
    
    Args:
        directory: Optional directory path (not used but kept for compatibility)
                  with existing function signatures
    
    Returns:
        tuple: (total_count, counts_by_type) where counts_by_type is a dictionary
               with keys 'cnn', 'rf', 'ensemble', and 'total'
    """
    # First, ensure the model registry is synchronized with the filesystem
    synchronize_model_registry()
    
    # Get model counts directly from the registry
    model_counts = get_model_counts_from_registry()
    
    # Log the counts retrieved from the registry
    logger.info("Model counts from registry - Total: %d, CNN: %d, RF: %d, Ensemble: %d",
               model_counts['total'], model_counts['cnn'], model_counts['rf'], model_counts['ensemble'])
    
    return model_counts['total'], model_counts
