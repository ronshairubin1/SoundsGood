import os
import json
import logging
from glob import glob
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

def get_model_dir(model_id, model_type):
    """Get the directory path for a model"""
    base_dir = os.path.join(Config.BASE_DIR, 'data', 'models', model_type.lower())
    return os.path.join(base_dir, model_id)

def get_cnn_model_path(model_id, dictionary_name=None, version=None):
    """
    Get the path to a CNN model file
    
    This function now handles two different calling patterns:
    1. Single argument: just the model_id (used by newer code)
    2. Three arguments: base_dir, dictionary_name, version (used by older code)
    """
    # If dictionary_name is provided, we're using the old 3-parameter pattern
    if dictionary_name is not None:
        # The old pattern expected (base_dir, dictionary_name, version)
        # So model_id is actually base_dir in this case
        base_dir = model_id
        # Construct an id in the format expected by the newer pattern
        if version:
            full_model_id = f"{dictionary_name}_cnn_{version}"
        else:
            full_model_id = f"{dictionary_name}_cnn_v1"
        
        # Log for debugging
        print(f"Legacy call to get_cnn_model_path with constructed id: {full_model_id}")
        
        # Try the newer path pattern first
        cnn_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'cnn', full_model_id, f"{full_model_id}.h5")
        
        # If that doesn't exist, return the legacy path
        if not os.path.exists(cnn_path):
            legacy_path = os.path.join(base_dir, dictionary_name, 'CNN', version or 'v1', 'cnn_model.h5')
            print(f"Using legacy path: {legacy_path}")
            return legacy_path
        
        return cnn_path
    
    # Original single-argument implementation
    model_dir = get_model_dir(model_id, 'cnn')
    return os.path.join(model_dir, f"{model_id}.h5")

def get_rf_model_path(model_id):
    """Get the path to a Random Forest model file"""
    model_dir = get_model_dir(model_id, 'rf')
    return os.path.join(model_dir, f"{model_id}.joblib")

def get_ensemble_model_path(base_dir, dictionary_name, version="v1"):
    """
    Returns the .joblib path for Ensemble model
    Example:  models/Two_words/Ensemble/v1/ensemble_model.joblib
    """
    folder = get_model_dir(base_dir, dictionary_name, "Ensemble", version)
    return os.path.join(folder, "ensemble_model.joblib")

def save_model_metadata(folder_path, metadata):
    """Save model metadata to a JSON file"""
    metadata_file = os.path.join(folder_path, f"{os.path.basename(folder_path)}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata_file

def load_model_metadata(model_id, model_type):
    """Load model metadata from its JSON file"""
    model_dir = get_model_dir(model_id, model_type)
    metadata_file = os.path.join(model_dir, f"{model_id}_metadata.json")
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata for {model_id}: {str(e)}")
    
    return None

def get_models_registry_path():
    """Get the path to the models.json registry file"""
    return os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')

def update_model_registry(model_id, model_type, dictionary_name=None, metadata=None, is_best=False):
    """Add or update a model in the models.json registry"""
    registry_path = get_models_registry_path()
    
    # Create or load the registry
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except Exception as e:
            logger.error(f"Error loading model registry: {str(e)}")
            registry = {"models": {"cnn": {}, "rf": {}, "ensemble": {}}, "counts": {"cnn": 0, "rf": 0, "ensemble": 0, "total": 0}}
    else:
        registry = {"models": {"cnn": {}, "rf": {}, "ensemble": {}}, "counts": {"cnn": 0, "rf": 0, "ensemble": 0, "total": 0}}
    
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
    
    # Prepare model entry
    model_entry = {
        "id": model_id,
        "name": f"{dictionary_name} {model_type.upper()} Model ({model_id.split('_')[-1]})" if dictionary_name else model_id,
        "type": model_type,
        "dictionary": dictionary_name,
        "created_at": current_time
    }
    
    if model_type == "cnn":
        model_entry["file_path"] = f"cnn/{model_id}/{model_id}.h5"
    else:
        model_entry["file_path"] = f"{model_type}/{model_id}/{model_id}.joblib"
    
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
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Updated model registry with {model_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating model registry: {str(e)}")
        return False

def synchronize_model_registry():
    """
    Scan the model folders and update models.json to match the actual files on disk.
    This ensures the registry is in sync with the filesystem.
    """
    logger.info("Starting model registry synchronization...")
    
    # Define the registry path
    registry_path = get_models_registry_path()
    
    # Create a new registry structure
    registry = {
        "models": {
            "cnn": {},
            "rf": {},
            "ensemble": {}
        },
        "counts": {
            "cnn": 0,
            "rf": 0,
            "ensemble": 0,
            "total": 0
        },
        "best_models": {}
    }
    
    # Load existing registry if it exists (to preserve any data not in the files)
    existing_registry = None
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                existing_registry = json.load(f)
            # Copy best_models data if it exists
            if existing_registry and "best_models" in existing_registry:
                registry["best_models"] = existing_registry["best_models"]
        except Exception as e:
            logger.error(f"Error loading existing registry: {str(e)}")
    
    # Scan for model types (cnn, rf, ensemble)
    for model_type in ["cnn", "rf", "ensemble"]:
        base_dir = os.path.join(Config.BASE_DIR, 'data', 'models', model_type)
        if not os.path.exists(base_dir):
            logger.info(f"Model directory {base_dir} does not exist, creating it")
            os.makedirs(base_dir, exist_ok=True)
            continue
        
        # Find all model directories
        model_dirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
        
        for model_id in model_dirs:
            model_dir = os.path.join(base_dir, model_id)
            
            # Check for the model file
            model_file = None
            if model_type == "cnn":
                model_file = os.path.join(model_dir, f"{model_id}.h5")
            else:
                model_file = os.path.join(model_dir, f"{model_id}.joblib")
            
            if not os.path.exists(model_file):
                logger.warning(f"Model file for {model_id} not found at {model_file}")
                continue
            
            # Check for metadata file
            metadata_file = os.path.join(model_dir, f"{model_id}_metadata.json")
            metadata = None
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading metadata for {model_id}: {str(e)}")
            
            # Extract dictionary name from model_id
            parts = model_id.split('_')
            dictionary_name = parts[0] if len(parts) > 1 else "Unknown"
            
            # Create model entry
            model_entry = {
                "id": model_id,
                "name": f"{dictionary_name} {model_type.upper()} Model ({parts[-1]})" if len(parts) > 1 else model_id,
                "type": model_type,
                "dictionary": dictionary_name,
                "file_path": f"{model_type}/{model_id}/{model_id}.{'h5' if model_type == 'cnn' else 'joblib'}"
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
                        file_ctime = os.path.getctime(model_file)
                        model_entry["created_at"] = datetime.fromtimestamp(file_ctime).isoformat()
                    except:
                        model_entry["created_at"] = datetime.now().isoformat()
                
                # Preserve is_best if it exists
                if "is_best" in existing_entry:
                    model_entry["is_best"] = existing_entry["is_best"]
            else:
                # Use file creation time for created_at
                try:
                    file_ctime = os.path.getctime(model_file)
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
    
    # Update counts
    for model_type in registry["models"]:
        registry["counts"][model_type] = len(registry["models"][model_type])
    registry["counts"]["total"] = sum(registry["counts"].values())
    
    # Check for best_cnn_model.h5 and identify what it links to
    best_model_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'best_cnn_model.h5')
    if os.path.exists(best_model_path) and os.path.islink(best_model_path):
        try:
            real_path = os.path.realpath(best_model_path)
            # Extract the model ID from the path
            # Path format: .../data/models/cnn/DictName_cnn_timestamp/DictName_cnn_timestamp.h5
            path_parts = real_path.split(os.path.sep)
            if len(path_parts) >= 2:
                model_id = path_parts[-2]  # Get the directory name which should be the model ID
                
                # Extract dictionary name
                dict_name = model_id.split('_')[0] if '_' in model_id else None
                
                if dict_name and model_id in registry["models"]["cnn"]:
                    # Mark this model as the best for its dictionary
                    registry["models"]["cnn"][model_id]["is_best"] = True
                    registry["best_models"][dict_name] = model_id
                    logger.info(f"Identified best_cnn_model.h5 as {model_id} for dictionary {dict_name}")
        except Exception as e:
            logger.error(f"Error processing best_cnn_model.h5: {str(e)}")
    
    # Save the updated registry
    try:
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Model registry synchronized successfully. Found {registry['counts']['total']} models.")
        return True
    except Exception as e:
        logger.error(f"Error saving synchronized registry: {str(e)}")
        return False
