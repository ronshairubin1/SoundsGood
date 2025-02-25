import os
import json
import datetime

def get_model_dir(base_dir, dictionary_name, model_type, version="v1"):
    """
    Build a folder path like:
      base_dir / dictionary_name / model_type / version
    Example:  models/Two_words/CNN/v1
    """
    safe_dict = dictionary_name.replace(' ', '_')
    path = os.path.join(base_dir, safe_dict, model_type, version)
    os.makedirs(path, exist_ok=True)
    return path

def get_cnn_model_path(base_dir, dictionary_name, version="v1"):
    """
    Returns the .h5 path for CNN model
    Example:  models/Two_words/CNN/v1/cnn_model.h5
    """
    folder = get_model_dir(base_dir, dictionary_name, "CNN", version)
    return os.path.join(folder, "cnn_model.h5")

def get_rf_model_path(base_dir, dictionary_name, version="v1"):
    """
    Returns the .joblib path for RF model
    Example:  models/Two_words/RF/v1/rf_model.joblib
    """
    folder = get_model_dir(base_dir, dictionary_name, "RF", version)
    return os.path.join(folder, "rf_model.joblib")

def get_ensemble_model_path(base_dir, dictionary_name, version="v1"):
    """
    Example for an ensemble approach, if you want a single file:
    models/Two_words/ENSEMBLE/v1/ensemble_model.json
    or some other approach. Right now, just return a JSON (or .h5).
    """
    folder = get_model_dir(base_dir, dictionary_name, "ENSEMBLE", version)
    return os.path.join(folder, "ensemble_info.json")

def save_model_metadata(folder_path, metadata):
    """
    Creates/updates a file named 'model_info.json' in `folder_path`
    containing training parameters, augmentation info, etc.
    """
    path = os.path.join(folder_path, "model_info.json")
    # Add a timestamp to the metadata
    if "timestamp" not in metadata:
        metadata["timestamp"] = str(datetime.datetime.now())
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return path
