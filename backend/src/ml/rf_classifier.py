# File: src/ml/rf_classifier.py

import joblib
import os
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRF

class RandomForestClassifier:
    """
    A RandomForest-based classifier adapted from old_code/model.py
    """

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.classes_ = None
        self.feature_count = None
        os.makedirs(model_dir, exist_ok=True)

    def train(self, X, y):
        """
        Train a Random Forest classifier
        """
        try:
            self.model = SKRF(
                n_estimators=200,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
            self.classes_ = self.model.classes_
            self.feature_count = X.shape[1]
            logging.info(f"RF model trained with {len(self.classes_)} classes.")
            return True
        except Exception as e:
            logging.error(f"Error training RandomForest: {str(e)}")
            return False

    def predict(self, X):
        if self.model is None:
            logging.error("RandomForest model not trained or loaded.")
            return None, None
        try:
            X = np.array(X)
            if X.shape[1] != self.feature_count:
                logging.error("Feature count mismatch for RF.")
                return None, None
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            return predictions, probabilities
        except Exception as e:
            logging.error(f"Error in RF prediction: {str(e)}")
            return None, None

    def get_top_predictions(self, X, top_n=3):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probs = self.model.predict_proba(X)
        top_indices = np.argsort(probs, axis=1)[:, -top_n:][:, ::-1]
        top_probabilities = np.take_along_axis(probs, top_indices, axis=1)
        top_labels = self.model.classes_[top_indices]
        results = []
        for labels, pvals in zip(top_labels, top_probabilities):
            pred_list = []
            for label, val in zip(labels, pvals):
                pred_list.append({"sound": label, "probability": float(val)})
            results.append(pred_list)
        return results

    def save(self, filename='rf_sound_classifier.joblib'):
        if self.model is None:
            logging.error("No RF model to save.")
            return False
        try:
            path = os.path.join(self.model_dir, filename)
            joblib.dump({
                'model': self.model,
                'classes': self.classes_,
                'feature_count': self.feature_count
            }, path)
            logging.info(f"RF model saved to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving RF model: {str(e)}")
            return False

    def load(self, filename='rf_sound_classifier.joblib'):
        try:
            path = os.path.join(self.model_dir, filename)
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.classes_ = model_data['classes']
            self.feature_count = model_data['feature_count']
            logging.info(f"RF model loaded from {path}")
            return True
        except Exception as e:
            logging.error(f"Error loading RF model: {str(e)}")
            return False
def get_ensemble_model_path(dictionary_name=None):
    """
    Get the path to an ensemble model file.
    
    Args:
        dictionary_name: Optional dictionary name
        
    Returns:
        string: Path to the model file
    """
    # Base models directory
    base_dir = os.path.join(Config.BASE_DIR, 'backend', 'data', 'models', 'ensemble')
    
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

def save_model_metadata(folder_path, metadata):
    """Save model metadata to a JSON file in the specified folder"""
    metadata_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return metadata_path

def load_model_metadata(model_id, model_type):
    """
    Load metadata for a model.
    Handles both new folder structure and legacy paths.
    
    Args:
        model_id: ID of the model
        model_type: Type of model (cnn, rf, ensemble)
        
    Returns:
        dict: Metadata dictionary or None if not found
    """
    # Base models directory
    base_models_dir = os.path.join(Config.BASE_DIR, 'backend', 'data', 'models')
    
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
    
    # Fallback to legacy structure
    legacy_path = os.path.join(base_models_dir, model_type, f"{model_id}_metadata.json")
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata from {legacy_path}: {str(e)}")
    
    return None

def get_models_registry_path():
    """Get the path to the models.json registry file"""
    # Check for registry in backend path
    backend_path = os.path.join(Config.BASE_DIR, 'backend', 'data', 'models', 'models.json')
    if os.path.exists(backend_path):
        return backend_path
    
    # Fallback to non-backend path
    return os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')

def get_ensemble_model_path(dictionary_name=None):
    """
    Get the path to an ensemble model file.
    
    Args:
        dictionary_name: Optional dictionary name
        
    Returns:
        string: Path to the model file
    """
    # Base models directory
    base_dir = os.path.join(Config.BASE_DIR, 'backend', 'data', 'models', 'ensemble')
    
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

def save_model_metadata(folder_path, metadata):
    """Save model metadata to a JSON file in the specified folder"""
    metadata_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return metadata_path

def load_model_metadata(model_id, model_type):
    """
    Load metadata for a model.
    Handles both new folder structure and legacy paths.
    
    Args:
        model_id: ID of the model
        model_type: Type of model (cnn, rf, ensemble)
        
    Returns:
        dict: Metadata dictionary or None if not found
    """
    # Base models directory
    base_models_dir = os.path.join(Config.BASE_DIR, 'backend', 'data', 'models')
    
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
    
    # Fallback to legacy structure
    legacy_path = os.path.join(base_models_dir, model_type, f"{model_id}_metadata.json")
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata from {legacy_path}: {str(e)}")
    
    return None

def get_models_registry_path():
    """Get the path to the models.json registry file"""
    # Check for registry in backend path
    backend_path = os.path.join(Config.BASE_DIR, 'backend', 'data', 'models', 'models.json')
    if os.path.exists(backend_path):
        return backend_path
    
    # Fallback to non-backend path
    return os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')
