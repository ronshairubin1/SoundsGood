"""
Core model definitions and utilities for model loading/creation.
"""

import os
import json
import logging
import tensorflow as tf
import joblib
import numpy as np

class ModelBase:
    """Base class for all models."""
    
    def __init__(self, model_dir='models'):
        """
        Initialize the model base.
        
        Args:
            model_dir (str): Directory for model storage
        """
        self.model_dir = model_dir
        self.model = None
        self.metadata = {}
        
    def load(self, model_path, metadata=None):
        """
        Load a model and its metadata.
        
        Args:
            model_path (str): Path to the model file
            metadata (dict, optional): Metadata to use directly
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if metadata:
                self.metadata = metadata
            else:
                # Try to load metadata file
                metadata_path = model_path.rstrip('.h5').rstrip('.joblib') + '_metadata.json'
                logging.info(f"Looking for metadata at: {metadata_path}")
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                        logging.info(f"Loaded metadata from {metadata_path}: {self.metadata}")
                else:
                    logging.warning(f"No metadata file found at {metadata_path}")
            
            # Normalize class names for consistency
            # Convert 'class_names' or 'classes' to 'sound_classes' 
            if 'class_names' in self.metadata and 'sound_classes' not in self.metadata:
                self.metadata['sound_classes'] = self.metadata['class_names']
                logging.info(f"Normalized 'class_names' to 'sound_classes': {self.metadata['sound_classes']}")
            elif 'classes' in self.metadata and 'sound_classes' not in self.metadata:
                self.metadata['sound_classes'] = self.metadata['classes']
                logging.info(f"Normalized 'classes' to 'sound_classes': {self.metadata['sound_classes']}")
            
            # Ensure input_shape is present
            if 'input_shape' not in self.metadata and self.__class__.__name__ == 'CNNModel':
                logging.warning("No input_shape in metadata, will try to infer from model after loading")
            
            # Actual model loading implemented in subclasses
            success = self._load_model(model_path)
            
            # Try to infer input shape if not in metadata
            if success and 'input_shape' not in self.metadata and self.__class__.__name__ == 'CNNModel':
                if hasattr(self.model, 'input_shape'):
                    self.metadata['input_shape'] = self.model.input_shape[1:]  # Remove batch dimension
                    logging.info(f"Inferred input_shape from model: {self.metadata['input_shape']}")
                
            return success
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            return False
    
    def save(self, path, metadata=None):
        """
        Save model and metadata.
        
        Args:
            path (str): Path to save the model
            metadata (dict, optional): Additional metadata to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if metadata:
                self.metadata.update(metadata)
            
            # Save metadata to JSON file
            metadata_path = path.rstrip('.h5').rstrip('.joblib') + '_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)
                logging.info(f"Saved metadata to {metadata_path}")
            
            # Actual model saving implemented in subclasses
            return self._save_model(path)
        except Exception as e:
            logging.error(f"Error saving model: {e}", exc_info=True)
            return False
    
    def _load_model(self, path):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_model")
    
    def _save_model(self, path):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _save_model")


class CNNModel(ModelBase):
    """CNN model implementation."""
    
    def __init__(self, model_dir='models'):
        super().__init__(model_dir=model_dir)
    
    def _load_model(self, path):
        try:
            logging.info(f"Loading CNN model from path: {path}")
            self.model = tf.keras.models.load_model(path)
            
            # Special handling for EhOh models
            if 'EhOh' in path:
                logging.info(f"Detected EhOh model: {path}")
                
                # If this is an EhOh model and we have class_names but no sound_classes
                if 'class_names' in self.metadata and 'sound_classes' not in self.metadata:
                    self.metadata['sound_classes'] = self.metadata['class_names']
                    logging.info(f"Added sound_classes from class_names for EhOh model: {self.metadata['sound_classes']}")
                
                # If we have num_classes but no input_shape
                if 'num_classes' in self.metadata and 'input_shape' not in self.metadata:
                    try:
                        # Try to infer input shape from model
                        if hasattr(self.model, 'input_shape'):
                            self.metadata['input_shape'] = self.model.input_shape[1:]  # Remove batch dimension
                            logging.info(f"Inferred input_shape for EhOh model: {self.metadata['input_shape']}")
                    except Exception as e:
                        logging.warning(f"Could not infer input_shape for EhOh model: {e}")
                
                # Add default preprocessing params if not present
                if 'preprocessing_params' not in self.metadata:
                    self.metadata['preprocessing_params'] = {
                        "sample_rate": 22050,
                        "n_mels": 128,
                        "n_fft": 2048,
                        "hop_length": 512,
                        "sound_threshold": 0.01,
                        "min_silence_duration": 0.5,
                        "trim_silence": True,
                        "normalize_audio": True
                    }
                    logging.info(f"Added default preprocessing_params for EhOh model")
            
            logging.info(f"Successfully loaded CNN model from {path}")
            return True
        except Exception as e:
            logging.error(f"Error loading CNN model: {e}", exc_info=True)
            return False
    
    def _save_model(self, path):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logging.info(f"Saved CNN model to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving CNN model: {e}", exc_info=True)
            return False
    
    def predict(self, X):
        """
        Make a prediction with the CNN model.
        
        Args:
            X: Input features, shape (batch_size, height, width, channels)
            
        Returns:
            Predictions from the model
        """
        return self.model.predict(X)


class RFModel(ModelBase):
    """Random Forest model implementation."""
    
    def __init__(self, model_dir='models'):
        super().__init__(model_dir=model_dir)
    
    def _load_model(self, path):
        try:
            self.model = joblib.load(path)
            logging.info(f"Loaded RF model from {path}")
            return True
        except Exception as e:
            logging.error(f"Error loading RF model: {e}", exc_info=True)
            return False
    
    def _save_model(self, path):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logging.info(f"Saved RF model to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving RF model: {e}", exc_info=True)
            return False
    
    def predict(self, X):
        """
        Make a prediction with the RF model.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        try:
            # Get class labels
            predicted_probs = self.model.predict_proba(X)
            predicted_classes = [self.model.classes_[np.argmax(probs)] for probs in predicted_probs]
            return predicted_classes, predicted_probs
        except Exception as e:
            logging.error(f"Error in RF prediction: {e}", exc_info=True)
            return None, None


class EnsembleModel(ModelBase):
    """Ensemble model implementation combining CNN and RF."""
    
    def __init__(self, model_dir='models'):
        super().__init__(model_dir=model_dir)
        self.rf_model = None
        self.cnn_model = None
        self.class_names = []
        self.rf_weight = 0.5
    
    def _load_model(self, path):
        try:
            # For ensemble, path should point to a directory containing both models
            if os.path.isdir(path):
                # Load RF model
                rf_path = os.path.join(path, 'rf_model.joblib')
                if os.path.exists(rf_path):
                    self.rf_model = joblib.load(rf_path)
                    logging.info(f"Loaded RF model from {rf_path}")
                
                # Load CNN model
                cnn_path = os.path.join(path, 'cnn_model.h5')
                if os.path.exists(cnn_path):
                    self.cnn_model = tf.keras.models.load_model(cnn_path)
                    logging.info(f"Loaded CNN model from {cnn_path}")
                
                if self.rf_model and self.cnn_model:
                    return True
                else:
                    logging.error("Ensemble requires both RF and CNN models")
                    return False
            else:
                logging.error(f"Expected directory for ensemble model: {path}")
                return False
        except Exception as e:
            logging.error(f"Error loading ensemble model: {e}", exc_info=True)
            return False
    
    def _save_model(self, path):
        try:
            # Ensure directory exists
            os.makedirs(path, exist_ok=True)
            
            # Save RF model
            if self.rf_model:
                rf_path = os.path.join(path, 'rf_model.joblib')
                joblib.dump(self.rf_model, rf_path)
                logging.info(f"Saved RF model to {rf_path}")
            
            # Save CNN model
            if self.cnn_model:
                cnn_path = os.path.join(path, 'cnn_model.h5')
                self.cnn_model.save(cnn_path)
                logging.info(f"Saved CNN model to {cnn_path}")
            
            return True
        except Exception as e:
            logging.error(f"Error saving ensemble model: {e}", exc_info=True)
            return False
    
    def predict(self, X_rf, X_cnn):
        """
        Make a prediction using both RF and CNN models.
        
        Args:
            X_rf: Input features for RF model
            X_cnn: Input features for CNN model
            
        Returns:
            List of (class, confidence) tuples
        """
        try:
            results = []
            
            # Get RF predictions
            if self.rf_model:
                rf_probs = self.rf_model.predict_proba(X_rf)
            else:
                rf_probs = None
            
            # Get CNN predictions
            if self.cnn_model:
                cnn_probs = self.cnn_model.predict(X_cnn)
            else:
                cnn_probs = None
            
            # Combine predictions
            if rf_probs is not None and cnn_probs is not None:
                # Weights for ensemble (RF vs CNN)
                for i in range(len(X_rf)):
                    # Combine probabilities
                    combined_probs = (1 - self.rf_weight) * cnn_probs[i] + self.rf_weight * rf_probs[i]
                    top_idx = np.argmax(combined_probs)
                    top_class = self.class_names[top_idx]
                    top_conf = combined_probs[top_idx]
                    results.append((top_class, top_conf))
            elif rf_probs is not None:
                # RF only
                for i in range(len(X_rf)):
                    top_idx = np.argmax(rf_probs[i])
                    top_class = self.class_names[top_idx]
                    top_conf = rf_probs[i][top_idx]
                    results.append((top_class, top_conf))
            elif cnn_probs is not None:
                # CNN only
                for i in range(len(X_cnn)):
                    top_idx = np.argmax(cnn_probs[i])
                    top_class = self.class_names[top_idx]
                    top_conf = cnn_probs[i][top_idx]
                    results.append((top_class, top_conf))
            
            return results
        except Exception as e:
            logging.error(f"Error in ensemble prediction: {e}", exc_info=True)
            return []


def create_model(model_type, model_dir='models'):
    """
    Factory function to create an appropriate model instance based on type.
    
    Args:
        model_type (str): Type of model ('cnn', 'rf', 'ensemble')
        model_dir (str): Directory for model storage
        
    Returns:
        ModelBase: An instance of the requested model type
    """
    from src.core.ml_algorithms import create_model
    return create_model(model_type, model_dir=model_dir) 