import os
import logging
import numpy as np
import time
import json
from collections import defaultdict

from src.core.ml_algorithms import create_model
from src.services.inference_analysis_service import InferenceAnalysisService
from src.core.audio.processor import AudioProcessor

class InferenceService:
    """
    Service for performing inference with trained models.
    Handles model loading, caching, and prediction.
    """
    
    def __init__(self, model_dir='data/models'):
        """
        Initialize the InferenceService.
        
        Args:
            model_dir (str): Directory containing model files
        """
        self.model_dir = model_dir
        self.loaded_models = {}  # Cache for loaded models
        # Initialize analysis service
        self.analysis_service = InferenceAnalysisService(max_history=100)
        # Initialize audio processor for consistent preprocessing
        self.audio_processor = AudioProcessor(
            sample_rate=8000,
            enable_loudness_normalization=False
        )
        self.last_prediction = None
    
    def load_model(self, model_type, model_name):
        """
        Load a trained model into memory.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', or 'ensemble')
            model_name (str): Name of the model file (without extension)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Create a key for the model cache
        model_key = f"{model_name}_{model_type}"
        
        # Check if model is already loaded
        if model_key in self.loaded_models:
            logging.info(f"Model {model_key} is already loaded")
            return True
        
        try:
            # Load models.json to find the model's specific directory
            models_json_path = os.path.join(self.model_dir, "models.json")
            if os.path.exists(models_json_path):
                with open(models_json_path, 'r') as f:
                    models_data = json.load(f)
                
                # Find the model in models.json
                model_id = None
                model_file_path = None
                
                # Normalize model name for comparison (model names may include _cnn suffix)
                normalized_name = model_name.replace("_" + model_type, "")
                
                # Look for the model in the models data
                for model_id, model_info in models_data.get("models", {}).get(model_type, {}).items():
                    # Check if this is the model we're looking for
                    model_id_base = model_id.replace("_" + model_type, "")
                    if model_id_base == normalized_name or model_id == model_name or model_id == model_name + "." + model_type:
                        model_file_path = model_info.get("file_path")
                        break
                
                if model_file_path:
                    # New structure: models are in their own directories
                    # model_file_path will be something like "cnn/EhOh_cnn_20250301113350/model.h5"
                    model_dir = os.path.dirname(os.path.join(self.model_dir, model_file_path))
                    
                    # Create the model with the specific model directory
                    model = create_model(model_type, model_dir=model_dir)
                    
                    # The filename is now standardized as model.h5 or model.joblib
                    filename = "model.h5" if model_type == "cnn" else "model.joblib"
                    
                    # Load the model
                    if model.load(filename):
                        # Store in cache
                        self.loaded_models[model_key] = model
                        logging.info(f"Successfully loaded model {model_key} from {model_dir}")
                        return True
            
            # Fall back to old method if models.json loading failed
            # Create the model object with the general models directory
            model = create_model(model_type, model_dir=self.model_dir)
            
            # For backward compatibility, try different file paths
            # 1. Try the new nested structure first
            model_path = os.path.join(self.model_dir, model_type, model_name)
            if not os.path.exists(model_path):
                # 2. Try with extension
                if model_type == 'cnn':
                    model_path = os.path.join(self.model_dir, model_type, model_name + '.h5')
                elif model_type == 'rf':
                    model_path = os.path.join(self.model_dir, model_type, model_name + '.joblib')
            
            # Attempt to load the model
            logging.info(f"Trying to load model from: {model_path}")
            model.load(model_path)
            
            # Store in cache
            self.loaded_models[model_key] = model
            logging.info(f"Successfully loaded model {model_key}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model {model_name} ({model_type}): {e}")
            return False
    
    def predict(self, features, model_type='cnn', model_name=None):
        """
        Make a prediction using the specified model.
        
        Args:
            features: Input features for prediction
            model_type (str): Type of model to use
            model_name (str): Name of the model to use
            
        Returns:
            dict: Prediction results with class probabilities
        """
        if model_name is None:
            # Use the default model if none specified
            logging.warning("No model specified, using default model")
            # TODO: Implement default model selection logic
            return None
        
        # Load the model if not already loaded
        model_key = f"{model_name}_{model_type}"
        if model_key not in self.loaded_models:
            if not self.load_model(model_type, model_name):
                logging.error(f"Failed to load model {model_key}")
                return None
        
        model = self.loaded_models[model_key]
        
        # Get class names
        class_names = model.get_class_names()
        
        # Track prediction time
        start_time = time.time()
        
        # Make prediction
        try:
            prediction = model.predict(features)
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return None
        
        end_time = time.time()
        
        # Calculate prediction time
        prediction_time = end_time - start_time
        
        # Create result dictionary
        result = {
            'prediction': prediction,
            'model_type': model_type,
            'model_name': model_name,
            'timestamp': time.time(),
            'prediction_time': prediction_time
        }
        
        # Add class information if available
        predicted_class = np.argmax(prediction)
        predicted_label = None
        
        if class_names and predicted_class < len(class_names):
            predicted_label = class_names[predicted_class]
        else:
            predicted_label = f"Class {predicted_class}"
        
        # Add to result
        result['predicted_class'] = predicted_class
        result['predicted_label'] = predicted_label
        result['confidence'] = float(np.max(prediction))
        
        # Record prediction in analysis service
        self.analysis_service.record_prediction({
            'predicted_class': predicted_label,
            'confidence': float(np.max(prediction)),
            'response_time': prediction_time,
            'model_type': model_type,
            'model_name': model_name
        })
        
        # Save as last prediction
        self.last_prediction = result
        
        return result
    
    def get_inference_stats(self):
        """
        Get statistics about predictions made.
        
        Returns:
            dict: Prediction statistics
        """
        return self.analysis_service.get_inference_stats()
    
    def get_recent_predictions(self, count=10):
        """
        Get the most recent predictions.
        
        Args:
            count (int): Number of predictions to return
            
        Returns:
            list: List of recent predictions
        """
        return self.analysis_service.get_recent_predictions(count)
    
    def get_class_accuracy(self, class_name):
        """
        Get the accuracy for a specific class.
        
        Args:
            class_name (str): Name of the class
            
        Returns:
            float: Accuracy for the specified class
        """
        return self.analysis_service.get_class_accuracy(class_name)
    
    def detect_drift(self, window_size=100):
        """
        Detect potential model drift based on recent accuracy.
        
        Args:
            window_size (int): Number of recent predictions to analyze
            
        Returns:
            dict: Drift analysis results
        """
        return self.analysis_service.detect_drift(window_size)
    
    def record_feedback(self, feedback_data):
        """
        Record user feedback about a prediction.
        
        Args:
            feedback_data (dict): Dictionary containing feedback details
                Required keys: 'predicted_class', 'actual_class', 'is_correct'
                
        Returns:
            dict: Updated inference statistics
        """
        return self.analysis_service.record_feedback(feedback_data)
    
    def get_confidence_distribution(self):
        """
        Calculate confidence distribution for predictions.
        
        Returns:
            dict: Confidence distribution statistics
        """
        return self.analysis_service.get_confidence_distribution()
    
    def reset_stats(self):
        """
        Reset all inference statistics.
        
        Returns:
            dict: Fresh inference statistics
        """
        return self.analysis_service.reset_stats()
    
    def clear_cache(self):
        """
        Clear the model cache to free memory.
        
        Returns:
            int: Number of models unloaded
        """
        count = len(self.loaded_models)
        self.loaded_models = {}
        logging.info(f"Cleared {count} models from cache")
        return count
    
    def get_last_prediction(self):
        """
        Get the most recent prediction.
        
        Returns:
            dict: Last prediction result
        """
        return self.last_prediction 