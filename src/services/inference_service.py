import os
import logging
import numpy as np
import joblib
import time
from threading import Thread

from src.core.models import create_model
from src.core.audio.processor import AudioProcessor
from config import Config

class InferenceService:
    """
    Service for performing inference with trained models.
    This service handles audio preprocessing, feature extraction,
    and prediction using trained models.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the inference service.
        
        Args:
            model_dir (str): Directory containing trained models
        """
        self.model_dir = model_dir
        self.audio_processor = AudioProcessor(sample_rate=16000)  # Consistent sample rate
        self.loaded_models = {}
        self.inference_stats = {}
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
            # Create the model object
            model = create_model(model_type, model_dir=self.model_dir)
            
            # Load the saved model
            model_path = os.path.join(self.model_dir, model_name)
            
            # Add extension if not provided
            if model_type == 'cnn' and not model_name.endswith('.h5'):
                model_path += '.h5'
            elif model_type == 'rf' and not model_name.endswith('.joblib'):
                model_path += '.joblib'
            
            model.load(model_path)
            
            # Store in cache
            self.loaded_models[model_key] = model
            logging.info(f"Successfully loaded model {model_key}")
            
            return True
        except Exception as e:
            logging.error(f"Error loading model {model_name}_{model_type}: {e}")
            return False
    
    def unload_model(self, model_type, model_name):
        """
        Unload a model from memory.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', or 'ensemble')
            model_name (str): Name of the model file
            
        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        model_key = f"{model_name}_{model_type}"
        
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            logging.info(f"Unloaded model {model_key}")
            return True
        else:
            logging.warning(f"Model {model_key} was not loaded")
            return False
    
    def predict(self, audio_file, model_type, model_name, top_n=3):
        """
        Make a prediction using a loaded model.
        
        Args:
            audio_file (str): Path to the audio file
            model_type (str): Type of model ('cnn', 'rf', or 'ensemble')
            model_name (str): Name of the model file
            top_n (int): Number of top predictions to return
            
        Returns:
            dict: Prediction results
        """
        model_key = f"{model_name}_{model_type}"
        
        # Load model if not already loaded
        if model_key not in self.loaded_models:
            success = self.load_model(model_type, model_name)
            if not success:
                return {
                    'success': False,
                    'error': f"Failed to load model {model_key}"
                }
        
        model = self.loaded_models[model_key]
        
        try:
            start_time = time.time()
            
            # Process audio based on model type
            if model_type == 'cnn':
                features = self.audio_processor.process_audio_for_cnn(audio_file)
                # Ensure features is a 3D array (batch size, height, width)
                features = np.expand_dims(features, axis=0)
            elif model_type == 'rf':
                features_dict = self.audio_processor.process_audio_for_rf(audio_file)
                # Get feature names in correct order
                feature_names = self.audio_processor.get_feature_names()
                # Convert dict to vector
                features = np.array([[features_dict[name] for name in feature_names]])
            elif model_type == 'ensemble':
                # For ensemble, we need both CNN and RF features
                cnn_features = self.audio_processor.process_audio_for_cnn(audio_file)
                cnn_features = np.expand_dims(cnn_features, axis=0)
                
                rf_features_dict = self.audio_processor.process_audio_for_rf(audio_file)
                feature_names = self.audio_processor.get_feature_names()
                rf_features = np.array([[rf_features_dict[name] for name in feature_names]])
                
                features = {
                    'cnn': cnn_features,
                    'rf': rf_features
                }
            else:
                return {
                    'success': False,
                    'error': f"Unknown model type: {model_type}"
                }
            
            # Make the prediction
            prediction = model.predict(features)
            
            # Get top N predictions
            if hasattr(model, 'get_top_predictions'):
                top_predictions = model.get_top_predictions(features, top_n)
            else:
                # Get class indices and probabilities
                class_indices = np.argsort(prediction['probabilities'][0])[::-1][:top_n]
                top_probabilities = prediction['probabilities'][0][class_indices]
                
                # Map indices to class names
                class_names = model.get_class_names()
                top_class_names = [class_names[idx] for idx in class_indices]
                
                top_predictions = [
                    {
                        'class': class_name,
                        'probability': float(prob)
                    }
                    for class_name, prob in zip(top_class_names, top_probabilities)
                ]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store prediction for later reference
            self.last_prediction = {
                'model': model_key,
                'predicted_class': prediction['class'],
                'probability': float(prediction['probability']),
                'top_predictions': top_predictions,
                'processing_time': processing_time
            }
            
            # Store inference stats
            self.inference_stats = {
                'model': model_key,
                'model_type': model_type,
                'processing_time': processing_time,
                'audio_file': audio_file
            }
            
            return {
                'success': True,
                'predicted_class': prediction['class'],
                'probability': float(prediction['probability']),
                'top_predictions': top_predictions,
                'processing_time': processing_time
            }
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def batch_predict(self, audio_files, model_type, model_name):
        """
        Make predictions on multiple audio files.
        
        Args:
            audio_files (list): List of audio file paths
            model_type (str): Type of model ('cnn', 'rf', or 'ensemble')
            model_name (str): Name of the model file
            
        Returns:
            dict: Batch prediction results
        """
        results = []
        
        for audio_file in audio_files:
            result = self.predict(audio_file, model_type, model_name)
            results.append({
                'file': audio_file,
                'result': result
            })
        
        return {
            'success': True,
            'results': results
        }
    
    def predict_async(self, audio_file, model_type, model_name, callback=None, top_n=3):
        """
        Make a prediction asynchronously in a separate thread.
        
        Args:
            audio_file (str): Path to the audio file
            model_type (str): Type of model ('cnn', 'rf', or 'ensemble')
            model_name (str): Name of the model file
            callback (callable): Function to call when prediction is complete
            top_n (int): Number of top predictions to return
            
        Returns:
            bool: True if prediction started, False otherwise
        """
        def predict_thread_fn():
            result = self.predict(audio_file, model_type, model_name, top_n=top_n)
            
            if callback:
                callback(result)
        
        thread = Thread(target=predict_thread_fn)
        thread.daemon = True
        thread.start()
        
        return True
    
    def get_inference_stats(self):
        """
        Get statistics from the last inference.
        
        Returns:
            dict: Inference statistics
        """
        return self.inference_stats
    
    def get_last_prediction(self):
        """
        Get the result of the last prediction.
        
        Returns:
            dict: Last prediction result
        """
        return self.last_prediction
    
    def get_available_models(self):
        """
        Get a list of available models in the model directory.
        
        Returns:
            dict: Available models by type
        """
        available_models = {
            'cnn': [],
            'rf': [],
            'ensemble': []
        }
        
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                return available_models
            
            # List files in model directory
            for filename in os.listdir(self.model_dir):
                filepath = os.path.join(self.model_dir, filename)
                
                # Skip directories
                if os.path.isdir(filepath):
                    continue
                
                # Determine model type based on extension
                if filename.endswith('.h5'):
                    model_name = filename[:-3]  # Remove extension
                    available_models['cnn'].append(model_name)
                elif filename.endswith('.joblib'):
                    model_name = filename[:-7]  # Remove extension
                    available_models['rf'].append(model_name)
                elif os.path.isdir(filepath) and any(f.endswith('.h5') or f.endswith('.joblib') for f in os.listdir(filepath)):
                    # This is likely an ensemble model directory
                    available_models['ensemble'].append(os.path.basename(filepath))
            
            return available_models
        
        except Exception as e:
            logging.error(f"Error getting available models: {e}")
            return available_models
    
    def analyze_audio(self, audio_file):
        """
        Analyze an audio file without making a prediction.
        This provides information about audio characteristics.
        
        Args:
            audio_file (str): Path to the audio file
            
        Returns:
            dict: Audio analysis results
        """
        try:
            # Load the audio
            y, sr = self.audio_processor.load_audio(audio_file)
            
            # Check if sound is detected
            has_sound = self.audio_processor.detect_sound(y)
            
            # Get audio features
            features = self.audio_processor.process_audio_for_rf(audio_file, return_dict=True)
            
            # Calculate basic audio statistics
            duration = len(y) / sr
            max_amplitude = np.max(np.abs(y))
            mean_amplitude = np.mean(np.abs(y))
            
            # Get sound boundaries if sound is detected
            start_idx, end_idx = (0, 0)
            sound_duration = 0
            
            if has_sound:
                start_idx, end_idx = self.audio_processor.detect_sound_boundaries(y)
                sound_duration = (end_idx - start_idx) / sr
            
            return {
                'success': True,
                'has_sound': has_sound,
                'duration': duration,
                'sound_duration': sound_duration,
                'max_amplitude': float(max_amplitude),
                'mean_amplitude': float(mean_amplitude),
                'start_time': float(start_idx / sr),
                'end_time': float(end_idx / sr),
                'features': features
            }
            
        except Exception as e:
            logging.error(f"Error analyzing audio: {e}")
            return {
                'success': False,
                'error': str(e)
            } 