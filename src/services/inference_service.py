import os
import logging
import numpy as np
import joblib
import time
from threading import Thread
import json

from src.core.ml_algorithms import create_model
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
            # Get the base directory using absolute paths
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            logging.info(f"Base directory for model loading: {base_dir}")
            
            # First, check if this model is in the models.json registry
            models_json_path = os.path.join(base_dir, 'data', 'models', 'models.json')
            model_path = None
            
            if os.path.exists(models_json_path):
                try:
                    with open(models_json_path, 'r') as f:
                        models_registry = json.load(f)
                    
                    # Find the model in the registry
                    if 'models' in models_registry and model_type in models_registry['models'] and model_name in models_registry['models'][model_type]:
                        model_data = models_registry['models'][model_type][model_name]
                        
                        # Get the file path from the registry
                        if 'file_path' in model_data:
                            file_path = model_data['file_path']
                            # Construct absolute path
                            model_path = os.path.join(base_dir, 'data', 'models', file_path)
                            logging.info(f"Found model path in registry: {model_path}")
                except Exception as e:
                    logging.warning(f"Error reading models.json: {e}. Will fall back to traditional path construction.")
            
            # If no model path found in registry, try traditional path construction
            if not model_path:
                logging.info(f"Model {model_name} not found in registry, trying traditional path construction")
                
                # Create the model object
                model = create_model(model_type, model_dir=self.model_dir)
                
                # Try standard model path
                model_path = os.path.join(self.model_dir, model_name)
                
                # Add extension if not provided
                if model_type == 'cnn' and not model_name.endswith('.h5'):
                    model_path += '.h5'
                elif model_type == 'rf' and not model_name.endswith('.joblib'):
                    model_path += '.joblib'
                
                logging.info(f"Using traditional model path: {model_path}")
                
                # If model file doesn't exist at traditional path, try the new directory structure
                if not os.path.exists(model_path) and model_type == 'cnn':
                    # Try the new subdirectory structure: data/models/cnn/model_name/model_name.h5
                    new_model_path = os.path.join(base_dir, 'data', 'models', model_type, model_name, f"{model_name}.h5")
                    logging.info(f"Traditional path not found, trying new directory structure: {new_model_path}")
                    
                    if os.path.exists(new_model_path):
                        model_path = new_model_path
            
            # Create the model object
            model = create_model(model_type, model_dir=self.model_dir)
            
            # Load the saved model
            logging.info(f"Loading model from: {model_path}")
            model.load(model_path)
            
            # Store in cache
            self.loaded_models[model_key] = model
            logging.info(f"Successfully loaded model {model_key}")
            
            return True
        except Exception as e:
            logging.error(f"Error loading model {model_name}_{model_type}: {e}", exc_info=True)
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
        
        # Print the starting model directory for debugging
        print(f"Current self.model_dir: {self.model_dir}")
        print(f"Absolute path: {os.path.abspath(self.model_dir)}")
        print(f"Parent directory: {os.path.dirname(os.path.abspath(self.model_dir))}")
        
        try:
            # First try to read models from the models.json registry
            # Use absolute paths to avoid any issues with relative paths
            # Go up THREE levels: first from __file__ to services, then from services to src, then from src to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            print(f"Project root directory: {base_dir}")
            models_json_path = os.path.join(base_dir, 'data', 'models', 'models.json')
            logging.info(f"Looking for models.json at absolute path: {models_json_path}")
            
            # Check if models.json exists
            if os.path.exists(models_json_path) and os.path.isfile(models_json_path):
                try:
                    logging.info(f"Reading models from registry file: {models_json_path}")
                    with open(models_json_path, 'r') as f:
                        models_registry = json.load(f)
                    
                    # Log the structure of models_registry to debug
                    logging.info(f"Models registry structure: {list(models_registry.keys())}")
                    
                    # Extract models from the registry
                    if 'models' in models_registry:
                        logging.info(f"Model types in registry: {list(models_registry['models'].keys())}")
                        
                        if 'cnn' in models_registry['models']:
                            cnn_models = models_registry['models']['cnn']
                            logging.info(f"Found {len(cnn_models)} CNN models in registry: {list(cnn_models.keys())}")
                            
                            # Extract model names
                            for model_id, model_data in cnn_models.items():
                                # Check if file exists before adding
                                file_path = model_data.get('file_path', '')
                                full_path = os.path.join(base_dir, 'data', 'models', file_path)
                                logging.info(f"Checking model file: {full_path}")
                                
                                if os.path.exists(full_path):
                                    logging.info(f"Adding model {model_id} with verified path: {full_path}")
                                    available_models['cnn'].append(model_id)
                                else:
                                    logging.warning(f"Model file not found at {full_path}, skipping")
                        
                        # Handle RF and ensemble models similarly if needed
                    
                    # If we found models in the registry, return them
                    if any(len(models) > 0 for models in available_models.values()):
                        logging.info(f"Returning {sum(len(models) for models in available_models.values())} models from registry")
                        return available_models
                    else:
                        logging.warning("No models found in registry that match existing files")
                    
                except Exception as e:
                    logging.error(f"Error reading models.json: {e}", exc_info=True)
            else:
                logging.warning(f"Models registry file not found at {models_json_path}")
            
            # Fall back to direct directory scanning
            logging.info(f"No models found in registry or registry not available, scanning directories")
            
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                logging.warning(f"Original model directory {self.model_dir} does not exist")
                
                # Try the new data/models structure with absolute path
                new_models_dir = os.path.join(base_dir, 'data', 'models')
                logging.info(f"Checking models directory at absolute path: {new_models_dir}")
                
                if os.path.exists(new_models_dir):
                    # Check CNN models directory
                    cnn_dir = os.path.join(new_models_dir, 'cnn')
                    logging.info(f"Checking CNN models directory: {cnn_dir}")
                    
                    if os.path.exists(cnn_dir) and os.path.isdir(cnn_dir):
                        # List all subdirectories (each should be a model)
                        model_dirs = os.listdir(cnn_dir)
                        logging.info(f"Found {len(model_dirs)} potential model directories in {cnn_dir}")
                        
                        for model_dir in model_dirs:
                            model_path = os.path.join(cnn_dir, model_dir)
                            if os.path.isdir(model_path):
                                # Look for h5 files in this directory
                                h5_files = [f for f in os.listdir(model_path) if f.endswith('.h5')]
                                if h5_files:
                                    logging.info(f"Found CNN model in directory: {model_dir} with file: {h5_files[0]}")
                                    available_models['cnn'].append(model_dir)
                                else:
                                    logging.warning(f"No h5 files found in directory: {model_path}")
                    else:
                        logging.warning(f"CNN models directory not found: {cnn_dir}")
                else:
                    logging.warning(f"New models directory not found: {new_models_dir}")
                
                return available_models
            
            # List files in model directory (original method)
            for filename in os.listdir(self.model_dir):
                filepath = os.path.join(self.model_dir, filename)
                
                # Skip directories
                if os.path.isdir(filepath):
                    continue
                
                # Determine model type based on extension
                if filename.endswith('.h5'):
                    model_name = filename[:-3]  # Remove extension
                    available_models['cnn'].append(model_name)
                    logging.info(f"Found CNN model file: {filename}")
                elif filename.endswith('.joblib'):
                    model_name = filename[:-7]  # Remove extension
                    available_models['rf'].append(model_name)
                    logging.info(f"Found RF model file: {filename}")
                elif os.path.isdir(filepath) and any(f.endswith('.h5') or f.endswith('.joblib') for f in os.listdir(filepath)):
                    # This is likely an ensemble model directory
                    available_models['ensemble'].append(os.path.basename(filepath))
                    logging.info(f"Found ensemble model directory: {filename}")
            
            return available_models
        
        except Exception as e:
            logging.error(f"Error getting available models: {e}", exc_info=True)
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
    
    def debug_models_json(self):
        """
        Debug helper to directly check models.json and model files.
        Outputs detailed information about file paths, existence, and content.
        """
        print("\n========= DEBUG MODELS JSON =========")
        
        # Get the base directory using absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(f"Project root directory: {base_dir}")
        print(f"Current model_dir: {self.model_dir}")
        print(f"Absolute model_dir: {os.path.abspath(self.model_dir)}")
        
        # Check models.json path and existence
        models_json_path = os.path.join(base_dir, 'data', 'models', 'models.json')
        print(f"Models JSON path: {models_json_path}")
        print(f"Path exists: {os.path.exists(models_json_path)}")
        print(f"Is file: {os.path.isfile(models_json_path) if os.path.exists(models_json_path) else 'N/A'}")
        
        # If the file exists, read and parse it
        if os.path.exists(models_json_path) and os.path.isfile(models_json_path):
            try:
                with open(models_json_path, 'r') as f:
                    file_content = f.read()
                    print(f"File size: {len(file_content)} bytes")
                    
                    # Try to parse as JSON
                    models_registry = json.loads(file_content)
                    print(f"Successfully parsed JSON with keys: {list(models_registry.keys())}")
                    
                    # Check if it has models key
                    if 'models' in models_registry:
                        print(f"Models types: {list(models_registry['models'].keys())}")
                        
                        # Check CNN models
                        if 'cnn' in models_registry['models']:
                            cnn_models = models_registry['models']['cnn']
                            print(f"Found {len(cnn_models)} CNN models: {list(cnn_models.keys())}")
                            
                            # Check a sample model's file path
                            if cnn_models:
                                sample_model_id = list(cnn_models.keys())[0]
                                sample_model = cnn_models[sample_model_id]
                                print(f"Sample model: {sample_model_id}")
                                print(f"Sample model data: {sample_model}")
                                
                                # Check file path
                                file_path = sample_model.get('file_path', '')
                                full_path = os.path.join(base_dir, 'data', 'models', file_path)
                                print(f"Full path: {full_path}")
                                print(f"Path exists: {os.path.exists(full_path)}")
                                
                                # Check the directory structure
                                dir_path = os.path.dirname(full_path)
                                print(f"Directory path: {dir_path}")
                                print(f"Directory exists: {os.path.exists(dir_path)}")
                                if os.path.exists(dir_path):
                                    print(f"Directory contents: {os.listdir(dir_path)}")
                    else:
                        print("No 'models' key in JSON")
            except Exception as e:
                print(f"Error reading models.json: {e}")
        
        # Check model directories directly
        new_models_dir = os.path.join(base_dir, 'data', 'models')
        print(f"\nModels directory: {new_models_dir}")
        print(f"Directory exists: {os.path.exists(new_models_dir)}")
        
        if os.path.exists(new_models_dir):
            print(f"Directory contents: {os.listdir(new_models_dir)}")
            
            # Check CNN directory
            cnn_dir = os.path.join(new_models_dir, 'cnn')
            print(f"\nCNN directory: {cnn_dir}")
            print(f"Directory exists: {os.path.exists(cnn_dir)}")
            
            if os.path.exists(cnn_dir):
                cnn_contents = os.listdir(cnn_dir)
                print(f"CNN directory contains {len(cnn_contents)} items")
                
                # Check a few model directories
                sample_count = min(3, len(cnn_contents))
                for i in range(sample_count):
                    model_dir = os.path.join(cnn_dir, cnn_contents[i])
                    if os.path.isdir(model_dir):
                        print(f"\nSample model directory {i+1}: {model_dir}")
                        print(f"Directory contents: {os.listdir(model_dir)}")
                        
                        # Check for h5 files
                        h5_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
                        print(f"H5 files: {h5_files}")
                        
                        # Check for metadata files
                        metadata_files = [f for f in os.listdir(model_dir) if f.endswith('_metadata.json')]
                        print(f"Metadata files: {metadata_files}")
        
        print("========= END DEBUG =========\n") 