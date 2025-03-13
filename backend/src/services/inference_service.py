"""Service for performing inference with trained models.

This module provides the InferenceService class which handles loading models, making
predictions, and managing the model cache for ML models (CNN, RF, Ensemble).
"""

import json
import logging
import os
import time
import joblib
from threading import Thread

import numpy as np

# Use a consistent import approach to get the appropriate processor
# First try to import directly from backend (new structure)
_using_new_structure = False
try:
    from backend.audio.processor import SoundProcessor as AudioProcessor
    logging.info("Using SoundProcessor from backend.audio.processor")
    _using_new_structure = True
except ImportError:
    # If not found, use the bridge which will eventually delegate to SoundProcessor
    from backend.src.core.audio.processor_bridge import AudioProcessor
    logging.info("Using AudioProcessor bridge from backend.src.core.audio.processor_bridge")

from backend.src.core.ml_algorithms import create_model
from backend.src.ml.model_paths import (
    get_cnn_model_path,
    get_rf_model_path,
    get_ensemble_model_path,
    get_models_registry_path,
    synchronize_model_registry
)
from backend.config import Config

# Log which audio processor structure is being used
logging.getLogger(__name__).info(f"Using {'new' if _using_new_structure else 'legacy'} audio processor structure")

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
        # Keep the sample rate consistent with the configuration
        self.audio_processor = AudioProcessor(sample_rate=Config.SAMPLE_RATE)
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
            
            # Get the models.json registry path using helper function
            models_json_path = get_models_registry_path()
            logging.info(f"Using models registry at: {models_json_path}")
            model_path = None
            class_names = None
            model_metadata = None
            
            # Check if models.json exists and try to find the model there first
            if os.path.exists(models_json_path):
                try:
                    # Re-synchronize the registry first to ensure we're using the latest data
                    synchronize_model_registry()
                    logging.info("Synchronized model registry for fresh model data")
                    
                    with open(models_json_path, 'r') as f:
                        models_registry = json.load(f)
                    
                    # Find the model in the registry
                    if 'models' in models_registry and model_type in models_registry['models'] and model_name in models_registry['models'][model_type]:
                        model_data = models_registry['models'][model_type][model_name]
                        model_metadata = model_data
                        logging.info(f"Found model {model_name} in registry with data: {model_data}")
                        
                        # Get the file path from the registry
                        if 'file_path' in model_data:
                            file_path = model_data['file_path']
                            # Construct absolute path - models.json should have relative paths from models directory
                            model_path = os.path.join(base_dir, 'data', 'models', file_path)
                            logging.info(f"Found model path in registry: {model_path}")
                            
                            # If class names are available in the model metadata, use them
                            if 'class_names' in model_data:
                                class_names = model_data['class_names']
                                logging.info(f"Found class names in registry: {class_names}")
                except Exception as e:
                    logging.warning(f"Error reading models.json: {e}. Will fall back to traditional path construction.")
                    logging.exception(e)  # Log full traceback for debugging
            
            # If no model path found in registry, try traditional path construction
            if not model_path or not os.path.exists(model_path):
                logging.info(f"Model path not found in registry or doesn't exist, trying traditional paths")
                
                # Try the model directory structure based on model type
                models_dir = os.path.join(base_dir, 'data', 'models')
                model_type_dir = os.path.join(models_dir, model_type)
                
                # Get appropriate file extension based on model type
                if model_type == 'cnn':
                    ext = '.h5'
                elif model_type in ['rf', 'ensemble']:
                    ext = '.joblib'
                else:
                    ext = ''
                
                # First, check if the model exists in the backend/data/models directory (preferred location)
                backend_models_dir = os.path.join(base_dir, 'backend', 'data', 'models')
                backend_model_type_dir = os.path.join(backend_models_dir, model_type)
                
                # Try in backend model type directory (preferred structure)
                backend_path = os.path.join(backend_model_type_dir, f"{model_name}{ext}")
                if os.path.exists(backend_path):
                    model_path = backend_path
                    logging.info(f"Found model at backend path: {model_path}")
                else:
                    # Try in standard model directory (new structure)
                    potential_path = os.path.join(model_type_dir, f"{model_name}{ext}")
                    if os.path.exists(potential_path):
                        model_path = potential_path
                        logging.info(f"Found model at standard path: {model_path}")
                    else:
                        # Try nested directory structure (e.g., cnn/model_name/model_name.h5)
                        potential_path = os.path.join(model_type_dir, model_name, f"{model_name}{ext}")
                        if os.path.exists(potential_path):
                            model_path = potential_path
                            logging.info(f"Found model at nested path: {model_path}")
                        else:
                            # Try nested structure in backend (e.g., backend/data/models/cnn/model_name/model_name.h5)
                            backend_nested_path = os.path.join(backend_model_type_dir, model_name, f"{model_name}{ext}")
                            if os.path.exists(backend_nested_path):
                                model_path = backend_nested_path
                                logging.info(f"Found model at backend nested path: {model_path}")
                            else:
                                # Try older/legacy paths as last resort
                                legacy_path = os.path.join(self.model_dir, f"{model_name}{ext}")
                                if os.path.exists(legacy_path):
                                    model_path = legacy_path
                                    logging.info(f"Found model at legacy path: {model_path}")
                                elif model_type == 'rf' and os.path.exists(os.path.join(self.model_dir, f"{model_name}.pkl")):
                                    # Check legacy pickle format for RF models
                                    model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                                    logging.info(f"Found RF model in legacy .pkl format: {model_path}")
            
            # If we still can't find the model, log error and return False
            if not model_path or not os.path.exists(model_path):
                logging.error(f"Could not find model file for {model_type} model '{model_name}' after trying all possible paths")
                return False
            
            # Create the model object based on model type
            model = create_model(model_type, model_dir=self.model_dir)
            
            # Load the saved model with additional metadata if available
            logging.info(f"Loading model from: {model_path}")
            if not os.path.exists(model_path):
                logging.error(f"Model file does not exist at path: {model_path}")
                return False
                
            try:
                if class_names is not None:
                    model.load(model_path, class_names=class_names)
                else:
                    model.load(model_path)
                logging.info(f"Successfully loaded model file from {model_path}")
            except Exception as e:
                logging.error(f"Error loading model file from {model_path}: {str(e)}")
                return False
            
            # Store in cache along with metadata if available
            self.loaded_models[model_key] = model
            if model_metadata:
                # Also store metadata with the model for reference
                self.loaded_models[f"{model_key}_metadata"] = model_metadata
                
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
            dict: Prediction results including:
            - success: Whether prediction succeeded
            - predicted_class: Top predicted class name
            - probability: Confidence score for top prediction
            - top_predictions: List of top N predictions with class and probability
            - processing_time: Time taken for inference in seconds
            - model_metadata: Information about the model used (if available)
        """
        model_key = f"{model_name}_{model_type}"
        metadata_key = f"{model_key}_metadata"
        
        # Load model if not already loaded
        if model_key not in self.loaded_models:
            logging.info(f"Model {model_key} not loaded yet, loading now...")
            success = self.load_model(model_type, model_name)
            if not success:
                logging.error(f"Failed to load model {model_key}")
                return {
                    'success': False,
                    'error': f"Failed to load model {model_key}"
                }
        
        model = self.loaded_models[model_key]
        model_metadata = self.loaded_models.get(metadata_key, None)
        
        try:
            start_time = time.time()
            logging.info(f"Starting prediction with {model_key} on audio file: {audio_file}")
            
            # Process audio based on model type
            if model_type == 'cnn':
                features = self.audio_processor.process_audio_for_cnn(audio_file)
                # Ensure features is a 3D array (batch size, height, width)
                features = np.expand_dims(features, axis=0)
                logging.info(f"Processed CNN features with shape: {features.shape}")
            elif model_type == 'rf':
                features_dict = self.audio_processor.process_audio_for_rf(audio_file)
                # Get feature names in correct order
                feature_names = self.audio_processor.get_feature_names()
                # Convert dict to vector
                features = np.array([[features_dict[name] for name in feature_names]])
                logging.info(f"Processed RF features with shape: {features.shape}")
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
                logging.info(f"Processed ensemble features: CNN shape {cnn_features.shape}, RF shape {rf_features.shape}")
            else:
                logging.error(f"Unknown model type: {model_type}")
                return {
                    'success': False,
                    'error': f"Unknown model type: {model_type}"
                }
            
            # Make the prediction
            logging.info(f"Calling model.predict with processed features")
            prediction = model.predict(features)
            
            # Check if prediction has the expected format
            if not isinstance(prediction, dict) or 'class' not in prediction or 'probability' not in prediction:
                logging.warning(f"Unexpected prediction format: {prediction}")
                # Try to convert to expected format if possible
                if isinstance(prediction, np.ndarray):
                    # Probably raw probabilities, convert to proper format
                    class_names = model.get_class_names()
                    if class_names and len(class_names) == prediction.shape[1]:
                        class_idx = np.argmax(prediction[0])
                        prediction = {
                            'class': class_names[class_idx],
                            'probability': float(prediction[0][class_idx]),
                            'probabilities': prediction[0]
                        }
            
            # Get top N predictions
            if hasattr(model, 'get_top_predictions'):
                logging.info(f"Using model's get_top_predictions method")
                top_predictions = model.get_top_predictions(features, top_n)
            else:
                # Get class indices and probabilities
                class_names = model.get_class_names()
                logging.info(f"Available class names: {class_names}")
                
                if 'probabilities' in prediction and len(prediction['probabilities']) > 0:
                    # Ensure probabilities are in the expected format
                    if isinstance(prediction['probabilities'], np.ndarray):
                        probs = prediction['probabilities'][0] if prediction['probabilities'].ndim > 1 else prediction['probabilities']
                    else:
                        probs = prediction['probabilities']
                        
                    # Get indices of top N classes
                    class_indices = np.argsort(probs)[::-1][:top_n]
                    top_probabilities = [float(probs[idx]) for idx in class_indices]
                    
                    # Map indices to class names
                    top_class_names = [class_names[idx] for idx in class_indices]
                    
                    top_predictions = [
                        {
                            'class': class_name,
                            'probability': float(prob)
                        }
                        for class_name, prob in zip(top_class_names, top_probabilities)
                    ]
                else:
                    # Fallback if probabilities not available
                    top_predictions = [{
                        'class': prediction['class'],
                        'probability': float(prediction['probability'])
                    }]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logging.info(f"Prediction completed in {processing_time:.3f} seconds")
            
            # Prepare response with model metadata if available
            result = {
                'success': True,
                'predicted_class': prediction['class'],
                'probability': float(prediction['probability']),
                'top_predictions': top_predictions,
                'processing_time': processing_time
            }
            
            # Add model metadata if available
            if model_metadata:
                result['model_metadata'] = {
                    'model_id': model_name,
                    'model_type': model_type,
                    'dictionary_name': model_metadata.get('dictionary_name', 'unknown'),
                    'creation_date': model_metadata.get('creation_date', ''),
                    'class_count': len(class_names) if class_names else 0
                }
                # Add accuracy if available
                if 'accuracy' in model_metadata:
                    result['model_metadata']['accuracy'] = model_metadata['accuracy']
            
            # Store prediction for later reference
            self.last_prediction = result.copy()
            
            # Store inference stats
            self.inference_stats = {
                'model': model_key,
                'model_type': model_type,
                'processing_time': processing_time,
                'audio_file': os.path.basename(audio_file),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
        
        except Exception as e:
            logging.error(f"Error during prediction with {model_key}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'model': model_key,
                'audio_file': os.path.basename(audio_file)
            }
    
    def batch_predict(self, audio_files, model_type, model_name, top_n=3):
        """
        Make predictions on multiple audio files.
        
        Args:
            audio_files (list): List of audio file paths
            model_type (str): Type of model ('cnn', 'rf', or 'ensemble')
            model_name (str): Name of the model file
            top_n (int): Number of top predictions to return for each file
            
        Returns:
            dict: Batch prediction results including:
            - success: Whether batch processing succeeded
            - results: List of individual results for each file
            - summary: Summary statistics about the batch prediction
        """
        results = []
        successful_predictions = 0
        failed_predictions = 0
        total_processing_time = 0
        start_time = time.time()
        
        logging.info(f"Starting batch prediction for {len(audio_files)} files using {model_type} model '{model_name}'")
        
        try:
            # Load model once for all predictions if not already loaded
            model_key = f"{model_name}_{model_type}"
            if model_key not in self.loaded_models:
                logging.info(f"Loading model {model_key} for batch prediction")
                success = self.load_model(model_type, model_name)
                if not success:
                    logging.error(f"Failed to load model {model_key} for batch prediction")
                    return {
                        'success': False,
                        'error': f"Failed to load model {model_key}"
                    }
            
            # Process each audio file
            for audio_file in audio_files:
                try:
                    logging.info(f"Processing file: {os.path.basename(audio_file)}")
                    result = self.predict(audio_file, model_type, model_name, top_n=top_n)
                    
                    if result['success']:
                        successful_predictions += 1
                        total_processing_time += result['processing_time']
                    else:
                        failed_predictions += 1
                        
                    results.append({
                        'file': os.path.basename(audio_file),
                        'result': result
                    })
                except Exception as e:
                    logging.error(f"Error processing file {audio_file}: {e}")
                    failed_predictions += 1
                    results.append({
                        'file': os.path.basename(audio_file),
                        'result': {
                            'success': False,
                            'error': str(e)
                        }
                    })
            
            # Calculate total batch processing time
            batch_processing_time = time.time() - start_time
            
            # Prepare summary statistics
            summary = {
                'total_files': len(audio_files),
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'avg_processing_time': total_processing_time / max(successful_predictions, 1),
                'batch_processing_time': batch_processing_time,
                'model': model_key,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logging.info(f"Batch prediction completed. Processed {successful_predictions}/{len(audio_files)} files successfully in {batch_processing_time:.2f} seconds")
            
            return {
                'success': True,
                'results': results,
                'summary': summary
            }
        except Exception as e:
            logging.error(f"Error during batch prediction: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'results': results,
                'processed_files': successful_predictions + failed_predictions,
                'total_files': len(audio_files)
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
            dict: Information about the started prediction task including:
            - success: Whether the async prediction task started
            - task_id: A unique identifier for this prediction task
            - model: The model identifier being used
            - file: The audio file being processed
        """
        # Generate a unique task ID
        task_id = f"predict_{int(time.time())}_{hash(audio_file) % 10000}"
        
        try:
            # Check if the model is already loaded or can be loaded
            model_key = f"{model_name}_{model_type}"
            if model_key not in self.loaded_models:
                # Try loading the model before starting the thread to catch early errors
                if not self.load_model(model_type, model_name):
                    logging.error(f"Failed to load model {model_key} for async prediction")
                    return {
                        'success': False,
                        'error': f"Failed to load model {model_key}",
                        'task_id': task_id,
                        'model': model_key,
                        'file': os.path.basename(audio_file)
                    }
            
            # Log the start of the async prediction task
            logging.info(f"Starting async prediction task {task_id} with model {model_key} on file {os.path.basename(audio_file)}")
            
            def predict_thread_fn():
                try:
                    # Log the start of the thread execution
                    logging.info(f"Executing async prediction task {task_id}")
                    
                    # Execute the prediction
                    result = self.predict(audio_file, model_type, model_name, top_n=top_n)
                    
                    # Add task_id to the result
                    result['task_id'] = task_id
                    
                    # Log completion
                    if result['success']:
                        logging.info(f"Async prediction task {task_id} completed successfully: {result['predicted_class']} ({result['probability']:.2f})")
                    else:
                        logging.error(f"Async prediction task {task_id} failed: {result.get('error', 'Unknown error')}")
                    
                    # Call the callback if provided
                    if callback:
                        try:
                            callback(result)
                        except Exception as callback_error:
                            logging.error(f"Error in callback for task {task_id}: {callback_error}")
                except Exception as e:
                    # Handle any exceptions in the thread
                    error_result = {
                        'success': False,
                        'error': str(e),
                        'task_id': task_id,
                        'model': model_key,
                        'file': os.path.basename(audio_file)
                    }
                    logging.error(f"Exception in async prediction task {task_id}: {e}", exc_info=True)
                    
                    # Call the callback with the error result
                    if callback:
                        try:
                            callback(error_result)
                        except Exception as callback_error:
                            logging.error(f"Error in error callback for task {task_id}: {callback_error}")
            
            # Create and start the thread
            thread = Thread(target=predict_thread_fn, name=f"Prediction-{task_id}")
            thread.daemon = True  # Set as daemon so it doesn't block program exit
            thread.start()
            
            # Return information about the started task
            return {
                'success': True,
                'task_id': task_id,
                'model': model_key,
                'file': os.path.basename(audio_file),
                'message': f"Started async prediction with {model_key}"
            }
        except Exception as e:
            logging.error(f"Failed to start async prediction: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'task_id': task_id,
                'model': f"{model_name}_{model_type}",
                'file': os.path.basename(audio_file)
            }
    
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
            dict: Available models by type and additional metadata including:
            - model_id: The unique identifier for the model
            - model_type: Type of model (cnn, rf, ensemble)
            - dictionary_name: Name of the dictionary this model is for
            - creation_date: When the model was created
            - class_names: The classes this model can predict
            - is_best: Whether this is the best model for this dictionary
        """
        available_models = {
            'cnn': [],
            'rf': [],
            'ensemble': []
        }
        
        try:
            # Get the registry path using our helper function
            registry_path = get_models_registry_path()
            logging.info(f"Looking for models in registry: {registry_path}")
            
            # Try to synchronize the registry to ensure it's up to date
            synchronize_model_registry()
            logging.info("Synchronized model registry to ensure up-to-date information")
            
            # Log the registry path to help debug
            logging.info(f"Looking for models registry at: {registry_path}")
            
            # Check if registry file exists
            if os.path.exists(registry_path) and os.path.isfile(registry_path):
                try:
                    # Read the registry file
                    with open(registry_path, 'r', encoding='utf-8') as f:
                        models_registry = json.load(f)
                    
                    # Extract models from the registry with detailed metadata
                    if 'models' in models_registry:
                        for model_type in ['cnn', 'rf', 'ensemble']:
                            if model_type in models_registry['models']:
                                model_dict = models_registry['models'][model_type]
                                logging.info(f"Found {len(model_dict)} {model_type.upper()} models in registry")
                                
                                for model_id, model_data in model_dict.items():
                                    # Get the file path
                                    file_path = model_data.get('file_path')
                                    if not file_path:
                                        logging.warning(f"Model {model_id} has no file_path in registry, skipping")
                                        continue
                                    
                                    # Construct full path - try different base directories
                                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                    models_dir = os.path.join(base_dir, 'backend', 'data', 'models')
                                    
                                    # Handle absolute vs relative paths
                                    if file_path.startswith('/'):
                                        full_path = file_path  # Already absolute
                                    else:
                                        # Try multiple model directory structures to find the file
                                        # First try direct join
                                        full_path = os.path.join(models_dir, file_path)
                                        if not os.path.exists(full_path):
                                            # Try without the type prefix if it's already in the path
                                            if model_type in file_path and file_path.startswith(model_type):
                                                # The file_path already includes the model_type subdirectory
                                                full_path = os.path.join(models_dir, file_path)
                                            else:
                                                # Try with model type as subdirectory
                                                full_path = os.path.join(models_dir, model_type, os.path.basename(file_path))
                                        
                                        logging.info(f"Resolving model path: {file_path} to {full_path}")
                                    
                                    # Verify the file exists
                                    logging.info(f"Checking if model file exists at: {full_path}")
                                    if os.path.exists(full_path):
                                        # Add model with extended metadata - handle multiple possible field names
                                        model_info = {
                                            'model_id': model_id,
                                            'model_type': model_type,
                                            'file_path': full_path,
                                            'dictionary_name': model_data.get('dictionary_name') or model_data.get('dictionary', 'unknown'),
                                            'creation_date': model_data.get('creation_date') or model_data.get('created_at', ''),
                                            'is_best': model_data.get('is_best', False),
                                            'class_names': model_data.get('class_names', [])
                                        }
                                        
                                        # Also include accuracy and other training stats if available
                                        if 'training_stats' in model_data:
                                            model_info['accuracy'] = model_data['training_stats'].get('accuracy', 0.0)
                                        
                                        # Add to available models
                                        available_models[model_type].append(model_info)
                                        logging.info(f"Added {model_type} model {model_id} from registry")
                                    else:
                                        logging.warning(f"Model file not found at {full_path} for {model_id}")
                except Exception as e:
                    logging.error(f"Error reading models.json: {e}", exc_info=True)
            
            # If we found models from the registry, we're done
            if any(len(models) > 0 for models in available_models.values()):
                total_models = sum(len(models) for models in available_models.values())
                logging.info(f"Found {total_models} models from registry")
                return available_models
            
            # If no models found in registry, fall back to directory scanning
            logging.info("No models found in registry, scanning directories for model files")
            self._scan_directories_for_models(available_models)
            
            return available_models
            
        except Exception as e:
            logging.error(f"Error getting available models: {e}", exc_info=True)
            return available_models
    
    def _scan_directories_for_models(self, available_models):
        """
        Helper method to scan directories for model files when registry lookup fails.
        This is a fallback mechanism only.
        
        Args:
            available_models (dict): Dictionary to populate with found models
        """
        try:
            # Get the base directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Define directories to scan in order of priority
            models_dirs = [
                # First priority: backend/data/models is the new standard location
                os.path.join(base_dir, 'backend', 'data', 'models'),
                # Second priority: data/models as an alternative location
                os.path.join(base_dir, 'data', 'models'),
                # Legacy location: models directory, lowest priority
                os.path.join(base_dir, 'models')
            ]
            
            for models_dir in models_dirs:
                if not os.path.exists(models_dir):
                    continue
                    
                logging.info(f"Scanning for model files in: {models_dir}")
                
                # Define model type specific extensions and directories
                model_configs = {
                    'cnn': {'ext': '.h5', 'subdir': 'cnn'},
                    'rf': {'ext': '.joblib', 'subdir': 'rf'}, 
                    'ensemble': {'ext': '.joblib', 'subdir': 'ensemble'}
                }
                
                # Scan for each model type
                for model_type, config in model_configs.items():
                    ext = config['ext']
                    subdir = os.path.join(models_dir, config['subdir'])
                    
                    # Skip if subdirectory doesn't exist
                    if not os.path.exists(subdir):
                        continue
                        
                    logging.info(f"Searching for {ext} files in {subdir}")
                    
                    # Walk the directory recursively
                    for root, _, files in os.walk(subdir):
                        # Skip features_cache directory
                        if 'features_cache' in root:
                            continue
                            
                        for file in files:
                            # Check the correct extension
                            if file.endswith(ext):
                                try:
                                    # Get file path and model ID
                                    file_path = os.path.join(root, file)
                                    model_id = file[:-len(ext)]  # Remove extension
                                    
                                    # Try to get metadata from accompanying JSON file
                                    metadata_path = file_path.replace(ext, '_metadata.json')
                                    model_info = {'model_id': model_id, 'model_type': model_type, 'file_path': file_path}
                                    
                                    # Check for metadata file
                                    if os.path.exists(metadata_path):
                                        try:
                                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                                metadata = json.load(f)
                                            
                                            # Add metadata fields to model info
                                            if 'class_names' in metadata:
                                                model_info['class_names'] = metadata['class_names']
                                            if 'creation_date' in metadata:
                                                model_info['creation_date'] = metadata['creation_date']
                                            if 'dictionary_name' in metadata:
                                                model_info['dictionary_name'] = metadata['dictionary_name']
                                                
                                            # Add accuracy if available in training stats
                                            if 'training_stats' in metadata and 'accuracy' in metadata['training_stats']:
                                                model_info['accuracy'] = metadata['training_stats']['accuracy']
                                        except Exception as e:
                                            logging.warning(f"Error reading metadata for {model_id}: {e}")
                                    
                                    # Add model to available models
                                    available_models[model_type].append(model_info)
                                    logging.info(f"Added {model_type} model {model_id} from file system scan")
                                except Exception as e:
                                    logging.warning(f"Error processing model {file}: {e}")
        except Exception as e:
            logging.error(f"Error scanning for models: {e}", exc_info=True)
        
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