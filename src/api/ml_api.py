import os
import logging
import json
from flask import request, jsonify, send_file
import datetime

from src.services.training_service import TrainingService
from src.services.inference_service import InferenceService
from config import Config

class MlApi:
    """
    API class that handles machine learning routes.
    This class integrates the training and inference services with Flask routes.
    """
    
    def __init__(self, app, model_dir='models'):
        """
        Initialize the ML API.
        
        Args:
            app: Flask application
            model_dir (str): Directory for model storage
        """
        self.app = app
        self.model_dir = model_dir
        self.training_service = TrainingService(model_dir=model_dir)
        self.inference_service = InferenceService(model_dir=model_dir)
        
        # Register routes
        self._register_routes()
        
        logging.info("ML API initialized")
    
    def _register_routes(self):
        """Register all API routes with the Flask app."""
        
        # Training routes
        @self.app.route('/api/ml/train', methods=['POST'])
        def train_model():
            return self.handle_train_model()
        
        @self.app.route('/api/ml/train/status', methods=['GET'])
        def training_status():
            return self.handle_training_status()
        
        @self.app.route('/api/ml/train/stats', methods=['GET'])
        def training_stats():
            return self.handle_training_stats()
        
        # Prediction routes
        @self.app.route('/api/ml/predict', methods=['POST'])
        def predict():
            return self.handle_predict()
        
        @self.app.route('/api/ml/predict/batch', methods=['POST'])
        def batch_predict():
            return self.handle_batch_predict()
        
        @self.app.route('/api/ml/predict/last', methods=['GET'])
        def last_prediction():
            return self.handle_last_prediction()
        
        # Model management routes
        @self.app.route('/api/ml/models', methods=['GET'])
        def list_models():
            return self.handle_list_models()
        
        @self.app.route('/api/ml/models/<model_type>/<model_name>', methods=['DELETE'])
        def delete_model(model_type, model_name):
            return self.handle_delete_model(model_type, model_name)
        
        # Add model metadata endpoint
        @self.app.route('/api/ml/model_metadata/<model_id>', methods=['GET'])
        def get_model_metadata_api(model_id):
            print(f"\n\n===== DEBUG: API MODEL METADATA REQUEST =====")
            print(f"Using API route for model_id: {model_id}")
            
            # Forward to the blueprint function
            from src.routes.ml_routes import get_model_metadata_direct
            return get_model_metadata_direct(model_id)
            
        # Add start_listening endpoint
        @self.app.route('/api/start_listening', methods=['POST'])
        def start_listening_api():
            print(f"\n\n===== DEBUG: API START LISTENING REQUEST =====")
            # Forward to the blueprint function
            from src.routes.ml_routes import start_listening
            return start_listening()
            
        # Add stop_listening endpoint
        @self.app.route('/api/ml/stop_listening', methods=['POST'])
        def stop_listening_api():
            print(f"\n\n===== DEBUG: API STOP LISTENING REQUEST =====")
            # Forward to the blueprint function
            from src.routes.ml_routes import stop_listening
            return stop_listening()
        
        # Add prediction_stream endpoint
        @self.app.route('/api/ml/prediction_stream')
        def prediction_stream_api():
            print(f"\n\n===== DEBUG: API PREDICTION STREAM REQUEST =====")
            # Forward to the blueprint function
            from src.routes.ml_routes import prediction_stream
            return prediction_stream()
        
        # Add inference_statistics endpoint
        @self.app.route('/api/ml/inference_statistics')
        def inference_statistics_api():
            print(f"\n\n===== DEBUG: API INFERENCE STATISTICS REQUEST =====")
            # Forward to the blueprint function
            from src.routes.ml_routes import inference_statistics
            return inference_statistics()
        
        # Add record_feedback endpoint
        @self.app.route('/api/ml/record_feedback', methods=['POST'])
        def record_feedback_api():
            print(f"\n\n===== DEBUG: API RECORD FEEDBACK REQUEST =====")
            # Forward to the blueprint function
            from src.routes.ml_routes import record_feedback
            return record_feedback()
        
        # Add save_analysis endpoint
        @self.app.route('/api/ml/save_analysis', methods=['POST'])
        def save_analysis_api():
            print(f"\n\n===== DEBUG: API SAVE ANALYSIS REQUEST =====")
            # Forward to the blueprint function
            from src.routes.ml_routes import save_analysis
            return save_analysis()
        
        # Audio analysis routes
        @self.app.route('/api/ml/analyze', methods=['POST'])
        def analyze_audio():
            return self.handle_analyze_audio()
    
    def handle_train_model(self):
        """Handle the train model request."""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No JSON data provided'
                }), 400
            
            model_type = data.get('model_type', 'ensemble')
            dict_name = data.get('dict_name', 'unknown')
            
            # Get dictionary classes from dictionary service
            from src.services.dictionary_service import DictionaryService
            dictionary_service = DictionaryService()
            
            # Add debug logs
            logging.info(f"Training requested for dictionary: '{dict_name}'")
            logging.info(f"Request data: {data}")
            
            # Get dictionary object
            dictionary = dictionary_service.get_dictionary(dict_name)
            
            # Add more debug logs
            if dictionary:
                logging.info(f"Found dictionary: {dictionary.get('name')} with keys: {list(dictionary.keys())}")
            else:
                logging.error(f"Dictionary '{dict_name}' not found")
            
            if not dictionary:
                return jsonify({
                    'success': False,
                    'error': f'Dictionary "{dict_name}" not found'
                }), 404
            
            # Set audio_dir to the training_sounds folder and pass classes list
            audio_dir = Config.TRAINING_SOUNDS_DIR
            
            # Check for classes or sounds key and use the appropriate one
            classes = []
            if 'classes' in dictionary and dictionary['classes']:
                classes = dictionary['classes']
                logging.info(f"Using classes from 'classes' key: {classes}")
            elif 'sounds' in dictionary and dictionary['sounds']:
                classes = dictionary['sounds']
                logging.info(f"Using classes from 'sounds' key: {classes}")
            else:
                logging.error(f"Dictionary has no classes or sounds defined: {dictionary}")
                
            if not classes:
                return jsonify({
                    'success': False,
                    'error': f'Dictionary "{dict_name}" has no classes defined'
                }), 400
            
            # Additional training parameters
            train_params = {
                'dict_name': dict_name,
                'classes': classes,
                'epochs': data.get('epochs', 50),
                'batch_size': data.get('batch_size', 32),
                'n_estimators': data.get('n_estimators', 100),
                'max_depth': data.get('max_depth'),
                'rf_weight': data.get('rf_weight', 0.5),
                'use_class_weights': data.get('use_class_weights', True)
            }
            
            logging.info(f"Starting training with parameters: {train_params}")
            
            # Start training asynchronously
            success = self.training_service.train_model_async(
                model_type, audio_dir, **train_params
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Training started for {model_type} model with data from {audio_dir}'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Training is already in progress'
                }), 409
        
        except Exception as e:
            logging.error(f"Error in train_model: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def handle_training_status(self):
        """Handle the training status request."""
        try:
            is_training = self.training_service.is_training_in_progress()
            return jsonify({
                'success': True,
                'is_training': is_training
            })
        
        except Exception as e:
            logging.error(f"Error in training_status: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def handle_training_stats(self):
        """Handle the training stats request."""
        try:
            stats = self.training_service.get_training_stats()
            
            # Add file errors if available
            if hasattr(self.training_service, 'file_errors'):
                stats['file_errors'] = self.training_service.file_errors
            
            # Add MFCC normalization stats if available
            if hasattr(self.training_service.audio_processor, 'mfcc_stats'):
                stats['mfcc_stats'] = self.training_service.audio_processor.mfcc_stats
            
            return jsonify({
                'success': True,
                'stats': stats
            })
        
        except Exception as e:
            logging.error(f"Error in training_stats: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def handle_predict(self):
        """Handle the predict request."""
        try:
            # Check if file was uploaded
            if 'audio' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No audio file provided'
                }), 400
            
            audio_file = request.files['audio']
            
            # Get model type and name from form data
            model_type = request.form.get('model_type', 'ensemble')
            model_name = request.form.get('model_name')
            
            if not model_name:
                return jsonify({
                    'success': False,
                    'error': 'No model name provided'
                }), 400
            
            # Save the uploaded file temporarily
            temp_path = os.path.join('/tmp', audio_file.filename)
            audio_file.save(temp_path)
            
            # Get top_n parameter
            top_n = int(request.form.get('top_n', 3))
            
            # Make the prediction
            result = self.inference_service.predict(
                temp_path, model_type, model_name, top_n=top_n
            )
            
            # Clean up the temporary file
            os.remove(temp_path)
            
            return jsonify(result)
        
        except Exception as e:
            logging.error(f"Error in predict: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def handle_batch_predict(self):
        """Handle the batch predict request."""
        try:
            # Get model type and name from form data
            model_type = request.form.get('model_type', 'ensemble')
            model_name = request.form.get('model_name')
            
            if not model_name:
                return jsonify({
                    'success': False,
                    'error': 'No model name provided'
                }), 400
            
            # Check if files were uploaded
            if 'audio_files' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No audio files provided'
                }), 400
            
            audio_files = request.files.getlist('audio_files')
            
            # Save the uploaded files temporarily
            temp_paths = []
            for audio_file in audio_files:
                temp_path = os.path.join('/tmp', audio_file.filename)
                audio_file.save(temp_path)
                temp_paths.append(temp_path)
            
            # Make the batch prediction
            result = self.inference_service.batch_predict(
                temp_paths, model_type, model_name
            )
            
            # Clean up the temporary files
            for temp_path in temp_paths:
                os.remove(temp_path)
            
            return jsonify(result)
        
        except Exception as e:
            logging.error(f"Error in batch_predict: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def handle_last_prediction(self):
        """Handle the last prediction request."""
        try:
            last_prediction = self.inference_service.get_last_prediction()
            
            if last_prediction:
                return jsonify({
                    'success': True,
                    'prediction': last_prediction
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No predictions have been made yet'
                }), 404
        
        except Exception as e:
            logging.error(f"Error in last_prediction: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def handle_list_models(self):
        """Handle the list models request."""
        try:
            # Debug the models.json file and directory structure
            self.inference_service.debug_models_json()
            
            # Get available models from InferenceService
            models_by_type = self.inference_service.get_available_models()
            logging.info(f"Retrieved models from inference service: {models_by_type}")
            
            # Get current dictionary using DictionaryService
            from src.services.dictionary_service import DictionaryService
            dictionary_service = DictionaryService()
            
            # Get the active dictionary name from the dictionaries object
            active_dict_name = dictionary_service.dictionaries.get('active_dictionary')
            logging.info(f"Active dictionary from dictionary_service: {active_dict_name}")
            
            if not active_dict_name:
                logging.warning("No active dictionary found, defaulting to first available")
                # Fallback: use the first dictionary if no active one
                dictionaries = dictionary_service.dictionaries.get('dictionaries', {})
                if dictionaries:
                    active_dict_name = next(iter(dictionaries.keys()))
                    logging.info(f"Using fallback dictionary: {active_dict_name}")
                else:
                    active_dict_name = "Unknown"
                    logging.warning("No dictionaries found, using 'Unknown'")
            
            # Get the dictionary details
            active_dict = dictionary_service.get_dictionary(active_dict_name)
            if not active_dict:
                logging.warning(f"Could not find dictionary details for '{active_dict_name}'")
                active_dict = {"name": active_dict_name}
            
            dict_name = active_dict.get('name', 'Unknown')
            dict_name_normalized = dict_name.replace(' ', '_')
            
            logging.info(f"Using dictionary: {dict_name} (normalized: {dict_name_normalized})")
            
            # Transform the model data to match what the predict page expects
            # Include ALL models regardless of dictionary
            transformed_models = {}
            
            # Find the original models.json file to get class names for each model
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_json_path = os.path.join(base_dir, 'data', 'models', 'models.json')
            models_registry = None
            
            if os.path.exists(models_json_path):
                try:
                    with open(models_json_path, 'r') as f:
                        models_registry = json.load(f)
                    logging.info(f"Successfully loaded models registry from {models_json_path}")
                except Exception as e:
                    logging.error(f"Error reading models.json: {e}", exc_info=True)
            
            # Create a dictionary to organize models by dictionary name
            models_by_dict = {}
            
            # Process CNN models - Include ALL models
            if 'cnn' in models_by_type and models_by_type['cnn']:
                transformed_models['cnn'] = []
                
                for model_name in models_by_type['cnn']:
                    logging.info(f"Processing model {model_name}")
                    
                    # Get dictionary name and timestamp from model name
                    parts = model_name.split('_')
                    if len(parts) >= 3:
                        model_dict_name = parts[0]
                        timestamp = parts[-1]
                    else:
                        model_dict_name = "Unknown"
                        timestamp = ""
                    
                    # Get model data (including class_names) from models registry if available
                    model_class_names = None
                    created_at = None
                    
                    if models_registry and 'models' in models_registry and 'cnn' in models_registry['models'] and model_name in models_registry['models']['cnn']:
                        model_data = models_registry['models']['cnn'][model_name]
                        model_dict_name = model_data.get('dictionary', model_dict_name)
                        model_class_names = model_data.get('class_names', None)
                        created_at = model_data.get('created_at', None)
                        logging.info(f"Found class names for model {model_name}: {model_class_names}")
                    
                    # Format timestamp for display
                    formatted_date = "Unknown date"
                    if created_at:
                        try:
                            # Parse ISO format datetime 
                            dt = datetime.datetime.fromisoformat(created_at)
                            formatted_date = dt.strftime("%B %d, %Y %I:%M%p")
                        except Exception as e:
                            logging.error(f"Error parsing created_at date: {e}")
                    elif timestamp and len(timestamp) >= 14:
                        try:
                            # Parse YYYYMMDDHHMMSS format
                            year = timestamp[0:4]
                            month = timestamp[4:6]
                            day = timestamp[6:8]
                            hour = timestamp[8:10] 
                            minute = timestamp[10:12]
                            second = timestamp[12:14]
                            
                            # Convert to datetime object
                            dt = datetime.datetime(int(year), int(month), int(day), 
                                                int(hour), int(minute), int(second))
                            formatted_date = dt.strftime("%B %d, %Y %I:%M%p")
                        except Exception as e:
                            logging.error(f"Error parsing timestamp: {e}")
                            formatted_date = timestamp
                    
                    model_obj = {
                        'id': model_name,
                        'name': f"{model_dict_name} CNN Model",
                        'display_name': f"{model_dict_name} CNN Model - {formatted_date}",
                        'type': 'cnn',
                        'dictionary': model_dict_name,
                        'class_names': model_class_names,
                        'timestamp': timestamp,
                        'created_at': created_at,
                        'formatted_date': formatted_date
                    }
                    
                    # Add to transformed_models list
                    transformed_models['cnn'].append(model_obj)
                    
                    # Group by dictionary for later filtering
                    if model_dict_name not in models_by_dict:
                        models_by_dict[model_dict_name] = []
                    models_by_dict[model_dict_name].append(model_obj)
            
            # For each dictionary, mark the latest model
            for dict_name, dict_models in models_by_dict.items():
                # Sort models by timestamp (descending)
                dict_models.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '', reverse=True)
                
                # Mark the latest model
                if dict_models:
                    dict_models[0]['is_latest'] = True
                    logging.info(f"Latest model for {dict_name}: {dict_models[0]['id']}")
            
            # Debug the transformed models
            logging.info(f"Transformed models for predict page: {transformed_models}")
            
            return jsonify({
                'success': True,
                'models': transformed_models,
                'models_by_dict': models_by_dict
            })
        
        except Exception as e:
            logging.error(f"Error in list_models: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def handle_delete_model(self, model_type, model_name):
        """Handle the delete model request."""
        try:
            # Determine file extension based on model type
            extension = ''
            if model_type == 'cnn':
                extension = '.h5'
            elif model_type == 'rf':
                extension = '.joblib'
            
            # Construct model path
            model_path = os.path.join(self.model_dir, model_name + extension)
            
            # Check if model exists
            if not os.path.exists(model_path):
                return jsonify({
                    'success': False,
                    'error': f'Model {model_name} of type {model_type} not found'
                }), 404
            
            # Unload model if it's loaded
            self.inference_service.unload_model(model_type, model_name)
            
            # Delete the model file
            os.remove(model_path)
            
            return jsonify({
                'success': True,
                'message': f'Model {model_name} of type {model_type} deleted'
            })
        
        except Exception as e:
            logging.error(f"Error in delete_model: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def handle_analyze_audio(self):
        """Handle the analyze audio request."""
        try:
            # Check if file was uploaded
            if 'audio' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No audio file provided'
                }), 400
            
            audio_file = request.files['audio']
            
            # Save the uploaded file temporarily
            temp_path = os.path.join('/tmp', audio_file.filename)
            audio_file.save(temp_path)
            
            # Analyze the audio
            result = self.inference_service.analyze_audio(temp_path)
            
            # Clean up the temporary file
            os.remove(temp_path)
            
            return jsonify(result)
        
        except Exception as e:
            logging.error(f"Error in analyze_audio: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500 