import os
import logging
import json
from flask import request, jsonify, send_file

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
            audio_dir = data.get('audio_dir')
            dict_name = data.get('dict_name', 'unknown')
            
            # Additional training parameters
            train_params = {
                'dict_name': dict_name,
                'epochs': data.get('epochs', 50),
                'batch_size': data.get('batch_size', 32),
                'n_estimators': data.get('n_estimators', 100),
                'max_depth': data.get('max_depth'),
                'rf_weight': data.get('rf_weight', 0.5),
                'use_class_weights': data.get('use_class_weights', True)
            }
            
            if not audio_dir:
                return jsonify({
                    'success': False,
                    'error': 'No audio directory provided'
                }), 400
            
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
            models = self.inference_service.get_available_models()
            return jsonify({
                'success': True,
                'models': models
            })
        
        except Exception as e:
            logging.error(f"Error in list_models: {e}")
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