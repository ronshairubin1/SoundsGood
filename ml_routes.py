from flask import render_template, jsonify, redirect, url_for, flash, current_app, request, Response
from flask_login import login_required
import os
import json
from datetime import datetime
import time

# Global variable to store the latest prediction
LATEST_PREDICTION = None

@ml_blueprint.route('/predict', methods=['GET'])
@login_required
def predict_page():
    """Render the prediction page"""
    # Get active dictionary
    active_dict = get_active_dictionary()
    if not active_dict:
        flash('No active dictionary found. Please create and activate a dictionary first.', 'danger')
        return redirect(url_for('main.index'))
    
    return render_template('predict.html', active_dict=active_dict)

@ml_blueprint.route('/api/ml/models', methods=['GET'])
def get_available_models():
    """API to get all available models for prediction"""
    try:
        models = []
        models_dir = current_app.config['MODELS_DIR']
        
        # Log debugging information
        current_app.logger.debug(f"Looking for models in directory: {models_dir}")
        
        if not os.path.exists(models_dir):
            current_app.logger.error(f"Models directory not found: {models_dir}")
            return jsonify({
                'success': False, 
                'message': 'Models directory not found',
                'models': []
            }), 404
            
        # List all files in the directory for debugging
        all_files = os.listdir(models_dir)
        current_app.logger.debug(f"Files in models directory: {all_files}")
        
        # Scan for model files directly in the models directory
        for filename in all_files:
            file_path = os.path.join(models_dir, filename)
            current_app.logger.debug(f"Checking file: {filename}")
            
            # Skip if it's a directory or not a model file
            if os.path.isdir(file_path):
                current_app.logger.debug(f"Skipping directory: {filename}")
                continue
                
            if not filename.endswith('.h5') and not filename.endswith('.joblib'):
                current_app.logger.debug(f"Skipping non-model file: {filename}")
                continue
                
            # Check if it's a backup or metadata file
            if "_metadata" in filename or "_backup" in filename:
                current_app.logger.debug(f"Skipping metadata or backup file: {filename}")
                continue
                
            # Parse the model filename to get components
            # Format: {Dictionary}_{model_type}_{timestamp}.{extension}
            parts = os.path.splitext(filename)[0].split('_')
            current_app.logger.debug(f"Filename parts: {parts}")
            
            # We need at least dictionary and model_type
            if len(parts) < 2:
                current_app.logger.debug(f"Skipping file with too few parts: {filename}")
                continue
                
            # Extract components
            dictionary_name = parts[0]
            model_type = None
            
            # Determine model type from filename or extension
            if 'cnn' in filename.lower():
                model_type = 'cnn'
            elif 'rf' in filename.lower():
                model_type = 'rf'
            elif filename.endswith('.h5'):
                model_type = 'cnn'
            elif filename.endswith('.joblib'):
                model_type = 'rf'
            else:
                # Skip if we can't determine the model type
                current_app.logger.debug(f"Skipping file with unknown model type: {filename}")
                continue
                
            current_app.logger.debug(f"Model type determined as: {model_type}")
                
            # Check if metadata file exists
            metadata_file = os.path.splitext(filename)[0] + "_metadata.json"
            metadata_path = os.path.join(models_dir, metadata_file)
            current_app.logger.debug(f"Looking for metadata file: {metadata_file}")
            
            metadata = None
            sounds = []
            
            if os.path.exists(metadata_path):
                current_app.logger.debug(f"Metadata file found: {metadata_file}")
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check for both possible formats: 'classes' or 'class_names'
                    if metadata and 'classes' in metadata:
                        sounds = metadata['classes']
                        current_app.logger.debug(f"Found classes in metadata: {sounds}")
                    elif metadata and 'class_names' in metadata:
                        sounds = metadata['class_names']
                        current_app.logger.debug(f"Found class_names in metadata: {sounds}")
                    else:
                        current_app.logger.debug(f"No classes found in metadata: {metadata}")
                except Exception as e:
                    current_app.logger.error(f"Error reading metadata file {metadata_path}: {str(e)}")
            else:
                current_app.logger.debug(f"Metadata file not found: {metadata_file}")
                
                # Try to find metadata by looking for similarly named files
                for meta_filename in all_files:
                    if meta_filename.startswith(os.path.splitext(filename)[0]) and meta_filename.endswith("_metadata.json"):
                        current_app.logger.debug(f"Found alternative metadata file: {meta_filename}")
                        try:
                            meta_path = os.path.join(models_dir, meta_filename)
                            with open(meta_path, 'r') as f:
                                metadata = json.load(f)
                            
                            # Check for both possible formats
                            if metadata and 'classes' in metadata:
                                sounds = metadata['classes']
                                current_app.logger.debug(f"Found classes in alternative metadata: {sounds}")
                                break
                            elif metadata and 'class_names' in metadata:
                                sounds = metadata['class_names']
                                current_app.logger.debug(f"Found class_names in alternative metadata: {sounds}")
                                break
                        except Exception as e:
                            current_app.logger.error(f"Error reading alternative metadata file {meta_path}: {str(e)}")
            
            # Only include models with valid metadata and sounds
            if sounds:  # Only if we have sounds from metadata
                models.append({
                    'id': os.path.splitext(filename)[0],
                    'name': os.path.splitext(filename)[0],
                    'type': model_type,
                    'dictionary': dictionary_name,
                    'path': file_path,
                    'sounds': sounds,
                    'timestamp': parts[2] if len(parts) > 2 else None
                })
                current_app.logger.debug(f"Added model: {filename} with sounds: {sounds}")
        
        # Log the final models list
        current_app.logger.debug(f"Found {len(models)} models: {[m['name'] for m in models]}")
        
        # Sort models by timestamp (newest first)
        models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'models': models
        })
    
    except Exception as e:
        current_app.logger.error(f"Error getting models: {str(e)}")
        return jsonify({'success': False, 'message': str(e), 'models': []}), 500

@ml_blueprint.route('/api/ml/dictionary/sounds', methods=['GET'])
def get_dictionary_sounds():
    """API to get all sounds in the current dictionary"""
    try:
        # Get active dictionary
        active_dict = get_active_dictionary()
        if not active_dict:
            return jsonify({
                'success': False, 
                'message': 'No active dictionary found',
                'sounds': []
            }), 400
        
        return jsonify({
            'success': True,
            'sounds': active_dict.sounds
        })
    
    except Exception as e:
        current_app.logger.error(f"Error getting dictionary sounds: {str(e)}")
        return jsonify({'success': False, 'message': str(e), 'sounds': []}), 500

@ml_blueprint.route('/api/ml/start_listening', methods=['POST'])
def start_listening():
    """Start listening for real-time sound classification"""
    try:
        # Get model_id from request - this should be the full model ID, not just the type
        model_id = request.args.get('model')
        if not model_id:
            return jsonify({"status": "error", "message": "No model specified"}), 400
            
        # Extract model type from model_id
        model_type = None
        if 'cnn' in model_id:
            model_type = 'cnn'
        elif 'rf' in model_id:
            model_type = 'rf'
        else:
            model_type = 'ensemble'
            
        # Find the model file path
        models_dir = current_app.config['MODELS_DIR']
        model_file = None
        metadata_file = None
        
        # First look for exact file match
        for ext in ['.h5', '.joblib']:
            path = os.path.join(models_dir, model_id + ext)
            if os.path.exists(path):
                model_file = path
                metadata_file = os.path.join(models_dir, model_id + "_metadata.json")
                break
                
        if not model_file:
            return jsonify({"status": "error", "message": "Model file not found"}), 404
            
        if not os.path.exists(metadata_file):
            return jsonify({"status": "error", "message": "Model metadata file not found"}), 404
            
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            current_app.logger.error(f"Error loading model metadata: {str(e)}")
            return jsonify({"status": "error", "message": f"Error loading model metadata: {str(e)}"}), 500
            
        # Check for required metadata fields
        sound_classes = None
        input_shape = None
        
        if 'class_names' in metadata:
            sound_classes = metadata['class_names']
        elif 'classes' in metadata:
            sound_classes = metadata['classes']
        else:
            return jsonify({"status": "error", "message": "No classes found in metadata"}), 400
            
        if 'input_shape' in metadata:
            input_shape = metadata['input_shape']
        else:
            return jsonify({"status": "error", "message": "No input shape found in metadata"}), 400
            
        # Get preprocessing parameters from metadata if available
        preprocessing_params = metadata.get('preprocessing_params', {})
        
        # Global variable to store the latest prediction
        global LATEST_PREDICTION
        LATEST_PREDICTION = None
        
        # Prediction callback function
        def prediction_callback(prediction):
            global LATEST_PREDICTION
            LATEST_PREDICTION = prediction
            
            # Record statistics if 'actual_sound' is provided (feedback)
            if 'actual_sound' in prediction:
                # Initialize statistics if not present
                if not hasattr(current_app, 'inference_stats'):
                    current_app.inference_stats = {
                        'total_predictions': 0,
                        'confidence_levels': [],
                        'class_counts': {},
                        'confusion_matrix': {},
                        'misclassifications': [],
                        'correct_classifications': []
                    }
                
                stats = current_app.inference_stats
                
                # Update total predictions
                stats['total_predictions'] += 1
                
                # Update confidence levels
                if 'confidence' in prediction:
                    stats['confidence_levels'].append(prediction['confidence'])
                
                # Update class counts
                actual_sound = prediction['actual_sound']
                if actual_sound not in stats['class_counts']:
                    stats['class_counts'][actual_sound] = 0
                stats['class_counts'][actual_sound] += 1
                
                # Update confusion matrix
                if actual_sound not in stats['confusion_matrix']:
                    stats['confusion_matrix'][actual_sound] = {}
                
                predicted_sound = prediction['class']
                if predicted_sound not in stats['confusion_matrix'][actual_sound]:
                    stats['confusion_matrix'][actual_sound][predicted_sound] = 0
                stats['confusion_matrix'][actual_sound][predicted_sound] += 1
                
                # Record classification
                classification = {
                    'timestamp': datetime.now().isoformat(),
                    'predicted': predicted_sound,
                    'actual': actual_sound,
                    'confidence': prediction['confidence']
                }
                
                if predicted_sound == actual_sound:
                    stats['correct_classifications'].append(classification)
                else:
                    stats['misclassifications'].append(classification)
        
        # Initialize the sound detector based on model type
        from src.ml.sound_detector_ensemble import SoundDetectorEnsemble
        from src.ml.sound_detector_rf import SoundDetectorRF
        from src.core.models import create_model
        
        # Create and initialize the sound detector with metadata parameters
        sound_detector = None
        
        if model_type == 'ensemble':
            # Create and load the ensemble model
            ensemble_model = create_model('ensemble')
            ensemble_model.load(model_file, metadata=metadata)  # Pass metadata to ensure consistent preprocessing
            sound_detector = SoundDetectorEnsemble(ensemble_model, preprocessing_params=preprocessing_params)
            sound_detector.class_names = sound_classes
            sound_detector.input_shape = input_shape
        elif model_type == 'rf':
            # Create and load the RF model
            rf_model = create_model('rf')
            rf_model.load(model_file, metadata=metadata)  # Pass metadata to ensure consistent preprocessing
            sound_detector = SoundDetectorRF(rf_model, preprocessing_params=preprocessing_params)
            sound_detector.class_names = sound_classes
        else:  # cnn
            # Create and load the CNN model
            cnn_model = create_model('cnn')
            cnn_model.load(model_file, metadata=metadata)  # Pass metadata to ensure consistent preprocessing
            # Use SoundDetector for CNN
            from src.ml.inference import SoundDetector
            sound_detector = SoundDetector(cnn_model, sound_classes, input_shape=input_shape, preprocessing_params=preprocessing_params)
        
        # Store the sound detector in the application context
        current_app.sound_detector = sound_detector
        
        # Log the setup
        current_app.logger.info(f"Initialized sound detector for model: {model_id}")
        current_app.logger.info(f"Using sound classes: {sound_classes}")
        current_app.logger.info(f"Using input shape: {input_shape}")
        current_app.logger.info(f"Using preprocessing params: {preprocessing_params}")
        
        # Start listening
        success = sound_detector.start_listening(callback=prediction_callback)
        
        if success:
            return jsonify({
                "status": "success", 
                "message": f"Started listening with {model_type} model",
                "sound_classes": sound_classes
            })
        else:
            return jsonify({"status": "error", "message": "Failed to start listening"}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error in start_listening: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@ml_blueprint.route('/api/ml/stop_listening', methods=['POST'])
def stop_listening():
    """Stop listening for real-time sound classification"""
    try:
        if hasattr(current_app, 'sound_detector'):
            sound_detector = current_app.sound_detector
            sound_detector.stop_listening()
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "No active sound detector found"}), 404
    except Exception as e:
        current_app.logger.error(f"Error stopping listener: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@ml_blueprint.route('/api/ml/prediction_stream')
def prediction_stream():
    """Stream real-time predictions using server-sent events"""
    try:
        global LATEST_PREDICTION
        
        def stream_predictions():
            while True:
                data = {}
                
                if LATEST_PREDICTION:
                    data['prediction'] = LATEST_PREDICTION
                    LATEST_PREDICTION = None
                
                if data:
                    yield f"data: {json.dumps(data)}\n\n"
                else:
                    yield ": heartbeat\n\n"
                
                time.sleep(0.1)
        
        return Response(stream_predictions(), mimetype='text/event-stream')
    except Exception as e:
        current_app.logger.error(f"Error in prediction_stream: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@ml_blueprint.route('/api/ml/record_feedback', methods=['POST'])
def record_feedback():
    """Record user feedback for a prediction"""
    try:
        data = request.get_json()
        predicted_sound = data.get('predicted_sound')
        actual_sound = data.get('actual_sound')
        confidence = data.get('confidence')
        
        if not all([predicted_sound, actual_sound, confidence is not None]):
            return jsonify({"status": "error", "message": "Missing required data"}), 400
        
        # Create prediction object with actual sound for statistics
        prediction = {
            'class': predicted_sound,
            'confidence': confidence,
            'actual_sound': actual_sound
        }
        
        # Process the prediction with the callback
        global LATEST_PREDICTION
        LATEST_PREDICTION = prediction
        
        return jsonify({"status": "success"})
    except Exception as e:
        current_app.logger.error(f"Error recording feedback: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@ml_blueprint.route('/api/ml/inference_statistics')
def inference_statistics():
    """Get inference statistics"""
    try:
        # Initialize statistics if not present
        if not hasattr(current_app, 'inference_stats'):
            current_app.inference_stats = {
                'total_predictions': 0,
                'confidence_levels': [],
                'class_counts': {},
                'confusion_matrix': {},
                'misclassifications': [],
                'correct_classifications': []
            }
        
        stats = current_app.inference_stats
        
        # Calculate average confidence
        if not stats['confidence_levels']:
            avg_conf = 0.0
        else:
            avg_conf = sum(stats['confidence_levels']) / len(stats['confidence_levels'])
        
        # Calculate class accuracy
        class_accuracy = {}
        cm = stats.get('confusion_matrix', {})
        for actual_sound in cm:
            total = sum(cm[actual_sound].values())
            correct = cm[actual_sound].get(actual_sound, 0)
            if total > 0:
                class_accuracy[actual_sound] = {
                    'accuracy': correct / total,
                    'total_samples': total,
                    'correct_samples': correct
                }
            else:
                class_accuracy[actual_sound] = {
                    'accuracy': 0.0,
                    'total_samples': 0,
                    'correct_samples': 0
                }
        
        return jsonify({
            'total_predictions': stats.get('total_predictions', 0),
            'average_confidence': avg_conf,
            'class_counts': stats.get('class_counts', {}),
            'class_accuracy': class_accuracy,
            'confusion_matrix': cm
        })
    except Exception as e:
        current_app.logger.error(f"Error getting inference statistics: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@ml_blueprint.route('/api/ml/save_analysis', methods=['POST'])
def save_analysis():
    """Save the current inference analysis to a file"""
    try:
        if not hasattr(current_app, 'inference_stats'):
            return jsonify({"status": "error", "message": "No inference statistics available"}), 404
        
        stats = current_app.inference_stats
        
        # Get active dictionary
        active_dict = get_active_dictionary()
        dict_name = active_dict.name if active_dict else "unknown"
        
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'dictionary': dict_name,
            'confusion_matrix': stats.get('confusion_matrix', {}),
            'misclassifications': stats.get('misclassifications', []),
            'correct_classifications': stats.get('correct_classifications', []),
            'total_predictions': stats.get('total_predictions', 0),
            'confidence_levels': stats.get('confidence_levels', []),
            'class_counts': stats.get('class_counts', {})
        }
        
        # Create directory for analysis files
        analysis_dir = os.path.join(current_app.config['DATA_DIR'], 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"inference_analysis_{dict_name}_{timestamp}.json"
        filepath = os.path.join(analysis_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        return jsonify({
            "status": "success",
            "message": f"Analysis saved to {filename}"
        })
    except Exception as e:
        current_app.logger.error(f"Error saving analysis: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

def get_dictionary_by_name(dictionary_name):
    """
    Helper function to get a dictionary by name
    
    Args:
        dictionary_name (str): The name of the dictionary to find
        
    Returns:
        dict or None: The dictionary object if found, None otherwise
    """
    try:
        # First try to import this function if it exists elsewhere
        try:
            from config import get_dictionary_by_name as config_get_dictionary
            return config_get_dictionary(dictionary_name)
        except ImportError:
            pass
            
        # Otherwise use a local implementation
        dictionaries = current_app.config.get('DICTIONARIES', [])
        for dictionary in dictionaries:
            if dictionary.get('name') == dictionary_name:
                return dictionary
        return None
    except Exception as e:
        current_app.logger.error(f"Error getting dictionary by name '{dictionary_name}': {str(e)}")
        return None

@ml_blueprint.route('/api/ml/dictionary/<dictionary_name>/sounds', methods=['GET'])
def get_dictionary_sounds_by_name(dictionary_name):
    """API to get all sounds in a specific dictionary by name"""
    try:
        # Get the dictionary by name
        target_dict = get_dictionary_by_name(dictionary_name)
        
        if not target_dict:
            return jsonify({
                'success': False, 
                'message': f'Dictionary "{dictionary_name}" not found',
                'sounds': []
            }), 404
        
        return jsonify({
            'success': True,
            'dictionary': dictionary_name,
            'sounds': target_dict.get('sounds', [])
        })
    
    except Exception as e:
        current_app.logger.error(f"Error getting sounds for dictionary {dictionary_name}: {str(e)}")
        return jsonify({'success': False, 'message': str(e), 'sounds': []}), 500