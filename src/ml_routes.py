# Import the new model classes
from src.core.models import create_model, CNNModel, RandomForestModel as RFModel, EnsembleModel
from flask import Blueprint, request, jsonify, current_app, Response
import os
import json
import time
import traceback
from datetime import datetime

# Create a blueprint for ML routes
ml_blueprint = Blueprint('ml', __name__)

# Global variables
detector = None
current_model_path = None
current_model_type = None
current_model_dict = None

# Update the start_listening endpoint to use model classes and metadata
@ml_blueprint.route('/api/start_listening', methods=['POST'])
def start_listening():
    """
    Start listening for sound and making predictions using the specified model.
    """
    try:
        global detector, current_model_path, current_model_type, current_model_dict
        
        # Get model_id from request
        data = request.get_json()
        current_app.logger.info(f"🎧 LISTENING DEBUG: Received JSON data: {data}")
        
        model_id = data.get('model_id')
        current_app.logger.info(f"🎧 LISTENING DEBUG: Received request to start listening with model_id: {model_id}")
        
        if not model_id:
            current_app.logger.error("❌ LISTENING ERROR: No model_id provided in request")
            return jsonify({'status': 'error', 'message': 'No model_id provided'}), 400
        
        # Get models directory from app config
        models_dir = current_app.config.get('MODELS_DIR', 'models')
        current_app.logger.info(f"🎧 LISTENING DEBUG: Using models directory: {models_dir}")
        current_app.logger.info(f"🎧 LISTENING DEBUG: Absolute path: {os.path.abspath(models_dir)}")
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            current_app.logger.error(f"❌ LISTENING ERROR: Models directory not found: {models_dir}")
            return jsonify({'status': 'error', 'message': f'Models directory not found: {models_dir}'}), 404
        
        # Parse the model_id to get dictionary and model type
        parts = model_id.split('_')
        current_app.logger.info(f"🎧 LISTENING DEBUG: Parsed model_id parts: {parts}")
        
        if len(parts) < 3:
            current_app.logger.error(f"❌ LISTENING ERROR: Invalid model_id format: {model_id}, not enough parts")
            return jsonify({'status': 'error', 'message': 'Invalid model_id format'}), 400
        
        dict_name = parts[0]
        model_type = parts[1].lower()  # cnn, rf, or ens
        timestamp = parts[2]  # Extract the timestamp
        
        current_app.logger.info(f"🎧 LISTENING DEBUG: Extracted dictionary name: {dict_name}")
        current_app.logger.info(f"🎧 LISTENING DEBUG: Extracted model type: {model_type}")
        current_app.logger.info(f"🎧 LISTENING DEBUG: Extracted timestamp: {timestamp}")
        
        # Check if dictionary directory exists
        dict_dir = os.path.join(models_dir, dict_name)
        if os.path.exists(dict_dir):
            current_app.logger.info(f"🎧 LISTENING DEBUG: Dictionary directory exists: {dict_dir}")
            try:
                dict_files = os.listdir(dict_dir)
                current_app.logger.info(f"🎧 LISTENING DEBUG: Files in dictionary: {dict_files}")
            except Exception as e:
                current_app.logger.error(f"❌ LISTENING ERROR: Cannot list files in dictionary directory: {str(e)}")
        else:
            current_app.logger.warning(f"⚠️ LISTENING WARNING: Dictionary directory not found: {dict_dir}, will try root directory")
        
        # Construct model path based on model type
        model_path = None
        model_obj = None
        
        if model_type == 'cnn':
            # Try both in dictionary folder and root
            dict_model_path = os.path.join(models_dir, dict_name, f"{model_id}.h5")
            root_model_path = os.path.join(models_dir, f"{model_id}.h5")
            
            current_app.logger.info(f"🎧 LISTENING DEBUG: Checking for CNN model at: {dict_model_path}")
            current_app.logger.info(f"🎧 LISTENING DEBUG: Alternative path: {root_model_path}")
            
            if os.path.exists(dict_model_path):
                model_path = dict_model_path
                model_obj = create_model('cnn', model_dir=models_dir)
                current_app.logger.info(f"✅ LISTENING SUCCESS: Found CNN model in dictionary folder: {model_path}")
            elif os.path.exists(root_model_path):
                model_path = root_model_path
                model_obj = create_model('cnn', model_dir=models_dir)
                current_app.logger.info(f"✅ LISTENING SUCCESS: Found CNN model in root folder: {model_path}")
            else:
                current_app.logger.error(f"❌ LISTENING ERROR: CNN model file not found at either location")
        
        elif model_type == 'rf':
            # Try both in dictionary folder and root
            dict_model_path = os.path.join(models_dir, dict_name, f"{model_id}.joblib")
            root_model_path = os.path.join(models_dir, f"{model_id}.joblib")
            
            current_app.logger.info(f"🎧 LISTENING DEBUG: Checking for RF model at: {dict_model_path}")
            current_app.logger.info(f"🎧 LISTENING DEBUG: Alternative path: {root_model_path}")
            
            if os.path.exists(dict_model_path):
                model_path = dict_model_path
                model_obj = create_model('rf', model_dir=models_dir)
                current_app.logger.info(f"✅ LISTENING SUCCESS: Found RF model in dictionary folder: {model_path}")
            elif os.path.exists(root_model_path):
                model_path = root_model_path
                model_obj = create_model('rf', model_dir=models_dir)
                current_app.logger.info(f"✅ LISTENING SUCCESS: Found RF model in root folder: {model_path}")
            else:
                current_app.logger.error(f"❌ LISTENING ERROR: RF model file not found at either location")
        
        elif model_type in ['ens', 'ensemble']:
            # For ensemble, look for directory
            dict_model_path = os.path.join(models_dir, dict_name, model_id)
            root_model_path = os.path.join(models_dir, model_id)
            
            current_app.logger.info(f"🎧 LISTENING DEBUG: Checking for Ensemble model at: {dict_model_path}")
            current_app.logger.info(f"🎧 LISTENING DEBUG: Alternative path: {root_model_path}")
            
            if os.path.exists(dict_model_path) and os.path.isdir(dict_model_path):
                model_path = dict_model_path
                model_obj = create_model('ensemble', model_dir=models_dir)
                current_app.logger.info(f"✅ LISTENING SUCCESS: Found Ensemble model in dictionary folder: {model_path}")
            elif os.path.exists(root_model_path) and os.path.isdir(root_model_path):
                model_path = root_model_path
                model_obj = create_model('ensemble', model_dir=models_dir)
                current_app.logger.info(f"✅ LISTENING SUCCESS: Found Ensemble model in root folder: {model_path}")
            else:
                current_app.logger.error(f"❌ LISTENING ERROR: Ensemble model directory not found at either location")
        else:
            current_app.logger.error(f"❌ LISTENING ERROR: Unsupported model type: {model_type}")
            return jsonify({'status': 'error', 'message': f'Unsupported model type: {model_type}'}), 400
        
        # Check if model path was found
        if not model_path:
            current_app.logger.error(f"❌ LISTENING ERROR: Model file not found for {model_id}")
            return jsonify({'status': 'error', 'message': 'Model file not found'}), 404
        
        # Check if model file exists
        if not os.path.exists(model_path):
            current_app.logger.error(f"❌ LISTENING ERROR: Model file not found: {model_path}")
            return jsonify({'status': 'error', 'message': 'Model file not found'}), 404
        
        # Look for metadata file
        metadata_path = model_path.rstrip('.h5').rstrip('.joblib') + '_metadata.json'
        current_app.logger.info(f"🎧 LISTENING DEBUG: Looking for metadata file at: {metadata_path}")
        
        if not os.path.exists(metadata_path):
            current_app.logger.warning(f"⚠️ LISTENING WARNING: Metadata file not found: {metadata_path}")
            
            # Try alternative filename patterns
            model_dir = os.path.dirname(model_path)
            model_basename = os.path.basename(model_path)
            
            # Try alternative: same directory, different extension pattern
            alt_patterns = [
                os.path.splitext(model_basename)[0] + "_metadata.json",
                model_basename + "_metadata.json",
                model_basename.replace('.h5', '_metadata.json').replace('.joblib', '_metadata.json')
            ]
            
            for pattern in alt_patterns:
                alt_path = os.path.join(model_dir, pattern)
                current_app.logger.info(f"🎧 LISTENING DEBUG: Trying alternative metadata path: {alt_path}")
                
                if os.path.exists(alt_path):
                    metadata_path = alt_path
                    current_app.logger.info(f"✅ LISTENING SUCCESS: Found metadata at alternative path: {metadata_path}")
                    break
        
        if not os.path.exists(metadata_path):
            current_app.logger.error(f"❌ LISTENING ERROR: Could not find metadata file for model: {model_path}")
            return jsonify({'status': 'error', 'message': 'Model metadata file not found'}), 404
        
        # Load the model using our model classes
        current_app.logger.info(f"🎧 LISTENING DEBUG: Loading model from {model_path}")
        
        success = model_obj.load(model_path)
        if not success:
            current_app.logger.error(f"❌ LISTENING ERROR: Failed to load model: {model_path}")
            return jsonify({'status': 'error', 'message': 'Failed to load model'}), 500
        
        # Log metadata for debugging
        current_app.logger.info(f"🎧 LISTENING DEBUG: Model metadata keys: {list(model_obj.metadata.keys())}")
        
        # Get sound classes from metadata
        sound_classes = None
        if 'sound_classes' in model_obj.metadata:
            sound_classes = model_obj.metadata['sound_classes']
            current_app.logger.info(f"🎧 LISTENING DEBUG: Found sound_classes in metadata with {len(sound_classes)} classes")
        elif 'classes' in model_obj.metadata:
            sound_classes = model_obj.metadata['classes']
            current_app.logger.info(f"🎧 LISTENING DEBUG: Found classes in metadata with {len(sound_classes)} classes")
        elif 'class_names' in model_obj.metadata:
            sound_classes = model_obj.metadata['class_names']
            current_app.logger.info(f"🎧 LISTENING DEBUG: Found class_names in metadata with {len(sound_classes)} classes")
        
        if not sound_classes:
            current_app.logger.error(f"❌ LISTENING ERROR: No sound classes found in metadata for model: {model_path}")
            return jsonify({'status': 'error', 'message': 'No sound classes in model metadata'}), 500
        
        # Get input shape from metadata
        input_shape = model_obj.metadata.get('input_shape')
        if not input_shape:
            current_app.logger.warning(f"⚠️ LISTENING WARNING: No input shape in metadata for model: {model_path}")
            
            # Try to infer input shape
            if model_type == 'cnn' and hasattr(model_obj.model, 'input_shape'):
                input_shape = model_obj.model.input_shape[1:]  # Remove batch dimension
                current_app.logger.info(f"🎧 LISTENING DEBUG: Inferred input shape from model: {input_shape}")
            else:
                current_app.logger.error(f"❌ LISTENING ERROR: Cannot determine input shape for model: {model_path}")
                return jsonify({'status': 'error', 'message': 'No input shape in model metadata'}), 500
        
        # Get preprocessing parameters from metadata
        preprocessing_params = model_obj.metadata.get('preprocessing_params', {})
        if not preprocessing_params:
            current_app.logger.warning(f"⚠️ LISTENING WARNING: No preprocessing parameters in metadata for model: {model_path}")
            # Default parameters if not specified
            preprocessing_params = {
                "sample_rate": 22050,
                "n_mels": 128,
                "n_fft": 2048,
                "hop_length": 512,
                "sound_threshold": 0.01,
                "min_silence_duration": 0.5,
                "trim_silence": True,
                "normalize_audio": True
            }
            current_app.logger.info(f"🎧 LISTENING DEBUG: Using default preprocessing parameters: {preprocessing_params}")
        else:
            current_app.logger.info(f"🎧 LISTENING DEBUG: Using preprocessing parameters from metadata: {preprocessing_params}")
        
        # Create appropriate detector based on model type
        if model_type == 'cnn':
            from ml.inference import SoundDetector
            detector = SoundDetector(
                model=model_obj.model,
                sound_classes=sound_classes,
                input_shape=input_shape,
                preprocessing_params=preprocessing_params
            )
            current_app.logger.info(f"✅ LISTENING SUCCESS: Created CNN sound detector with {len(sound_classes)} classes")
        elif model_type == 'rf':
            from ml.sound_detector_rf import SoundDetectorRF
            detector = SoundDetectorRF(
                model=model_obj.model,
                sound_classes=sound_classes,
                preprocessing_params=preprocessing_params
            )
            current_app.logger.info(f"✅ LISTENING SUCCESS: Created RF sound detector with {len(sound_classes)} classes")
        elif model_type in ['ens', 'ensemble']:
            # For ensemble models, we need to handle both CNN and RF components
            from ml.sound_detector_ensemble import SoundDetectorEnsemble
            detector = SoundDetectorEnsemble(
                models=model_obj.models,
                sound_classes=sound_classes,
                preprocessing_params=preprocessing_params
            )
            current_app.logger.info(f"✅ LISTENING SUCCESS: Created Ensemble sound detector with {len(sound_classes)} classes")
        
        # Start the detector
        detector.start_listening()
        current_app.logger.info(f"✅ LISTENING SUCCESS: Started listening with {model_type} model")
        
        current_model_path = model_path
        current_model_type = model_type
        current_model_dict = dict_name
        
        return jsonify({
            'status': 'success', 
            'message': f'Started listening with {model_type} model',
            'sound_classes': sound_classes
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"❌ LISTENING ERROR: Unexpected error in start_listening: {str(e)}")
        current_app.logger.error(f"❌ LISTENING ERROR: Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500