# SoundClassifier_v08/src/routes/ml_routes.py

LATEST_PREDICTION = None

# Standard library imports
import os
import json
import logging
import time
import threading
import io
import glob
import traceback
import uuid
from datetime import datetime
from threading import Thread

# Flask imports
from flask import (
    Blueprint, render_template, request, session,
    redirect, url_for, flash, jsonify, Response, current_app
)

# Third-party imports
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
from scipy.io import wavfile
from tensorflow.keras.models import load_model

# Local imports
from config import Config
from src.services.training_service import TrainingService
# Import model classes
from src.ml.rf_classifier import RandomForestClassifier
from src.ml.ensemble_classifier import EnsembleClassifier
from src.ml.inference import predict_sound, SoundDetector

# Sound processing imports
from src.ml.sound_detector_rf import SoundDetectorRF
from src.ml.model_paths import get_cnn_model_path
from backend.features.extractor import FeatureExtractor

# Initialize blueprint
ml_bp = Blueprint('ml', __name__)


@ml_bp.before_app_request
def init_app_inference_stats():
    """Initialize inference statistics in the app context."""
    # Skip if outside application context or already initialized
    if not hasattr(current_app, '_get_current_object'):
        return

    app = current_app._get_current_object()

    # Only set up once
    if not hasattr(app, 'inference_stats'):
        try:
            # Setup inference statistics dictionary
            app.inference_stats = {
                'total_predictions': 0,
                'true_positives': 0,
                'false_positives': 0,
                'class_stats': {},
                'recent_predictions': [],
                'confusion_matrix': {}
            }
            logging.debug("Initialized current_app.inference_stats.")
        except Exception as e:
            logging.error("Error initializing inference stats: %s", str(e))

    # Minimal addition:
    if not hasattr(current_app, 'latest_prediction'):
        current_app.latest_prediction = None
        logging.debug("Initialized current_app.latest_prediction = None.")


# Store your dictionary here at module level
INFERENCE_STATS_DEFAULT = {
    'total_predictions': 0,
    'class_counts': {},
    'confidence_levels': []
}

# Global references (optional)
sound_detector = None
detector_lock = threading.Lock()


# For SSE
class DebugLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        self.logs.append(self.format(record))


debug_log_handler = DebugLogHandler()
debug_log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
debug_log_handler.setFormatter(formatter)
logging.getLogger().addHandler(debug_log_handler)


def prediction_callback(prediction):
    """
    Callback function to handle new predictions from SoundDetector.
    Updates inference statistics and stores the latest prediction.
    
    Args:
        prediction (dict): Prediction result
    """
    # Check if we're in an application context
    if not current_app:
        logging.warning("prediction_callback called outside application context")
        return

    try:
        # Store the prediction in the global application context
        current_app.latest_prediction = prediction

        # Update inference statistics if event is a prediction
        if prediction.get('event') == 'prediction':
            pred_data = prediction.get('prediction', {})
            pred_class = pred_data.get('class')
            confidence = pred_data.get('confidence', 0)

            # Update total count
            stats = current_app.inference_stats
            stats['total_predictions'] += 1

            # Update class statistics
            if pred_class:
                if pred_class not in stats.get('class_stats', {}):
                    stats['class_stats'][pred_class] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'correct': 0,
                        'incorrect': 0
                    }

                stats['class_stats'][pred_class]['count'] += 1

                # Update average confidence (running average)
                current_avg = stats['class_stats'][pred_class]['avg_confidence']
                current_count = stats['class_stats'][pred_class]['count']
                new_avg = ((current_avg * (current_count - 1)) + confidence) / current_count
                stats['class_stats'][pred_class]['avg_confidence'] = new_avg

            # Add to recent predictions (keep last 100)
            recent = stats.get('recent_predictions', [])
            recent.append({
                'class': pred_class,
                'confidence': confidence,
                'timestamp': time.time()
            })
            if len(recent) > 100:
                recent.pop(0)
            stats['recent_predictions'] = recent

            # Initialize confusion matrix entry if needed
            if pred_class:
                if 'confusion_matrix' not in stats:
                    stats['confusion_matrix'] = {}

    except Exception as e:
        logging.error("Error updating inference stats: %s", str(e))
        logging.error(traceback.format_exc())


def get_sound_list():
    active_dict = Config.get_dictionary()
    if not active_dict or 'sounds' not in active_dict:
        return []
    return active_dict['sounds']


def get_goodsounds_dir_path():
    """Return the path to the good sounds directory."""
    return Config.TRAINING_SOUNDS_DIR


@ml_bp.route('/train_model', methods=['GET', 'POST'])
def train_model():
    """
    Legacy route for model training, now redirects to the unified training approach.
    
    GET: Renders the training form
    POST: Processes the training request using the unified approach
    
    Returns:
        Rendered template or redirect to training status
    """
    if 'username' not in session:
        return redirect(url_for('index'))

    if request.method == 'GET':
        # Redirect to the unified training route
        flash('Using the improved unified training approach for better consistency and performance.', 'info')
        return redirect(url_for('ml.train_unified_model'))

    # For POST requests, also redirect to the unified training route
    return redirect(url_for('ml.train_unified_model'))


@ml_bp.route('/train_unified_model', methods=['GET', 'POST'])
def train_unified_model():
    """
    Route for training models using the unified approach.
    Uses the TrainingService to train CNN, RF, or Ensemble models with unified preprocessing.
    """
    if 'username' not in session:
        return redirect(url_for('index'))

    # Use the existing TrainingService instance
    training_service = TrainingService()

    if request.method == 'GET':
        # Load available dictionaries
        dictionaries = Config.get_dictionaries()
        dict_names = [d['name'] for d in dictionaries if 'name' in d]

        logging.info("Available dictionaries for training: %s", dict_names)
        logging.info("Dictionary directory: %s", os.path.join(Config.DATA_DIR, 'dictionaries'))
        dictionary_files = os.listdir(os.path.join(Config.DATA_DIR, 'dictionaries'))
        logging.info("Dictionary files in directory: %s", dictionary_files)

        return render_template(
            'train_unified.html',
            dictionaries=dict_names
        )

    logging.info("Received unified training form submission: %s", request.form)

    # Extract form data
    model_type = request.form.get('model_type', 'cnn')
    dict_name = request.form.get('dict_name')

    logging.info("Model type: %s, Dictionary name: %s", model_type, dict_name)

    # Validate model type
    if model_type not in ['cnn', 'rf', 'ensemble']:
        logging.error("Invalid model type: %s", model_type)
        flash(f'Invalid model type: {model_type}', 'error')
        return redirect(url_for('ml.train_unified_model'))

    # Load dictionary data
    dictionaries = Config.get_dictionaries()
    selected_dict = None
    for d in dictionaries:
        if d.get('name') == dict_name:
            selected_dict = d
            logging.info("Found matching dictionary: %s", selected_dict.get('name'))
            break

    if not selected_dict:
        logging.error("Dictionary not found in list: %s", dict_name)
        logging.info("Available dictionaries: %s", [d.get('name') for d in dictionaries])
        flash(f'Dictionary not found: {dict_name}', 'error')
        return redirect(url_for('ml.train_unified_model'))

    sounds = selected_dict.get('sounds', [])
    if not sounds:
        logging.error("No sounds found in dictionary: %s", dict_name)
        flash(f'No sounds found in dictionary: {dict_name}', 'error')
        return redirect(url_for('ml.train_unified_model'))

    sounds_dir = get_goodsounds_dir_path()
    if not sounds_dir or not os.path.exists(sounds_dir):
        logging.error("Training sounds directory not found: %s", sounds_dir)
        flash('Training sounds directory not found', 'error')
        return redirect(url_for('ml.train_unified_model'))

    logging.info("Found sounds directory: %s with sounds: %s", sounds_dir, sounds)

    # Set up training parameters based on model type
    training_params = {
        'model_type': model_type,
        'audio_dir': sounds_dir,
        'save': True,
        'model_name': f"{dict_name}_{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'class_names': sounds,
        'dict_name': dict_name
    }

    # Add model-specific parameters
    if model_type == 'cnn':
        epochs = int(request.form.get('epochs', 50))
        batch_size = int(request.form.get('batch_size', 32))
        use_class_weights = request.form.get('use_class_weights') == 'on'
        use_data_augmentation = request.form.get('use_data_augmentation') == 'on'

        training_params.update({
            'epochs': epochs,
            'batch_size': batch_size,
            'use_class_weights': use_class_weights,
            'use_data_augmentation': use_data_augmentation
        })

        logging.info("CNN training parameters: epochs=%d, batch_size=%d, use_class_weights=%s, use_data_augmentation=%s",
                     epochs, batch_size, use_class_weights, use_data_augmentation)

    elif model_type == 'rf':
        n_estimators = int(request.form.get('n_estimators', 100))
        max_depth = request.form.get('max_depth')
        if max_depth and max_depth.strip():
            max_depth = int(max_depth)
        else:
            max_depth = None

        training_params.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth
        })

        logging.info("RF training parameters: n_estimators=%d, max_depth=%s", n_estimators, max_depth)

    # Get current application object for async training
    app = current_app._get_current_object()

    if not hasattr(app, 'training_progress'):
        app.training_progress = 0

    if not hasattr(app, 'training_status'):
        app.training_status = "Not started"

    try:
        # Define an async training function that runs in a separate thread
        def train_async_unified():
            with app.app_context():
                try:
                    logging.info("Starting async training for %s model with classes: %s", model_type, sounds)
                    app.training_progress = 0
                    app.training_status = "Training in progress"

                    # Define progress callback function
                    def progress_callback(progress, status_message):
                        app.training_progress = progress
                        if status_message:
                            app.training_status = status_message
                        logging.info("Training progress: %d%%, Status: %s", progress, status_message)

                    logging.info("Calling train_unified with params: %s", training_params)
                    training_params['progress_callback'] = progress_callback

                    result = training_service.train_unified(**training_params)

                    if result and result.get('status') == 'success':
                        logging.info("Training completed successfully: %s", result.get('message'))
                        app.training_status = "Completed"
                        app.training_progress = 100
                    else:
                        error_msg = result.get('message') if result and 'message' in result else "Unknown error"
                        logging.error("Training failed: %s", error_msg)
                        app.training_status = f"Failed: {error_msg}"
                except Exception as e:
                    logging.error("Error in async training: %s", str(e))
                    traceback.print_exc()
                    app.training_status = f"Error: {str(e)}"

        # Start the async training thread
        thread = Thread(target=train_async_unified)
        thread.daemon = True
        thread.start()

        flash(f'Training started for {model_type} model using unified extractor', 'success')
        return redirect(url_for('ml.training_status'))
    except Exception as e:
        logging.error("Failed to start training thread: %s", str(e))
        flash(f'Failed to start training: {str(e)}', 'error')
        return redirect(url_for('ml.train_unified_model'))


def build_training_stats_and_history(
    X_train, y_train,
    X_val, y_val,
    class_names,
    dataset_stats,
    model_summary_str,
    keras_history
):
    # Basic dataset details
    training_stats = {
        'input_shape': str(X_train.shape[1:]),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'total_samples': int(len(X_train) + len(X_val)),
        'class_names': class_names,
        'model_summary_str': model_summary_str,
        # Merge in anything from dataset_stats if present
        'original_counts': dataset_stats.get('original_counts', {}),
        'augmented_counts': dataset_stats.get('augmented_counts', {})
    }

    # Convert the Keras history (keras_history.history) to a plain dictionary so it's JSON-serializable
    if hasattr(keras_history, 'history'):
        training_history = dict(keras_history.history)
    else:
        training_history = {}
    # Minimal addition below:
    if hasattr(keras_history, 'history') and 'loss' in keras_history.history:
        training_history['epochs'] = len(keras_history.history['loss'])
    else:
        training_history['epochs'] = 0

    return training_stats, training_history


@ml_bp.route('/predict_hub')
def predict_hub():
    return render_template('predict_hub.html')


@ml_bp.route('/predict')
def predict():
    if 'username' not in session:
        session['username'] = 'guest'

    active_dict = Config.get_dictionary()
    if not active_dict or 'sounds' not in active_dict:
        flash("No active dictionary found. Please create or select a dictionary first.")
        return redirect(url_for('ml.list_dictionaries'))

    return render_template('predict.html', active_dict=active_dict)


@ml_bp.route('/predict_cnn', methods=['GET', 'POST'])
def predict_cnn():
    """
    Handle CNN model prediction requests.

    GET: Renders the inference form
    POST: Processes audio and returns predictions

    Returns:
        Rendered template or JSON response with predictions
    """
    active_dict = Config.get_dictionary()
    if 'username' not in session:
        session['username'] = 'guest'

    dict_name = active_dict['name']
    expected_model_path = get_cnn_model_path('models', dict_name.replace(' ', '_'), 'v1')
    logging.info(f"Looking for CNN model for: {dict_name}")
    logging.info(f"Expected path: {expected_model_path}")

    if not os.path.exists(expected_model_path):
        models_dir = 'models'
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            logging.info(f"Available models: {files}")

            dict_key = dict_name.replace(' ', '_').lower()
            for f in files:
                if f.endswith('.h5') and dict_key in f.lower():
                    logging.info(f"Found potential model match: {f}")
                    expected_model_path = os.path.join(models_dir, f)
                    break

    if request.method == 'POST':
        try:
            use_microphone = (request.form.get('input_type') == 'microphone')

            if not os.path.exists(expected_model_path):
                flash(f"No CNN model found for dictionary: {dict_name}")
                return render_template('inference.html', active_dict=active_dict)

            model = load_model(expected_model_path)
            class_names = active_dict['sounds']

            # Handle audio input
            if use_microphone:
                # Use predict_sound function which handles microphone input
                pred_class, confidence = predict_sound(model, None, class_names, use_microphone=True)

                if pred_class:
                    return jsonify({
                        "predictions": [{
                            "sound": pred_class,
                            "probability": float(confidence)
                        }]
                    })

        except Exception as e:
            logging.error(f"Error during CNN prediction: {e}", exc_info=True)
            flash(str(e), "error")

    return render_template('inference.html', active_dict=active_dict)

@ml_bp.route('/predict_rf', methods=['POST'])
def predict_rf():
    active_dict = Config.get_dictionary()
    class_names = active_dict['sounds']
    
    rf_path = os.path.join('models', f"{active_dict['name'].replace(' ', '_')}_rf.joblib")
    if not os.path.exists(rf_path):
        return jsonify({"error": "No RF model"}), 400
    
    rf = RandomForestClassifier(model_dir='models')
    rf.load(filename=os.path.basename(rf_path))
    
    file = request.files.get('audio')
    if not file:
        return jsonify({"error": "No audio file"}), 400
    
    filename = f"rf_predict_{uuid.uuid4().hex[:8]}.wav"
    temp_path = os.path.join(Config.UPLOADED_SOUNDS_DIR, filename)
    file.save(temp_path)
    
    sp = SoundProcessor(sample_rate=16000)  # noqa: F821
    audio_data, _ = librosa.load(temp_path, sr=16000)
    
    start_idx, end_idx, has_sound = sp.detect_sound_boundaries(audio_data)
    trimmed_data = audio_data[start_idx:end_idx]
    if len(trimmed_data) == 0:
        return jsonify({"error": "No sound detected in audio"}), 400

    wavfile.write(temp_path, 16000, np.int16(trimmed_data * 32767))
    
    extractor = FeatureExtractor(sample_rate=16000)
    all_features = extractor.extract_features(temp_path)
    if not all_features:
        return jsonify({"error": "Feature extraction failed"}), 500
        
    feats = extractor.extract_features_for_model(all_features, model_type='rf')
    if not feats:
        return jsonify({"error": "Feature extraction failed"}), 500

    feature_names = list(feats.keys())
    row = [feats[fn] for fn in feature_names]
    
    preds, probs = rf.predict([row])
    if preds is None:
        return jsonify({"error": "RF predict() returned None"}), 500
    
    predicted_class = preds[0]
    confidence = float(np.max(probs[0]))
    
    return jsonify({
        "predictions": [{
            "sound": predicted_class,
            "probability": confidence
        }]
    })

@ml_bp.route('/predict_ensemble', methods=['POST'])
def predict_ensemble():
    active_dict = Config.get_dictionary()
    class_names = active_dict['sounds']

    cnn_path = os.path.join('models', f"{active_dict['name'].replace(' ', '_')}_model.h5")
    if not os.path.exists(cnn_path):
        return jsonify({"error": "No CNN model"}), 400
    cnn_model = load_model(cnn_path)

    rf_path = os.path.join('models', f"{active_dict['name'].replace(' ', '_')}_rf.joblib")
    if not os.path.exists(rf_path):
        return jsonify({"error": "No RF model"}), 400

    rf = RandomForestClassifier(model_dir='models')
    rf.load(filename=os.path.basename(rf_path))

    ensemble = EnsembleClassifier(rf, cnn_model, class_names, rf_weight=0.5)

    file = request.files.get('audio')
    if not file:
        return jsonify({"error": "No audio file"}), 400

    filename = f"ensemble_predict_{uuid.uuid4().hex[:8]}.wav"
    temp_path = os.path.join(Config.UPLOADED_SOUNDS_DIR, filename)
    file.save(temp_path)

    # Use the unified FeatureExtractor for both CNN and RF features
    extractor = FeatureExtractor(sample_rate=16000)
    all_features = extractor.extract_features(temp_path)

    if all_features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    # Extract model-specific features
    ensemble_features = extractor.extract_features_for_model(all_features, model_type='ensemble')

    if ensemble_features is None:
        return jsonify({"error": "Feature extraction failed for ensemble"}), 500

    # Get features for each model type
    cnn_features = ensemble_features['cnn']
    rf_features = ensemble_features['rf']

    # Prepare CNN features
    X_cnn = np.expand_dims(cnn_features, axis=0)

    # Prepare RF features
    feature_names = list(rf_features.keys())
    X_rf = [rf_features[fn] for fn in feature_names]

    # Clean up temporary file
    os.remove(temp_path)

    # Get predictions
    top_preds = ensemble.get_top_predictions(X_rf, X_cnn, top_n=1)[0]

    return jsonify({
        "predictions": top_preds
    })

@ml_bp.route('/predict_sound', methods=['POST'])
def predict_sound_endpoint():
    active_dict = Config.get_dictionary()
    dict_name = active_dict['name']
    model_path = os.path.join('models', f"{dict_name.replace(' ', '_')}_model.h5")
    if not os.path.exists(model_path):
        return jsonify({"error": f"No model for dictionary {dict_name}"}), 400

    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file"}), 400

    model = load_model(model_path)
    class_names = active_dict['sounds']

    filename = f"predict_temp_{uuid.uuid4().hex[:8]}.wav"
    temp_path = os.path.join(Config.UPLOADED_SOUNDS_DIR, filename)
    with open(temp_path, 'wb') as f:
        f.write(audio_file.read())

    pred_class, confidence = predict_sound(model, temp_path, class_names, use_microphone=False)
    os.remove(temp_path)

    return jsonify({
        "predictions": [{
            "sound": pred_class,
            "probability": float(confidence)
        }]
    })


@ml_bp.route('/start_listening', methods=['POST'])
def start_listening():
    """
    Start real-time sound detection and classification.
    
    Request body:
    {
        "model_id": "model_identifier",
        "use_ambient_noise": true|false
    }
    
    Returns:
        JSON with start status and model info
    """
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    # Response template
    response = {
        'status': 'success',
        'message': 'Started listening',
        'model_type': None,
        'model_id': None,
        'sound_classes': []
    }

    # Access sound_detector as module-level variable
    global sound_detector

    try:
        logging.info("========== START LISTENING API CALLED ==========")

        # Parse request data
        data = request.json
        model_id = data.get('model_id')
        use_ambient_noise = data.get('use_ambient_noise', False)

        logging.info("Request data: model_id=%s, use_ambient_noise=%s",
                     model_id, use_ambient_noise)

        # Validate model_id
        if not model_id:
            logging.error("No model_id provided in request")
            return jsonify({
                'status': 'error',
                'message': 'model_id is required'
            }), 400

        # Default to CNN model if model type not specified
        model_choice = data.get('model_type', '').lower()
        if model_choice not in ['cnn', 'rf', 'ensemble']:
            model_choice = 'cnn'

        logging.info("Starting listening with model ID: %s, type: %s, use_ambient_noise: %s",
                     model_id, model_choice, use_ambient_noise)

        # Get class names from model registry or active dictionary
        class_names = []
        models_json_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')

        if os.path.exists(models_json_path):
            try:
                with open(models_json_path, 'r') as f:
                    models_registry = json.load(f)

                model_data = None
                # Look for the model in the registry
                for mtype, models in models_registry.get('models', {}).items():
                    if model_id in models:
                        model_data = models[model_id]
                        break

                if model_data and 'class_names' in model_data:
                    class_names = model_data['class_names']
                    logging.info("Found %d class names from registry for %s",
                                 len(class_names), model_id)
            except Exception as e:
                logging.error("Error loading models.json: %s", str(e))

        # If no class names found in model registry, use active dictionary
        if not class_names:
            active_dict = Config.get_dictionary()
            class_names = active_dict.get('sounds', [])
            logging.info("Using %d class names from active dictionary", len(class_names))

        # Ensure we have class names
        if not class_names:
            return jsonify({
                'status': 'error',
                'message': 'No sounds in dictionary or model metadata'
            }), 400

        # Initialize the appropriate model
        if model_choice == 'rf':
            # Find and load the RF model
            rf_path = None

            if model_data and 'file_path' in model_data:
                rf_path = model_data['file_path']
            else:
                # Look for models with consistent naming
                rf_path = f"models/{model_id}.joblib"

            if not os.path.exists(rf_path):
                logging.error("RF model file not found at %s", rf_path)
                return jsonify({
                    'status': 'error',
                    'message': f'No RF model file found at {rf_path}'
                }), 400

            logging.info("Loading RF model from %s", rf_path)
            rf_classifier = RandomForestClassifier()
            rf_classifier.load(rf_path)

            # Create and start the RF detector
            sound_detector = SoundDetectorRF(
                model=rf_classifier,
                sound_classes=class_names,
                callback=prediction_callback,
                sample_rate=16000
            )

            sound_detector.start_listening(measure_ambient=use_ambient_noise)

            # Update response
            response['model_type'] = 'rf'
            response['model_id'] = model_id
            response['sound_classes'] = class_names

        else:  # Default to CNN
            # Find and load the CNN model
            model_path = None

            if model_data and 'file_path' in model_data:
                model_path = model_data['file_path']
            else:
                # Look for models with consistent naming
                model_path = f"models/{model_id}.h5"

            if not os.path.exists(model_path):
                logging.error("Model file not found at %s", model_path)
                return jsonify({
                    'status': 'error',
                    'message': f'Model file not found at {model_path}'
                }), 404

            logging.info("Loading CNN model from %s", model_path)
            try:
                logging.info("Loading Keras model...")
                model = load_model(model_path)
                logging.info("Model loaded successfully with input shape: %s", model.input_shape)
            except Exception as e:
                logging.error("Error loading model: %s", str(e))
                traceback.print_exc()
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to load model: {str(e)}'
                }), 500

            # Create and start the CNN detector
            try:
                sound_detector = SoundDetector(
                    model=model,
                    sound_classes=class_names,
                    callback=prediction_callback,
                    sample_rate=16000
                )

                sound_detector.start_listening(measure_ambient=use_ambient_noise)

                # Update response
                response['model_type'] = 'cnn'
                response['model_id'] = model_id
                response['sound_classes'] = class_names
            except Exception as e:
                logging.error("Error creating/starting SoundDetector: %s", str(e))
                traceback.print_exc()
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to start sound detector: {str(e)}'
                }), 500

        return jsonify(response)

    except Exception as e:
        logging.error("Error in start_listening: %s", str(e))
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/stop_listening', methods=['POST'])
def stop_listening():
    """
    Stop real-time sound detection and classification.

    Returns:
        JSON with stop status
    """
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    global sound_detector
    try:
        if sound_detector:
            result = sound_detector.stop_listening()
            sound_detector = None
            return jsonify({
                'status': 'success',
                'message': 'Listening stopped'
            })

        return jsonify({
            'status': 'error',
            'message': 'No active listener'
        })
    except Exception as e:
        logging.error("Error stopping listener: %s", str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@ml_bp.route('/prediction_stream')
def prediction_stream():
    """
    Stream predictions as server-sent events.
    This endpoint allows clients to receive real-time prediction updates.

    Returns:
        A stream of server-sent events containing prediction data
    """
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    def stream_predictions():
        """Generator function that yields prediction data as SSE events"""
        while True:
            data = {}
            # Access the latest prediction from the application context
            if hasattr(current_app, 'latest_prediction') and current_app.latest_prediction:
                data['prediction'] = current_app.latest_prediction
                current_app.latest_prediction = None  # Clear after sending

                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.2)  # Throttle rate to reduce client load

    response = Response(
        stream_predictions(),
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable proxy buffering
    return response


@ml_bp.route('/inference_statistics')
def inference_statistics():
    try:
        stats = None

        if hasattr(current_app, 'inference_stats'):
            stats = current_app.inference_stats
        elif hasattr(current_app, 'ml_api') and hasattr(current_app.ml_api, 'inference_service'):
            stats = current_app.ml_api.inference_service.inference_stats
        elif hasattr(current_app, 'inference_service'):
            stats = current_app.inference_service.get_inference_stats()

        if stats is None:
            logging.warning("No inference stats found on app, creating minimal stats")
            stats = {
                'total_predictions': 0,
                'class_counts': {},
                'confidence_levels': [],
                'average_confidence': 0,
                'classes_found': [],
                'status': 'initialized'
            }
            current_app.inference_stats = stats

        if 'average_confidence' not in stats and stats.get('confidence_levels'):
            confidences = stats.get('confidence_levels', [])
            stats['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0

        if 'classes_found' not in stats:
            stats['classes_found'] = list(stats.get('class_counts', {}).keys())

        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error getting inference statistics: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"Failed to get inference statistics: {str(e)}"
        }), 500


@ml_bp.route('/record_feedback', methods=['POST'])
def record_feedback():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    try:
        data = request.get_json()

        required_fields = ['predicted_class', 'actual_class']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400

        if 'is_correct' not in data:
            data['is_correct'] = (data['predicted_class'] == data['actual_class'])

        updated_stats = current_app.inference_service.record_feedback(data)

        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded successfully',
            'updated_stats': updated_stats
        })
    except Exception as e:
        logging.error(f"Error recording feedback: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to record feedback: {str(e)}"
        }), 500


@ml_bp.route('/save_analysis', methods=['POST'])
def save_analysis():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    try:
        stats = current_app.inference_service.get_inference_stats()
        dict_name = Config.get_dictionary().get('name', 'unknown')

        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'dictionary': dict_name,
            **stats
        }

        os.makedirs(Config.ANALYSIS_DIR, exist_ok=True)
        filename = f"analysis_{dict_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(Config.ANALYSIS_DIR, filename)

        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        return jsonify({
            'status': 'success',
            'message': 'Analysis data saved',
            'filepath': filepath
        })
    except Exception as e:
        logging.error(f"Error saving analysis data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to save analysis data: {str(e)}"
        }), 500


@ml_bp.route('/view_analysis')
def view_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))

    if not os.path.exists(Config.ANALYSIS_DIR):
        return render_template('view_analysis.html', analysis_files=[])

    analysis_files = []
    for fname in os.listdir(Config.ANALYSIS_DIR):
        if fname.endswith('.json'):
            path = os.path.join(Config.ANALYSIS_DIR, fname)
            with open(path, 'r') as f:
                data = json.load(f)
            analysis_files.append({
                'filename': fname,
                'timestamp': data.get('timestamp', ''),
                'dictionary': data.get('dictionary', ''),
                'total_predictions': data.get('total_predictions', 0)
            })
    analysis_files.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template('view_analysis.html', analysis_files=analysis_files)


@ml_bp.route('/get_analysis/<filename>')
def get_analysis(filename):
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    filepath = os.path.join(Config.ANALYSIS_DIR, filename)

    if not os.path.abspath(filepath).startswith(os.path.abspath(Config.ANALYSIS_DIR)):
        return jsonify({'status': 'error', 'message': 'Invalid filename'}), 400

    if not os.path.exists(filepath):
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/training_status')
def training_status():
    if not hasattr(current_app, 'training_progress'):
        current_app.training_progress = 0
    if not hasattr(current_app, 'training_status'):
        current_app.training_status = 'Not started'

    status_data = {
        'progress': current_app.training_progress,
        'status': current_app.training_status
    }

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
              request.accept_mimetypes.best_match(['application/json', 'text/html']) == 'application/json'

    if is_ajax:
        return jsonify(status_data)
    else:
        return render_template('training_status.html')


@ml_bp.route('/record', methods=['POST'])
def record():
    if 'username' not in session:
        return redirect(url_for('index'))

    sound = request.form.get('sound')
    audio_data = request.files.get('audio')
    current_app.logger.debug(f"Recording attempt. sound={sound}, has audio={audio_data is not None}")
    if sound and audio_data:
        try:
            audio = AudioSegment.from_file(audio_data, format="webm")
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
        except Exception as e:
            current_app.logger.error(f"Error converting audio: {e}", exc_info=True)
            return "Error converting audio", 500

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"{sound}_{session['username']}_{timestamp}.wav"
            temp_path = os.path.join(Config.TEMP_DIR, temp_filename)
            with open(temp_path, 'wb') as f:
                f.write(wav_io.read())

            processor = Audio_Chunker()  # noqa: F821
            chopped_files = processor.chop_recording(temp_path)
            os.remove(temp_path)

            if not chopped_files:
                current_app.logger.error("No valid sound chunks found")
                flash("No valid sound chunks found in your recording.")
                return redirect(url_for('index'))

            return redirect(url_for('ml.verify_chunks', timestamp=timestamp))
        except Exception as e:
            current_app.logger.error(f"Error processing recording: {e}", exc_info=True)
            return "Error processing recording", 500

    flash("Sound or audio data missing.")
    return redirect(url_for('index'))


@ml_bp.route('/verify/<timestamp>')
def verify_chunks(timestamp):
    if 'username' not in session:
        return redirect(url_for('index'))

    chunks = [f for f in os.listdir(Config.TEMP_DIR) if timestamp in f]
    if not chunks:
        flash('No chunks found to verify or all processed.')
        return redirect(url_for('index'))
    return render_template('verify.html', chunks=chunks)


@ml_bp.route('/process_verification', methods=['POST'])
def process_verification():
    if 'username' not in session:
        return redirect(url_for('index'))

    chunk_file = request.form.get('chunk_file')
    is_good = request.form.get('is_good') == 'true'
    if not chunk_file:
        return redirect(url_for('index'))

    parts = chunk_file.split('_')
    if len(parts) < 3:
        flash("File name does not match expected pattern.")
        return redirect(url_for('index'))
    timestamp = parts[-2]

    if is_good:
        sound = parts[0]
        username = session['username']
        sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
        os.makedirs(sound_dir, exist_ok=True)

        existing_count = len([
            f for f in os.listdir(sound_dir)
            if f.startswith(f"{sound}_{username}_")
        ])
        new_filename = f"{sound}_{username}_{existing_count + 1}.wav"
        new_path = os.path.join(sound_dir, new_filename)
        os.rename(os.path.join(Config.TEMP_DIR, chunk_file), new_path)
        flash(f"Chunk saved as {new_filename}")
    else:
        os.remove(os.path.join(Config.TEMP_DIR, chunk_file))
        flash("Chunk deleted.")

    return redirect(url_for('ml.verify_chunks', timestamp=timestamp))


@ml_bp.route('/manage_dictionaries')
def list_dictionaries():
    logging.info("=== DICTIONARIES PAGE REQUESTED ===")

    active_dict = Config.get_dictionary()
    if not active_dict:
        flash("No active dictionary.")
        return redirect(url_for('index'))
    dictionaries = Config.get_dictionaries()

    sound_stats = {}
    if active_dict and 'sounds' in active_dict:
        for sound in active_dict['sounds']:
            sound_stats[sound] = {}

    return render_template('dictionaries.html',
                           dictionaries=dictionaries,
                           active_dictionary=active_dict,
                           sound_stats=sound_stats)


@ml_bp.route('/save_dictionary', methods=['POST'])
def save_dictionary():
    if not session.get('is_admin'):
        return redirect(url_for('index'))

    name = request.form.get('name')
    sounds_str = request.form.get('sounds')
    if not name or not sounds_str:
        flash("Please provide dictionary name and sounds")
        return redirect(url_for('ml.list_dictionaries'))

    sounds = [s.strip() for s in sounds_str.split(',') if s.strip()]
    new_dict = {"name": name, "sounds": sounds}
    dictionaries = Config.get_dictionaries()

    found = False
    for d in dictionaries:
        if d['name'] == name:
            d['sounds'] = sounds
            found = True
            break
    if not found:
        dictionaries.append(new_dict)

    Config.save_dictionaries(dictionaries)
    Config.set_active_dictionary(new_dict)
    flash("Dictionary saved and activated.")
    return redirect(url_for('ml.list_dictionaries'))


@ml_bp.route('/make_active', methods=['POST'])
def make_active():
    if not session.get('is_admin'):
        return redirect(url_for('index'))

    name = request.form.get('name')
    if not name:
        flash('Dictionary name is required')
        return redirect(url_for('ml.list_dictionaries'))

    dictionaries = Config.get_dictionaries()
    selected_dict = None
    for d in dictionaries:
        if d['name'] == name:
            selected_dict = d
            break
    if not selected_dict:
        flash('Dictionary not found')
        return redirect(url_for('ml.list_dictionaries'))

    # Set the selected dictionary as active.
    Config.get_dictionary(selected_dict)
    # Create directories for each sound.
    for sound in selected_dict['sounds']:
        os.makedirs(os.path.join(Config.TRAINING_SOUNDS_DIR, sound), exist_ok=True)

    flash(f'Dictionary "{name}" is now active')
    return redirect(url_for('ml.list_dictionaries'))


@ml_bp.route('/set_active_dictionary', methods=['POST'])
def set_active_dictionary():
    if not session.get('is_admin'):
        return redirect(url_for('index'))

    name = request.form.get('name')
    if name:
        dictionaries = Config.get_dictionaries()
        for d in dictionaries:
            if d['name'] == name:
                Config.set_active_dictionary(d)
                flash(f'Activated dictionary: {name}')
                break
    return redirect(url_for('ml.list_dictionaries'))


@ml_bp.route('/ml.list_recordings')
def list_recordings():
    if 'username' not in session:
        return redirect(url_for('index'))
    active_dict = Config.get_dictionary()
    if not active_dict:
        flash("No active dictionary.")
        return redirect(url_for('index'))

    if session.get('is_admin'):
        recordings_by_sound = {}
        for sound in active_dict['sounds']:
            sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
            if os.path.exists(sound_dir):
                files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]
                if files:
                    recordings_by_sound[sound] = files
        return render_template('list_recordings.html', recordings=recordings_by_sound)
    else:
        user = session['username']
        recordings_by_sound = {}
        for sound in active_dict['sounds']:
            sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
            if not os.path.exists(sound_dir):
                continue
            sound_files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]
            user_files = [f for f in sound_files if f.split('_')[1] == user]
            if user_files:
                recordings_by_sound[sound] = user_files
        return render_template('list_recordings.html', recordings=recordings_by_sound)


@ml_bp.route('/ml.get_sound_stats')
def get_sound_stats():
    if not session.get('is_admin'):
        return redirect(url_for('index'))

    active_dict = Config.get_dictionary()
    if not active_dict:
        return jsonify({})

    sound_stats = {}
    for sound in active_dict['sounds']:
        sound_stats[sound] = {'system_total': 0, 'user_total': 0}
        sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
        if os.path.exists(sound_dir):
            files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]
            sound_stats[sound]['system_total'] = len(files)
    return json.dumps(sound_stats)


@ml_bp.route('/ml.upload_sounds')
def upload_sounds():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('upload_sounds.html',
                           sounds=Config.get_dictionary()['sounds'])


@ml_bp.route('/ml.process_uploads', methods=['POST'])
def process_uploads():
    if 'username' not in session:
        return redirect(url_for('index'))

    sound = request.form.get('sound')
    files = request.files.getlist('files')
    if not sound or not files:
        flash("Please select both a sound and files.")
        return redirect(url_for('ml.upload_sounds'))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processor = SoundProcessor()  # noqa: F821
    all_chunks = []

    for file in files:
        if file.filename.lower().endswith('.wav'):
            temp_path = os.path.join(
                Config.TEMP_DIR, f"{sound}_{session['username']}_{timestamp}_temp.wav"
            )
            file.save(temp_path)
            chunks = processor.chop_recording(temp_path)
            all_chunks.extend(chunks)
            os.remove(temp_path)

    if not all_chunks:
        flash("No valid sound chunks found in uploads.")
        return redirect(url_for('ml.upload_sounds'))

    return redirect(url_for('ml.verify_chunks', timestamp=timestamp))


def gather_training_stats_for_rf():
    """
    Get RF training statistics (legacy method).
    Returns:
        dict: RF training statistics
    """
    global training_stats, training_history, model_summary_str
    if training_stats is None:
        return {}
    return training_stats


@ml_bp.route("/rf_training_summary")
def rf_training_summary():
    global training_stats, training_history, model_summary_str

    dict_name = Config.get_dictionary().get('name', 'Unknown')
    goodsounds_dir = get_goodsounds_dir_path()

    try:
        if training_stats and training_stats.get('model_type') == 'rf':
            return render_template("rf_training_summary_stats.html",
                                   training_stats=training_stats)

        models_dir = os.path.join('models', 'rf', dict_name.replace(' ', '_'))
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            if model_files:
                model_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(models_dir, x)),
                    reverse=True
                )
                model_base = model_files[0].replace('.joblib', '')
                metadata_file = os.path.join(models_dir, f"{model_base}_metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        training_stats = json.load(f)
                    return render_template("rf_training_summary_stats.html",
                                           training_stats=training_stats)

        logging.warning("No RF model found, training a new one")
        training_service = TrainingService()
        from src.core.ml_algorithms.rf import RandomForestModel
        model = RandomForestModel(
            model_dir=os.path.join('models', 'rf', dict_name.replace(' ', '_'))
        )
        result = training_service.train_model(
            'rf',
            goodsounds_dir,
            save=True,
            dict_name=dict_name,
            n_estimators=100
        )
        if result.get('success', False):
            training_stats = result.get('stats', {})
            return render_template("rf_training_summary_stats.html",
                                   training_stats=training_stats)
        else:
            flash(f"RF training failed: {result.get('error', 'Unknown error')}")
            return redirect(url_for('ml.train_model'))
    except Exception as e:
        logging.error(f"Error in RF training: {str(e)}", exc_info=True)
        flash(f"Error during RF training: {str(e)}")
        return redirect(url_for('ml.train_model'))


#########################
# Minimal Helper Functions
#########################
def gather_training_stats_for_cnn():
    global training_stats
    if training_stats is None:
        return {}
    return training_stats


def get_cnn_history():
    global training_history
    if training_history is None:
        return {}
    return training_history


@ml_bp.route("/cnn_training_summary")
def cnn_training_summary():
    global training_stats, training_history, model_summary_str

    training_service = TrainingService()
    stats = gather_training_stats_for_cnn()
    history = get_cnn_history()
    model_summary = stats.get('model_summary', model_summary_str)

    return render_template(
        "cnn_training_summary_stats.html",
        training_stats=stats,
        training_history=history,
        model_summary=model_summary
    )


def gather_training_stats_for_ensemble():
    global training_stats
    if training_stats is None:
        return {}
    return training_stats


@ml_bp.route("/ensemble_training_summary")
def ensemble_training_summary():
    global training_stats

    training_service = TrainingService()
    stats = gather_training_stats_for_ensemble()

    return render_template("ensemble_training_summary_stats.html",
                           training_stats=stats)


def gather_training_stats_for_all():
    global training_stats
    if training_stats is None:
        return {}
    return training_stats


@ml_bp.route("/training_model_comparisons")
def training_model_comparisons():
    global training_stats

    training_service = TrainingService()
    stats = gather_training_stats_for_all()

    return render_template("training_model_comparisons_stats.html",
                           training_stats=stats)


@ml_bp.route('/model_summary')
def model_summary():
    return render_template('model_summary_hub.html')


@ml_bp.route('/api/ml/models')
def get_available_models():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    try:
        active_dict = Config.get_dictionary()
        dict_name = active_dict.get('name', 'Unknown')
        models_json_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')
        available_models = []

        if os.path.exists(models_json_path):
            try:
                with open(models_json_path, 'r') as f:
                    models_registry = json.load(f)
                logging.info(f"Successfully loaded models registry from {models_json_path}")

                if 'models' in models_registry and 'cnn' in models_registry['models']:
                    cnn_models = models_registry['models']['cnn']

                    for model_id, model_data in cnn_models.items():
                        file_path = model_data.get('file_path', '')
                        model_path = os.path.join(Config.BASE_DIR, 'data', 'models', file_path)
                        class_names = []
                        metadata_path = model_path.replace('.h5', '_metadata.json')
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                class_names = metadata.get('class_names', [])
                                logging.info(f"Loaded {len(class_names)} class names from {metadata_path}")
                            except Exception as e:
                                logging.warning(f"Error loading metadata for {model_id}: {e}")
                        if not class_names and 'class_names' in model_data:
                            class_names = model_data['class_names']
                            logging.info(f"Using {len(class_names)} class names from model_data")
                        elif not class_names and 'sounds' in active_dict:
                            class_names = active_dict['sounds']
                            logging.info(f"Using {len(class_names)} class names from active dictionary")

                        available_models.append({
                            'id': model_id,
                            'name': f"{dict_name} CNN Model",
                            'type': 'cnn',
                            'dictionary': dict_name.replace(' ', '_'),
                            'class_names': class_names
                        })
                else:
                    logging.warning(f"No models found for '{dict_name}', adding fallback")
                    class_names = active_dict.get('sounds', [])
                    available_models = [{
                        'id': f"{dict_name.replace(' ', '_')}_model",
                        'name': f"{dict_name} Default Model",
                        'type': 'cnn',
                        'dictionary': dict_name.replace(' ', '_'),
                        'class_names': class_names
                    }]
            except Exception as e:
                logging.error(f"Error processing models.json: {e}", exc_info=True)
        return jsonify({
            'success': True,
            'models': available_models,
            'dictionary_name': dict_name
        })
    except Exception as e:
        logging.error(f"Error getting available models: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'success': False
        }), 500


@ml_bp.route('/api/dictionary/sounds')
def get_dictionary_sounds():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    try:
        active_dict = Config.get_dictionary()
        sounds = active_dict.get('sounds', [])
        return jsonify({'sounds': sounds})
    except Exception as e:
        logging.error(f"Error getting dictionary sounds: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@ml_bp.route('/views/analysis/<analysis_id>')
def view_analysis_detail(analysis_id):
    try:
        analysis_path = os.path.join(Config.ANALYSIS_DIR, f"analysis_{analysis_id}.json")
        if not os.path.exists(analysis_path):
            return render_template('error.html', error_message=f"Analysis {analysis_id} not found"), 404
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        return render_template('analysis.html', analysis=analysis_data)
    except Exception as e:
        logging.error(f"Error viewing analysis: {e}")
        return render_template('error.html', error_message=f"Error viewing analysis: {e}"), 500


@ml_bp.route('/api/analysis/<analysis_id>')
def get_analysis_data(analysis_id):
    try:
        analysis_path = os.path.join(Config.ANALYSIS_DIR, f"analysis_{analysis_id}.json")
        if not os.path.exists(analysis_path):
            return jsonify({"error": f"Analysis {analysis_id} not found"}), 404
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        return jsonify(analysis_data)
    except Exception as e:
        logging.error(f"Error getting analysis data: {e}")
        return jsonify({"error": f"Error getting analysis data: {e}"}), 500


@ml_bp.route('/api/analysis')
def list_analyses():
    try:
        analysis_dir = Config.ANALYSIS_DIR
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir, exist_ok=True)
            return jsonify({"analyses": []})
        analysis_files = glob.glob(os.path.join(analysis_dir, "analysis_*.json"))
        analyses = []
        for file_path in analysis_files:
            file_name = os.path.basename(file_path)
            analysis_id = file_name.replace("analysis_", "").replace(".json", "")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    analyses.append({
                        "id": analysis_id,
                        "name": data.get("model_name", "Unknown"),
                        "date": data.get("date", "Unknown"),
                        "accuracy": data.get("accuracy", 0),
                        "dictionary": data.get("dictionary", "Unknown")
                    })
            except Exception as e:
                logging.error(f"Error reading analysis file {file_path}: {e}")
        return jsonify({"analyses": analyses})
    except Exception as e:
        logging.error(f"Error listing analyses: {e}")
        return jsonify({"error": f"Error listing analyses: {e}"}), 500


@ml_bp.route('/confidence_distribution')
def confidence_distribution():
    try:
        distribution = current_app.inference_service.get_confidence_distribution()
        return jsonify(distribution)
    except Exception as e:
        logging.error(f"Error getting confidence distribution: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to get confidence distribution: {str(e)}"
        }), 500


@ml_bp.route('/detect_drift')
def detect_drift():
    try:
        window_size = request.args.get('window_size', 100, type=int)
        drift_analysis = current_app.inference_service.detect_drift(window_size)
        return jsonify(drift_analysis)
    except Exception as e:
        logging.error(f"Error detecting model drift: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to detect model drift: {str(e)}"
        }), 500


@ml_bp.route('/recent_predictions')
def recent_predictions():
    try:
        count = request.args.get('count', 10, type=int)
        predictions = current_app.inference_service.get_recent_predictions(count)
        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'predictions': predictions
        })
    except Exception as e:
        logging.error(f"Error getting recent predictions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to get recent predictions: {str(e)}"
        }), 500


@ml_bp.route('/class_accuracy/<class_name>')
def class_accuracy(class_name):
    try:
        accuracy = current_app.inference_service.get_class_accuracy(class_name)
        return jsonify({
            'status': 'success',
            'class': class_name,
            'accuracy': accuracy
        })
    except Exception as e:
        logging.error(f"Error getting class accuracy: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to get class accuracy: {str(e)}"
        }), 500


@ml_bp.route('/reset_inference_stats', methods=['POST'])
def reset_inference_stats():
    try:
        fresh_stats = current_app.inference_service.reset_stats()
        return jsonify({
            'status': 'success',
            'message': 'Inference statistics reset successfully',
            'stats': fresh_stats
        })
    except Exception as e:
        logging.error(f"Error resetting inference statistics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to reset inference statistics: {str(e)}"
        }), 500


@ml_bp.route('/model_metadata/<model_id>', methods=['GET'])
def get_model_metadata_direct(model_id):
    print("\n\n===== DEBUG: MODEL METADATA REQUEST =====")
    print(f"Requested model_id: {model_id}")
    print(f"Blueprint name: {ml_bp.name}")
    print(f"Full request path: {request.path}")
    print(f"Request method: {request.method}")

    if not model_id:
        print("DEBUG: No model_id provided")
        return jsonify({
            'status': 'error',
            'message': 'No model_id provided'
        }), 400

    try:
        models_json_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')
        print(f"DEBUG: Looking for models.json at: {models_json_path}")
        print(f"DEBUG: File exists: {os.path.exists(models_json_path)}")

        if os.path.exists(models_json_path):
            print("DEBUG: models.json exists, attempting to load it")
            with open(models_json_path, 'r') as f:
                registry = json.load(f)
            print(f"DEBUG: Registry keys: {list(registry.keys())}")
            if 'models' in registry:
                print(f"DEBUG: Model types in registry: {list(registry['models'].keys())}")

            model_data = None
            for model_type, models in registry.get('models', {}).items():
                print(f"DEBUG: Checking models of type {model_type}, count: {len(models)}")
                if model_id in models:
                    print(f"DEBUG: Found model {model_id} in type {model_type}")
                    model_data = models[model_id]
                    break

            if model_data:
                print(f"DEBUG: Model data keys: {list(model_data.keys())}")
                if 'class_names' in model_data:
                    print(f"DEBUG: Found class_names in model_data: {model_data['class_names']}")
                    return jsonify({
                        'status': 'success',
                        'metadata': model_data
                    })
                else:
                    print("DEBUG: model_data exists but does not contain class_names")

        print(f"DEBUG: Attempting to find metadata file for model: {model_id}")
        model_parts = model_id.split('_')
        print(f"DEBUG: Model parts: {model_parts}")

        if len(model_parts) >= 3:
            model_type = model_parts[1]
            print(f"DEBUG: Extracted model_type: {model_type}")

            if model_type.lower() == 'cnn':
                metadata_path = os.path.join(
                    Config.BASE_DIR, 'data', 'models', 'cnn', model_id, f'{model_id}_metadata.json'
                )
            elif model_type.lower() == 'rf':
                metadata_path = os.path.join(
                    Config.BASE_DIR, 'data', 'models', 'rf', model_id, f'{model_id}_metadata.json'
                )
            elif model_type.lower() in ['ens', 'ensemble']:
                metadata_path = os.path.join(
                    Config.BASE_DIR, 'data', 'models', 'ensemble', model_id, f'{model_id}_metadata.json'
                )
            else:
                print(f"DEBUG: Unknown model type: {model_type}")
                return jsonify({
                    'status': 'error',
                    'message': f'Unknown model type: {model_type}'
                }), 400

            print(f"DEBUG: Looking for metadata file at: {metadata_path}")
            print(f"DEBUG: Metadata file exists: {os.path.exists(metadata_path)}")

            if os.path.exists(metadata_path):
                print("DEBUG: Metadata file exists, loading content")
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"DEBUG: Metadata keys: {list(metadata.keys())}")
                    print(f"DEBUG: class_names in metadata: {'class_names' in metadata}")
                    if 'class_names' in metadata:
                        print(f"DEBUG: class_names value: {metadata['class_names']}")
                    return jsonify({
                        'status': 'success',
                        'metadata': metadata
                    })
                except Exception as file_error:
                    print(f"DEBUG: Error reading metadata file: {str(file_error)}")
                    return jsonify({
                        'status': 'error',
                        'message': f'Error reading metadata file: {str(file_error)}'
                    }), 500
            else:
                print(f"DEBUG: Metadata file not found at path: {metadata_path}")
                return jsonify({
                    'status': 'error',
                    'message': f'Metadata file not found for model {model_id}'
                }), 404
        else:
            print(f"DEBUG: Invalid model ID format: {model_id}, cannot extract parts")

        return jsonify({
            'status': 'error',
            'message': f'No metadata found for model {model_id}'
        }), 404

    except Exception as e:
        print(f"DEBUG: Unhandled error in model_metadata: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving model metadata: {str(e)}'
        }), 500
