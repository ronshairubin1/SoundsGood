# SoundClassifier_v08/src/routes/ml_routes.py

LATEST_PREDICTION = None

# Global variables for training statistics and history
training_stats = None
training_history = None
model_summary_str = None

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
from backend.config import Config
from backend.src.services.training_service import TrainingService
# Import model classes
from backend.src.ml.rf_classifier import RandomForestClassifier
from backend.src.ml.ensemble_classifier import EnsembleClassifier
from backend.src.ml.inference import predict_sound, SoundDetector
from backend.audio.augmentor import AudioAugmentor

# Sound processing imports
from backend.src.ml.sound_detector_rf import SoundDetectorRF
from backend.src.ml.model_paths import get_cnn_model_path
from backend.features.extractor import FeatureExtractor

# Initialize blueprint
ml_bp = Blueprint('ml', __name__)


# Function to log augmentation results to a file
def log_augmentation_results(dict_name, results):
    """Log detailed augmentation results to a file for reference and debugging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use a relative path to logs directory at the project root (same level as .gitignore)
    log_dir = os.path.join('logs', 'augmentations')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"augmentation_{dict_name}_{timestamp}.log")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Augmentation Results for Dictionary: {dict_name} ===\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total original files: {results['total_original_files']}\n")
        f.write(f"Total augmented files generated: {results['total_augmented_generated']}\n\n")
        
        if results.get('files_with_errors', []):
            f.write(f"=== {len(results['files_with_errors'])} Files with Errors ===\n")
            for error_file in results['files_with_errors']:
                f.write(f"- {error_file}\n")
            f.write("\n")
        
        if results.get('file_details', {}):
            f.write("=== Detailed File Information ===\n")
            for orig_file, details in results['file_details'].items():
                f.write(f"Original: {orig_file}\n")
                if details.get('augmented_files', []):
                    f.write(f"  Generated {len(details['augmented_files'])} augmented files:\n")
                    for aug_file in details['augmented_files']:
                        f.write(f"  - {os.path.basename(aug_file)}\n")
                if details.get('error'):
                    f.write(f"  ERROR: {details['error']}\n")
                f.write("\n")
    
    logging.info("Augmentation results logged to %s", log_file)
    return log_file


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
        log_message = self.format(record)
        self.logs.append(log_message)
        
        # Also add to app context if available
        try:
            if current_app and hasattr(current_app, 'training_debug_logs'):
                current_app.training_debug_logs.append(log_message)
        except Exception:
            # Ignore errors if outside app context
            pass


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


# Legacy training route removed - all training now uses the unified approach


@ml_bp.route('/train_model', methods=['GET', 'POST'])
def train_model():
    """
    Main route for model training.
    
    GET: Renders the training form
    POST: Processes the training request
    
    Returns:
        Rendered template or redirect to training status
    """
    if 'username' not in session:
        return redirect(url_for('index'))

    # Use the existing TrainingService instance
    training_service = TrainingService()

    if request.method == 'GET':
        # Load available dictionaries
        dictionaries = Config.get_dictionaries()
        dict_names = [d['name'] for d in dictionaries if 'name' in d]
        
        # Set the list of model types
        model_types = ['cnn', 'rf', 'ensemble']

        # Get sounds for the currently selected dictionary (if any)
        selected_dict = request.args.get('dict')
        sounds = []
        if selected_dict:
            selected_dict_data = None
            for d in dictionaries:
                if d.get('name') == selected_dict:
                    selected_dict_data = d
                    break
            
            if selected_dict_data:
                sounds = selected_dict_data.get('sounds', [])

        # Get augmentation status if requested
        sound_analysis = None
        if request.args.get('analyze') == 'true':
            audio_dir = get_goodsounds_dir_path()
            sound_analysis = training_service.analyze_sound_file_status(audio_dir)
        
        # Check for feature extraction status if requested
        feature_status = None
        extract_done = False
        if request.args.get('check_features') == 'true' or request.args.get('extract_features') == 'true':
            if selected_dict:
                # Get the sound classes for this dictionary
                class_dirs = []
                for sound in sounds:
                    class_dir = os.path.join(get_goodsounds_dir_path(), sound)
                    if os.path.exists(class_dir):
                        class_dirs.append(class_dir)
                
                # Get status of feature extraction
                feature_status = {
                    "total_files": 0,
                    "files_with_features": 0,
                    "files_missing_features": 0,
                    "classes": {}
                }
                
                # Check each class directory
                for class_dir in class_dirs:
                    class_name = os.path.basename(class_dir)
                    feature_status["classes"][class_name] = {
                        "total_files": 0,
                        "files_with_features": 0,
                        "files_missing_features": 0,
                        "missing_files": []
                    }
                    
                    # Check all WAV files in this class
                    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                    feature_status["total_files"] += len(wav_files)
                    feature_status["classes"][class_name]["total_files"] = len(wav_files)
                    
                    for wav_file in wav_files:
                        file_path = os.path.join(class_dir, wav_file)
                        # Check if features exist by attempting to load them
                        from src.ml.feature_extractor import FeatureExtractor
                        feature_extractor = FeatureExtractor()
                        has_features = feature_extractor.check_features_exist(file_path)
                        
                        if has_features:
                            feature_status["files_with_features"] += 1
                            feature_status["classes"][class_name]["files_with_features"] += 1
                        else:
                            feature_status["files_missing_features"] += 1
                            feature_status["classes"][class_name]["files_missing_features"] += 1
                            feature_status["classes"][class_name]["missing_files"].append(wav_file)
                            
                            # Extract features if requested
                            if request.args.get('extract_features') == 'true':
                                try:
                                    # Create a feature extractor if needed
                                    from src.ml.feature_extractor import FeatureExtractor
                                    feature_extractor = FeatureExtractor()
                                    feature_extractor.extract_features(file_path)
                                    extract_done = True
                                except Exception as e:
                                    current_app.logger.error(f"Error extracting features for {file_path}: {e}")
                
                # If we extracted features, recheck the status
                if extract_done:
                    # Reset counters
                    feature_status["files_with_features"] = 0
                    feature_status["files_missing_features"] = 0
                    for class_name in feature_status["classes"]:
                        feature_status["classes"][class_name]["files_with_features"] = 0
                        feature_status["classes"][class_name]["files_missing_features"] = 0
                        feature_status["classes"][class_name]["missing_files"] = []
                    
                    # Recheck each class directory
                    for class_dir in class_dirs:
                        class_name = os.path.basename(class_dir)
                        wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                        
                        for wav_file in wav_files:
                            file_path = os.path.join(class_dir, wav_file)
                            # Check if features exist by attempting to load them
                            from src.ml.feature_extractor import FeatureExtractor
                            feature_extractor = FeatureExtractor()
                            has_features = feature_extractor.check_features_exist(file_path)
                            
                            if has_features:
                                feature_status["files_with_features"] += 1
                                feature_status["classes"][class_name]["files_with_features"] += 1
                            else:
                                feature_status["files_missing_features"] += 1
                                feature_status["classes"][class_name]["files_missing_features"] += 1
                                feature_status["classes"][class_name]["missing_files"].append(wav_file)
            
        # Render the standard training UI
        return render_template(
            'train.html',  # Updated template name to reflect standard training
            dictionaries=dict_names,
            model_types=model_types,
            sounds=sounds,
            selected_dict=selected_dict,
            sound_analysis=sound_analysis,
            feature_status=feature_status,
            extract_done=extract_done,
            audio_dir=get_goodsounds_dir_path()
        )

    # Handle POST request for training
    logging.info("Received training form submission: %s", request.form)

    # Extract all form data in the main request thread
    model_type = request.form.get('model_type', 'cnn')
    dict_name = request.form.get('dict_name')

    logging.info("Model type: %s, Dictionary name: %s", model_type, dict_name)

    # Validate model type
    if model_type not in ['cnn', 'rf', 'ensemble']:
        logging.error("Invalid model type: %s", model_type)
        flash(f'Invalid model type: {model_type}', 'error')
        return redirect(url_for('ml.train_model'))

    # Get dictionary classes for efficient feature extraction
    dict_classes = []
    try:
        dictionaries = Config.get_dictionaries()
        for d in dictionaries:
            if d.get('name') == dict_name:
                dict_classes = d.get('sounds', [])
                break
        logging.info(f"Found {len(dict_classes)} classes for dictionary {dict_name}: {dict_classes}")
    except Exception as e:
        logging.warning(f"Error getting dictionary classes: {str(e)}")
        
    # Check if augmentation options are enabled
    include_augmented = request.form.get('include_augmented') == 'on'
    generate_augmentations = request.form.get('generate_augmentations') == 'on'
    target_augmentation_count = int(request.form.get('target_augmentation_count', 27))
    
    # Capture all model-specific parameters outside the thread
    training_params = {
        'model_type': model_type,
        'dict_name': dict_name,
        'dict_classes': dict_classes,
        'save': True,
        # Add augmentation parameters
        'include_augmented': include_augmented,
        'generate_augmentations': generate_augmentations,
        'target_augmentation_count': target_augmentation_count
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
    elif model_type == 'rf':
        n_estimators = int(request.form.get('n_estimators', 100))
        max_depth = request.form.get('max_depth')
        if max_depth and max_depth.strip() and max_depth.lower() != 'none':
            max_depth = int(max_depth)
        else:
            max_depth = None
            
        training_params.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth
        })

    # Get current application object for async training
    app = current_app._get_current_object()

    if not hasattr(app, 'training_progress'):
        app.training_progress = 0

    if not hasattr(app, 'training_status'):
        app.training_status = "Not started"
        
    # Initialize debug logs storage
    if not hasattr(app, 'training_debug_logs'):
        app.training_debug_logs = []

    # Create a special log handler to capture logs just for this training session
    class TrainingLogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            with app.app_context():
                # Store in app context and also print to console
                if hasattr(app, 'training_debug_logs'):
                    app.training_debug_logs.append(log_entry)
                print(f"[TRAINING] {log_entry}")
    
    # Configure and add the training log handler
    training_log_handler = TrainingLogHandler()
    training_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    training_log_handler.setFormatter(formatter)
    logging.getLogger().addHandler(training_log_handler)

    try:
        # Define an async training function that runs in a separate thread
        def train_async_unified(params):
            with app.app_context():
                try:
                    # Clear previous logs
                    app.training_debug_logs = []
                    
                    # Log basic training info
                    log_message = f"Starting training for {params['dict_name']} dictionary using {params['model_type']} model"
                    logging.info(log_message)
                    app.training_debug_logs.append(log_message)
                    
                    # Explicitly log to console as well
                    print(f"[TRAINING] {log_message}")
                    
                    # Add specific diagnostic information about the training environment
                    import sys
                    env_info = f"Python version: {sys.version}\nRunning in environment: {sys.prefix}"
                    logging.info(f"Environment information: {env_info}")
                    app.training_debug_logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {env_info}")
                    
                    # Display training parameters
                    logging.info(f"Training parameters: {json.dumps(params, indent=2, default=str)}")
                    
                    # Handle EhOh dictionary specifically
                    if params['dict_name'] == 'EhOh':
                        logging.info("Setting up EhOh dictionary training...")
                        
                        # Use the classes directly from the dictionaries.json definition
                        # rather than trying to find a directory
                        try:
                            from backend.data.dictionary_manager import DictionaryManager
                            dict_manager = DictionaryManager()
                            ehoh_dict = dict_manager.get_dictionary('EhOh')
                            
                            if ehoh_dict and 'classes' in ehoh_dict:
                                params['dict_classes'] = ehoh_dict['classes']
                                logging.info(f"Using classes from dictionary definition: {params['dict_classes']}")
                            else:
                                # Fallback to hardcoded classes if dictionary definition is incomplete
                                params['dict_classes'] = ['eh', 'oh']
                                logging.info(f"Using hardcoded classes: {params['dict_classes']}")
                            
                            # Verify that features exist for these classes
                            from backend.features.extractor import FeatureExtractor
                            feature_extractor = FeatureExtractor()
                            logging.info(f"Using feature cache directory: {feature_extractor.cache_dir}")
                            
                            # Check if features directory exists
                            if os.path.exists(feature_extractor.cache_dir):
                                logging.info(f"Features directory found at {feature_extractor.cache_dir}")
                            else:
                                logging.warning(f"Features directory not found at {feature_extractor.cache_dir}")
                                logging.info("Training will proceed but may fail if features are not available")
                                
                        except Exception as e:
                            logging.error(f"Error setting up EhOh dictionary: {str(e)}")
                            # Continue with hardcoded classes as fallback
                            params['dict_classes'] = ['eh', 'oh']
                            logging.info(f"Falling back to hardcoded classes: {params['dict_classes']}")
                    
                    app.training_progress = 0
                    app.training_status = "Training in progress"

                    # Enhanced progress callback function
                    def progress_callback(progress, status_message):
                        app.training_progress = progress
                        if status_message:
                            app.training_status = status_message
                            # Add to debug logs
                            log_entry = f"Training progress: {progress}%, Status: {status_message}"
                            logging.info(log_entry)
                            if hasattr(app, 'training_debug_logs'):
                                app.training_debug_logs.append(log_entry)
                    
                    # Create a full copy of params and add runtime parameters
                    full_params = params.copy()
                    
                    # Process checkbox values from form
                    # Convert string 'true'/'false' to boolean if needed
                    include_augmented_value = params.get('include_augmented', 'false')
                    if isinstance(include_augmented_value, bool):
                        include_augmented = include_augmented_value
                    else:
                        include_augmented = include_augmented_value.lower() in ['true', 'on', '1']
                    
                    full_params.update({
                        'audio_dir': Config.get_training_sounds_dir(),
                        'model_name': f"{params['dict_name']}_{params['model_type']}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'progress_callback': progress_callback,
                        'include_augmented': include_augmented
                    })
                    
                    # Log if augmented data is being included
                    aug_log = f"Including augmented data: {include_augmented}"
                    logging.info(aug_log)
                    if hasattr(app, 'training_debug_logs'):
                        app.training_debug_logs.append(aug_log)
                    
                    # Log the dictionary classes that will be used for feature extraction
                    dict_classes = full_params.get('dict_classes', [])
                    class_log = f"Training with {len(dict_classes)} dictionary classes: {dict_classes}"
                    logging.info(class_log)
                    if hasattr(app, 'training_debug_logs'):
                        app.training_debug_logs.append(class_log)

                    # Call the training method
                    result = training_service.train_unified(**full_params)
                    
                    if result.get('status') == 'success':
                        app.training_progress = 100
                        # Check if we used an existing model
                        if result.get('used_existing'):
                            app.training_status = "Training complete! (Used existing identical model)"
                        else:
                            app.training_status = "Training complete!"
                    else:
                        error_msg = f"Training failed: {result.get('message', 'Unknown error')}"
                        app.training_status = error_msg
                        logging.error(error_msg)
                        
                        # Add additional diagnostics for failures
                        if "No valid samples" in result.get('message', ''):
                            # Special diagnostics for the common error
                            logging.error("Detailed diagnostics for 'No valid samples' error:")
                            audio_dir = Config.get_training_sounds_dir()
                            dict_dir = os.path.join(audio_dir, params['dict_name'])
                            logging.error(f"Dictionary directory: {dict_dir}")
                            
                            # Check for required dictionary classes
                            if os.path.exists(dict_dir):
                                actual_classes = [d for d in os.listdir(dict_dir) 
                                                if os.path.isdir(os.path.join(dict_dir, d))]
                                logging.error(f"Available classes: {actual_classes}")
                                logging.error(f"Required classes: {dict_classes}")
                                
                                # Check for mismatch
                                missing_classes = [c for c in dict_classes if c not in actual_classes]
                                if missing_classes:
                                    logging.error(f"Missing required classes: {missing_classes}")
                                
                                # Check each class directory
                                for cls in actual_classes:
                                    class_dir = os.path.join(dict_dir, cls)
                                    files = [f for f in os.listdir(class_dir) 
                                           if f.endswith('.wav')]
                                    logging.error(f"Class '{cls}' has {len(files)} audio files")
                        
                except Exception as e:
                    error_msg = f"Error during training: {str(e)}"
                    app.training_status = error_msg
                    logging.error(error_msg)
                    logging.error("Exception details:", exc_info=True)
                    
                    # Capture stack trace
                    import traceback
                    stack_trace = traceback.format_exc()
                    logging.error(f"Stack trace: {stack_trace}")
                    
                    if hasattr(app, 'training_debug_logs'):
                        app.training_debug_logs.append(error_msg)
                        app.training_debug_logs.append(f"Stack trace: {stack_trace}")
        
        # Start the training thread with the captured parameters
        training_thread = threading.Thread(target=train_async_unified, args=(training_params,))
        training_thread.daemon = True
        training_thread.start()
        
        # Store dictionary name and model type for the training status page
        app.training_dictionary_name = dict_name
        app.training_model_type = model_type
        
        # Redirect to the training status page
        return redirect(url_for('ml.training_status'))
        
    except Exception as e:
        flash(f'Error starting training: {str(e)}', 'error')
        logging.error("Error starting training: %s", e, exc_info=True)
        return redirect(url_for('ml.train_model'))


@ml_bp.route('/generate_augmentations', methods=['GET', 'POST'])
def generate_augmentations():
    """
    Generate augmented versions of sound files for a selected dictionary.
    
    GET: Redirects to the training page
    POST: Processes the augmentation generation request
    
    Returns:
        Redirect to the training page with augmentation results
    """
    if 'username' not in session:
        return redirect(url_for('index'))
        
    # Get dictionary parameter
    dict_name = request.args.get('dict')
    if not dict_name:
        flash('Please select a dictionary first', 'warning')
        return redirect(url_for('ml.train_model'))
        
    # Get the sounds for this dictionary
    dictionaries = Config.get_dictionaries()
    selected_dict_data = None
    for d in dictionaries:
        if d.get('name') == dict_name:
            selected_dict_data = d
            break
            
    if not selected_dict_data:
        flash(f'Dictionary not found: {dict_name}', 'error')
        return redirect(url_for('ml.train_model'))
        
    sounds = selected_dict_data.get('sounds', [])
    if not sounds:
        flash(f'No sounds defined in dictionary: {dict_name}', 'warning')
        return redirect(url_for('ml.train_model'))
        
    # Initialize services
    training_service = TrainingService()
    audio_dir = get_goodsounds_dir_path()
    
    # Create a list of class directories to process based on the dictionary sounds
    class_dirs = []
    for sound in sounds:
        class_dir = os.path.join(audio_dir, sound)
        if os.path.exists(class_dir):
            class_dirs.append(class_dir)
            
    if not class_dirs:
        flash(f'No sound class directories found for dictionary: {dict_name}', 'warning')
        return redirect(url_for('ml.train_model'))
        
    # Generate augmentations for all files
    try:
        augmentation_results = training_service.generate_augmentations(audio_dir)
        
        # Add log messages
        from flask import current_app
        if hasattr(current_app, 'training_debug_logs'):
            current_app.training_debug_logs.append(f"Generated {augmentation_results['total_augmented_generated']} augmented files from {augmentation_results['total_original_files']} original files")
            
            if augmentation_results['files_with_errors']:
                current_app.training_debug_logs.append(f"Errors occurred for {len(augmentation_results['files_with_errors'])} files")
        
        # Log detailed results to a file
        log_file = log_augmentation_results(dict_name, augmentation_results)
        if hasattr(current_app, 'training_debug_logs'):
            current_app.training_debug_logs.append(f"Detailed augmentation results logged to {log_file}")
                
        # Set flash messages with the results instead of storing in session
        flash(f"Successfully generated {augmentation_results['total_augmented_generated']} augmented files from {augmentation_results['total_original_files']} original files", 'success')
        
        # Only store a small summary in the session, not the entire results object
        # This prevents session cookie size issues
        session['augmentation_summary'] = {
            'total_original_files': augmentation_results['total_original_files'],
            'total_augmented_generated': augmentation_results['total_augmented_generated'],
            'error_count': len(augmentation_results.get('files_with_errors', []))
        }
        
        # Redirect to training page with analysis to show the results
        return redirect(url_for('ml.train', analyze='true', dict=dict_name))
        
    except Exception as e:
        flash(f'Error generating augmentations: {str(e)}', 'error')
        logging.error("Error generating augmentations: %s", e, exc_info=True)
        return redirect(url_for('ml.train_model'))


@ml_bp.route('/train', methods=['GET', 'POST'])
def train():
    """
    Main route for model training, using the unified training approach.
    
    GET: Renders the training form
    POST: Processes the training request using the unified approach
    
    Returns:
        Rendered template or redirect to training status
    """
    # Always perform sound file analysis for GET requests
    if request.method == 'GET':
        training_service = TrainingService()
        audio_dir = get_goodsounds_dir_path()
        analysis = training_service.analyze_sound_file_status(audio_dir)
        
        # Load available dictionaries - same as in train_model
        dictionaries = Config.get_dictionaries()
        dict_names = [d['name'] for d in dictionaries if 'name' in d]
        
        # Set the list of model types
        model_types = ['cnn', 'rf', 'ensemble']

        # Get sounds for the currently selected dictionary (if any)
        selected_dict = request.args.get('dict')
        sounds = []
        if selected_dict:
            selected_dict_data = next((d for d in dictionaries if d.get('name') == selected_dict), None)
            if selected_dict_data:
                sounds = selected_dict_data.get('sounds', [])
        
        # Get augmentation summary from session if available
        augmentation_summary = None
        if 'augmentation_summary' in session:
            augmentation_summary = session.pop('augmentation_summary')
        
        # Pass analysis results along with dictionary and model info to the template
        return render_template('train.html', 
                              analysis=analysis,
                              audio_dir=audio_dir,
                              dictionaries=dict_names,
                              model_types=model_types,
                              sounds=sounds,
                              selected_dict=selected_dict,
                              augmentation_summary=augmentation_summary)
    
    return train_model()


@ml_bp.route('/train_unified_model', methods=['GET', 'POST'])
def train_unified_model():
    """
    Legacy route for unified training model.
    
    Redirects to the main /train route which is now the only training interface.
    All training now uses the unified training approach.
    """
    # Show a message about the consolidated training approach
    flash('All training now uses the unified approach. Redirecting to the training interface.', 'info')
    return redirect(url_for('ml.train'))


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
        
        # Initialize model_data to avoid reference error
        model_data = None

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
                preprocessing_params={
                    "sample_rate": 16000,
                    "sound_threshold": 0.008,
                    "chunk_duration": 0.5,
                    "overlap": 0.4
                },
                use_ambient_noise=use_ambient_noise
            )

            sound_detector.start_listening(callback=prediction_callback)
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
                # Use the model_paths module to get the correct path for folder-based structure
                from backend.src.ml.model_paths import get_cnn_model_path
                model_path = get_cnn_model_path(model_id=model_id)
                
                # If not found using the helper, fall back to direct path construction
                if not model_path:
                    # Check if it's using the new folder structure
                    base_models_dir = os.path.join(Config.BASE_DIR, 'backend', 'data', 'models')
                    folder_path = os.path.join(base_models_dir, model_id)
                    if os.path.isdir(folder_path):
                        model_path = os.path.join(folder_path, f"{model_id}.h5")
                    else:
                        # Legacy fallback
                        model_path = os.path.join(Config.MODELS_DIR, f"{model_id}.h5")

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
                    preprocessing_params={
                        "sample_rate": 16000,
                        "sound_threshold": 0.008,
                        "chunk_duration": 0.5,
                        "overlap": 0.4
                    },
                    use_ambient_noise=use_ambient_noise
                )

                sound_detector.start_listening(callback=prediction_callback)

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
    
    # Initialize debug logs container if it doesn't exist
    if not hasattr(current_app, 'training_debug_logs'):
        current_app.training_debug_logs = []
    
    # Add current status to logs to ensure we have something to display
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status_log = f"{current_time} - INFO - Current training status: {current_app.training_status}, Progress: {current_app.training_progress}%"
    current_app.training_debug_logs.append(status_log)
    
    # Manually add a meaningful log if we're just starting
    if current_app.training_status == 'Not started':
        current_app.training_debug_logs.append(f"{current_time} - INFO - Waiting for training to start. Select a dictionary and model type to begin.")
    
    # Get the most recent logs from the handler
    recent_logs = debug_log_handler.logs[-100:] if debug_log_handler.logs else []
    
    # Also add the logs from the application container
    app_logs = getattr(current_app, 'training_debug_logs', [])
    
    # Combine all logs, filter out HTTP access logs which are just noise
    filtered_logs = [log for log in app_logs + recent_logs if not ('/training_status HTTP/1.1" 200' in log or 'GET /static/' in log)]
    
    # Remove duplicates and sort
    all_logs = list(set(filtered_logs))
    try:
        all_logs.sort()
    except Exception:
        pass
        
    # Take only the most recent 150 logs
    all_logs = all_logs[-150:]
    debug_log_text = '\n'.join(all_logs)
    
    # Add info about the dictionary classes and structure directly in the logs
    logging.info(f"Preparing debug information with {len(all_logs)} log entries")
    
    # Get dictionary class information if available
    dict_name = getattr(current_app, 'training_dictionary_name', '')
    dict_classes = []
    class_info_text = ''
    
    # Check for EhOh dictionary specifically since that's giving issues
    if dict_name == 'EhOh':
        # Get base training directory
        training_dir = Config.get_training_sounds_dir()
        
        # Get dictionary specific directory
        dict_dir = os.path.join(training_dir, dict_name)
        
        if os.path.exists(dict_dir):
            # List all classes in the dictionary
            try:
                classes = [d for d in os.listdir(dict_dir) if os.path.isdir(os.path.join(dict_dir, d))]
                class_counts = {}
                
                # Count files in each class
                for cls in classes:
                    class_dir = os.path.join(dict_dir, cls)
                    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                    
                    class_counts[cls] = len(files)
                
                # Format class information
                class_info_lines = [f"Dictionary: {dict_name}"]
                class_info_lines.append(f"Total classes: {len(classes)}")
                class_info_lines.append("Classes and file counts:")
                for cls, count in class_counts.items():
                    class_info_lines.append(f"  - {cls}: {count} files")
                
                class_info_text = '\n'.join(class_info_lines)
            except Exception as e:
                class_info_text = f"Error getting class info: {str(e)}"
    
    # Import TrainingService to get epoch data
    from src.services.training_service import TrainingService
    
    # Add debug, class information, and epoch data to the status data
    status_data = {
        'progress': current_app.training_progress,
        'status': current_app.training_status,
        'dictionary_name': dict_name,
        'model_type': getattr(current_app, 'training_model_type', ''),
        'debug_logs': debug_log_text,
        'class_info': class_info_text,
        'epoch_data': TrainingService.latest_epoch_data
    }

    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
              request.accept_mimetypes.best_match(['application/json', 'text/html']) == 'application/json'

    if is_ajax:
        return jsonify(status_data)
    else:
        return render_template('training_status.html')


@ml_bp.route('/training_dataset')
def training_dataset():
    """Get information about the dataset used for the current/last training session.
    
    Returns:
        JSON response with dataset information including files and class distribution
    """
    # Get the current dictionary name from the app context
    dict_name = getattr(current_app, 'training_dictionary_name', None)
    
    if not dict_name:
        return jsonify({
            'error': 'No active training session found'
        }), 404
    
    # Find the dictionary data
    dictionaries = Config.get_dictionaries()
    dict_data = None
    for d in dictionaries:
        if d.get('name') == dict_name:
            dict_data = d
            break
    
    if not dict_data:
        return jsonify({
            'error': f'Dictionary {dict_name} not found'
        }), 404
    
    # Get the classes (sounds) from the dictionary
    sounds = dict_data.get('sounds', [])
    
    # Try to get dataset files from app context
    files = getattr(current_app, 'dataset_files', [])
    class_distribution = getattr(current_app, 'class_distribution', {})
    
    # If no files in app context, try to scan the directory
    if not files:
        try:
            logging.info("No dataset files in app context, scanning directories...")
            for sound_class in sounds:
                # Get the directory for this class
                class_dir = os.path.join(Config.get_training_sounds_dir(), dict_name, sound_class)
                
                if not os.path.exists(class_dir):
                    continue
                    
                # Get all audio files in this directory
                class_files = [f for f in os.listdir(class_dir) 
                             if f.endswith(('.wav', '.mp3', '.ogg')) and not f.startswith('.')]
                
                # Track class distribution
                class_distribution[sound_class] = len(class_files)
                
                # Add file info to the list
                for file_name in class_files:
                    file_path = os.path.join(class_dir, file_name)
                    file_info = {
                        'path': file_path,
                        'class': sound_class,
                        'filename': file_name,
                        'split': 'unknown'  # We don't know the split if loading from disk
                    }
                        
                    files.append(file_info)
            
            logging.info(f"Found {len(files)} files by scanning directories")
        except Exception as e:
            logging.error(f"Error getting dataset files: {str(e)}")
            return jsonify({
                'error': f'Error accessing dataset files: {str(e)}'
            }), 500
    
    # Calculate whether the dataset is balanced
    is_balanced = False
    if class_distribution:
        counts = list(class_distribution.values())
        min_count = min(counts)
        max_count = max(counts)
        # Consider balanced if the max/min ratio is less than 1.5
        is_balanced = (max_count / min_count if min_count > 0 else float('inf')) < 1.5
    
    # Count training and validation files
    training_files = sum(1 for f in files if f.get('split') == 'training')
    validation_files = sum(1 for f in files if f.get('split') == 'validation')
    
    # Return dataset summary and files
    return jsonify({
        'summary': {
            'total_files': len(files),
            'training_files': training_files,
            'validation_files': validation_files,
            'classes': sounds,
            'class_count': len(sounds),
            'class_distribution': class_distribution,
            'is_balanced': is_balanced
        },
        'files': files
    })


@ml_bp.route('/training_analysis')
def training_analysis():
    """Generate a prose analysis of the training results.
    
    Returns:
        JSON response with analysis text
    """
    # Get necessary data for analysis
    dict_name = getattr(current_app, 'training_dictionary_name', None)
    model_type = getattr(current_app, 'training_model_type', None)
    
    # Import TrainingService to get epoch data
    from src.services.training_service import TrainingService
    epoch_data = TrainingService.latest_epoch_data if hasattr(TrainingService, 'latest_epoch_data') else []
    
    if not dict_name or not model_type or not epoch_data:
        return jsonify({
            'analysis': '<p>No training data available to analyze.</p>'
        })
    
    # Generate the analysis HTML
    analysis_html = '<div class="training-analysis">'
    
    # Model overview
    model_type_display = {
        'cnn': 'Convolutional Neural Network',
        'rf': 'Random Forest',
        'ensemble': 'Ensemble Model'
    }.get(model_type, model_type)
    
    analysis_html += f"<h4>Model Overview</h4>"
    analysis_html += f"<p>A <strong>{model_type_display}</strong> model was trained on the <strong>{dict_name}</strong> sound dictionary.</p>"
    
    # Training performance
    if epoch_data:
        analysis_html += f"<h4>Training Performance</h4>"
        final_epoch = epoch_data[-1]
        initial_epoch = epoch_data[0]
        
        # Accuracy analysis
        final_acc = final_epoch.get('accuracy', 0) * 100
        final_val_acc = final_epoch.get('val_accuracy', 0) * 100
        
        if final_val_acc > 85:
            acc_assessment = "excellent"
        elif final_val_acc > 70:
            acc_assessment = "good"
        elif final_val_acc > 50:
            acc_assessment = "moderate"
        else:
            acc_assessment = "poor"
            
        analysis_html += f"<p>The model achieved <strong>{final_val_acc:.2f}%</strong> validation accuracy "
        analysis_html += f"after {len(epoch_data)} training epochs, which is "
        analysis_html += f"<span class='text-{'success' if acc_assessment in ['excellent', 'good'] else 'warning'}'>{acc_assessment}</span>.</p>"
        
        # Overfitting analysis
        train_val_diff = final_acc - final_val_acc
        if train_val_diff > 20:
            analysis_html += f"<p><strong>Warning:</strong> There appears to be significant overfitting. "
            analysis_html += f"The training accuracy ({final_acc:.2f}%) is much higher than validation accuracy ({final_val_acc:.2f}%).</p>"
            analysis_html += f"<p><strong>Recommendation:</strong> Consider using data augmentation, reducing model complexity, "
            analysis_html += f"or adding more diverse training samples.</p>"
        
        # Check for poor validation performance
        if final_val_acc <= 50:
            analysis_html += f"<p><strong>Critical Issue:</strong> The model is showing poor validation performance "
            analysis_html += f"({final_val_acc:.2f}%), which suggests it may not be learning meaningful patterns.</p>"
            analysis_html += f"<p><strong>Possible causes:</strong></p>"
            analysis_html += "<ul>"
            analysis_html += "<li>Insufficient training data</li>"
            analysis_html += "<li>Poor feature extraction</li>"
            analysis_html += "<li>Class imbalance</li>"
            analysis_html += "<li>Excessive noise in audio samples</li>"
            analysis_html += "</ul>"
        
        # Learning curve analysis
        initial_loss = initial_epoch.get('loss', 0)
        final_loss = final_epoch.get('loss', 0)
        loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
        
        if loss_reduction > 50:
            analysis_html += f"<p>The model showed <span class='text-success'>significant learning</span> during training, "
            analysis_html += f"with a {loss_reduction:.1f}% reduction in loss.</p>"
        elif loss_reduction > 20:
            analysis_html += f"<p>The model showed <span class='text-primary'>moderate learning</span> during training, "
            analysis_html += f"with a {loss_reduction:.1f}% reduction in loss.</p>"
        else:
            analysis_html += f"<p><strong>Warning:</strong> The model showed <span class='text-danger'>limited learning</span> "
            analysis_html += f"during training, with only a {loss_reduction:.1f}% reduction in loss.</p>"
            analysis_html += f"<p><strong>Recommendation:</strong> Try adjusting the learning rate, model architecture, "
            analysis_html += f"or feature extraction method.</p>"
    
    # Recommendations
    analysis_html += f"<h4>Next Steps</h4>"
    analysis_html += "<ol>"
    analysis_html += "<li>Test the model on new audio samples to validate its real-world performance.</li>"
    
    if epoch_data:
        final_val_acc = epoch_data[-1].get('val_accuracy', 0) * 100
        if final_val_acc < 70:
            analysis_html += "<li>Try collecting more training data for each class.</li>"
        if final_val_acc > 70:
            analysis_html += "<li>Consider exporting the model for use in your application.</li>"
    
    analysis_html += "</ol>"
    
    analysis_html += "</div>"
    
    return jsonify({
        'analysis': analysis_html
    })


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

    # Check if we're receiving JSON data (new format) or form data (old format)
    if request.is_json:
        data = request.get_json()
        chunk_id = data.get('chunkId')
        is_good = data.get('isGood', False)
        augmentation = data.get('augmentation', None)
        
        # Get the chunk filename from the ID
        chunks = [f for f in os.listdir(Config.TEMP_DIR) if chunk_id in f]
        if not chunks:
            return jsonify({"success": False, "error": "Chunk not found"})
        
        chunk_file = chunks[0]
        parts = chunk_file.split('_')
        if len(parts) < 3:
            return jsonify({"success": False, "error": "Invalid filename format"})
            
        timestamp = parts[-2]
        sound = parts[0]
        
        if is_good:
            username = session['username']
            sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
            os.makedirs(sound_dir, exist_ok=True)

            existing_count = len([
                f for f in os.listdir(sound_dir)
                if f.startswith(f"{sound}_{username}_") and not f.endswith("_aug.wav")
            ])
            new_filename = f"{sound}_{username}_{existing_count + 1}.wav"
            new_path = os.path.join(sound_dir, new_filename)
            os.rename(os.path.join(Config.TEMP_DIR, chunk_file), new_path)
            
            # Handle augmentation if requested
            if augmentation and augmentation.get('type') != 'none':
                from src.services.training_service import TrainingService
                
                # Create a training service instance
                training_service = TrainingService()
                
                # Get augmentation parameters
                pitch_variations = augmentation.get('pitch', 0)
                stretch_variations = augmentation.get('stretch', 0)
                noise_variations = augmentation.get('noise', 0)
                
                # Generate augmentations
                try:
                    generated_files = training_service.generate_augmentations(
                        new_path,
                        pitch_count=pitch_variations,
                        stretch_count=stretch_variations,
                        noise_count=noise_variations
                    )
                    
                    current_app.logger.info(f"Generated {len(generated_files)} augmented files for {new_filename}")
                except Exception as e:
                    current_app.logger.error(f"Error generating augmentations: {e}", exc_info=True)
            
            # Return success response
            return jsonify({"success": True, "message": f"Chunk saved as {new_filename}"})
        else:
            os.remove(os.path.join(Config.TEMP_DIR, chunk_file))
            return jsonify({"success": True, "message": "Chunk discarded"})
    
    # Legacy form-based processing
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
            if f.startswith(f"{sound}_{username}_") and not f.endswith("_aug.wav")
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

    name = request.args.get('dictionary') or request.args.get('dict')
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
# DEPRECATED: Legacy route - will be consolidated with unified training in future updates
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
        from backend.src.core.ml_algorithms.rf import RandomForestModel
        model = RandomForestModel(
            model_dir=os.path.join('models', 'rf', dict_name.replace(' ', '_'))
        )
        # Use the unified training approach instead of legacy train_model
        result = training_service.train_unified(
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
# Legacy helper function for CNN training summary
# DEPRECATED: Will be consolidated with the unified training approach in future updates
def gather_training_stats_for_cnn():
    global training_stats
    if training_stats is None:
        return {}
    return training_stats


# Legacy helper function for CNN training history
# DEPRECATED: Will be consolidated with the unified training approach in future updates
def get_cnn_history():
    global training_history
    if training_history is None:
        return {}
    return training_history


@ml_bp.route("/cnn_training_summary")
# DEPRECATED: Legacy route - will be consolidated with unified training in future updates
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


# Routes for sound file analysis and augmentation management
@ml_bp.route('/analyze_sound_files', methods=['GET', 'POST'])
def analyze_sound_files():
    """
    Analyze sound files to check augmentation and feature extraction status.
    
    GET: Renders a form to select directory to analyze
    POST: Performs analysis and returns results
    
    Returns:
        JSON with analysis results
    """
    training_service = TrainingService()
    
    if request.method == 'POST':
        try:
            data = request.get_json() or {}
            audio_dir = data.get('audio_dir') or get_goodsounds_dir_path()
            
            # Perform analysis
            analysis_results = training_service.analyze_sound_file_status(audio_dir)
            
            # Add some additional useful information
            analysis_results['timestamp'] = datetime.now().isoformat()
            analysis_results['audio_dir'] = audio_dir
            
            return jsonify({
                'status': 'success',
                'analysis': analysis_results
            })
            
        except Exception as e:
            logging.error(f"Error analyzing sound files: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }), 500
    
    # GET request - render form
    return render_template('sound_analysis.html')

@ml_bp.route('/manage_augmentations', methods=['POST'])
def manage_augmentations():
    """
    Generate additional augmentations for sound files.
    
    Request body:
    {
        "audio_dir": "path/to/audio/dir",
        "target_count": 27,  # Target number of augmentations per original file
        "extract_features": true  # Whether to also extract features
    }
    
    Returns:
        JSON with results of the augmentation operation
    """
    training_service = TrainingService()
    
    try:
        data = request.get_json() or {}
        audio_dir = data.get('audio_dir') or get_goodsounds_dir_path()
        target_count = int(data.get('target_count', 27))
        extract_features = data.get('extract_features', True)
        
        # Perform augmentation and feature extraction
        results = training_service.update_sound_files(
            audio_dir, 
            target_augmentation_count=target_count,
            extract_features=extract_features
        )
        
        # Add some additional useful information
        results['timestamp'] = datetime.now().isoformat()
        results['audio_dir'] = audio_dir
        results['target_count'] = target_count
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        logging.error(f"Error managing augmentations: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


# Legacy helper function for ensemble training summary
# DEPRECATED: Will be consolidated with the unified training approach in future updates
def gather_training_stats_for_ensemble():
    global training_stats
    if training_stats is None:
        return {}
    return training_stats


@ml_bp.route("/ensemble_training_summary")
# DEPRECATED: Legacy route - will be consolidated with unified training in future updates
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
        # Get active dictionary information
        active_dict = Config.get_dictionary()
        dict_name = active_dict.get('name', 'Unknown')
        
        # Create InferenceService instance to retrieve models from filesystem
        from src.services.inference_service import InferenceService
        inference_service = InferenceService()
        
        # Get all available models directly from the filesystem
        # This will scan directories for models even if models.json doesn't exist or is empty
        models_data = inference_service.get_available_models()
        
        # Format the models for the frontend
        available_models = []
        
        # Process CNN models first (prioritize these for the Real-Time predictor)
        if 'cnn' in models_data and models_data['cnn']:
            logging.info(f"Found {len(models_data['cnn'])} CNN models")
            
            for model_info in models_data['cnn']:
                model_id = model_info.get('model_id')
                dictionary_name = model_info.get('dictionary_name', dict_name).replace(' ', '_')
                
                # Get class names from model data or fall back to active dictionary
                class_names = model_info.get('class_names', [])
                if not class_names and 'sounds' in active_dict:
                    class_names = active_dict['sounds']
                    logging.info(f"Using {len(class_names)} class names from active dictionary")
                
                available_models.append({
                    'id': model_id,
                    'name': f"{dictionary_name} CNN Model",
                    'type': 'cnn',
                    'dictionary': dictionary_name,
                    'class_names': class_names
                })
        
        # Add RF models if available
        if 'rf' in models_data and models_data['rf']:
            logging.info(f"Found {len(models_data['rf'])} RF models")
            
            for model_info in models_data['rf']:
                model_id = model_info.get('model_id')
                dictionary_name = model_info.get('dictionary_name', dict_name).replace(' ', '_')
                
                class_names = model_info.get('class_names', [])
                if not class_names and 'sounds' in active_dict:
                    class_names = active_dict['sounds']
                
                available_models.append({
                    'id': model_id,
                    'name': f"{dictionary_name} RF Model",
                    'type': 'rf',
                    'dictionary': dictionary_name,
                    'class_names': class_names
                })
        
        # Add ensemble models if available
        if 'ensemble' in models_data and models_data['ensemble']:
            logging.info(f"Found {len(models_data['ensemble'])} Ensemble models")
            
            for model_info in models_data['ensemble']:
                model_id = model_info.get('model_id')
                dictionary_name = model_info.get('dictionary_name', dict_name).replace(' ', '_')
                
                class_names = model_info.get('class_names', [])
                if not class_names and 'sounds' in active_dict:
                    class_names = active_dict['sounds']
                
                available_models.append({
                    'id': model_id,
                    'name': f"{dictionary_name} Ensemble Model",
                    'type': 'ensemble',
                    'dictionary': dictionary_name,
                    'class_names': class_names
                })
        
        # If no models found at all, add a fallback model based on active dictionary
        if not available_models:
            logging.warning(f"No models found for '{dict_name}', adding fallback")
            class_names = active_dict.get('sounds', [])
            available_models = [{
                'id': f"{dict_name.replace(' ', '_')}_model",
                'name': f"{dict_name} Default Model",
                'type': 'cnn',
                'dictionary': dict_name.replace(' ', '_'),
                'class_names': class_names
            }]
        
        # Return the models
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

@ml_bp.route('/training', methods=['GET', 'POST'])
def training_view():
    """
    Advanced version of the model training interface.
    
    This route uses the same backend processing as the standard train route
    but renders the more advanced training.html template.
    
    GET: Renders the advanced training form
    POST: Processes the training request using the unified approach
    
    Returns:
        Rendered template or redirect to training status
    """
    # Process POST requests by forwarding to train_model
    if request.method == 'POST':
        return train_model()
    
    # For GET requests, copy the exact same processing as the train route
    # Always perform sound file analysis for GET requests
    training_service = TrainingService()
    audio_dir = get_goodsounds_dir_path()
    analysis = training_service.analyze_sound_file_status(audio_dir)
    
    # Load available dictionaries
    dictionaries = Config.get_dictionaries()
    dict_names = [d['name'] for d in dictionaries if 'name' in d]
    
    # Set the list of model types
    model_types = ['cnn', 'rf', 'ensemble']

    # Get sounds for the currently selected dictionary (if any)
    selected_dict = request.args.get('dictionary') or request.args.get('dict')
    sounds = []
    if selected_dict:
        selected_dict_data = next((d for d in dictionaries if d.get('name') == selected_dict), None)
        if selected_dict_data:
            sounds = selected_dict_data.get('sounds', [])
    
    # Get augmentation summary from session if available
    augmentation_summary = None
    if 'augmentation_summary' in session:
        augmentation_summary = session.pop('augmentation_summary')
    
    # Get extract_features flag from query parameters
    extract_features = request.args.get('extract_features') == 'true'
    
    # Get extract_done flag from session if available
    extract_done = session.pop('extract_done', False)
    
    # Check feature extraction status if requested
    feature_status = None
    if request.args.get('check_features') == 'true' or extract_features:
        if selected_dict:
            # Get the sound classes for this dictionary
            class_dirs = []
            for sound in sounds:
                class_dir = os.path.join(get_goodsounds_dir_path(), sound)
                if os.path.exists(class_dir):
                    class_dirs.append(class_dir)
            
            # Get status of feature extraction
            feature_status = {
                "total_files": 0,
                "files_with_features": 0,
                "files_missing_features": 0,
                "classes": {}
            }
            
            # Check each class directory
            for class_dir in class_dirs:
                class_name = os.path.basename(class_dir)
                feature_status["classes"][class_name] = {
                    "total_files": 0,
                    "files_with_features": 0,
                    "files_missing_features": 0,
                    "missing_files": []
                }
                
                # Check all WAV files in this class
                wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                feature_status["total_files"] += len(wav_files)
                feature_status["classes"][class_name]["total_files"] = len(wav_files)
                
                for wav_file in wav_files:
                    file_path = os.path.join(class_dir, wav_file)
                    # Check if features exist by attempting to load them
                    from src.ml.feature_extractor import FeatureExtractor
                    feature_extractor = FeatureExtractor()
                    has_features = feature_extractor.check_features_exist(file_path)
                    
                    if has_features:
                        feature_status["files_with_features"] += 1
                        feature_status["classes"][class_name]["files_with_features"] += 1
                    else:
                        feature_status["files_missing_features"] += 1
                        feature_status["classes"][class_name]["files_missing_features"] += 1
                        feature_status["classes"][class_name]["missing_files"].append(wav_file)
                        
                        # Extract features if requested
                        if extract_features:
                            try:
                                # Create a feature extractor if needed
                                from src.ml.feature_extractor import FeatureExtractor
                                feature_extractor = FeatureExtractor()
                                feature_extractor.extract_features(file_path)
                                extract_done = True
                            except Exception as e:
                                current_app.logger.error(f"Error extracting features for {file_path}: {e}")
            
            # If we extracted features, recheck the status
            if extract_done:
                # Reset counters
                feature_status["files_with_features"] = 0
                feature_status["files_missing_features"] = 0
                for class_name in feature_status["classes"]:
                    feature_status["classes"][class_name]["files_with_features"] = 0
                    feature_status["classes"][class_name]["files_missing_features"] = 0
                    feature_status["classes"][class_name]["missing_files"] = []
                
                # Recheck each class directory
                for class_dir in class_dirs:
                    class_name = os.path.basename(class_dir)
                    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                    
                    for wav_file in wav_files:
                        file_path = os.path.join(class_dir, wav_file)
                        # Check if features exist by attempting to load them
                        from src.ml.feature_extractor import FeatureExtractor
                        feature_extractor = FeatureExtractor()
                        has_features = feature_extractor.check_features_exist(file_path)
                        
                        if has_features:
                            feature_status["files_with_features"] += 1
                            feature_status["classes"][class_name]["files_with_features"] += 1
                        else:
                            feature_status["files_missing_features"] += 1
                            feature_status["classes"][class_name]["files_missing_features"] += 1
                            feature_status["classes"][class_name]["missing_files"].append(wav_file)
            
        # Calculate feature status for the template if not already determined
        if feature_status is None:
            feature_status = {
                'total_files': analysis.get('total_files', 0),
                'files_with_features': analysis.get('files_with_features', 0),
                'files_missing_features': analysis.get('files_missing_features', 0)
            }
    
    # Check if there is a training in progress or completed
    training_in_progress = 'training_status' in session and session['training_status'] != 'completed'
    
    # Prepare dictionary data in the format expected by the training template
    # This matches the structure expected by the template's JavaScript
    formatted_dictionaries = []
    for d in dictionaries:
        if 'name' in d:
            dict_info = {
                'name': d['name'],
                'classes': d.get('sounds', []),
                'sample_count': sum(1 for _ in d.get('sounds', []))
            }
            formatted_dictionaries.append(dict_info)
            
    # Pass data to the template
    return render_template('training.html', 
                           analysis=analysis,
                           audio_dir=audio_dir,
                           dictionaries=formatted_dictionaries,
                           model_types=model_types,
                           sounds=sounds,
                           selected_dict=selected_dict,
                           augmentation_summary=augmentation_summary,
                           feature_status=feature_status,
                           extract_features=extract_features,
                           extract_done=extract_done,
                           training_in_progress=training_in_progress)
