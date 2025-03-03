# SoundClassifier_v08/src/routes/ml_routes.py

LATEST_PREDICTION = None

# Standard library imports
import os
import json
import logging
import time
import threading
from datetime import datetime
import uuid

# Flask imports
from flask import (
    Blueprint, render_template, request, session,
    redirect, url_for, flash, jsonify, Response, stream_with_context
)
from flask import current_app

# Scientific/ML imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import librosa

# Local application imports
from config import Config
from src.ml.cnn_classifier import build_model, build_dataset
from src.ml.rf_classifier import RandomForestClassifier
from src.ml.ensemble_classifier import EnsembleClassifier
from src.ml.inference import predict_sound, SoundDetector
from src.ml.audio_processing import SoundProcessor
from src.ml.trainer import Trainer
from src.ml.sound_detector_rf import SoundDetectorRF
from src.ml.sound_detector_ensemble import SoundDetectorEnsemble
from src.ml.feature_extractor import AudioFeatureExtractor
from src.ml.model_paths import get_model_dir, get_cnn_model_path, get_rf_model_path, save_model_metadata

# Global variables for training stats
training_stats = None
training_history = None
model_summary_str = None          

ml_bp = Blueprint('ml', __name__)

@ml_bp.before_app_request
def init_app_inference_stats():

    if not hasattr(current_app, 'inference_stats'):
        current_app.inference_stats = {
            'total_predictions': 0,
            'class_counts': {},
            'confidence_levels': [],
            'confusion_matrix': {},
            'misclassifications': [],
            'correct_classifications': []
        }
        logging.debug("Initialized current_app.inference_stats.")

    # Minimal addition:
    if not hasattr(current_app, 'latest_prediction'):
        LATEST_PREDICTION = None
        logging.debug("Initialized LATEST_PREDICTION = None.")

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

# Global variable to avoid app context for SSE:
#LATEST_PREDICTION = None

def prediction_callback(prediction):
    global LATEST_PREDICTION
    LATEST_PREDICTION = prediction
    logging.info(f"Got prediction: {prediction}")
   

    stats = current_app.inference_stats
    stats['total_predictions'] += 1
    c = prediction['class']
    conf = prediction['confidence']
    actual = prediction.get('actual_sound')

    stats['class_counts'].setdefault(c, 0)
    stats['class_counts'][c] += 1
    stats['confidence_levels'].append(conf)

    # More advanced confusion matrix if you want
    if 'confusion_matrix' not in stats:
        stats['confusion_matrix'] = {}
    if 'misclassifications' not in stats:
        stats['misclassifications'] = []
    if 'correct_classifications' not in stats:
        stats['correct_classifications'] = []

    cm = stats['confusion_matrix']
    if actual:
        if actual not in cm:
            cm[actual] = {}
        if c not in cm[actual]:
            cm[actual][c] = 0
        cm[actual][c] += 1

        detail = {
            'predicted': c,
            'actual': actual,
            'confidence': conf,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        if c == actual:
            stats['correct_classifications'].append(detail)
        else:
            stats['misclassifications'].append(detail)

    # Also store top-level prediction in a global var for SSE
    #global LATEST_PREDICTION
    #LATEST_PREDICTION = prediction

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
    global training_stats, training_history, model_summary_str

    if request.method == 'GET':
        # Display the form (train_model.html) with the dropdown to select method
        # Existing code: no changes here
        sounds = get_sound_list()  # Example function to list sounds
        return render_template('train_model.html', sounds=sounds)

    # POST: user clicked "Start Training" with a chosen method
    train_method = request.form.get('train_method', 'cnn')  # defaults to 'cnn'
    goodsounds_dir = get_goodsounds_dir_path()  # however you get your data path

    try:
        # Turn off debug if needed (per your existing code)
        original_debug = current_app.config['DEBUG']
        current_app.config['DEBUG'] = False

        # Create an instance of your Trainer (if you have one)
        trainer = Trainer(model_dir=os.path.join(Config.PROJECT_ROOT, 'models'))

        # We will store some results in these variables
        rf_result = None
        cnn_result = None
        ensemble_result = None

        if train_method == 'cnn':
            logging.info("# NEW LOG: Entering CNN training block (cnn only).")

            X, y, class_names, stats = build_dataset(goodsounds_dir)
            if X is None or y is None:
                flash("No training data found for CNN.")
                return redirect(url_for('ml.train_model'))

            # (B) Shuffle / split data
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]
            split_idx = int(len(X)*0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            input_shape = X_train.shape[1:]

            # (C) Build & train the CNN model
            model, model_summary_cnn = build_model(input_shape, num_classes=len(class_names))
            class_weights = {}
            total_samples = len(y_train)
            for i in range(len(class_names)):
                ccount = (y_train == i).sum()
                class_weights[i] = total_samples / (len(class_names)*ccount)

            callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                min_delta=0.001,
                restore_best_weights=True
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=5,
                min_lr=1e-5
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                class_weight=class_weights,
                callbacks=[callback, reduce_lr],
                batch_size=32,
                verbose=1
            )

            # (D) Optionally evaluate on val set
            val_loss, val_acc = model.evaluate(X_val, y_val)

            # (E) Build training_stats & training_history from the CNN
            training_stats, training_history = build_training_stats_and_history(
                X_train, y_train, X_val, y_val,
                class_names, stats, model_summary_cnn, history
            )
            model_summary_str = model_summary_cnn
            cnn_result = {'val_acc': val_acc, 'model': model}  # keep for ensemble?
            training_stats['classes'] = class_names
            training_stats['dictionary_name'] = Config.get_dictionary().get('name', 'Unknown')

            # Find this line where the model is saved and log it
            model_path = os.path.join('models', f"{Config.get_dictionary().get('name', 'Unknown').replace(' ', '_')}_model.h5")
            logging.info(f"Saving model to: {model_path}")
            model.save(model_path)

            return redirect(url_for('ml.cnn_training_summary'))

        elif train_method == 'rf':  # NEW
            logging.info("# NEW LOG: Entering RF training block (rf only).")

            # 1) Train the Random Forest
            success = trainer.train_rf(goodsounds_dir)
            rf_accuracy = trainer.get_rf_accuracy()
            class_names = trainer.get_rf_class_names()

            # ─────────────────────────────────────────────────────────────
            # 2) Retrieve X_train, X_val, stats from the trainer
            X_train, X_val, stats = trainer.get_rf_train_data()
            # ─────────────────────────────────────────────────────────────

            # 3) Construct training_stats
            training_stats = {
                'method': 'Random Forest Only',
                'rf_acc': rf_accuracy,
                'classes': class_names,
                'dictionary_name': Config.get_dictionary().get('name', 'Unknown'),

                # Now these train/val stats show the real split
                'train_samples': len(X_train) if X_train is not None else 0,
                'val_samples': len(X_val) if X_val is not None else 0,
                'total_samples': (len(X_train) + len(X_val)) if X_train is not None and X_val is not None else 0,

                # Use what you stored in trainer._stats
                'original_counts': stats.get('original_counts', {}),
                'augmented_counts': stats.get('augmented_counts', {}),
                'model_summary_str': model_summary_str,
                'training_history': training_history,
            }

            training_history = None
            model_summary_str = "Random Forest does not have a Keras-style summary."

            # 4) Redirect to summary
            return redirect(url_for('ml.rf_training_summary'))
        
        elif train_method == 'ensemble':  # NEW
            logging.info("# NEW LOG: Entering Ensemble training block (cnn+rf).")

            # You might need a current CNN model and a current RF model
            # If you want to train them freshly here, do so, or load from disk
            # Minimal approach: 
            success_rf = trainer.train_rf(goodsounds_dir)
            X, y, class_names, stats = build_dataset(goodsounds_dir)
            input_shape = X.shape[1:]
            model, model_summary_cnn = build_model(input_shape, len(class_names))
            # Maybe skip the full .fit() if you're just loading existing weights
            # or do a short training. Example:
            history = model.fit(X, y, epochs=2, verbose=1)  # short train or load

            # Then do your ensemble logic, e.g. weighted average:
            # predictions_cnn = model.predict(...)
            # predictions_rf = trainer.rf_classifier.predict_proba(...)
            # ensemble_preds = 0.5 * predictions_cnn + 0.5 * predictions_rf
            # ...
            ensemble_acc = 0.99  # example placeholder

            # Build training_stats for summary
            training_stats = {
                'method': 'Ensemble Only',
                'ensemble_acc': ensemble_acc,
                # Possibly store partial stats for CNN, RF as well
            }
            model_summary_str = "Ensemble combines CNN + RF results."
            training_history = None
            ensemble_result = {'val_acc': ensemble_acc}
            training_stats['classes'] = class_names
            training_stats['dictionary_name'] = Config.get_dictionary().get('name', 'Unknown')
            return redirect(url_for('ml.ensemble_training_summary'))

        elif train_method == 'all':  # NEW
            logging.info("# NEW LOG: Entering ALL training block (cnn + rf + ensemble).")

            #  A) Train CNN
            X, y, class_names, stats = build_dataset(goodsounds_dir)
            ...
            # The same CNN code from above, or
            model, model_summary_cnn, c_acc = train_cnn_briefly(X, y, class_names, stats)
            cnn_result = {'val_acc': c_acc, 'model': model}

            #  B) Train RF
            success_rf = trainer.train_rf(goodsounds_dir)
            rf_accuracy = trainer.get_rf_accuracy()

            #  C) Do ensemble
            # E.g. do a weighted average from both newly trained models
            # ensemble_acc = compute_ensemble_accuracy(model, trainer.rf_classifier)
            ensemble_acc = 0.98  # placeholder

            #  D) Build training_stats with all 3 results
            training_stats = {
                'method': 'All (CNN + RF + Ensemble)',
                'cnn_acc': c_acc,
                'rf_acc': rf_accuracy,
                'ensemble_acc': ensemble_acc,
                # add shapes, stats, etc. if needed
            }
            training_history = None
            model_summary_str = "Trained CNN, trained RF, computed ensemble."
            ensemble_result = {'val_acc': ensemble_acc}
            training_stats['classes'] = class_names
            training_stats['dictionary_name'] = Config.get_dictionary().get('name', 'Unknown')
            return redirect(url_for('ml.training_model_comparisons'))
            
        else:
            # If the user somehow selected something else, default to CNN or redirect
            logging.warning(f"Unknown train method {train_method}, defaulting to CNN.")
            return redirect(url_for('ml.train_model'))

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        flash(str(e), 'error')
    finally:
        current_app.config['DEBUG'] = original_debug

    # Once we finish training, redirect to the summary page
    # return redirect(url_for('ml.training_model_comparisons'))

# ... existing imports and code above ...

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
    """
    Main prediction page using the new real-time predictor interface.
    """
    if 'username' not in session:
        session['username'] = 'guest'
    
    active_dict = Config.get_dictionary()
    if not active_dict or 'sounds' not in active_dict:
        flash("No active dictionary found. Please create or select a dictionary first.")
        return redirect(url_for('ml.list_dictionaries'))
    
    return render_template('predict.html', active_dict=active_dict)

@ml_bp.route('/predict_cnn', methods=['GET', 'POST'])
def predict_cnn():
    active_dict = Config.get_dictionary()
    if 'username' not in session:
        session['username'] = 'guest'

    # Add logging to see model file details
    dict_name = active_dict['name']
    expected_model_path = get_cnn_model_path('models', dict_name.replace(' ', '_'), 'v1')
    logging.info(f"Looking for CNN model for: {dict_name}")
    logging.info(f"Expected path: {expected_model_path}")
    
    if not os.path.exists(expected_model_path):
        models_dir = 'models'
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            logging.info(f"Available models: {files}")
            
            # Try to find a suitable model by partial name match
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

            model = models.load_model(expected_model_path)
            class_names = active_dict['sounds']

            # Rest of your POST handling code...

        except Exception as e:
            logging.error(f"Error during CNN prediction: {e}", exc_info=True)
            flash(str(e), 'error')

    return render_template('inference.html', active_dict=active_dict)

@ml_bp.route('/predict_rf', methods=['POST'])
def predict_rf():

    # 1) Load the active dictionary and get the list of sounds
    active_dict = Config.get_dictionary()
    class_names = active_dict.get('sounds', [])

    # 2) Load the Random Forest model
    rf_path = os.path.join('models', f"{active_dict['name'].replace(' ', '_')}_rf.joblib")
    if not os.path.exists(rf_path):
        return jsonify({"error": "No random forest model for current dictionary."}), 400

    rf = RandomForestClassifier(model_dir='models')
    if not rf.load(filename=os.path.basename(rf_path)):
        return jsonify({"error": "Failed loading RF model."}), 500

    # 3) Read the uploaded audio file
    uploaded_file = request.files.get('audio')
    if not uploaded_file:
        return jsonify({"error": "No audio file"}), 400

    # Create a unique filename for the uploaded file
    filename = f"rf_predict_{uuid.uuid4().hex[:8]}.wav"
    temp_path = os.path.join(Config.UPLOADED_SOUNDS_DIR, filename)
    uploaded_file.save(temp_path)

    try:
        # 4) Load audio from the temporary file
        # Note: This call creates an instance of SoundProcessor so that we can use its methods.
        sp = SoundProcessor(sample_rate=16000)
        audio_data, _ = librosa.load(temp_path, sr=16000)

        # 5) Use SoundProcessor.detect_sound_boundaries() to trim silence
        start_idx, end_idx, has_sound = sp.detect_sound_boundaries(audio_data)
        trimmed_data = audio_data[start_idx:end_idx]
        if len(trimmed_data) == 0:
            return jsonify({"error": "No sound detected in audio"}), 400

        # Overwrite the temp file with the trimmed audio
        wavfile.write(temp_path, 16000, np.int16(trimmed_data * 32767))

        # 6) Extract classical features using AudioFeatureExtractor.
        # AudioFeatureExtractor.extract_features() expects a file path.
        extractor = AudioFeatureExtractor(sr=16000)
        feats = extractor.extract_features(temp_path)
        if not feats:
            return jsonify({"error": "Feature extraction failed"}), 500

        # 7) Create a feature row in the same order as expected by the RF model
        feature_names = extractor.get_feature_names()
        row = [feats[fn] for fn in feature_names]

        # 8) Run RandomForest prediction
        preds, probs = rf.predict([row])
        if preds is None:
            return jsonify({"error": "RF predict() returned None"}), 500

        predicted_class = preds[0]
        confidence = float(np.max(probs[0]))  # highest probability

        return jsonify({
            "predictions": [{
                "sound": predicted_class,
                "probability": confidence
            }]
        })

    except Exception as e:
        logging.error(f"Error during RF prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up the temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
         
@ml_bp.route('/predict_ensemble', methods=['POST'])
def predict_ensemble():
    active_dict = Config.get_dictionary()
    class_names = active_dict['sounds']
    
    # 1) load CNN
    cnn_path = os.path.join('models', f"{active_dict['name'].replace(' ','_')}_model.h5")
    if not os.path.exists(cnn_path):
        return jsonify({"error":"No CNN model"}), 400
    cnn_model = load_model(cnn_path)
    
    # 2) load RF
    rf_path = os.path.join('models', f"{active_dict['name'].replace(' ','_')}_rf.joblib")
    if not os.path.exists(rf_path):
        return jsonify({"error":"No RF model"}), 400
    
    rf = RandomForestClassifier(model_dir='models')
    rf.load(filename=os.path.basename(rf_path))
    
    # 3) create ensemble
    ensemble = EnsembleClassifier(rf, cnn_model, class_names, rf_weight=0.5)
    
    # 4) get the posted audio, do feature extraction for RF and for CNN
    file = request.files.get('audio')
    if not file:
        return jsonify({"error":"No audio file"}), 400
    
    # Create a unique filename for the uploaded file
    filename = f"ensemble_predict_{uuid.uuid4().hex[:8]}.wav"
    temp_path = os.path.join(Config.UPLOADED_SOUNDS_DIR, filename)
    file.save(temp_path)
    
    # For CNN: process with SoundProcessor
    sp = SoundProcessor(sample_rate=16000)
    wav_data, _ = librosa.load(temp_path, sr=16000)
    cnn_features = sp.process_audio(wav_data)
    if cnn_features is None:
        return jsonify({"error":"No features for CNN"}), 500
    X_cnn = np.expand_dims(cnn_features, axis=0)
    
    # For RF: use AudioFeatureExtractor
    feats = AudioFeatureExtractor(sr=16000).extract_features(temp_path)
    os.remove(temp_path)
    if not feats:
        return jsonify({"error":"Feature extraction failed for RF"}), 500
    
    feature_names = AudioFeatureExtractor(sr=16000).get_feature_names()
    row = [feats[fn] for fn in feature_names]
    X_rf = [row]
    
    # 5) get ensemble predictions
    top_preds = ensemble.get_top_predictions(X_rf, X_cnn, top_n=1)[0]
    # top_preds is something like [{"sound":..., "probability":...}]
    
    return jsonify({
        "predictions": top_preds
    })

@ml_bp.route('/predict_sound', methods=['POST'])
def predict_sound_endpoint():

    active_dict = Config.get_dictionary()
    dict_name = active_dict['name']
    model_path = os.path.join('models', f"{dict_name.replace(' ','_')}_model.h5")
    if not os.path.exists(model_path):
        return jsonify({"error": f"No model for dictionary {dict_name}"}), 400

    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file"}), 400

    model = models.load_model(model_path)
    class_names = active_dict['sounds']

    # Create a unique filename for the uploaded file
    filename = f"predict_temp_{uuid.uuid4().hex[:8]}.wav"
    temp_path = os.path.join(Config.UPLOADED_SOUNDS_DIR, filename)
    with open(temp_path, 'wb') as f:
        f.write(audio_file.read())

    pred_class, confidence = predict_sound(model, temp_path, class_names, use_microphone=False)
    os.remove(temp_path)  # Clean up temporary file after prediction

    # Return top-1 for now
    return jsonify({
        "predictions": [{
            "sound": pred_class,
            "probability": float(confidence)
        }]
    })

@ml_bp.route('/start_listening', methods=['POST'])
def start_listening():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    # Reset stats
    current_app.inference_stats = {
        'total_predictions': 0,
        'class_counts': {},
        'confidence_levels': [],
        'confusion_matrix': {},
        'misclassifications': [],
        'correct_classifications': []
    }

    global sound_detector
    try:
        # 1) figure out which model
        model_choice = request.args.get('model', 'cnn')  # "cnn", "rf", or "ensemble"
        
        active_dict = Config.get_dictionary()
        class_names = active_dict.get('sounds', [])
        dict_name = active_dict.get('name', 'Default')
        if not class_names:
            return jsonify({'status': 'error', 'message': 'No sounds in dictionary'})

        if model_choice == 'rf':
            # Load the RF model
            rf_path = os.path.join('models', f"{dict_name.replace(' ','_')}_rf.joblib")
            if not os.path.exists(rf_path):
                return jsonify({'status': 'error', 'message': 'No RF model available for dictionary.'}), 400

            rf_classifier = RandomForestClassifier(model_dir='models')
            rf_classifier.load(filename=os.path.basename(rf_path))

            sound_detector = SoundDetectorRF(rf_classifier)
            # Start listening with the same callback
            sound_detector.start_listening(callback=prediction_callback)

            return jsonify({'status': 'success', 'message': 'Real-time RF started'})

        elif model_choice == 'ensemble':
            
            # Load the CNN model
            cnn_path = os.path.join('models', f"{dict_name.replace(' ','_')}_model.h5")
            if not os.path.exists(cnn_path):
                return jsonify({'status': 'error', 'message': 'No CNN model found for this dictionary'}), 400
            cnn_model = load_model(cnn_path)

            # Load the RF model
            rf_path = os.path.join('models', f"{dict_name.replace(' ','_')}_rf.joblib")
            if not os.path.exists(rf_path):
                return jsonify({'status': 'error', 'message': 'No RF model found for this dictionary'}), 400

            rf_classifier = RandomForestClassifier(model_dir='models')
            rf_classifier.load(filename=os.path.basename(rf_path))

            # Create an EnsembleClassifier object
            ensemble_model = EnsembleClassifier(
                rf_classifier=rf_classifier,
                cnn_model=cnn_model,
                class_names=class_names,
                rf_weight=0.5
            )

            # Create the SoundDetectorEnsemble
            sound_detector = SoundDetectorEnsemble(ensemble_model)

            # Start listening, same callback as your CNN/RF routes
            success = sound_detector.start_listening(callback=prediction_callback)
            if not success:
                return jsonify({'status': 'error', 'message': 'Failed to start Ensemble detection'}), 500

            return jsonify({'status': 'success', 'message': 'Real-time Ensemble started'})
        else:
            # Add more detailed logging to find the model
            logging.info(f"Looking for CNN model for: {dict_name}")
            
            # Check for possible paths
            possible_paths = [
                # Path 1: Expected path from get_cnn_model_path()
                get_cnn_model_path('models', dict_name.replace(' ', '_'), 'v1'),
                
                # Path 2: Old format with _model.h5 suffix
                os.path.join('models', f"{dict_name.replace(' ', '_')}_model.h5"),
                
                # Path 3: Direct cnn_model.h5 file in dictionary folder
                os.path.join('models', dict_name.replace(' ', '_'), 'cnn_model.h5'),
                
                # Path 4: Any .h5 file in the models folder with dict name
                os.path.join('models', f"{dict_name.replace(' ', '_')}.h5"),
                
                # Path 5: Just try a fallback model
                os.path.join('models', 'audio_classifier.h5'),
            ]
            
            # Try each path
            model_path = None
            for path in possible_paths:
                logging.info(f"Checking path: {path}")
                if os.path.exists(path):
                    model_path = path
                    logging.info(f"Found model at: {model_path}")
                    break
            
            # Check if there's a directory with dict name and list its contents
            dict_dir = os.path.join('models', dict_name.replace(' ', '_'))
            if os.path.exists(dict_dir) and os.path.isdir(dict_dir):
                # List all files in this directory and subdirectories
                logging.info(f"Listing contents of {dict_dir}:")
                for root, dirs, files in os.walk(dict_dir):
                    for file in files:
                        if file.endswith('.h5'):
                            logging.info(f"Found .h5 file: {os.path.join(root, file)}")
                            if not model_path:  # use the first .h5 file if no path was found
                                model_path = os.path.join(root, file)
                                logging.info(f"Using model at: {model_path}")
            
            # If still not found, try any .h5 file in models dir
            if not model_path:
                models_dir = 'models'
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        if file.endswith('.h5'):
                            model_path = os.path.join(models_dir, file)
                            logging.info(f"Falling back to: {model_path}")
                            break
            
            if not os.path.exists(model_path):
                logging.error(f"No CNN model found for {dict_name} after trying all paths")
                return jsonify({
                    'status': 'error', 
                    'message': f'No CNN model found for {dict_name}. Please train a model first.'
                }), 400
            
            with tf.keras.utils.custom_object_scope({'BatchShape': lambda x: None}):
                model = tf.keras.models.load_model(model_path)
            
            sound_detector = SoundDetector(model, class_names)
            sound_detector.start_listening(callback=prediction_callback)
            
            return jsonify({'status': 'success'})

    except Exception as e:
        logging.error(f"Error in start_listening: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)})

@ml_bp.route('/stop_listening', methods=['POST'])
def stop_listening():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    global sound_detector
    try:
        if sound_detector:
            result = sound_detector.stop_listening()
            sound_detector = None
            return jsonify(result)
        else:
            return jsonify({"status":"error","message":"No active listener"})
    except Exception as e:
        logging.error(f"Error stopping listener: {e}", exc_info=True)
        return jsonify({"status":"error","message":str(e)})

@ml_bp.route('/prediction_stream')
def prediction_stream():

    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401

    def stream_predictions():
        last_index = 0
        while True:
            data = {}
            global LATEST_PREDICTION
            if LATEST_PREDICTION:
                data['prediction'] = LATEST_PREDICTION  
                LATEST_PREDICTION = None

            if len(debug_log_handler.logs) > last_index:
                data['log'] = debug_log_handler.logs[last_index]
                last_index = len(debug_log_handler.logs)

            if data:
                yield f"data: {json.dumps(data)}\n\n"
            else:
                yield ": heartbeat\n\n"
            time.sleep(0.1)

    return Response(stream_predictions(), mimetype='text/event-stream')

@ml_bp.route('/inference_statistics')
def inference_statistics():
    stats = current_app.inference_stats
    if not stats['confidence_levels']:
        avg_conf = 0.0
    else:
        avg_conf = sum(stats['confidence_levels']) / len(stats['confidence_levels'])

    # Class accuracy
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

    misclass_patterns = []
    for a_sound in cm:
        for p_sound, count in cm[a_sound].items():
            if a_sound != p_sound and count>0:
                misclass_patterns.append({
                    'actual': a_sound,
                    'predicted': p_sound,
                    'count': count
                })
    misclass_patterns.sort(key=lambda x: x['count'], reverse=True)

    return jsonify({
        'total_predictions': stats.get('total_predictions',0),
        'average_confidence': avg_conf,
        'class_counts': stats.get('class_counts',{}),
        'class_accuracy': class_accuracy,
        'confusion_matrix': cm,
        'misclassification_patterns': misclass_patterns,
        'recent_misclassifications': stats.get('misclassifications', [])[-10:],
        'recent_correct_classifications': stats.get('correct_classifications', [])[-10:]
    })

@ml_bp.route('/record_feedback', methods=['POST'])
def record_feedback():
    if 'username' not in session:
        return jsonify({'status':'error','message':'Please log in first'}), 401

    data = request.get_json()
    predicted_sound = data.get('predicted_sound')
    actual_sound = data.get('actual_sound')
    confidence = data.get('confidence')

    if not all([predicted_sound, actual_sound, confidence is not None]):
        return jsonify({'status':'error','message':'Missing data'}), 400

    # Fire the callback with "actual" so the confusion matrix updates
    cb_pred = {
        'class': predicted_sound,
        'confidence': confidence,
        'actual_sound': actual_sound
    }
    prediction_callback(cb_pred)
    return jsonify({'status':'success'})

@ml_bp.route('/save_analysis', methods=['POST'])
def save_analysis():
    if 'username' not in session:
        return jsonify({'status':'error','message':'Please log in first'}), 401

    stats = current_app.inference_stats
    dict_name = Config.get_dictionary().get('name','unknown')
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'dictionary': dict_name,
        'confusion_matrix': stats.get('confusion_matrix',{}),
        'misclassifications': stats.get('misclassifications',[]),
        'correct_classifications': stats.get('correct_classifications',[]),
        'total_predictions': (len(stats.get('misclassifications',[])) +
                              len(stats.get('correct_classifications',[]))),
        'confidence_levels': stats.get('confidence_levels',[]),
        'class_counts': stats.get('class_counts',{})
    }

    analysis_dir = os.path.join(Config.CONFIG_DIR, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    filename = f"analysis_{dict_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(analysis_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(analysis_data, f, indent=2)

    return jsonify({'status':'success','message':'Analysis data saved'})

@ml_bp.route('/view_analysis')
def view_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))

    analysis_dir = os.path.join(Config.CONFIG_DIR, 'analysis')
    if not os.path.exists(analysis_dir):
        return render_template('view_analysis.html', analysis_files=[])

    analysis_files = []
    for fname in os.listdir(analysis_dir):
        if fname.endswith('.json'):
            path = os.path.join(analysis_dir, fname)
            with open(path,'r') as f:
                data = json.load(f)
            analysis_files.append({
                'filename': fname,
                'timestamp': data.get('timestamp',''),
                'dictionary': data.get('dictionary',''),
                'total_predictions': data.get('total_predictions',0)
            })
    analysis_files.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template('view_analysis.html', analysis_files=analysis_files)

@ml_bp.route('/get_analysis/<filename>')
def get_analysis(filename):
    if 'username' not in session:
        return jsonify({'status':'error','message':'Please log in first'}), 401

    analysis_dir = os.path.join(Config.CONFIG_DIR, 'analysis')
    filepath = os.path.join(analysis_dir, filename)

    # Safety check
    if not os.path.abspath(filepath).startswith(os.path.abspath(analysis_dir)):
        return jsonify({'status':'error','message':'Invalid filename'}),400

    if not os.path.exists(filepath):
        return jsonify({'status':'error','message':'File not found'}),404

    try:
        with open(filepath,'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}),500

@ml_bp.route('/training_status')
def training_status():
    # So your front-end can poll. If you prefer a purely JS solution, okay.
    if hasattr(current_app, 'training_progress'):
        return jsonify({
            'progress': current_app.training_progress,
            'status': current_app.training_status
        })
    return jsonify({'progress':0,'status':'Not started'})

@ml_bp.route('/record', methods=['POST'])
def record():
    if 'username' not in session:
        return redirect(url_for('index'))

    sound = request.form.get('sound')
    audio_data = request.files.get('audio')
    current_app.logger.debug(f"Recording attempt. sound={sound}, has audio={audio_data is not None}")
    if sound and audio_data:
        try:
            # Convert webm->wav with pydub
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

            # Chop into chunks
            processor = Audio_Chunker()
            chopped_files = processor.chop_recording(temp_path)
            os.remove(temp_path)  # remove original big recording

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
        # Move chunk to the sounds directory
        sound = parts[0]
        username = session['username']
        sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
        os.makedirs(sound_dir, exist_ok=True)

        # Count how many are in that folder for that user
        existing_count = len([
            f for f in os.listdir(sound_dir)
            if f.startswith(f"{sound}_{username}_")
        ])
        new_filename = f"{sound}_{username}_{existing_count + 1}.wav"
        new_path = os.path.join(sound_dir, new_filename)
        os.rename(os.path.join(Config.TEMP_DIR, chunk_file), new_path)
        flash(f"Chunk saved as {new_filename}")
    else:
        # Delete
        os.remove(os.path.join(Config.TEMP_DIR, chunk_file))
        flash("Chunk deleted.")

    return redirect(url_for('ml.verify_chunks', timestamp=timestamp))

@ml_bp.route('/manage_dictionaries')
def list_dictionaries():
    """Render the dictionary listing page."""
    logging.info("=== DICTIONARIES PAGE REQUESTED ===")
    
    # Get dictionaries
    dictionaries = Config.get_dictionaries()
    active_dictionary = Config.get_dictionary()
    
    # Get sound stats
    sound_stats = {}
    if active_dictionary and 'sounds' in active_dictionary:
        for sound in active_dictionary['sounds']:
            sound_stats[sound] = {}
    
    return render_template('dictionaries.html',
                           dictionaries=dictionaries,
                           active_dictionary=active_dictionary,
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

    Config.get_dictionary(selected_dict)
    # Create directories for each sound
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

    if session.get('is_admin'):
        # Admin sees everything
        recordings_by_sound = {}
        active_dict = Config.get_dictionary()
        if not active_dict:
            flash("No active dictionary.")
            return redirect(url_for('index'))

        for sound in active_dict['sounds']:
            sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
            if os.path.exists(sound_dir):
                files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]
                if files:
                    recordings_by_sound[sound] = files
        return render_template('list_recordings.html', recordings=recordings_by_sound)
    else:
        # Show only user's recordings
        user = session['username']
        recordings_by_sound = {}
        active_dict = Config.get_dictionary()
        if not active_dict:
            flash("No active dictionary.")
            return redirect(url_for('index'))

        for sound in active_dict['sounds']:
            sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
            if not os.path.exists(sound_dir):
                continue
            sound_files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]
            # Filter for user
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
        sound_stats[sound] = {
            'system_total': 0,
            'user_total': 0
        }
        sound_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound)
        if os.path.exists(sound_dir):
            files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]
            sound_stats[sound]['system_total'] = len(files)
            # user_total is optional
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


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    processor = SoundProcessor()
    all_chunks = []

    for file in files:
        if file.filename.lower().endswith('.wav'):
            temp_path = os.path.join(Config.TEMP_DIR,
                                     f"{sound}_{session['username']}_{timestamp}_temp.wav")
            file.save(temp_path)
            chunks = processor.chop_recording(temp_path)
            all_chunks.extend(chunks)
            os.remove(temp_path)

    if not all_chunks:
        flash("No valid sound chunks found in uploads.")
        return redirect(url_for('ml.upload_sounds'))

    return redirect(url_for('ml.verify_chunks', timestamp=timestamp))


def gather_training_stats_for_rf():
    global training_stats, training_history, model_summary_str
    if training_stats is None:
        return {}
    return training_stats
    
@ml_bp.route("/rf_training_summary")
def rf_training_summary():
    global training_stats, training_history, model_summary_str

    # For example, you can gather stats from your Trainer or session
    trainer = Trainer()
    trainer.train_rf(goodsounds_dir)
    # Suppose 'training_stats' is a dictionary we either get from trainer or a global store
    training_stats = gather_training_stats_for_rf()  # define as needed
    return render_template("rf_training_summary_stats.html", training_stats=training_stats)

#########################
# Minimal Helper Functions (to read from global vars, for example)
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

    trainer = Trainer()
    training_stats = gather_training_stats_for_cnn()  # You implement or adapt
    training_history = get_cnn_history()              # So you can show epoch-by-epoch data
    # model_summary might also come from trainer
    model_summary = training_stats.get('model_summary', None)

    return render_template(
        "cnn_training_summary_stats.html",
        training_stats=training_stats,
        training_history=training_history,
        model_summary=model_summary
    )

def gather_training_stats_for_ensemble():
    global training_stats
    if training_stats is None:
        return {}
    # If you want, ensure training_stats['method'] is 'Ensemble Only' or 'All(...)'
    return training_stats

@ml_bp.route("/ensemble_training_summary")
def ensemble_training_summary():

    trainer = Trainer()
    training_stats = gather_training_stats_for_ensemble()
    return render_template("ensemble_training_summary_stats.html", training_stats=training_stats)

def gather_training_stats_for_all():
    global training_stats
    if training_stats is None:
        return {}
    # You could check training_stats['method'] == 'All (CNN + RF + Ensemble)'
    return training_stats

@ml_bp.route("/training_model_comparisons")
def training_model_comparisons():

    trainer = Trainer()
    training_stats = gather_all_model_stats()
    return render_template("training_model_comparisons_stats.html", training_stats=training_stats)

@ml_bp.route('/model_summary')
def model_summary():
  
    # We do not remove old code above, just replace with this new rendering
    return render_template('model_summary_hub.html')

@ml_bp.route('/api/models')
def get_available_models():
    """
    Get a list of available trained models for the current dictionary.
    Returns:
    {
        "models": [
            {"id": "cnn_model", "name": "CNN Model", "type": "cnn"},
            {"id": "rf_model", "name": "Random Forest", "type": "rf"}
        ],
        "dictionary_name": "Current Dictionary"
    }
    """
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401
    
    try:
        # Get current dictionary
        active_dict = Config.get_dictionary()
        dict_name = active_dict.get('name', 'Unknown')
        
        # Check models directory for available models
        models_dir = 'models'
        available_models = []
        
        if os.path.exists(models_dir):
            # Look for CNN models (*.h5 files)
            cnn_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            for file in cnn_files:
                if dict_name.replace(' ', '_') in file:
                    model_id = file.replace('.h5', '')
                    available_models.append({
                        'id': model_id,
                        'name': f"{dict_name} CNN Model",
                        'type': 'cnn'
                    })
            
            # Look for RF models (*.joblib files)
            rf_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            for file in rf_files:
                if dict_name.replace(' ', '_') in file:
                    model_id = file.replace('.joblib', '')
                    available_models.append({
                        'id': model_id,
                        'name': f"{dict_name} RF Model",
                        'type': 'rf'
                    })
            
            # Look for ensemble models (*.pkl files)
            ensemble_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            for file in ensemble_files:
                if dict_name.replace(' ', '_') in file:
                    model_id = file.replace('.pkl', '')
                    available_models.append({
                        'id': model_id,
                        'name': f"{dict_name} Ensemble Model",
                        'type': 'ensemble'
                    })
        
        # If no models found, add fallbacks
        if not available_models:
            available_models = [
                {
                    'id': f"{dict_name.replace(' ', '_')}_model",
                    'name': f"{dict_name} Default Model",
                    'type': 'cnn'
                }
            ]
        
        return jsonify({
            'models': available_models,
            'dictionary_name': dict_name
        })
    except Exception as e:
        logging.error(f"Error getting available models: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@ml_bp.route('/api/dictionary/sounds')
def get_dictionary_sounds():
    """
    Get a list of available sound classes for the current dictionary.
    Returns:
    {
        "sounds": ["cat", "dog", "bird"]
    }
    """
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401
    
    try:
        # Get current dictionary
        active_dict = Config.get_dictionary()
        sounds = active_dict.get('sounds', [])
        
        return jsonify({
            'sounds': sounds
        })
    except Exception as e:
        logging.error(f"Error getting dictionary sounds: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

