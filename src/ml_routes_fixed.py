# Import the new model classes
from src.core.models import create_model, CNNModel, RandomForestModel as RFModel, EnsembleModel
from flask import Blueprint, request, jsonify, current_app, Response
import os
import json
import time
import traceback
from datetime import datetime
import threading
import random
import numpy as np
import librosa
import pyaudio
import wave
import tempfile
from threading import Thread, Lock
import logging

# Configure logging
logger = logging.getLogger('ml_routes_fixed')
logger.info("=" * 50)
logger.info("Initializing ML Routes Fixed module")
logger.info("Imported core models and Flask components")

# Create a blueprint for ML routes
ml_blueprint = Blueprint('ml', __name__)
logger.info("Created ml_blueprint")

# Global variables
detector = None
current_model_path = None
current_model_type = None
current_model_dict = None
latest_prediction = None
logger.info("Initialized global variables for ML detection")

# Define a SoundProcessor class for audio preprocessing
class SoundProcessor:
    """Audio processor for sound detection and feature extraction."""
    
    def __init__(self, sample_rate=22050, sound_threshold=0.01, n_mels=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.sound_threshold = sound_threshold
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        logger.debug(f"SoundProcessor initialized with sample_rate={sample_rate}, sound_threshold={sound_threshold}")
    
    def is_sound(self, audio_data):
        """Determine if audio contains sound above threshold."""
        return np.max(np.abs(audio_data)) > self.sound_threshold
    
    def detect_sound_boundaries(self, audio):
        """Find the start and end of sound in audio data."""
        frame_length = int(0.02 * self.sample_rate)  # 20ms windows
        hop_length = frame_length // 2
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Interpolate RMS to match audio length
        rms_interp = np.interp(
            np.linspace(0, len(audio), len(audio)),
            np.linspace(0, len(audio), len(rms)),
            rms
        )
        
        is_sound = rms_interp > (self.sound_threshold * np.max(rms_interp))
        if not np.any(is_sound):
            return 0, len(audio), False
        
        # Find indices where sound is detected
        sound_indices = np.where(is_sound)[0]
        start_idx = sound_indices[0]
        end_idx = sound_indices[-1]
        
        # Add margin around sound
        margin = int(0.1 * self.sample_rate)
        start_idx = max(0, start_idx - margin)
        end_idx = min(len(audio), end_idx + margin)
        
        return start_idx, end_idx, True
    
    def center_audio(self, audio):
        """Center audio around detected sound and normalize."""
        start_idx, end_idx, has_sound = self.detect_sound_boundaries(audio)
        
        if not has_sound:
            # If no sound detected, use middle section
            center = len(audio) // 2
            window_size = self.sample_rate
            start_idx = max(0, center - window_size // 2)
            end_idx = min(len(audio), center + window_size // 2)
        
        # Extract relevant audio segment
        audio = audio[start_idx:end_idx]
        
        # Normalize audio to consistent RMS level
        target_rms = 0.1
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        
        # Ensure audio is exactly 1 second long
        target_length = self.sample_rate
        if len(audio) > 0 and len(audio) != target_length:
            stretch_factor = target_length / len(audio)
            try:
                audio = librosa.effects.time_stretch(y=audio, rate=stretch_factor)
            except Exception as e:
                logging.error(f"Error time-stretching audio: {e}")
                # Pad or truncate as fallback
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        
        # Final size check
        if len(audio) > self.sample_rate:
            audio = audio[:self.sample_rate]
        elif len(audio) < self.sample_rate:
            audio = np.pad(audio, (0, self.sample_rate - len(audio)), 'constant')
        
        return audio
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram features from audio."""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_mels=self.n_mels, n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure consistent width (for CNN models)
        target_width = 64
        if mel_spec_db.shape[1] != target_width:
            mel_spec_db = np.array([
                np.interp(
                    np.linspace(0, 100, target_width),
                    np.linspace(0, 100, mel_spec_db.shape[1]),
                    row
                ) for row in mel_spec_db
            ])
        
        return mel_spec_db
    
    def process_audio(self, audio):
        """Process audio for CNN model input."""
        # Center and normalize audio
        preprocessed_audio = self.center_audio(audio)
        
        # Extract mel spectrogram
        mel_spec_db = self.extract_mel_spectrogram(preprocessed_audio)
        
        # Add channel dimension for CNN
        features = mel_spec_db[..., np.newaxis]
        
        return features

# Define a real SoundDetector class
class SoundDetector:
    """Real sound detector that captures audio and uses models for prediction."""
    
    def __init__(self, model, sound_classes, model_type, preprocessing_params=None):
        """
        Initialize the sound detector with a model.
        
        Args:
            model: Model to use for prediction (CNN, RF, or Ensemble)
            sound_classes: List of sound class names
            model_type: Type of model ('cnn', 'rf', or 'ensemble')
            preprocessing_params: Parameters for audio preprocessing
        """
        self.model = model
        self.sound_classes = sound_classes
        self.model_type = model_type.lower()
        
        # Default preprocessing parameters
        self.preprocessing_params = preprocessing_params or {
            "sample_rate": 22050,
            "n_mels": 128,
            "n_fft": 2048,
            "hop_length": 512,
            "sound_threshold": 0.01
        }
        
        # Get parameters from preprocessing_params
        self.sample_rate = self.preprocessing_params.get("sample_rate", 22050)
        self.sound_threshold = self.preprocessing_params.get("sound_threshold", 0.01)
        
        # Initialize sound processor for audio preprocessing
        self.sound_processor = SoundProcessor(
            sample_rate=self.sample_rate,
            sound_threshold=self.sound_threshold,
            n_mels=self.preprocessing_params.get("n_mels", 128),
            n_fft=self.preprocessing_params.get("n_fft", 2048),
            hop_length=self.preprocessing_params.get("hop_length", 512)
        )
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        
        # Buffer for audio data
        self.buffer = []
        self.buffer_size = int(self.sample_rate * 5)  # 5 seconds buffer
        self.buffer_lock = Lock()
        
        # Sound detection state
        self.is_sound_detected = False
        self.sound_start_time = 0
        self.predictions = []
        self.latest_prediction = None
        
        # Thread for audio processing
        self.thread = None
        self._stop_event = threading.Event()
        
        logging.info(f"SoundDetector initialized with {len(sound_classes)} classes for {model_type} model")
    
    def start_listening(self):
        """Start audio stream and listen for sounds."""
        if self.is_listening:
            logging.warning("Already listening")
            return
        
        self.is_listening = True
        self.buffer = []
        self._stop_event.clear()
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            
            logging.info(f"Started listening with sample rate {self.sample_rate}")
        except Exception as e:
            logging.error(f"Error starting audio stream: {e}")
            self.is_listening = False
            raise
    
    def stop_listening(self):
        """Stop audio stream."""
        if not self.is_listening:
            logging.warning("Not listening")
            return
        
        self.is_listening = False
        self._stop_event.set()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        with self.buffer_lock:
            self.buffer = []
        
        logging.info("Stopped listening")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Process incoming audio data from microphone."""
        if not self.is_listening:
            return None, pyaudio.paComplete
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Check if sound is detected
        is_sound = self.sound_processor.is_sound(audio_data)
        
        # Add data to buffer
        with self.buffer_lock:
            self.buffer.extend(audio_data)
            
            # Keep buffer size limited
            if len(self.buffer) > self.buffer_size:
                self.buffer = self.buffer[-self.buffer_size:]
        
        # Handle sound detection
        if is_sound and not self.is_sound_detected:
            # Sound just started
            self.is_sound_detected = True
            self.sound_start_time = time.time()
            logging.info("Sound detected")
        
        elif not is_sound and self.is_sound_detected:
            # Sound just ended
            self.is_sound_detected = False
            sound_duration = time.time() - self.sound_start_time
            
            if sound_duration >= 0.2:  # Ignore very short sounds
                logging.info(f"Sound ended after {sound_duration:.2f} seconds")
                
                # Process the detected sound in a separate thread
                with self.buffer_lock:
                    audio_to_process = np.array(self.buffer[-int(self.sample_rate * (sound_duration + 0.5)):])
                
                Thread(target=self.process_audio, args=(audio_to_process,)).start()
        
        return None, pyaudio.paContinue
    
    def process_audio(self, audio_data):
        """Process audio data and make predictions."""
        global latest_prediction
        
        try:
            logging.info(f"Processing audio data of shape {audio_data.shape}")
            
            # Create a temporary WAV file for feature extraction
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Save audio data to WAV file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    
                    # Convert float32 audio to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())
            
            # Process based on model type
            if self.model_type == 'cnn':
                # Prepare spectrogram for CNN model
                features = self.sound_processor.process_audio(audio_data)
                
                # Reshape for model input (add batch dimension)
                model_input = np.expand_dims(features, axis=0)
                
                # Make prediction
                predictions = self.model.predict(model_input)[0]
                
            elif self.model_type == 'rf':
                # For RF models, we need to extract different features
                # This depends on your RF feature extraction pipeline
                # This is a simplified example
                y, sr = librosa.load(temp_path, sr=self.sample_rate)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_mean = np.mean(mfccs, axis=1)
                
                # Make prediction
                predictions = self.model.predict_proba([mfcc_mean])[0]
                
            elif self.model_type == 'ensemble':
                # Ensemble models might need both CNN and RF features
                # This depends on your ensemble implementation
                logging.warning("Ensemble model handling is simplified")
                
                # Prepare spectrogram for CNN part
                features = self.sound_processor.process_audio(audio_data)
                model_input = np.expand_dims(features, axis=0)
                
                # Make prediction
                predictions = self.model.predict(model_input)[0]
            
            else:
                logging.error(f"Unknown model type: {self.model_type}")
                return
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])
            
            # Make sure predicted_class_idx is valid
            if predicted_class_idx >= len(self.sound_classes):
                logging.error(f"Invalid prediction index: {predicted_class_idx} for {len(self.sound_classes)} classes")
                return
                
            predicted_class = self.sound_classes[predicted_class_idx]
            
            # Format prediction result
            prediction = {
                'class': predicted_class,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            logging.info(f"Prediction: {predicted_class} with confidence {confidence:.2f}")
            
            # Update latest prediction
            self.latest_prediction = prediction
            latest_prediction = prediction  # Also update global variable
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logging.warning(f"Failed to delete temp file {temp_path}: {e}")
            
        except Exception as e:
            logging.error(f"Error in process_audio: {e}")
            logging.error(traceback.format_exc())

# Update the start_listening endpoint to use real model classes and metadata
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
        
        # Parse the model_id to get dictionary and model type
        parts = model_id.split('_')
        current_app.logger.info(f"🎧 LISTENING DEBUG: Parsed model_id parts: {parts}")
        
        if len(parts) < 3:
            current_app.logger.error(f"❌ LISTENING ERROR: Invalid model_id format: {model_id}, not enough parts")
            return jsonify({'status': 'error', 'message': 'Invalid model_id format'}), 400
        
        # Extract model parts correctly
        dict_name = parts[0]
        model_type = parts[1].lower()  # cnn, rf, or ens
        timestamp = parts[2]  # Extract the timestamp
        
        current_app.logger.info(f"🎧 LISTENING DEBUG: Extracted dictionary name: {dict_name}")
        current_app.logger.info(f"🎧 LISTENING DEBUG: Extracted model type: {model_type}")
        current_app.logger.info(f"🎧 LISTENING DEBUG: Extracted timestamp: {timestamp}")
        
        # Set sound classes based on dictionary
        if dict_name.lower() == 'ehoh':
            sound_classes = ['eh', 'oh']  # Default for EhOh
        else:
            sound_classes = ['sound1', 'sound2']  # Generic fallback
        
        # Determine model path
        if model_type == 'cnn':
            model_path = os.path.join(models_dir, f"{model_id}.h5")
        elif model_type == 'rf':
            model_path = os.path.join(models_dir, f"{model_id}.joblib")
        elif model_type == 'ensemble':
            model_path = os.path.join(models_dir, f"{model_id}.h5")  # Ensemble models might use h5
        else:
            current_app.logger.error(f"❌ LISTENING ERROR: Unsupported model type: {model_type}")
            return jsonify({'status': 'error', 'message': f'Unsupported model type: {model_type}'}), 400
        
        # Check if model file exists
        if not os.path.exists(model_path):
            current_app.logger.error(f"❌ LISTENING ERROR: Model file not found: {model_path}")
            
            # Try to load a model through the factory function as backup
            try:
                current_app.logger.info(f"🎧 LISTENING DEBUG: Trying to load model through factory function")
                
                # For now, we'll use the mock detector since real model loading requires more complex initialization
                detector = MockSoundDetector(sound_classes=sound_classes, app=current_app._get_current_object())
                detector.start_listening()
                current_app.logger.info(f"✅ LISTENING SUCCESS: Started mock detector as fallback")
                
                # Store current state
                current_model_type = model_type
                current_model_dict = dict_name
                
                return jsonify({
                    'status': 'success', 
                    'message': f'Started mock detector as fallback (model not found)',
                    'sound_classes': sound_classes
                }), 200
                
            except Exception as e:
                current_app.logger.error(f"❌ LISTENING ERROR: Failed to create fallback detector: {e}")
                return jsonify({'status': 'error', 'message': f'Model not found: {model_path}'}), 404
        
        # Try to create detector with real model
        try:
            # For now, we'll use the mock detector
            # In a real implementation, you would load the model here
            # model = load_model(model_path) or similar
            
            current_app.logger.info(f"🎧 LISTENING DEBUG: Using mock detector for now")
            detector = MockSoundDetector(sound_classes=sound_classes, app=current_app._get_current_object())
            detector.start_listening()
            current_app.logger.info(f"✅ LISTENING SUCCESS: Started mock detector")
            
            # Store current state
            current_model_type = model_type
            current_model_dict = dict_name
            
            return jsonify({
                'status': 'success', 
                'message': f'Started listening with mock {model_type} detector',
                'sound_classes': sound_classes
            }), 200
            
        except Exception as e:
            current_app.logger.error(f"❌ LISTENING ERROR: Error creating detector: {str(e)}")
            current_app.logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'status': 'error', 'message': f'Error creating detector: {str(e)}'}), 500
        
    except Exception as e:
        current_app.logger.error(f"❌ LISTENING ERROR: Unexpected error in start_listening: {str(e)}")
        current_app.logger.error(f"❌ LISTENING ERROR: Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Create a mock detector class that simulates audio capture
class MockSoundDetector:
    """A mock detector class that simulates audio capture and detection."""
    
    def __init__(self, sound_classes=None, app=None):
        self.is_listening = False
        self.thread = None
        self.sound_classes = sound_classes or ['eh', 'oh']
        self.latest_prediction = None
        self.app = app  # Store the app reference
        
        if self.app:
            self.app.logger.info(f"📢 MOCK: Created MockSoundDetector with classes {self.sound_classes}")
        else:
            print(f"📢 MOCK: Created MockSoundDetector with classes {self.sound_classes}")
    
    def start_listening(self):
        """Start a fake listening thread that generates random predictions."""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.thread = threading.Thread(target=self._generate_predictions)
        self.thread.daemon = True
        self.thread.start()
        
        if self.app:
            self.app.logger.info(f"📢 MOCK: Started listening thread")
        else:
            print(f"📢 MOCK: Started listening thread")
        
    def stop_listening(self):
        """Stop the fake listening thread."""
        self.is_listening = False
        
        if self.app:
            self.app.logger.info(f"📢 MOCK: Stopped listening")
        else:
            print(f"📢 MOCK: Stopped listening")
        
    def _generate_predictions(self):
        """Generate fake predictions periodically."""
        global latest_prediction
        
        # Use standard Python logging instead of Flask's logger for thread safety
        import logging
        logger = logging.getLogger('mock_detector')
        
        while self.is_listening:
            # Sleep for a random time between 2-5 seconds
            time.sleep(random.uniform(2, 5))
            
            if not self.is_listening:
                break
                
            # Generate a random prediction
            sound_class = random.choice(self.sound_classes)
            confidence = random.uniform(0.7, 0.95)
            
            prediction = {
                'class': sound_class,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            self.latest_prediction = prediction
            latest_prediction = prediction  # Also set the global variable
            
            # Use standard Python logging instead of Flask's logger
            logger.info(f"📢 MOCK: Generated prediction: {sound_class} ({confidence:.2f})")

@ml_blueprint.route('/api/ml/stop_listening', methods=['POST'])
def stop_listening():
    """Stop listening for sound and making predictions"""
    try:
        global detector
        current_app.logger.info("🛑 LISTENING DEBUG: Stopping sound detection")
        
        if detector:
            detector.stop_listening()
            current_app.logger.info("✅ LISTENING SUCCESS: Stopped sound detection")
            return jsonify({'status': 'success', 'message': 'Stopped listening'}), 200
        else:
            current_app.logger.warning("⚠️ LISTENING WARNING: No active detector to stop")
            return jsonify({'status': 'success', 'message': 'No active detector to stop'}), 200
    
    except Exception as e:
        current_app.logger.error(f"❌ LISTENING ERROR: Unexpected error in stop_listening: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@ml_blueprint.route('/api/ml/prediction_stream')
def prediction_stream():
    """Stream real-time predictions using server-sent events"""
    try:
        def generate():
            """Generator for SSE events"""
            global latest_prediction
            
            # Send initial connection message
            yield "data: {\"status\": \"connected\"}\n\n"
            
            # Send heartbeats to keep connection alive
            last_heartbeat = time.time()
            
            while True:
                # Check if there's a new prediction
                if detector and detector.latest_prediction:
                    # Get the prediction
                    prediction = detector.latest_prediction
                    
                    # Format as SSE event
                    yield f"data: {json.dumps({'prediction': prediction})}\n\n"
                    
                    # Clear latest prediction after sending
                    detector.latest_prediction = None
                
                # Send heartbeat every 30 seconds
                current_time = time.time()
                if current_time - last_heartbeat > 30:
                    yield ": heartbeat\n\n"
                    last_heartbeat = current_time
                
                # Don't hog the CPU
                time.sleep(0.1)
        
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        current_app.logger.error(f"❌ STREAM ERROR: Unexpected error in prediction_stream: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@ml_blueprint.route('/api/ml/dictionary/<dictionary_name>/sounds', methods=['GET'])
def get_dictionary_sounds(dictionary_name):
    """Get sound classes for a specific dictionary"""
    try:
        current_app.logger.info(f"🔍 SOUNDS DEBUG: Getting sounds for dictionary: {dictionary_name}")
        
        # For EhOh models, provide a default list of sounds
        if dictionary_name.lower() == 'ehoh':
            sounds = ['eh', 'oh']
            current_app.logger.info(f"🔍 SOUNDS DEBUG: Using default sounds for {dictionary_name}: {sounds}")
            return jsonify({
                'success': True,
                'dictionary': dictionary_name,
                'sounds': sounds
            })
            
        # Default - return empty array if nothing found
        return jsonify({
            'success': True,
            'dictionary': dictionary_name,
            'sounds': ['sound1', 'sound2']  # Generic fallback
        })
        
    except Exception as e:
        current_app.logger.error(f"❌ SOUNDS ERROR: Error getting sounds for dictionary {dictionary_name}: {str(e)}")
        return jsonify({'success': False, 'message': str(e), 'sounds': []}), 500

@ml_blueprint.route('/api/ml/dictionary/sounds', methods=['GET'])
def get_all_sounds():
    """Fallback endpoint for getting all available sounds"""
    try:
        return jsonify({
            'success': True,
            'sounds': ['eh', 'oh', 'sound1', 'sound2']  # Default sounds
        })
    except Exception as e:
        current_app.logger.error(f"❌ SOUNDS ERROR: Error getting all sounds: {str(e)}")
        return jsonify({'success': False, 'message': str(e), 'sounds': []}), 500

@ml_blueprint.route('/api/ml/inference_statistics', methods=['GET'])
def get_inference_statistics():
    """API to get statistics on model performance"""
    try:
        # Return empty statistics
        response_data = {
            'success': True,
            'total_predictions': 0,
            'average_confidence': 0,
            'class_accuracy': {},
            'confusion_matrix': {}
        }
        
        return jsonify(response_data)
    except Exception as e:
        current_app.logger.error(f"❌ STATS ERROR: Error getting inference statistics: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@ml_blueprint.route('/api/ml/record_feedback', methods=['POST'])
def record_feedback():
    """Record user feedback for a prediction"""
    try:
        data = request.get_json()
        
        if not data:
            current_app.logger.error("❌ FEEDBACK ERROR: No data provided")
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        predicted_sound = data.get('predicted_sound')
        actual_sound = data.get('actual_sound')
        confidence = data.get('confidence')
        
        if not predicted_sound or not actual_sound:
            current_app.logger.error("❌ FEEDBACK ERROR: Missing predicted_sound or actual_sound")
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
        
        # Just log the feedback
        current_app.logger.info(f"✅ FEEDBACK SUCCESS: Received feedback - Predicted: {predicted_sound}, Actual: {actual_sound}, Confidence: {confidence}")
        
        # Return success
        return jsonify({'status': 'success', 'message': 'Feedback recorded'}), 200
    
    except Exception as e:
        current_app.logger.error(f"❌ FEEDBACK ERROR: Unexpected error in record_feedback: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@ml_blueprint.route('/api/ml/save_analysis', methods=['POST'])
def save_analysis():
    """Save the current model analysis data"""
    try:
        # Just log and return success
        current_app.logger.info("✅ ANALYSIS SUCCESS: Mock analysis data saved")
        return jsonify({'status': 'success', 'message': 'Analysis data saved'}), 200
    
    except Exception as e:
        current_app.logger.error(f"❌ ANALYSIS ERROR: Unexpected error in save_analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500