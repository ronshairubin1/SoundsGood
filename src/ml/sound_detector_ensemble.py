"""
Sound detector for ensemble models combining CNN and RF models.
"""

import os
import numpy as np
import logging
import pyaudio
import wave
import tempfile
import time
from threading import Thread, Lock
from .audio_processing import SoundProcessor
from backend.features.extractor import FeatureExtractor

class SoundDetectorEnsemble:
    """
    Sound detector that uses an ensemble of models (CNN + RF) to make predictions.
    This class manages audio input, processing, and prediction for ensemble models.
    """
    
    def __init__(self, models, sound_classes, preprocessing_params=None, use_ambient_noise=False):
        """
        Initialize the ensemble sound detector.
        
        Args:
            models (dict): Dictionary of models (cnn and rf)
            sound_classes (list): List of sound class names
            preprocessing_params (dict): Parameters for audio preprocessing
            use_ambient_noise (bool): Whether to measure ambient noise before listening
        """
        self.models = models
        self.sound_classes = sound_classes
        self.use_ambient_noise = use_ambient_noise
        self.ambient_noise_level = None
        
        # Default preprocessing parameters
        self.preprocessing_params = preprocessing_params or {
            "sample_rate": 22050,
            "n_mels": 128,
            "n_fft": 2048,
            "hop_length": 512,
            "sound_threshold": 0.01,
            "min_silence_duration": 0.5,
            "trim_silence": True,
            "normalize_audio": True
        }
        
        logging.info(f"Initializing SoundDetectorEnsemble with parameters: {self.preprocessing_params}")
        
        # Get parameters from preprocessing_params
        self.sample_rate = self.preprocessing_params.get("sample_rate", 22050)
        self.sound_threshold = self.preprocessing_params.get("sound_threshold", 0.01)
        self.min_silence_duration = self.preprocessing_params.get("min_silence_duration", 0.5)
        self.trim_silence = self.preprocessing_params.get("trim_silence", True)
        self.normalize_audio = self.preprocessing_params.get("normalize_audio", True)
        self.n_mels = self.preprocessing_params.get("n_mels", 128)
        self.n_fft = self.preprocessing_params.get("n_fft", 2048)
        self.hop_length = self.preprocessing_params.get("hop_length", 512)
        
        # Initialize sound processor for basic sound detection
        self.sound_processor = SoundProcessor(
            sample_rate=self.sample_rate,
            sound_threshold=self.sound_threshold
        )
        
        # Initialize the unified feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=self.sample_rate,
            use_cache=True
        )
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        self.callback = None
        
        # Audio buffer for storing sound data
        self.buffer = []
        self.buffer_size = self.sample_rate * 10  # 10 seconds max
        self.buffer_lock = Lock()
        
        # Sound detection parameters
        self.is_sound_detected = False
        self.sound_start_time = 0
        
        # Predictions storage
        self.predictions = []
        
        logging.info(f"SoundDetectorEnsemble initialized with {len(sound_classes)} classes and ambient_noise={use_ambient_noise}")
    
    def start_listening(self, callback=None):
        """
        Start audio stream and listen for sounds.
        
        Args:
            callback (function): Callback function for prediction results
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_listening:
            logging.warning("Already listening")
            return False
        
        self.is_listening = True
        self.buffer = []
        self.callback = callback  # Store callback for later use
        
        # Measure ambient noise if requested
        if self.use_ambient_noise:
            logging.info("Measuring ambient noise before starting ensemble detection...")
            # Create a temporary audio stream to measure ambient noise
            temp_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            # Collect 3 seconds of audio for ambient noise measurement
            ambient_frames = []
            for _ in range(0, int(self.sample_rate / 1024 * 3)):  # 3 seconds of frames
                try:
                    data = temp_stream.read(1024)
                    ambient_frames.append(np.frombuffer(data, dtype=np.float32))
                except Exception as e:
                    logging.error(f"Error reading ambient noise: {e}")
            
            # Close temporary stream
            temp_stream.stop_stream()
            temp_stream.close()
            
            # Process ambient noise frames
            if ambient_frames:
                ambient_audio = np.concatenate(ambient_frames)
                # Calculate RMS of ambient noise
                self.ambient_noise_level = np.sqrt(np.mean(ambient_audio**2))
                logging.info(f"Measured ambient noise level: {self.ambient_noise_level:.6f}")
                
                # Update sound threshold based on ambient noise
                if self.ambient_noise_level > 0:
                    # Use 2x ambient noise as threshold for sound detection (reduced from 3x) 
                    self.sound_threshold = max(self.sound_threshold, self.ambient_noise_level * 2.0)
                    logging.info(f"Updated sound threshold to {self.sound_threshold:.6f}")
                    
                    # Update the sound processor with the new threshold
                    self.sound_processor.sound_threshold = self.sound_threshold
            else:
                logging.warning("Failed to collect ambient noise samples")
        
        # Open audio stream for main listening
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        
        logging.info(f"Started ensemble listening with sample rate {self.sample_rate}")
        return True
    
    def stop_listening(self):
        """Stop audio stream."""
        if not self.is_listening:
            logging.warning("Not listening")
            return
        
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        with self.buffer_lock:
            self.buffer = []
        
        logging.info("Stopped listening")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Process incoming audio data.
        
        Args:
            in_data: Audio data from PyAudio
            frame_count: Number of frames
            time_info: Time information
            status: Status flag
        
        Returns:
            tuple: (None, paContinue)
        """
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
                audio_to_process = np.array(self.buffer[-int(self.sample_rate * (sound_duration + 0.5)):])
                Thread(target=self.process_audio, args=(audio_to_process,)).start()
        
        return None, pyaudio.paContinue
    
    def process_audio(self, audio_data):
        """
        Process audio data and make predictions.
        
        Args:
            audio_data: Audio data to process
            
        Returns:
            dict: Prediction results
        """
        try:
            # Process audio data
            logging.info(f"Processing audio data of shape {audio_data.shape}")
            
            # Extract features using the unified feature extractor
            # This handles both CNN and RF features in one call
            all_features = self.feature_extractor.extract_features(audio_data, is_file=False)
            
            # Get CNN features
            cnn_features = self.feature_extractor.extract_features_for_model(all_features, model_type='cnn')
            # Get RF features
            rf_features = self.feature_extractor.extract_features_for_model(all_features, model_type='rf')
            
            features = {
                'cnn': cnn_features,
                'rf': rf_features
            }
            
            if features is None or 'cnn' not in features or 'rf' not in features:
                logging.error("Failed to extract features")
                return {'class': 'error', 'confidence': 0}
            
            # Get CNN predictions
            cnn_model = self.models.get('cnn')
            if cnn_model and features['cnn'] is not None:
                # Add batch dimension if needed
                cnn_features = features['cnn']
                if len(cnn_features.shape) == 3:  # If missing batch dimension
                    cnn_features = np.expand_dims(cnn_features, axis=0)
                cnn_predictions = cnn_model.predict(cnn_features, verbose=0)[0]
            else:
                cnn_predictions = None
            
            # Get RF predictions
            rf_model = self.models.get('rf')
            if rf_model and features['rf'] is not None:
                rf_probabilities = rf_model.predict_proba([features['rf']])[0]
            else:
                rf_probabilities = None
            
            # Combine predictions
            if cnn_predictions is not None and rf_probabilities is not None:
                # Equal weighting for CNN and RF
                combined_probabilities = (cnn_predictions + rf_probabilities) / 2
            elif cnn_predictions is not None:
                combined_probabilities = cnn_predictions
            elif rf_probabilities is not None:
                combined_probabilities = rf_probabilities
            else:
                logging.error("Both models failed to produce predictions")
                return {'class': 'error', 'confidence': 0}
            
            # Get top prediction
            top_idx = np.argmax(combined_probabilities)
            top_class = self.sound_classes[top_idx]
            top_confidence = float(combined_probabilities[top_idx])
            
            # Create prediction result
            result = {
                'class': top_class,
                'confidence': top_confidence,
                'probabilities': {
                    self.sound_classes[i]: float(p) 
                    for i, p in enumerate(combined_probabilities)
                }
            }
            
            # Add to predictions list
            self.predictions.append(result)
            if len(self.predictions) > 5:
                self.predictions = self.predictions[-5:]
            
            logging.info(f"Ensemble prediction: {top_class} with confidence {top_confidence:.4f}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing audio: {e}", exc_info=True)
            return {'class': 'error', 'confidence': 0}
    
    def get_latest_prediction(self):
        """
        Get the latest prediction.
        
        Returns:
            dict: Latest prediction or None
        """
        if not self.predictions:
            return None
        return self.predictions[-1]

    def get_current_state(self):
        """
        Get the current state of the detector.
        
        Returns:
            dict: Current state information
        """
        return {
            'is_listening': self.is_listening,
            'is_sound_detected': self.is_sound_detected,
            'latest_prediction': self.get_latest_prediction()
        } 