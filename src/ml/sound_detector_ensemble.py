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
from .audio_feature_extractor import AudioFeatureExtractor

class SoundDetectorEnsemble:
    """
    Sound detector that uses an ensemble of models (CNN + RF) to make predictions.
    This class manages audio input, processing, and prediction for ensemble models.
    """
    
    def __init__(self, models, sound_classes, preprocessing_params=None):
        """
        Initialize the ensemble sound detector.
        
        Args:
            models (dict): Dictionary of models (cnn and rf)
            sound_classes (list): List of sound class names
            preprocessing_params (dict): Parameters for audio preprocessing
        """
        self.models = models
        self.sound_classes = sound_classes
        
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
        
        # Initialize sound processor
        self.sound_processor = SoundProcessor(
            sample_rate=self.sample_rate,
            sound_threshold=self.sound_threshold,
            min_silence_duration=self.min_silence_duration,
            trim_silence=self.trim_silence,
            normalize_audio=self.normalize_audio
        )
        
        # Initialize audio feature extractor
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
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
        
        logging.info(f"SoundDetectorEnsemble initialized with {len(sound_classes)} classes")
    
    def start_listening(self):
        """Start audio stream and listen for sounds."""
        if self.is_listening:
            logging.warning("Already listening")
            return
        
        self.is_listening = True
        self.buffer = []
        
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
            
            # Create a temporary WAV file for feature extraction
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Save audio data to WAV file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(4)  # 32-bit float
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data.tobytes())
            
            try:
                # Extract features for RF model
                rf_features = self.feature_extractor.extract_features(temp_path)
                
                # Extract spectrograms for CNN model
                spectrogram = self.feature_extractor.extract_mel_spectrogram(temp_path)
                
                # Make predictions
                predictions = {}
                
                # RF model prediction
                if 'rf' in self.models and self.models['rf'] is not None:
                    rf_pred = self.models['rf'].predict_proba([rf_features])[0]
                    predictions['rf'] = {
                        'class': self.sound_classes[np.argmax(rf_pred)],
                        'confidence': np.max(rf_pred),
                        'probabilities': {self.sound_classes[i]: float(p) for i, p in enumerate(rf_pred)}
                    }
                
                # CNN model prediction
                if 'cnn' in self.models and self.models['cnn'] is not None:
                    # Reshape spectrogram to match model input shape
                    spectrogram_input = np.expand_dims(spectrogram, axis=0)
                    
                    # Add channel dimension if needed
                    if len(spectrogram_input.shape) == 3:
                        spectrogram_input = np.expand_dims(spectrogram_input, axis=-1)
                    
                    cnn_pred = self.models['cnn'].predict(spectrogram_input)[0]
                    predictions['cnn'] = {
                        'class': self.sound_classes[np.argmax(cnn_pred)],
                        'confidence': float(np.max(cnn_pred)),
                        'probabilities': {self.sound_classes[i]: float(p) for i, p in enumerate(cnn_pred)}
                    }
                
                # Ensemble prediction (weighted average)
                if 'rf' in predictions and 'cnn' in predictions:
                    # Get probabilities for all classes
                    ensemble_probs = {}
                    
                    # Default weights
                    rf_weight = 0.5
                    cnn_weight = 0.5
                    
                    for cls in self.sound_classes:
                        rf_prob = predictions['rf']['probabilities'].get(cls, 0)
                        cnn_prob = predictions['cnn']['probabilities'].get(cls, 0)
                        ensemble_probs[cls] = (rf_weight * rf_prob) + (cnn_weight * cnn_prob)
                    
                    # Get top class
                    top_class = max(ensemble_probs, key=ensemble_probs.get)
                    
                    predictions['ensemble'] = {
                        'class': top_class,
                        'confidence': ensemble_probs[top_class],
                        'probabilities': ensemble_probs
                    }
                
                # Use best available prediction
                if 'ensemble' in predictions:
                    final_prediction = predictions['ensemble']
                elif 'cnn' in predictions:
                    final_prediction = predictions['cnn']
                elif 'rf' in predictions:
                    final_prediction = predictions['rf']
                else:
                    final_prediction = {'class': 'unknown', 'confidence': 0, 'probabilities': {}}
                
                # Save the most recent prediction
                self.predictions.append(final_prediction)
                if len(self.predictions) > 5:
                    self.predictions = self.predictions[-5:]
                
                logging.info(f"Prediction: {final_prediction['class']} with confidence {final_prediction['confidence']:.2f}")
                
                return final_prediction
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except Exception as e:
            logging.error(f"Error processing audio: {e}", exc_info=True)
            return {'class': 'error', 'confidence': 0, 'probabilities': {}}
    
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