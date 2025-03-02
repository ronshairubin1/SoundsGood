# SoundClassifier_v08/src/ml/inference.py

import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import threading
import time
import logging
from scipy.io import wavfile
from flask import current_app
import os
from src.ml.model_paths import get_cnn_model_path
import pyaudio
from threading import Thread, Lock
import tempfile
import wave
from .sound_processor import SoundProcessor

# Import shared constants and the SoundProcessor from audio_processing
from .constants import SAMPLE_RATE, AUDIO_DURATION, AUDIO_LENGTH, TEST_DURATION, TEST_FS, INT16_MAX
from .audio_processing import SoundProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)

# Print available audio devices for reference
print(sd.query_devices())

class SoundDetector:
    """
    Sound detector that listens for sounds and makes predictions using CNN models.
    This class manages audio input, processing, and prediction.
    """
    
    def __init__(self, model, sound_classes, input_shape=None, preprocessing_params=None):
        """
        Initialize the sound detector.
        
        Args:
            model: CNN model to use for prediction
            sound_classes (list): List of sound class names
            input_shape (tuple): Shape expected by the model (height, width, channels)
            preprocessing_params (dict): Parameters for audio preprocessing
        """
        self.model = model
        self.sound_classes = sound_classes
        self.input_shape = input_shape
        
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
        
        logging.info(f"Initializing SoundDetector with parameters: {self.preprocessing_params}")
        
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
            normalize_audio=self.normalize_audio,
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
        
        logging.info(f"SoundDetector initialized with {len(sound_classes)} classes")
    
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
    
    def reshape_features(self, features):
        """
        Reshape features to match the expected input shape of the model.
        
        Args:
            features: Features to reshape
            
        Returns:
            reshaped_features: Reshaped features
        """
        if self.input_shape:
            # Handle reshaping based on input_shape
            if len(features.shape) == 2:  # (height, width)
                if len(self.input_shape) == 3:  # (height, width, channels)
                    # Add channel dimension
                    features = np.expand_dims(features, axis=-1)
            
            # Add batch dimension if not present
            if len(features.shape) == len(self.input_shape):
                features = np.expand_dims(features, axis=0)
            
            # Check if shapes are compatible
            for i in range(len(self.input_shape)):
                if self.input_shape[i] is not None and features.shape[i+1] != self.input_shape[i]:
                    logging.warning(f"Feature shape mismatch: got {features.shape}, expected ({None}, {self.input_shape})")
                    # Try to resize
                    if i == 0:  # Height
                        features = np.resize(features, (features.shape[0], self.input_shape[0], features.shape[2], features.shape[3]))
                    elif i == 1:  # Width
                        features = np.resize(features, (features.shape[0], features.shape[1], self.input_shape[1], features.shape[3]))
                    elif i == 2:  # Channels
                        features = np.resize(features, (features.shape[0], features.shape[1], features.shape[2], self.input_shape[2]))
        
        return features
    
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
                # Extract features using sound processor
                features = self.sound_processor.extract_features(temp_path)
                
                if features is None:
                    logging.error("Failed to extract features")
                    return {'class': 'error', 'confidence': 0}
                
                # Reshape features to match model input shape
                features = self.reshape_features(features)
                
                # Make prediction
                predictions = self.model.predict(features, verbose=0)
                
                # Get top prediction
                top_idx = np.argmax(predictions[0])
                top_class = self.sound_classes[top_idx]
                top_confidence = float(predictions[0][top_idx])
                
                # Create prediction result
                result = {
                    'class': top_class,
                    'confidence': top_confidence,
                    'probabilities': {
                        self.sound_classes[i]: float(p) 
                        for i, p in enumerate(predictions[0])
                    }
                }
                
                # Add to predictions list
                self.predictions.append(result)
                if len(self.predictions) > 5:
                    self.predictions = self.predictions[-5:]
                
                logging.info(f"Prediction: {top_class} with confidence {top_confidence:.4f}")
                
                return result
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
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


def record_audio(duration=AUDIO_DURATION):
    """
    Records from the microphone for `duration` seconds 
    and returns a 1D NumPy array of float32 samples.
    """
    logging.info(f"Recording for {duration:.2f} second(s)...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,channels=1,blocking=True)
    sd.wait()
    return recording.flatten().astype(np.float32)


def predict_sound(model, input_source, class_names, use_microphone=False):
    """
    Predict sound from either a file path or microphone input (CNN approach).
    If `use_microphone` is True, it records from mic for AUDIO_DURATION.
    Otherwise, `input_source` is interpreted as a filepath (or raw data).
    """
    try:
        sp = SoundProcessor(sample_rate=SAMPLE_RATE)
        if use_microphone:
            audio = record_audio(AUDIO_DURATION)
        else:
            # Load from a file path, resampling to SAMPLE_RATE
            audio, _ = librosa.load(input_source, sr=SAMPLE_RATE)

        # Process
        features = sp.process_audio(audio)
        if features is None:
            return None, 0.0

        features = np.expand_dims(features, axis=0)
        predictions = model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_label = class_names[predicted_class]
        return predicted_label, confidence

    except Exception as e:
        logging.error(f"Error in predict_sound: {str(e)}", exc_info=True)
        return None, 0.0


def run_inference_loop(model, class_names):
    """
    Simple interactive loop for testing predictions on command line.
    Type 'mic' to record from microphone, 'file <path>' for a WAV file, or 'quit' to exit.
    """
    print("\nSound Prediction Mode")
    print("--------------------")
    print("Commands:")
    print("  'mic' - Record from microphone")
    print("  'file <path>' - Predict from audio file")
    print("  'quit' - Exit the program")

    while True:
        try:
            command = input("\nEnter command >>> ").strip().lower()
            if command == 'quit':
                print("Exiting...")
                break
            elif command == 'mic':
                label, conf = predict_sound(model, None, class_names, use_microphone=True)
                if label:
                    print(f"Predicted: '{label}' (confidence: {conf:.3f})")
            elif command.startswith('file '):
                file_path = command[5:].strip()
                label, conf = predict_sound(model, file_path, class_names, use_microphone=False)
                if label:
                    print(f"Predicted: '{label}' (confidence: {conf:.3f})")
            else:
                print("Unknown command. Use 'mic', 'file <path>', or 'quit'")
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def test_microphone():
    """
    Record for TEST_DURATION seconds at TEST_FS sample rate, then play it back and optionally save.
    """
    print(f"Recording for {TEST_DURATION} second(s) at {TEST_FS} Hz...")
    test_recording = sd.rec(int(TEST_DURATION * TEST_FS), samplerate=TEST_FS, channels=1)
    sd.wait()
    print("Recording complete. Playing back...")

    sd.play(test_recording, samplerate=TEST_FS)
    sd.wait()

    # Save the recording to a WAV file
    test_recording_int = np.int16(test_recording * INT16_MAX)
    wavfile.write("test_recording.wav", TEST_FS, test_recording_int)
    print("Playback complete.")


if __name__ == "__main__":
    # Old logic
    try:
        model = tf.keras.models.load_model("models/audio_classifier.h5")
        
        dictionary_name = "Two_words"  # or read from config
        version = "v1"
        new_cnn_path = get_cnn_model_path("models", dictionary_name, version)

        if os.path.exists(new_cnn_path):
            print(f"Loading new CNN path: {new_cnn_path}")
            model = tf.keras.models.load_model(new_cnn_path)
        else:
            print("New CNN path not found, falling back to 'audio_classifier.h5'")
            model = tf.keras.models.load_model("audio_classifier.h5")

        class_names = np.load("models/class_names.npy", allow_pickle=True)
        print(f"Loaded class names: {class_names}")

        run_inference_loop(model, class_names)
        test_microphone()

    except Exception as e:
        print(f"Error loading model or class names: {str(e)}")
