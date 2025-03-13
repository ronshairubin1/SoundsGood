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
from backend.src.ml.model_paths import get_cnn_model_path
import pyaudio
from threading import Thread, Lock
import tempfile
import wave
import traceback
import matplotlib.pyplot as plt

# Import shared constants and the SoundProcessor from audio_processing
from .constants import SAMPLE_RATE, AUDIO_DURATION, AUDIO_LENGTH, TEST_DURATION, TEST_FS, INT16_MAX
from .audio_processing import SoundProcessor
from backend.features.extractor import FeatureExtractor
from backend.config import Config

# Import our unified audio components
from backend.audio import AudioPreprocessor

# Set up logging
logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)  # Change from INFO to DEBUG to see more detailed logs

# Print available audio devices for reference
print(sd.query_devices())

class SoundDetector:
    """
    Sound detector that listens for sounds and makes predictions using CNN models.
    This class manages audio input, processing, and prediction.
    
    IMPORTANT: This class now uses ONLY the FeatureExtractor for feature extraction
    to ensure 100% consistency between training and inference.
    """
    
    def __init__(self, model, sound_classes, input_shape=None, preprocessing_params=None, use_ambient_noise=False):
        """
        Initialize the sound detector with the specified model.
        
        Args:
            model: Trained model for sound classification
            sound_classes: List of sound class names
            input_shape: Expected input shape for the model
            preprocessing_params: Parameters for audio preprocessing
            use_ambient_noise: Whether to use ambient noise for normalization
        """
        logging.info("========== INITIALIZING SOUND DETECTOR ==========")
        logging.info(f"Parameters: sound_classes={sound_classes[:5]}..., use_ambient_noise={use_ambient_noise}")
        
        try:
            self.model = model
            self.sound_classes = sound_classes
            self.input_shape = input_shape or (223, 64, 1)  # Default CNN shape
            self.preprocessing_params = preprocessing_params or {}
            self.use_ambient_noise = use_ambient_noise
            
            # Set default parameters
            self.sample_rate = self.preprocessing_params.get('sample_rate', SAMPLE_RATE)
            self.sound_threshold = self.preprocessing_params.get('sound_threshold', 0.01)
            self.ambient_noise_level = 0
            
            logging.info("Initializing audio processing components...")
            # Initialize the AudioPreprocessor for consistent preprocessing
            self.audio_preprocessor = AudioPreprocessor(
                sample_rate=self.sample_rate,
                sound_threshold=self.sound_threshold,
                min_silence_duration=0.3,
                target_duration=1.0
            )
            
            # Initialize the FeatureExtractor - THIS IS CRITICAL
            # This ensures that feature extraction is IDENTICAL to training
            self.feature_extractor = FeatureExtractor(
                sample_rate=self.sample_rate
            )
            logging.info("Audio processing components initialized successfully")
            
            # Audio setup
            self.format = pyaudio.paFloat32
            self.channels = 1
            self.chunk = int(self.sample_rate * 0.1)  # 100ms chunks
            
            logging.info(f"Initialized SoundDetector with sample_rate={self.sample_rate}, sound_threshold={self.sound_threshold}")
            
            # Initialize audio capture and processing
            logging.info("Initializing PyAudio...")
            self.p = pyaudio.PyAudio()
            self.stream = None
            self.is_listening = False
            self.thread = None
            self.lock = Lock()
            
            # Buffer to hold audio frames
            self.audio_buffer = []
            self.buffer_lock = Lock()
            
            # Utterance detection state
            self.is_utterance = False
            self.silence_counter = 0
            self.min_silence_frames = 5  # Number of silent frames to end utterance
            
            # Callback for real-time prediction
            self.callback = None
            self.latest_prediction = None
            
            # Statistics
            self.sounds_detected = 0
            self.last_sound_time = None
            
            # Load model if string path is provided instead of model object
            if isinstance(model, str):
                logging.info(f"Loading model from {model}")
                self.model = tf.keras.models.load_model(model)
                logging.info(f"Model loaded successfully")
                
            # Debug log model info
            try:
                logging.info(f"Model info: Input shape={self.model.input_shape}, Output shape={self.model.output_shape}")
            except:
                logging.info(f"Model info not available")
            
            # State flags
            self.has_detected_sound = False
            
            # Debug info
            if self.use_ambient_noise:
                logging.info(f"Ambient noise measurement enabled with initial threshold {self.sound_threshold}")
                
            logging.info("SoundDetector initialization complete")
        except Exception as e:
            logging.error(f"ERROR initializing SoundDetector: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise  # Re-raise to avoid silent failure
        
    def start_listening(self, callback=None):
        """Start audio stream and listen for sounds."""
        if self.is_listening:
            logging.warning("Already listening")
            return False
        
        self.is_listening = True
        self.audio_buffer = []
        self.callback = callback  # Store the callback if provided
        
        # Measure ambient noise if requested
        if self.use_ambient_noise:
            print("\n===== STARTING AMBIENT NOISE MEASUREMENT =====")
            logging.info("Measuring ambient noise before starting detection...")
            # Create a temporary audio stream to measure ambient noise
            temp_stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            # Collect 3 seconds of audio for ambient noise measurement
            ambient_frames = []
            print("Collecting ambient noise samples for 3 seconds...")
            for i in range(0, int(self.sample_rate / 1024 * 3)):  # 3 seconds of frames
                try:
                    data = temp_stream.read(1024)
                    ambient_frames.append(np.frombuffer(data, dtype=np.float32))
                    # Print progress every second
                    if i % int(self.sample_rate / 1024) == 0:
                        print(f"Ambient noise measurement progress: {i // int(self.sample_rate / 1024) + 1}/3 seconds")
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
                old_threshold = self.sound_threshold
                
                # Update sound threshold based on ambient noise
                if self.ambient_noise_level > 0:
                    # Use 2x ambient noise as threshold for sound detection (reduced from 3x)
                    self.sound_threshold = max(self.sound_threshold, self.ambient_noise_level * 2.0)
                
                print(f"\n===== AMBIENT NOISE MEASUREMENT COMPLETED =====")
                print(f"Ambient noise level: {self.ambient_noise_level:.6f}")
                print(f"Previous sound threshold: {old_threshold:.6f}")
                print(f"New sound threshold: {self.sound_threshold:.6f}")
                
                logging.info(f"Measured ambient noise level: {self.ambient_noise_level:.6f}")
                logging.info(f"Updated sound threshold from {old_threshold:.6f} to {self.sound_threshold:.6f}")
            else:
                print("\n===== AMBIENT NOISE MEASUREMENT FAILED =====")
                logging.warning("Failed to collect ambient noise samples")
        
        # Open audio stream for main listening
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.audio_callback
        )
        
        print("\n===== STARTING AUDIO DETECTION WITH THRESHOLD: {:.6f} =====".format(self.sound_threshold))
        logging.info(f"Started listening with sample rate {self.sample_rate} and threshold {self.sound_threshold:.6f}")
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
        
        with self.lock:
            self.audio_buffer = []
        
        logging.info("Stopped listening")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for the audio stream.
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Status flag
            
        Returns:
            tuple: (None, pyaudio.paContinue)
        """
        try:
            # Process audio data
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Check if this is sound or silence
            is_sound = self.sound_threshold < np.abs(audio_data).mean()
            
            # Handle utterance detection and buffering
            with self.lock:
                if is_sound and not self.is_utterance:
                    # Start of a new utterance
                    logging.debug("Starting to record new utterance")
                    self.is_utterance = True
                    self.audio_buffer = [audio_data]  # Start with this frame
                    self.silence_counter = 0
                    self.has_detected_sound = True
                    
                elif self.is_utterance:
                    # Continue recording utterance
                    self.audio_buffer.append(audio_data)
                    self.silence_counter += 1
                    
                    if is_sound:
                        # Reset silence counter if we hear sound
                        self.silence_counter = 0
                    else:
                        # Increment silence counter
                        self.silence_counter += 1
                    
                    # Check if utterance is complete (enough silence or max length)
                    if self.silence_counter >= self.min_silence_frames:
                        # Complete utterance - process it
                        logging.debug(f"Complete utterance detected with {len(self.audio_buffer)} frames, processing...")
                        self.process_complete_utterance()
                        
                        # Reset state
                        self.is_utterance = False
                        self.audio_buffer = []
                
            # Continue audio stream
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")
            traceback.print_exc()
            return (None, pyaudio.paContinue)
    
    def process_complete_utterance(self):
        """
        Process a complete utterance (sound segment) using the
        unified feature extractor, and makes a prediction.
        
        This method uses ONLY the unified FeatureExtractor to ensure
        features are extracted in the EXACT SAME WAY as during training.
        """
        try:
            if not self.audio_buffer:
                return
                
            # Concatenate audio frames into a single buffer
            audio_data = np.concatenate(self.audio_buffer)
            
            logging.info(f"Processing utterance with shape: {audio_data.shape}, min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean={np.mean(audio_data):.4f}")
            
            # Step 1: Preprocess audio with AudioPreprocessor
            # Use preprocess_recording for consistency with the rest of the codebase
            processed_segments = self.audio_preprocessor.preprocess_recording(audio_data)
            
            if not processed_segments:
                logging.warning("Failed to preprocess audio - skipping prediction")
                return
                
            # Use the first processed segment
            processed_audio = processed_segments[0]
            
            # Step 2: Extract features with the unified FeatureExtractor
            # This uses EXACTLY the same processing as during training
            all_features = self.feature_extractor.extract_features(processed_audio, is_file=False)
            features = self.feature_extractor.extract_features_for_model(all_features, model_type='cnn')
            
            if features is None:
                logging.error("Failed to extract features from audio data")
                return
                
            logging.info(f"Extracted features with shape: {features.shape}, min={np.min(features):.4f}, max={np.max(features):.4f}, mean={np.mean(features):.4f}")
            
            # We no longer need to add channel dimension here as it's handled in the feature extractor
            # Only add batch dimension if needed
            if len(features.shape) == 3:
                features = np.expand_dims(features, axis=0)
                logging.info(f"Added batch dimension: {features.shape}")
            
            logging.info(f"Final features for model: {features.shape}")
            
            # Make prediction
            predictions = self.model.predict(features, verbose=0)
            
            # Debug all prediction values
            class_confidence = {}
            for i, confidence in enumerate(predictions[0]):
                if i < len(self.sound_classes):
                    class_name = self.sound_classes[i]
                    class_confidence[class_name] = float(confidence)
                
            # Sort by confidence (highest first)
            sorted_predictions = sorted(class_confidence.items(), key=lambda x: x[1], reverse=True)
            top_5 = sorted_predictions[:5]
            logging.info(f"Top 5 predictions: {top_5}")
            
            # Find predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.sound_classes[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Store the latest prediction
            self.latest_prediction = {
                'class': predicted_class,
                'confidence': confidence,
                'timestamp': time.time(),
                'all_predictions': class_confidence
            }
            
            # Log prediction
            logging.info(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")
            
            # Call the callback if provided
            if self.callback:
                self.callback({
                    "event": "prediction",
                    "prediction": self.latest_prediction
                })
                
        except Exception as e:
            logging.error(f"Error processing utterance: {e}")
            traceback.print_exc()
    
    def reshape_features(self, features):
        """
        This method is now a simple wrapper that adds batch dimension
        if needed. The actual feature reshaping is handled by the
        FeatureExtractor.
        
        Args:
            features: Features from FeatureExtractor
            
        Returns:
            features: Features with batch dimension added if needed
        """
        # If features is None, return None
        if features is None:
            return None
        
        # If features already have batch dimension (4D)
        if len(features.shape) == 4:
            return features
        
        # Add batch dimension if needed (3D -> 4D)
        if len(features.shape) == 3:
            return np.expand_dims(features, axis=0)
        
        logging.error(f"Unexpected feature shape: {features.shape}")
        return None
    
    def process_audio(self, audio_data):
        """
        Process a single frame of audio data.
        
        Args:
            audio_data: Audio data to process
            
        Returns:
            dict: Prediction result
        """
        try:
            # Check if this is sound or silence
            is_sound = self.sound_threshold < np.abs(audio_data).mean()
            
            if not is_sound:
                return None
                
            # For individual frame processing, use the feature extractor
            all_features = self.feature_extractor.extract_features(audio_data, is_file=False)
            features = self.feature_extractor.extract_features_for_model(all_features, model_type='cnn')
            
            if features is None:
                return None
                
            # We no longer need to add channel dimension here as it's handled in the feature extractor
            # Only add batch dimension if needed
            if len(features.shape) == 3:
                features = np.expand_dims(features, axis=0)
            
            # Make prediction
            predictions = self.model.predict(features, verbose=0)
            
            # Find predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.sound_classes[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Return prediction
            return {
                'class': predicted_class,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            traceback.print_exc()
            return None
    
    def get_latest_prediction(self):
        """
        Get the latest prediction.
        
        Returns:
            dict: Latest prediction or None
        """
        if not self.latest_prediction:
            return None
        return self.latest_prediction
    
    def get_current_state(self):
        """
        Get the current state of the detector.
        
        Returns:
            dict: Current state information
        """
        return {
            'is_listening': self.is_listening,
            'is_sound_detected': self.has_detected_sound,
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
    test_recording_path = os.path.join(Config.TEST_SOUNDS_DIR, "test_recording.wav")
    wavfile.write(test_recording_path, TEST_FS, test_recording_int)
    print(f"Playback complete. Recording saved to {test_recording_path}")


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
