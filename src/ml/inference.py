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

# Import shared constants and the SoundProcessor from audio_processing
from .constants import SAMPLE_RATE, AUDIO_DURATION, AUDIO_LENGTH, TEST_DURATION, TEST_FS, INT16_MAX
from .audio_processing import SoundProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)

# Print available audio devices for reference
print(sd.query_devices())

class SoundDetector:
    """
    Continuously listens to the microphone for short bursts of sound.
    Accumulates audio in a queue; once enough audio is detected,
    runs inference with the loaded CNN model.
    """
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.audio_queue = []
        self.is_recording = False
        self.predictions = []
        self.callback = None
        self.stream = None
        self.thread = None
        self.buffer = bytearray()
        self.sample_width = 2  # 16-bit audio
        self.frame_duration_ms = 30   # Reduced for finer granularity
        self.frame_size = int(SAMPLE_RATE * self.frame_duration_ms / 1000)
        self.frame_duration = self.frame_duration_ms / 1000.0
        self.speech_buffer = bytearray()
        self.speech_detected = False
        self.silence_duration = 0
        self.silence_threshold_ms = 500  # For better word detection
        self.auto_stop = False
        self.is_speech_active = False
        self.speech_duration = 0
        self.min_sound_duration = 0.3  # Minimum duration for a complete sound

        # SoundProcessor with adjusted threshold
        self.sound_processor = SoundProcessor(sample_rate=SAMPLE_RATE)
        self.sound_processor.sound_threshold = 0.08  # Slightly lower threshold

        self.audio_queue_lock = threading.Lock()

        logging.info(f"SoundDetector initialized with classes: {class_names}")
        devices = sd.query_devices()
        logging.info("Available audio devices:")
        for i, device in enumerate(devices):
            logging.info(f"[{i}] {device['name']} (inputs: {device['max_input_channels']})")

        # Confidence threshold for final predictions
        self.confidence_threshold = 0.40
        self.amplitude_threshold = 0.08
        self.min_prediction_threshold = 0.3

        # Circular buffer for a small "pre-buffer"
        self.pre_buffer_duration_ms = 100
        self.pre_buffer_size = int(SAMPLE_RATE * self.pre_buffer_duration_ms / 1000)
        self.circular_buffer = np.zeros(self.pre_buffer_size, dtype=np.float32)
        self.buffer_index = 0

        self._stop_event = threading.Event()

    def process_audio(self):
        """
        Called typically after enough audio is accumulated in the queue.
        Concatenates all frames, extracts features, predicts with the model.
        """
        try:
            with self.audio_queue_lock:
                if not self.audio_queue:
                    logging.info("No audio data in queue to process")
                    return
                logging.info(f"Processing audio queue of size: {len(self.audio_queue)}")
                audio_data = np.concatenate(self.audio_queue)
                self.audio_queue.clear()

            logging.info(f"Concatenated audio shape: {audio_data.shape}")

            # Normalize to -1..1 if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

            # Extract features
            features = self.extract_features(audio_data)
            if features is None:
                logging.info("No features extracted (features=None).")
                return

            logging.info(f"Extracted features shape: {features.shape}")

            # Model prediction
            features = np.expand_dims(features, axis=0)
            predictions = self.model.predict(features, verbose=0)
            logging.info(f"Raw predictions: {predictions[0]}")

            top_idx = np.argmax(predictions[0])
            top_label = self.class_names[top_idx]
            top_conf = float(predictions[0][top_idx])

            if top_conf > self.confidence_threshold:
                logging.info(f"Prediction above threshold: {top_label} (conf: {top_conf:.4f})")
                self.predictions.append((top_label, top_conf))

                if self.callback:
                    self.callback({
                        "class": top_label,
                        "confidence": top_conf
                    })

            else:
                logging.info(f"Confidence {top_conf:.4f} < threshold {self.confidence_threshold}")
                # ============ ADDED BELOW ============ 
                # Even if confidence is low, let the callback happen,
                # so the user can see/confirm/correct the guess.
                if self.callback:
                    self.callback({
                        "class": top_label,
                        "confidence": top_conf,
                        "low_confidence": True
                    })
                # ============ END NEW CODE ============

        except Exception as e:
            logging.error(f"Error in process_audio: {e}", exc_info=True)

    def extract_features(self, audio):
        # Logging the input shape
        logging.info(f"Extracting features from audio shape: {audio.shape}")
        
        # Match the feature extraction exactly as done during training
        n_mfcc = 13  # Instead of the current value
        n_fft = 512  # Make sure this matches training
        hop_length = 256  # Make sure this matches training
        
        # Extract MFCCs with matching parameters
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # Ensure we get the expected shape (13, 32)
        if mfccs.shape[1] < 32:
            # Pad if too short
            mfccs = np.pad(mfccs, ((0, 0), (0, 32 - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > 32:
            # Truncate if too long
            mfccs = mfccs[:, :32]
        
        # Add channel dimension for CNN
        mfccs = mfccs.reshape(mfccs.shape[0], mfccs.shape[1], 1)
        
        logging.info(f"Feature shape: {mfccs.shape}")
        return mfccs

    def audio_callback(self, indata, frames, time_info, status):
        """
        Sounddevice callback. Called with new chunks of mic data.
        """
        try:
            if status:
                logging.warning(f"Audio callback status: {status}")

            audio_data = indata.flatten().astype(np.float32)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

            # Sound detection
            has_sound, _ = self.sound_processor.detect_sound(audio_data)

            # Update circular buffer
            start_idx = self.buffer_index
            end_idx = start_idx + len(audio_data)
            if end_idx > self.pre_buffer_size:
                first_part = self.pre_buffer_size - start_idx
                self.circular_buffer[start_idx:] = audio_data[:first_part]
                self.circular_buffer[:end_idx - self.pre_buffer_size] = audio_data[first_part:]
            else:
                self.circular_buffer[start_idx:end_idx] = audio_data
            self.buffer_index = (self.buffer_index + len(audio_data)) % self.pre_buffer_size

            # If there's sound
            if has_sound:
                if not self.is_speech_active:
                    logging.info("Sound detected!")
                    self.is_speech_active = True
                    self.speech_duration = 0

                    # Include the pre-buffer data for context
                    with self.audio_queue_lock:
                        pre_buffer = np.concatenate([
                            self.circular_buffer[self.buffer_index:],
                            self.circular_buffer[:self.buffer_index]
                        ])
                        self.audio_queue.append(pre_buffer)

                with self.audio_queue_lock:
                    self.audio_queue.append(audio_data)

                self.speech_duration += len(audio_data) / SAMPLE_RATE

                # If we've collected enough audio, process
                if self.speech_duration >= AUDIO_DURATION:
                    self.process_audio()
                    self.is_speech_active = False

            else:
                # If we were capturing speech but now silent, finalize
                if self.is_speech_active:
                    with self.audio_queue_lock:
                        self.audio_queue.append(audio_data)
                    self.process_audio()
                    self.is_speech_active = False
                    logging.info("Sound ended")

        except Exception as e:
            logging.error(f"Error in audio_callback: {str(e)}", exc_info=True)

    def start_listening(self, callback=None, auto_stop=False):
        """
        Start capturing audio from microphone in real-time,
        applying audio_callback, and processing when enough data is found.
        """
        if self.is_recording:
            return False

        try:
            self.is_recording = True
            self.callback = callback
            self.auto_stop = auto_stop
            self.predictions = []

            # Clear flags
            self.buffer = bytearray()
            self.speech_buffer = bytearray()
            self.speech_detected = False
            self.silence_duration = 0
            self.speech_duration = 0

            logging.info("Starting audio stream with:")
            logging.info(f"Sample rate: {SAMPLE_RATE}")
            logging.info(f"Frame duration: {self.frame_duration_ms} ms")
            logging.info(f"Frame size: {self.frame_size} samples")

            # Open stream
            self.stream = sd.InputStream(
                channels=1,
                dtype=np.float32,
                samplerate=SAMPLE_RATE,
                blocksize=self.frame_size,
                callback=self.audio_callback
            )

            # Start a thread that can occasionally run process_audio
            # (Though we also trigger it in audio_callback.)
            self.thread = threading.Thread(target=self.process_audio, daemon=True)
            self.thread.start()

            self.stream.start()
            return True

        except Exception as e:
            logging.error(f"Error starting listener: {e}", exc_info=True)
            self.is_recording = False
            return False

    def stop_listening(self):
        """
        Stop the stream, clear buffers, finalize.
        """
        try:
            self.is_recording = False

            with self.audio_queue_lock:
                self.audio_queue.clear()

            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if self.thread:
                if threading.current_thread() != self.thread:
                    self.thread.join()
                self.thread = None

            logging.info("Stopped listening successfully.")
            self.speech_buffer.clear()
            return {"status": "success", "message": "Stopped listening successfully"}

        except Exception as e:
            msg = f"Error stopping listener: {e}"
            logging.error(msg, exc_info=True)
            return {"status": "error", "message": msg}


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
