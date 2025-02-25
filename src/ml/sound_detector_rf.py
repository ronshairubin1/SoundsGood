import numpy as np
import logging
import threading
import sounddevice as sd
from .audio_processing import SoundProcessor
from .constants import SAMPLE_RATE, AUDIO_DURATION

class SoundDetectorRF:
    """
    Real-time sound detector that uses a RandomForestClassifier for inference.
    Similar to SoundDetector, but calls rf_model.predict() on extracted features.
    """
    def __init__(self, rf_model):
        self.rf_model = rf_model
        self.audio_queue = []
        self.callback = None
        self.is_speech_active = False
        self.speech_duration = 0
        self.min_sound_duration = 0.3

        # NEW: track recording state
        self.is_recording = False
        self.stream = None

        # Same thresholds / buffering as your CNN SoundDetector
        self.audio_queue_lock = threading.Lock()
        self.pre_buffer_duration_ms = 100
        self.buffer_index = 0
        self._stop_event = threading.Event()

        # Setup SoundProcessor
        self.sound_processor = SoundProcessor(sample_rate=SAMPLE_RATE)
        self.sound_processor.sound_threshold = 0.08
        self.circular_buffer = np.zeros(int(SAMPLE_RATE * self.pre_buffer_duration_ms / 1000), dtype=np.float32)

        # Confidence threshold is optional; RF returns predicted labels but not a direct "confidence"
        self.confidence_threshold = 0.4

        logging.info("SoundDetectorRF initialized.")

    def process_audio(self):
        try:
            with self.audio_queue_lock:
                if not self.audio_queue:
                    logging.info("No audio data in queue to process.")
                    return
                audio_data = np.concatenate(self.audio_queue)
                self.audio_queue.clear()

            # Normalize -1..1 if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

            # Extract features (assuming your rf_classifier expects a feature row)
            features = self.sound_processor.process_audio(audio_data)
            if features is None:
                logging.info("No features extracted.")
                return

            # The RF classifier expects a 2D array: shape (n_samples, n_features)
            features_2d = np.expand_dims(features, axis=0)

            # Call the RF model's predict
            predictions, probabilities = self.rf_model.predict(features_2d)
            top_label = predictions[0]   # The predicted class label
            # probabilities is shape (1, n_classes), so:
            top_conf = max(probabilities[0]) if probabilities is not None else 1.0

            # If you want a threshold:
            if top_conf >= self.confidence_threshold:
                logging.info(f"RF predict: {top_label} (conf: {top_conf:.4f})")
            else:
                logging.info(f"RF predict (low conf {top_conf:.4f}): {top_label}")

            # If there's a callback, pass the result
            if self.callback:
                self.callback({
                    "class": top_label,
                    "confidence": float(top_conf)
                })

        except Exception as e:
            logging.error(f"Error in SoundDetectorRF.process_audio: {e}", exc_info=True)

    def audio_callback(self, indata, frames, time_info, status):
        """
        This is similar to your CNN code, except we still call self.process_audio()
        once enough audio is gathered.
        """
        try:
            audio_data = indata.flatten().astype(np.float32)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

            has_sound, _ = self.sound_processor.detect_sound(audio_data)

            # Update the circular buffer (just like your CNN-based code)
            start_idx = self.buffer_index
            end_idx = start_idx + len(audio_data)
            if end_idx > len(self.circular_buffer):
                first_part = len(self.circular_buffer) - start_idx
                self.circular_buffer[start_idx:] = audio_data[:first_part]
                self.circular_buffer[:end_idx - len(self.circular_buffer)] = audio_data[first_part:]
            else:
                self.circular_buffer[start_idx:end_idx] = audio_data
            self.buffer_index = (self.buffer_index + len(audio_data)) % len(self.circular_buffer)

            # If there's sound
            if has_sound:
                if not self.is_speech_active:
                    logging.info("RF: Sound detected!")
                    self.is_speech_active = True
                    self.speech_duration = 0

                    # Include pre-buffer for context
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
                if self.is_speech_active:
                    # finalize
                    with self.audio_queue_lock:
                        self.audio_queue.append(audio_data)
                    self.process_audio()
                    self.is_speech_active = False
                    logging.info("RF: Sound ended.")

        except Exception as e:
            logging.error(f"Error in SoundDetectorRF.audio_callback: {e}", exc_info=True)

    def start_listening(self, callback=None, auto_stop=False):
        """
        Opens an audio stream (just like your CNN/Ensemble) so that audio_callback
        is continuously invoked, and we can queue/process data in real time.
        """
        if self.is_recording:
            return False
        try:
            self.is_recording = True
            self.callback = callback

            logging.info("Starting RF audio stream...")
            self.stream = sd.InputStream(
                channels=1,
                dtype=np.float32,
                samplerate=SAMPLE_RATE,
                blocksize=int(SAMPLE_RATE * 0.03),  # ~30 ms block
                callback=self.audio_callback
            )
            self.stream.start()
            return True
        except Exception as e:
            logging.error(f"Error starting RF listening: {e}", exc_info=True)
            self.is_recording = False
            return False

    def stop_listening(self):
        try:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            logging.info("Stopped RF listening.")
            return {"status": "success", "message": "RF detector stopped."}
        except Exception as e:
            logging.error(f"Error stopping RF listener: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
