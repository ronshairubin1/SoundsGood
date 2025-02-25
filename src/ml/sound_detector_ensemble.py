import numpy as np
import logging
import threading
import sounddevice as sd
from .audio_processing import SoundProcessor
from .feature_extractor import AudioFeatureExtractor
from .constants import SAMPLE_RATE, AUDIO_DURATION

class SoundDetectorEnsemble:
    """
    Real-time sound detector that uses an EnsembleClassifier for inference.
    It calls ensemble_model.predict(X_rf, X_cnn).
    """
    def __init__(self, ensemble_model):
        # ensemble_model is an EnsembleClassifier instance
        self.ensemble_model = ensemble_model
        
        # Queues and state
        self.audio_queue = []
        self.callback = None
        self.is_speech_active = False
        self.speech_duration = 0
        
        self.audio_queue_lock = threading.Lock()
        self.pre_buffer_duration_ms = 100
        self.buffer_index = 0
        self._stop_event = threading.Event()
        
        # This SoundProcessor might produce mel-spectrograms for the CNN
        self.sound_processor = SoundProcessor(sample_rate=SAMPLE_RATE)
        self.sound_processor.sound_threshold = 0.08
        
        # For the RF portion, we can use an 
        # (assuming you have something like this in your codebase)
        self.rf_extractor = AudioFeatureExtractor(sr=SAMPLE_RATE)
        self.feature_names = self.rf_extractor.get_feature_names()

        # Pre-buffer for short-latency detection
        self.circular_buffer = np.zeros(
            int(SAMPLE_RATE * self.pre_buffer_duration_ms / 1000),
            dtype=np.float32
        )

        self.confidence_threshold = 0.4
        self.stream = None
        self.is_recording = False

        logging.info("SoundDetectorEnsemble initialized.")

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

            # 1) Extract CNN features
            cnn_features = self.sound_processor.process_audio(audio_data)
            if cnn_features is None:
                logging.info("Could not extract CNN features.")
                return
            X_cnn = np.expand_dims(cnn_features, axis=0)

            # 2) Extract RF features
            # If your AudioFeatureExtractor requires a temp WAV file, you might write audio_data to disk
            # For an in-memory approach, adapt your AudioFeatureExtractor.  Example is simplified:
            temp_wav = "temp_ensemble_chunk.wav"
            self.rf_extractor.save_wav(audio_data, SAMPLE_RATE, temp_wav)
            feats = self.rf_extractor.extract_features(temp_wav)
            # os.remove(temp_wav)  # If you wish to clean up once done

            if not feats:
                logging.info("Could not extract RF features.")
                return
            row = [feats[fn] for fn in self.feature_names]
            X_rf = [row]

            # 3) Run Ensemble prediction
            # .predict(...) returns a list of (class_label, confidence)
            results = self.ensemble_model.predict(X_rf, X_cnn)
            predicted_label, top_conf = results[0]  # single row

            # 4) Decide if we pass it along
            if top_conf >= self.confidence_threshold:
                logging.info(
                    f"Ensemble predicted: {predicted_label} (conf: {float(top_conf):.4f})"
                )
            else:
                logging.info(
                    f"Ensemble predicted (low conf {float(top_conf):.4f}): {predicted_label}"
                )

            if self.callback:
                self.callback({
                    "class": predicted_label,
                    "confidence": float(top_conf)
                })

        except Exception as e:
            logging.error(f"Error in SoundDetectorEnsemble.process_audio: {e}", exc_info=True)

    def audio_callback(self, indata, frames, time_info, status):
        try:
            # Convert to float32
            audio_data = indata.flatten().astype(np.float32)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()

            has_sound, _ = self.sound_processor.detect_sound(audio_data)

            # Update our circular buffer
            start_idx = self.buffer_index
            end_idx = start_idx + len(audio_data)
            if end_idx > len(self.circular_buffer):
                first_part = len(self.circular_buffer) - start_idx
                self.circular_buffer[start_idx:] = audio_data[:first_part]
                self.circular_buffer[:end_idx - len(self.circular_buffer)] = audio_data[first_part:]
            else:
                self.circular_buffer[start_idx:end_idx] = audio_data
            self.buffer_index = (self.buffer_index + len(audio_data)) % len(self.circular_buffer)

            if has_sound:
                if not self.is_speech_active:
                    logging.info("Ensemble: Sound detected!")
                    self.is_speech_active = True
                    self.speech_duration = 0

                    with self.audio_queue_lock:
                        pre_buffer = np.concatenate([
                            self.circular_buffer[self.buffer_index:], 
                            self.circular_buffer[:self.buffer_index]
                        ])
                        self.audio_queue.append(pre_buffer)

                with self.audio_queue_lock:
                    self.audio_queue.append(audio_data)

                self.speech_duration += len(audio_data) / SAMPLE_RATE

                if self.speech_duration >= AUDIO_DURATION:
                    self.process_audio()
                    self.is_speech_active = False
            else:
                if self.is_speech_active:
                    with self.audio_queue_lock:
                        self.audio_queue.append(audio_data)
                    self.process_audio()
                    self.is_speech_active = False
                    logging.info("Ensemble: Sound ended.")

        except Exception as e:
            logging.error(f"Error in SoundDetectorEnsemble.audio_callback: {e}", exc_info=True)

    def start_listening(self, callback=None, auto_stop=False):
        if self.is_recording:
            return False

        try:
            self.is_recording = True
            self.callback = callback

            logging.info("Starting ensemble audio stream...")

            self.stream = sd.InputStream(
                channels=1, dtype=np.float32, samplerate=SAMPLE_RATE,
                blocksize=int(SAMPLE_RATE * 0.03),  # 30 ms chunk
                callback=self.audio_callback
            )
            self.stream.start()
            return True
        except Exception as e:
            logging.error(f"Error starting ensemble listening: {e}", exc_info=True)
            self.is_recording = False
            return False

    def stop_listening(self):
        try:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            logging.info("Stopped ensemble listening.")
        except Exception as e:
            logging.error(f"Error stopping ensemble listening: {e}", exc_info=True) 