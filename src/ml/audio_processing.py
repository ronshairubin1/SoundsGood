# SoundClassifier_v08/src/ml/audio_processing.py

import numpy as np
import librosa
from scipy.signal import find_peaks
from .constants import SAMPLE_RATE

class SoundProcessor:
    """
    Unified SoundProcessor for audio preprocessing in both training and inference.
    This ensures consistency: the same trimming, centering, RMS normalization,
    time-stretch, and mel-spectrogram creation are used.
    """
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.sound_threshold = 0.1  # Threshold for sound detection

    def detect_sound(self, audio):
        """Detect if audio contains significant sound."""
        frame_rms = np.sqrt(np.mean(audio**2))
        peaks, _ = find_peaks(np.abs(audio), height=self.sound_threshold)
        has_sound = frame_rms > self.sound_threshold or len(peaks) > 0
        sound_location = None
        if len(peaks) > 0:
            sound_location = peaks[np.argmax(np.abs(audio)[peaks])]
        return has_sound, sound_location

    def detect_sound_boundaries(self, audio):
        frame_length = int(0.02 * self.sample_rate)  # 20ms windows
        hop_length = frame_length // 2
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Interpolate RMS
        rms_interp = np.interp(
            np.linspace(0, len(audio), len(audio)),
            np.linspace(0, len(audio), len(rms)),
            rms
        )
        is_sound = rms_interp > (self.sound_threshold * np.max(rms_interp))
        if not np.any(is_sound):
            return 0, len(audio), False
        
        sound_indices = np.where(is_sound)[0]
        start_idx = sound_indices[0]
        end_idx = sound_indices[-1]
        
        margin = int(0.1 * self.sample_rate)
        start_idx = max(0, start_idx - margin)
        end_idx = min(len(audio), end_idx + margin)
        return start_idx, end_idx, True

    def center_audio(self, audio):
        start_idx, end_idx, has_sound = self.detect_sound_boundaries(audio)
        if not has_sound:
            center = len(audio) // 2
            window_size = self.sample_rate
            start_idx = max(0, center - window_size // 2)
            end_idx = min(len(audio), center + window_size // 2)
        audio = audio[start_idx:end_idx]

        target_rms = 0.1
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)

        # Time-stretch to exactly 1 second
        target_length = self.sample_rate
        if len(audio) > 0:
            stretch_factor = target_length / len(audio)
            audio = librosa.effects.time_stretch(y=audio, rate=stretch_factor)
        
        if len(audio) > self.sample_rate:
            audio = audio[:self.sample_rate]
        elif len(audio) < self.sample_rate:
            audio = np.pad(audio, (0, self.sample_rate - len(audio)), 'constant')
        return audio

    def extract_features(self, audio):
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_mels=64, n_fft=1024,
            hop_length=256
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db[1:, :]  # remove the first mel band
        target_width = 64
        if mel_spec_db.shape[1] != target_width:
            mel_spec_db = np.array([np.interp(
                np.linspace(0, 100, target_width),
                np.linspace(0, 100, mel_spec_db.shape[1]),
                row
            ) for row in mel_spec_db])
        features = mel_spec_db[..., np.newaxis]
        return features

    def process_audio(self, audio):
        preprocessed_audio = self.center_audio(audio)
        features = self.extract_features(preprocessed_audio)
        return features
