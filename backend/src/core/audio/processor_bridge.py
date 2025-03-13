"""
Bridge module to facilitate transition from src.core.audio.processor to backend.audio.processor.

This module provides backward compatibility for code that imports AudioProcessor
from src.core.audio.processor by wrapping the new SoundProcessor functionality.
"""

import logging
import warnings
import os
import numpy as np
import librosa
import time
from scipy import signal
from scipy.signal import find_peaks
import joblib

# Set up logger
logger = logging.getLogger(__name__)

# Import the new SoundProcessor if available, or provide a basic implementation
try:
    from backend.audio.processor import SoundProcessor
    logger.info("Successfully imported SoundProcessor from backend.audio.processor")
    HAVE_SOUND_PROCESSOR = True
except ImportError:
    logger.warning("Could not import SoundProcessor from backend.audio.processor. Using fallback implementation.")
    HAVE_SOUND_PROCESSOR = False
    
    # Define a minimal SoundProcessor for fallback
    class SoundProcessor:
        """Minimal fallback implementation if the real SoundProcessor is not available."""
        def __init__(self, sample_rate=16000, sound_threshold=0.02, min_silence_duration=0.1,
                    trim_silence=True, normalize_audio=True):
            self.sample_rate = sample_rate
            self.sound_threshold = sound_threshold
            self.min_silence_duration = min_silence_duration
            self.trim_silence = trim_silence
            self.normalize_audio = normalize_audio
            logger.warning("Using fallback SoundProcessor implementation. Functionality will be limited.")
            
        def process_audio(self, y):
            """Basic processing of audio signal."""
            # Implement basic functionality
            return y
            
        def extract_features(self, y, sr=None):
            """Extract basic features."""
            if sr is None:
                sr = self.sample_rate
            return {}
        
        def is_sound(self, y):
            """Check if audio contains sound above threshold."""
            return np.max(np.abs(y)) > self.sound_threshold
            
        def detect_sound_boundaries(self, y):
            """Return start and end indices of sound."""
            return 0, len(y) - 1
            
        def center_audio(self, y):
            """Center audio in frame."""
            return y

class AudioProcessor:
    """
    Legacy AudioProcessor class that wraps the new SoundProcessor.
    
    This class maintains the old API while delegating to the new implementation.
    It issues deprecation warnings to encourage migration to the new API.
    """
    
    def __init__(self, sample_rate=8000, sound_threshold=0.02, silence_threshold=0.02,
                 min_chunk_duration=0.5, min_silence_duration=0.1, max_silence_duration=0.5,
                 ambient_noise_level=None, sound_multiplier=3.0, sound_end_multiplier=2.0,
                 padding_duration=0.01, enable_stretching=False, target_chunk_duration=1.0,
                 auto_stop_after_silence=10.0, enable_loudness_normalization=False):
        """
        Initialize the legacy AudioProcessor which wraps the new SoundProcessor.
        
        Issues a deprecation warning and delegates to the new implementation.
        """
        warnings.warn(
            "AudioProcessor from src.core.audio.processor is deprecated. "
            "Please update your imports to use 'from backend.audio.processor import SoundProcessor'.",
            DeprecationWarning, stacklevel=2
        )
        
        # Store parameters for compatibility
        self.sample_rate = sample_rate
        self.sound_threshold = sound_threshold
        self.silence_threshold = silence_threshold
        self.min_chunk_duration = min_chunk_duration
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        self.target_chunk_duration = target_chunk_duration
        self.enable_loudness_normalization = enable_loudness_normalization
        
        # For backward compatibility, store these even if not used directly
        self._ambient_noise_level = ambient_noise_level
        self._sound_multiplier = sound_multiplier
        self._sound_end_multiplier = sound_end_multiplier
        self._padding_duration = padding_duration
        self._enable_stretching = enable_stretching
        self._auto_stop_after_silence = auto_stop_after_silence
        
        # Initialize the new SoundProcessor with compatible parameters
        self.processor = SoundProcessor(
            sample_rate=sample_rate,
            sound_threshold=sound_threshold,
            min_silence_duration=min_silence_duration,
            trim_silence=True,
            normalize_audio=enable_loudness_normalization
        )
        
        # Cache for feature names when processing for RF models
        self._feature_names = None
        
        logger.info("AudioProcessor bridge initialized with SoundProcessor")
    
    # Core audio processing methods
    
    def load_audio(self, file_path, sr=None):
        """Load audio from a file path."""
        if sr is None:
            sr = self.sample_rate
            
        try:
            y, sr = librosa.load(file_path, sr=sr)
            return y
        except Exception as e:
            logger.error(f"Error loading audio from {file_path}: {str(e)}")
            return None
    
    def detect_sound(self, y):
        """Detect if audio contains sound above threshold."""
        return self.processor.is_sound(y)
    
    def detect_sound_boundaries(self, y):
        """Detect sound start and end boundaries."""
        return self.processor.detect_sound_boundaries(y)
    
    def center_audio(self, y, padding_ratio=0.1):
        """Center audio in the frame with padding."""
        return self.processor.center_audio(y)
    
    def extract_mel_spectrogram(self, y, normalize=True):
        """Extract mel spectrogram features."""
        # Implement if SoundProcessor doesn't have this method
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sample_rate, 
            n_mels=64,
            n_fft=1024, 
            hop_length=256
        )
        
        # Convert to log scale (dB)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        if normalize:
            # Normalize to range [0, 1]
            log_S = (log_S - log_S.min()) / (log_S.max() - log_S.min() + 1e-10)
            
        return log_S
    
    def process_audio_for_cnn(self, file_path, training_mode=True):
        """Process audio for CNN model input."""
        # Load and preprocess audio
        audio = self.load_audio(file_path)
        
        if audio is None:
            return None
            
        # Process audio using bridge to new processor
        processed_audio = self.processor.process_audio(audio)
        
        # Extract mel spectrogram features
        mel_spec = self.extract_mel_spectrogram(processed_audio)
        
        # Reshape for CNN input (add channel dimension)
        mel_spec = np.expand_dims(mel_spec, axis=-1)
        
        return mel_spec
    
    def process_audio_for_rf(self, file_path, return_dict=False, training_mode=True):
        """Process audio for Random Forest model input."""
        # Load audio
        audio = self.load_audio(file_path)
        
        if audio is None:
            logger.error(f"Could not load audio from {file_path}")
            return None
            
        # Process audio
        processed_audio = self.processor.process_audio(audio)
        
        # Extract features using the SoundProcessor or fallback implementation
        if hasattr(self.processor, 'extract_features_for_rf'):
            features = self.processor.extract_features_for_rf(processed_audio, self.sample_rate)
        else:
            # Fallback implementation - extract basic acoustic features
            features = self._extract_acoustic_features(processed_audio)
        
        # Cache feature names
        if self._feature_names is None and return_dict:
            self._feature_names = list(features.keys())
        
        return features if return_dict else np.array(list(features.values()))
    
    def get_feature_names(self):
        """Get feature names for RF model features."""
        if self._feature_names is None:
            # If not cached, create a sample feature set and extract names
            dummy_audio = np.zeros(int(self.sample_rate * 1.0))  # 1 second of silence
            features = self.process_audio_for_rf(None, return_dict=True)
            if features is None:
                # If we can't get real features, return a default set of names
                self._feature_names = [f"feature_{i}" for i in range(20)]  
            else:
                self._feature_names = list(features.keys())
        
        return self._feature_names
    
    def _extract_acoustic_features(self, y):
        """Extract basic acoustic features from audio sample. Fallback implementation."""
        features = {}
        
        # Amplitude features
        features['rms'] = np.sqrt(np.mean(y**2))
        features['peak_amplitude'] = np.max(np.abs(y))
        
        # Zero crossing rate
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Spectral features
        if len(y) > 0:  # Ensure non-empty audio
            # Spectral centroid
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)
            features['spectral_centroid_mean'] = np.mean(spec_centroid)
            features['spectral_centroid_std'] = np.std(spec_centroid)
            
            # Spectral bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate)
            features['spectral_bandwidth_mean'] = np.mean(spec_bw)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)
            features['spectral_rolloff_mean'] = np.mean(rolloff)
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13)
            for i in range(min(13, mfccs.shape[0])):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # If no features were extracted, provide some defaults
        if not features:
            features = {
                'rms': 0.0,
                'peak_amplitude': 0.0,
                'zero_crossing_rate': 0.0,
                'spectral_centroid_mean': 0.0,
                'spectral_centroid_std': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'spectral_rolloff_mean': 0.0
            }
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 0.0
        
        return features 