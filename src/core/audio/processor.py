import os
import numpy as np
import librosa
import logging
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import find_peaks
import scipy.signal as signal
from scipy import stats
import pyloudnorm as pyln

class AudioProcessor:
    """
    Unified audio processing class that handles:
    - Audio loading and normalization
    - Sound detection and centering
    - Feature extraction for different model types
    - Audio chunking based on silences
    
    This class consolidates functionality from multiple existing classes
    to ensure consistent processing for both training and inference.
    """
    
    def __init__(self, sample_rate=16000, sound_threshold=0.02, silence_threshold=0.02,
                 min_chunk_duration=0.5, min_silence_duration=0.1, max_silence_duration=0.5):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate (int): Sample rate to use for audio processing
            sound_threshold (float): Threshold for sound detection
            silence_threshold (float): Threshold for silence detection
            min_chunk_duration (float): Minimum duration for audio chunks in seconds
            min_silence_duration (float): Minimum silence duration to split on in seconds
            max_silence_duration (float): Maximum silence duration to include in chunks in seconds
        """
        self.sr = sample_rate
        self.sound_threshold = sound_threshold
        self.silence_threshold = silence_threshold
        self.min_chunk_duration = min_chunk_duration
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        
        # Classical feature parameters
        self.n_mfcc = 13
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
        # Feature names for RandomForest
        self._feature_names = [
            'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5',
            'mfcc_mean_6', 'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9', 'mfcc_mean_10',
            'mfcc_mean_11', 'mfcc_mean_12', 'mfcc_mean_13',
            'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3', 'mfcc_std_4', 'mfcc_std_5',
            'mfcc_std_6', 'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9', 'mfcc_std_10',
            'mfcc_std_11', 'mfcc_std_12', 'mfcc_std_13',
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max',
            'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std', 
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'formant1_mean', 'formant2_mean', 'formant3_mean',
            'formant1_std', 'formant2_std', 'formant3_std'
        ]
    
    def load_audio(self, file_path):
        """
        Load audio from a file path.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            return y, sr
        except Exception as e:
            logging.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def detect_sound(self, y):
        """
        Determine if significant sound is present in the audio.
        
        Args:
            y (numpy.ndarray): Audio signal
            
        Returns:
            bool: True if sound is detected, False otherwise
        """
        # Check if audio contains sound above threshold
        return np.max(np.abs(y)) > self.sound_threshold
    
    def detect_sound_boundaries(self, y):
        """
        Find the start and end indices of sound in the audio.
        
        Args:
            y (numpy.ndarray): Audio signal
            
        Returns:
            tuple: (start_idx, end_idx)
        """
        # Compute amplitude envelope
        amplitude_envelope = np.abs(y)
        
        # Find where signal exceeds threshold
        sound_indices = np.where(amplitude_envelope > self.sound_threshold)[0]
        
        if len(sound_indices) == 0:
            return 0, len(y)
        
        # Get the first and last indices where sound is detected
        start_idx = sound_indices[0]
        end_idx = sound_indices[-1]
        
        return start_idx, end_idx
    
    def center_audio(self, y, padding_ratio=0.1):
        """
        Center the audio based on detected sound boundaries.
        This also normalizes the audio to consistent RMS level.
        
        Args:
            y (numpy.ndarray): Audio signal
            padding_ratio (float): Ratio of signal length to add as padding
            
        Returns:
            numpy.ndarray: Centered and normalized audio
        """
        if not self.detect_sound(y):
            # If no sound detected, return the original audio
            return y
        
        # Find sound boundaries
        start_idx, end_idx = self.detect_sound_boundaries(y)
        
        # Calculate padding based on signal length
        padding = int(len(y) * padding_ratio)
        
        # Adjust start and end indices with padding
        start_idx = max(0, start_idx - padding)
        end_idx = min(len(y), end_idx + padding)
        
        # Extract the sound portion
        y_centered = y[start_idx:end_idx]
        
        # Normalize volume using pyloudnorm
        meter = pyln.Meter(self.sr)
        loudness = meter.integrated_loudness(y_centered)
        
        # Target loudness in LUFS (loudness units relative to full scale)
        target_loudness = -23.0
        
        # Normalize if we can measure loudness
        if np.isfinite(loudness):
            y_normalized = pyln.normalize.loudness(y_centered, loudness, target_loudness)
        else:
            # Fallback to peak normalization if loudness measurement fails
            max_amp = np.max(np.abs(y_centered))
            if max_amp > 0:
                y_normalized = y_centered / max_amp * 0.9
            else:
                y_normalized = y_centered
        
        return y_normalized
    
    def extract_mel_spectrogram(self, y, normalize=True):
        """
        Extract mel spectrogram features for CNN models.
        
        Args:
            y (numpy.ndarray): Audio signal
            normalize (bool): Whether to normalize the spectrogram
            
        Returns:
            numpy.ndarray: Mel spectrogram features
        """
        # Check if audio is empty or too short
        if len(y) < self.hop_length:
            logging.warning("Audio is too short for feature extraction")
            # Pad with zeros
            y = np.pad(y, (0, self.hop_length - len(y)))
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize if requested
        if normalize:
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db
    
    def extract_classical_features(self, y, return_dict=True):
        """
        Extract classical audio features for RandomForest models.
        
        Args:
            y (numpy.ndarray): Audio signal
            return_dict (bool): Whether to return features as a dictionary
            
        Returns:
            dict or numpy.ndarray: Audio features
        """
        features = {}
        
        try:
            # Check if audio is empty or too short
            if len(y) < self.hop_length:
                logging.warning("Audio is too short for feature extraction")
                # Pad with zeros
                y = np.pad(y, (0, self.hop_length - len(y)))
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=self.sr, 
                n_mfcc=self.n_mfcc, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Calculate MFCC statistics (mean and std for each coefficient)
            for i in range(self.n_mfcc):
                features[f'mfcc_mean_{i+1}'] = np.mean(mfccs[i])
                features[f'mfcc_std_{i+1}'] = np.std(mfccs[i])
            
            # Extract pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(
                y=y, 
                sr=self.sr, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Calculate pitch statistics
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Avoid zero/silent frames
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_min'] = np.min(pitch_values)
                features['pitch_max'] = np.max(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_min'] = 0
                features['pitch_max'] = 0
            
            # Extract RMS energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Extract zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Extract spectral centroid
            cent = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)[0]
            features['spectral_centroid_mean'] = np.mean(cent)
            features['spectral_centroid_std'] = np.std(cent)
            
            # Extract spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, hop_length=self.hop_length)[0]
            features['spectral_rolloff_mean'] = np.mean(rolloff)
            features['spectral_rolloff_std'] = np.std(rolloff)
            
            # Extract formants (use LPC for approximate formant analysis)
            # This is a simple approximation; more advanced formant extraction may be needed
            order = 12  # LPC order (higher for more formants)
            lpc = librosa.lpc(y, order)
            
            # Find roots of the LPC polynomial
            roots = np.roots(lpc)
            
            # Keep only roots with angle in upper half of unit circle
            roots = roots[np.imag(roots) >= 0]
            
            # Convert to frequency and bandwidth
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = angles * self.sr / (2 * np.pi)
            
            # Sort by frequency
            formants = sorted(freqs)
            
            # Store first 3 formants (or pad with zeros if fewer)
            for i in range(3):
                if i < len(formants):
                    features[f'formant{i+1}_mean'] = formants[i]
                    features[f'formant{i+1}_std'] = 0  # Can't calculate std from one value
                else:
                    features[f'formant{i+1}_mean'] = 0
                    features[f'formant{i+1}_std'] = 0
            
            # Return features as dictionary or array
            if return_dict:
                return features
            else:
                # Convert to array in the same order as feature_names
                return np.array([features[name] for name in self._feature_names])
        
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            
            # Return empty features
            if return_dict:
                return {name: 0 for name in self._feature_names}
            else:
                return np.zeros(len(self._feature_names))
    
    def get_feature_names(self):
        """
        Get the list of feature names for classical features.
        
        Returns:
            list: Feature names
        """
        return self._feature_names
    
    def chop_recording(self, file_path, output_dir, min_samples=4000):
        """
        Split a WAV file into chunks based on silence detection.
        
        Args:
            file_path (str): Path to WAV file
            output_dir (str): Directory to save chunks
            min_samples (int): Minimum number of samples per chunk
            
        Returns:
            list: Paths to saved chunks
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get filename without extension for naming chunks
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load audio
        y, sr = self.load_audio(file_path)
        
        min_silence_samples = int(self.min_silence_duration * sr)
        max_silence_samples = int(self.max_silence_duration * sr)
        min_chunk_samples = int(self.min_chunk_duration * sr)
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms = np.repeat(rms, self.hop_length)
        rms = rms[:len(y)]  # Ensure same length as audio
        
        # Find silent regions (where RMS is below threshold)
        is_silent = rms < self.silence_threshold
        
        # Find transitions from sound to silence and silence to sound
        transitions = np.diff(is_silent.astype(int))
        
        # Adjust tolerance for short silences/sounds
        for i in range(len(transitions) - 1):
            if transitions[i] == 1 and transitions[i+1] == -1:  # Sound to silence to sound
                gap = i + 1 - np.where(transitions[:i] == -1)[0][-1] if len(np.where(transitions[:i] == -1)[0]) > 0 else i + 1
                if gap < min_silence_samples:
                    # Too short silence, undo transitions
                    transitions[i] = 0
                    transitions[i+1] = 0
        
        # Indices where silence starts and ends
        silence_starts = np.where(transitions == 1)[0]
        silence_ends = np.where(transitions == -1)[0]
        
        # Ensure proper pairing
        if len(silence_starts) > 0 and len(silence_ends) > 0:
            if silence_ends[0] < silence_starts[0]:
                silence_starts = np.insert(silence_starts, 0, 0)
                
            if silence_starts[-1] > silence_ends[-1]:
                silence_ends = np.append(silence_ends, len(y) - 1)
                
            # Trim to matching pairs
            min_len = min(len(silence_starts), len(silence_ends))
            silence_starts = silence_starts[:min_len]
            silence_ends = silence_ends[:min_len]
        
        # Find valid silence regions (long enough but not too long)
        valid_silences = []
        for start, end in zip(silence_starts, silence_ends):
            duration = end - start
            if min_silence_samples <= duration <= max_silence_samples:
                valid_silences.append((start, end))
        
        print(f"Found {len(valid_silences)} valid silences in recording")
        
        # Split audio at silences
        chunks = []
        if len(valid_silences) == 0:
            # No valid silences, use the whole recording
            if len(y) >= min_chunk_samples:
                chunks.append((0, len(y)))
        else:
            # Add first chunk if needed
            if valid_silences[0][0] >= min_chunk_samples:
                chunks.append((0, valid_silences[0][0]))
                
            # Add middle chunks
            for i in range(len(valid_silences) - 1):
                start = valid_silences[i][1]
                end = valid_silences[i+1][0]
                if end - start >= min_chunk_samples:
                    chunks.append((start, end))
                    
            # Add last chunk if needed
            if len(y) - valid_silences[-1][1] >= min_chunk_samples:
                chunks.append((valid_silences[-1][1], len(y)))
        
        print(f"Created {len(chunks)} chunks from recording")
        
        # Save chunks to disk
        saved_paths = []
        for i, (start, end) in enumerate(chunks):
            chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i+1}.wav")
            
            # Extract chunk
            chunk = y[start:end]
            
            # Normalize chunk amplitude
            if np.max(np.abs(chunk)) > 0:
                chunk = chunk / np.max(np.abs(chunk)) * 0.9
                
            # Save chunk
            sf.write(chunk_path, chunk, sr)
            saved_paths.append(chunk_path)
        
        return saved_paths
    
    def load_and_preprocess_audio(self, file_path):
        """
        Load and preprocess audio from a file.
        Combines loading, centering, and normalization in one step.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            numpy.ndarray: Preprocessed audio
        """
        y, sr = self.load_audio(file_path)
        y_processed = self.center_audio(y)
        return y_processed
    
    def process_audio_for_cnn(self, file_path):
        """
        Process audio for CNN model input.
        Combines loading, preprocessing, and mel spectrogram extraction.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            numpy.ndarray: Mel spectrogram features
        """
        y_processed = self.load_and_preprocess_audio(file_path)
        mel_spec = self.extract_mel_spectrogram(y_processed)
        return mel_spec
    
    def process_audio_for_rf(self, file_path, return_dict=False):
        """
        Process audio for RandomForest model input.
        Combines loading, preprocessing, and classical feature extraction.
        
        Args:
            file_path (str): Path to audio file
            return_dict (bool): Whether to return features as a dictionary
            
        Returns:
            dict or numpy.ndarray: Audio features
        """
        y_processed = self.load_and_preprocess_audio(file_path)
        features = self.extract_classical_features(y_processed, return_dict=return_dict)
        return features 