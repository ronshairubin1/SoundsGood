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
    
    def __init__(self, sample_rate=8000, sound_threshold=0.02, silence_threshold=0.02,
                 min_chunk_duration=0.5, min_silence_duration=0.1, max_silence_duration=0.5,
                 ambient_noise_level=None, sound_multiplier=3.0, sound_end_multiplier=2.0,
                 padding_duration=0.01, enable_stretching=False, target_chunk_duration=1.0,
                 auto_stop_after_silence=10.0, enable_loudness_normalization=False):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate (int): Sample rate to use for audio processing
            sound_threshold (float): Threshold for sound detection
            silence_threshold (float): Threshold for silence detection
            min_chunk_duration (float): Minimum duration for audio chunks in seconds
            min_silence_duration (float): Minimum silence duration to split on in seconds
            max_silence_duration (float): Maximum silence duration to include in chunks in seconds
            ambient_noise_level (float): Ambient noise level (if None, use silence_threshold)
            sound_multiplier (float): Multiple of ambient noise to consider as sound
            sound_end_multiplier (float): Multiple of ambient noise to extend sound boundaries
            padding_duration (float): Padding to add at beginning and end of chunks in seconds
            enable_stretching (bool): Whether to stretch short chunks to target duration
            target_chunk_duration (float): Target duration for stretching in seconds
            auto_stop_after_silence (float): Stop recording after this many seconds of silence
            enable_loudness_normalization (bool): Whether to apply loudness normalization
        """
        self.sr = sample_rate
        self.sound_threshold = sound_threshold
        self.silence_threshold = silence_threshold
        self.min_chunk_duration = min_chunk_duration
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        
        # New parameters for enhanced processing
        self.ambient_noise_level = ambient_noise_level or silence_threshold
        self.sound_multiplier = sound_multiplier
        self.sound_end_multiplier = sound_end_multiplier
        self.padding_duration = padding_duration
        self.enable_stretching = enable_stretching
        self.target_chunk_duration = target_chunk_duration
        self.auto_stop_after_silence = auto_stop_after_silence
        self.enable_loudness_normalization = enable_loudness_normalization
        
        # Classical feature parameters - REDUCED BLOCK SIZE for short audio
        self.n_mfcc = 14  # Increased from 13 to 14 MFCC coefficients
        self.n_mels = 64  # Reduced from 128
        self.n_fft = 512  # Reduced from 2048 to handle shorter audio
        self.hop_length = 128  # Reduced from 512
        
        # Delta computation parameters
        self.compute_deltas = True  # Whether to compute delta features
        self.compute_delta_deltas = True  # Whether to compute delta-delta features
        self.delta_width = 9  # Width of the window for computing deltas (must be odd)
        
        # Counter for stretched files - new addition
        self.stretched_count = 0
        
        # Dynamically generate feature names based on MFCC settings
        self._feature_names = self._generate_feature_names()
        
        # Track processing errors
        self.processing_errors = []
        
        # Track MFCC normalization stats
        self.mfcc_stats = {
            'before_normalization': {
                'min': None,
                'max': None,
                'mean': None,
                'std': None
            },
            'after_normalization': {
                'min': None,
                'max': None,
                'mean': None,
                'std': None
            },
            'coefficients': []  # Store individual coefficient values
        }
    
    def _generate_feature_names(self):
        """
        Dynamically generate feature names based on MFCC settings.
        
        Returns:
            list: Feature names for all extracted features
        """
        feature_names = []
        
        # Basic MFCC features (means and standard deviations)
        for i in range(1, self.n_mfcc + 1):
            feature_names.append(f'mfcc_mean_{i}')
        
        for i in range(1, self.n_mfcc + 1):
            feature_names.append(f'mfcc_std_{i}')
        
        # Delta MFCC features
        if self.compute_deltas:
            for i in range(1, self.n_mfcc + 1):
                feature_names.append(f'mfcc_delta_mean_{i}')
            
            for i in range(1, self.n_mfcc + 1):
                feature_names.append(f'mfcc_delta_std_{i}')
        
        # Delta-delta MFCC features
        if self.compute_delta_deltas:
            for i in range(1, self.n_mfcc + 1):
                feature_names.append(f'mfcc_delta2_mean_{i}')
            
            for i in range(1, self.n_mfcc + 1):
                feature_names.append(f'mfcc_delta2_std_{i}')
        
        # Add other acoustic features
        feature_names.extend([
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max',
            'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std', 
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'formant1_mean', 'formant2_mean', 'formant3_mean',
            'formant1_std', 'formant2_std', 'formant3_std'
        ])
        
        return feature_names
    
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
        original_length = len(y)
        logging.info(f"center_audio: original length = {original_length} samples")
        
        if not self.detect_sound(y):
            # If no sound detected, return the full original audio to avoid 
            # returning an empty segment
            logging.warning("No significant sound detected in audio, using full signal")
            return y
        
        # Find sound boundaries
        start_idx, end_idx = self.detect_sound_boundaries(y)
        logging.info(f"center_audio: detected sound from {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
        
        # Make sure we have at least some meaningful audio content
        min_required_samples = max(self.n_fft, self.hop_length * 4)
        if end_idx - start_idx < min_required_samples:
            logging.warning(f"Detected sound segment too short ({end_idx - start_idx} samples), using full signal")
            return y
        
        # Calculate padding based on signal length
        padding = int(len(y) * padding_ratio)
        
        # Adjust start and end indices with padding
        start_idx = max(0, start_idx - padding)
        end_idx = min(len(y), end_idx + padding)
        
        logging.info(f"center_audio: after padding, using {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
        
        # For safety, check if centering would make audio too short
        if end_idx - start_idx < min_required_samples:
            logging.warning(f"Centering would make audio too short ({end_idx - start_idx} samples), using full signal")
            return y
        
        # Extract the sound portion
        y_centered = y[start_idx:end_idx]
        
        # Apply normalization based on settings
        if self.enable_loudness_normalization:
            # Normalize volume using pyloudnorm
            try:
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
                        
                logging.info(f"Applied loudness normalization (target: {target_loudness} LUFS)")
            except Exception as e:
                error_msg = f"Error in loudness normalization: {e}. Using peak normalization instead."
                logging.warning(error_msg)
                self.processing_errors.append({
                    'type': 'loudness_normalization',
                    'message': error_msg
                })
                
                # If pyloudnorm fails (e.g., "Audio must have length greater than the block size")
                # Fall back to simple peak normalization
                max_amp = np.max(np.abs(y_centered))
                if max_amp > 0:
                    y_normalized = y_centered / max_amp * 0.9
                else:
                    y_normalized = y_centered
        else:
            # Skip loudness normalization, just use peak normalization
            logging.info("Loudness normalization disabled, using peak normalization")
            max_amp = np.max(np.abs(y_centered))
            if max_amp > 0:
                y_normalized = y_centered / max_amp * 0.9
            else:
                y_normalized = y_centered
        
        logging.info(f"center_audio: final length = {len(y_normalized)} samples")
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
        original_len = len(y)
        use_smaller_fft = False
        n_fft = self.n_fft
        
        # Always log the original audio length
        logging.info(f"Processing audio with length {original_len} samples ({original_len/self.sr:.3f} seconds)")
        
        # Ensure minimum audio length for processing
        min_required_samples = max(self.n_fft, self.hop_length * 4)
        
        # Check if audio is empty or too short for n_fft
        if len(y) < min_required_samples:
            logging.warning(f"Audio is too short for feature extraction (length: {len(y)}, needs: {min_required_samples})")
            
            # Time-stretch the audio if it has at least some content
            if len(y) > 0:
                # Target length should be more generous to ensure success
                target_length = min_required_samples * 2
                
                try:
                    # For extremely short files, repeat the audio before stretching
                    if len(y) < 100:  # If extremely short (less than 100 samples)
                        repeats_needed = max(1, int(200 / len(y)))
                        logging.info(f"Audio extremely short ({len(y)} samples), repeating {repeats_needed} times before stretching")
                        y = np.tile(y, repeats_needed)
                    
                    # Calculate stretch rate (target length / current length)
                    # This is confusing: in librosa, rate < 1 makes audio LONGER (it stretches it)
                    # So we need current_length / target_length to make it longer
                    stretch_rate = len(y) / target_length
                    
                    # A clear explanation of what's happening
                    logging.info(f"Time-stretching: original length={len(y)} samples, " +
                                f"target length={target_length} samples, stretch rate={stretch_rate:.3f}")
                    logging.info(f"A stretch rate < 1.0 makes audio LONGER in time, which is what we want here")
                    
                    # Time stretch the audio to make it longer (rate < 1 makes it longer)
                    y = librosa.effects.time_stretch(y, rate=stretch_rate)
                    logging.info(f"Successfully time-stretched audio from {original_len} to {len(y)} samples")
                    self.stretched_count += 1
                    
                    # Double-check the result is long enough
                    if len(y) < min_required_samples:
                        # If still too short after stretching, pad it
                        logging.warning(f"Audio still too short after stretching ({len(y)} samples), padding with zeros")
                        y = np.pad(y, (0, min_required_samples - len(y)))
                        
                except Exception as e:
                    logging.error(f"Error time-stretching audio: {e}, falling back to padding")
                    # If time stretching fails, fall back to padding
                    padding_needed = min_required_samples - len(y)
                    y = np.pad(y, (0, padding_needed))
                    logging.info(f"Padded audio from {original_len} to {len(y)} samples instead")
            else:
                # If empty, just create a minimal audio array filled with a small amount of noise
                logging.warning("Empty audio, creating minimal audio with small noise")
                # Small random noise instead of zeros (better for processing)
                y = np.random.randn(min_required_samples) * 0.01
                
        # Also check for hop_length as before
        elif len(y) < self.hop_length * 4:  # Need at least a few frames
            logging.warning(f"Audio is too short for multiple hop lengths ({len(y)} samples)")
            # Pad with zeros
            y = np.pad(y, (0, (self.hop_length * 4) - len(y)))
            logging.info(f"Padded audio from {original_len} to {len(y)} samples for multiple hop lengths")
        
        # Compute mel spectrogram with potentially adjusted n_fft
        hop_length = min(self.hop_length, n_fft // 4) if use_smaller_fft else self.hop_length
        
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=self.sr, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                n_mels=self.n_mels
            )
            
            # Convert to decibels
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize if requested
            if normalize:
                mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Ensure spectrograms have at least 13 time steps for consistent shape
            if mel_spec_db.shape[1] < 13:
                # Pad to at least 13 time steps
                padding = ((0, 0), (0, 13 - mel_spec_db.shape[1]))
                mel_spec_db = np.pad(mel_spec_db, padding, mode='constant')
                logging.info(f"Padded spectrogram time steps from {mel_spec.shape[1]} to 13")
            
            # Add channel dimension for CNN (mel_bands, time_steps) -> (time_steps, mel_bands, 1)
            # NOTE: Transposing to (time_steps, mel_bands, 1) for CNN - this is important!
            # Original shape: (mel_bands, time_steps)
            logging.info(f"Original spectrogram shape: {mel_spec_db.shape}")
            
            mel_spec_db = np.transpose(mel_spec_db)  # Now shape is (time_steps, mel_bands)
            logging.info(f"Transposed spectrogram shape: {mel_spec_db.shape}")
            
            mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)  # Add channel dimension
            logging.info(f"Final spectrogram shape with channel dimension: {mel_spec_db.shape}")
            
            logging.info(f"Processed audio of length {original_len} to mel spectrogram shape {mel_spec_db.shape}")
            return mel_spec_db
            
        except Exception as e:
            logging.error(f"Error in melspectrogram extraction: {e}")
            # For problematic files, create a placeholder spectrogram with the right dimensions
            # Using 13 time steps, n_mels mel bands, 1 channel
            placeholder = np.zeros((13, self.n_mels, 1))
            logging.info(f"Created placeholder spectrogram of shape {placeholder.shape}")
            return placeholder
    
    def extract_classical_features(self, y, return_dict=True, normalize_mfcc=True, exclude_first_mfcc=True):
        """
        Extract classical audio features for RandomForest models.
        
        Args:
            y (numpy.ndarray): Audio signal
            return_dict (bool): Whether to return features as a dictionary
            normalize_mfcc (bool): Whether to normalize MFCC coefficients
            exclude_first_mfcc (bool): Whether to exclude the first MFCC coefficient (energy)
            
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
            
            # Store original MFCC stats before normalization
            self.mfcc_stats['before_normalization']['min'] = float(np.min(mfccs))
            self.mfcc_stats['before_normalization']['max'] = float(np.max(mfccs))
            self.mfcc_stats['before_normalization']['mean'] = float(np.mean(mfccs))
            self.mfcc_stats['before_normalization']['std'] = float(np.std(mfccs))
            
            # Store individual coefficient values (take mean across time frames to get one value per coefficient)
            self.mfcc_stats['coefficients'] = [float(np.mean(mfccs[i])) for i in range(mfccs.shape[0])]
            
            # Normalize MFCCs if requested
            if normalize_mfcc:
                # Determine which coefficients to normalize
                start_idx = 1 if exclude_first_mfcc else 0
                
                if exclude_first_mfcc:
                    logging.info("Excluding first MFCC coefficient (energy/loudness) from normalization")
                
                # Normalize each MFCC coefficient (except possibly the first)
                if start_idx < len(mfccs):
                    mfccs_to_normalize = mfccs[start_idx:]
                    # Normalize to zero mean and unit variance
                    mfccs_mean = np.mean(mfccs_to_normalize, axis=1, keepdims=True)
                    mfccs_std = np.std(mfccs_to_normalize, axis=1, keepdims=True) + 1e-8  # Avoid division by zero
                    mfccs[start_idx:] = (mfccs_to_normalize - mfccs_mean) / mfccs_std
                    
                    logging.info(f"Normalized MFCC coefficients {start_idx+1}-{self.n_mfcc}")
                
                # Store normalized MFCC stats
                self.mfcc_stats['after_normalization']['min'] = float(np.min(mfccs))
                self.mfcc_stats['after_normalization']['max'] = float(np.max(mfccs))
                self.mfcc_stats['after_normalization']['mean'] = float(np.mean(mfccs))
                self.mfcc_stats['after_normalization']['std'] = float(np.std(mfccs))
            
            # Calculate MFCC statistics (mean and std for each coefficient)
            for i in range(self.n_mfcc):
                features[f'mfcc_mean_{i+1}'] = np.mean(mfccs[i])
                features[f'mfcc_std_{i+1}'] = np.std(mfccs[i])
            
            # Compute delta and delta-delta features if enabled
            if self.compute_deltas:
                # MODIFICATION: Exclude the first coefficient when computing deltas
                # Extract all coefficients except the first
                mfccs_for_delta = mfccs[1:] if exclude_first_mfcc else mfccs
                
                if exclude_first_mfcc:
                    logging.info("Excluding first MFCC coefficient (energy/loudness) from delta calculation")
                
                # Compute first-order delta (velocity) features
                delta_mfccs = librosa.feature.delta(mfccs_for_delta, width=self.delta_width)
                
                logging.info(f"Computed MFCC delta features with width={self.delta_width}")
                
                # Apply separate normalization to delta features
                delta_mean = np.mean(delta_mfccs, axis=1, keepdims=True)
                delta_std = np.std(delta_mfccs, axis=1, keepdims=True) + 1e-8  # Avoid division by zero
                delta_mfccs_normalized = (delta_mfccs - delta_mean) / delta_std
                logging.info("Applied separate normalization to delta features")
                
                # Calculate delta statistics
                num_delta_feats = delta_mfccs.shape[0]  # Number of features (could be n_mfcc-1 if excluding first coeff)
                for i in range(num_delta_feats):
                    # Offset the index by 1 if we excluded the first coefficient
                    coeff_index = i + (1 if exclude_first_mfcc else 0) + 1
                    features[f'mfcc_delta_mean_{coeff_index}'] = np.mean(delta_mfccs_normalized[i])
                    features[f'mfcc_delta_std_{coeff_index}'] = np.std(delta_mfccs_normalized[i])
                
                # If first coefficient was excluded, set its delta values to 0 for completeness
                if exclude_first_mfcc:
                    features['mfcc_delta_mean_1'] = 0.0
                    features['mfcc_delta_std_1'] = 0.0
                
                # Compute second-order delta (acceleration) features
                if self.compute_delta_deltas:
                    delta2_mfccs = librosa.feature.delta(delta_mfccs, width=self.delta_width)
                    
                    logging.info(f"Computed MFCC delta-delta features with width={self.delta_width}")
                    
                    # Apply separate normalization to delta-delta features
                    delta2_mean = np.mean(delta2_mfccs, axis=1, keepdims=True)
                    delta2_std = np.std(delta2_mfccs, axis=1, keepdims=True) + 1e-8  # Avoid division by zero
                    delta2_mfccs_normalized = (delta2_mfccs - delta2_mean) / delta2_std
                    logging.info("Applied separate normalization to delta-delta features")
                    
                    # Calculate delta-delta statistics
                    for i in range(num_delta_feats):
                        # Offset the index by 1 if we excluded the first coefficient
                        coeff_index = i + (1 if exclude_first_mfcc else 0) + 1
                        features[f'mfcc_delta2_mean_{coeff_index}'] = np.mean(delta2_mfccs_normalized[i])
                        features[f'mfcc_delta2_std_{coeff_index}'] = np.std(delta2_mfccs_normalized[i])
                    
                    # If first coefficient was excluded, set its delta-delta values to 0 for completeness
                    if exclude_first_mfcc:
                        features['mfcc_delta2_mean_1'] = 0.0
                        features['mfcc_delta2_std_1'] = 0.0
            
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
                return np.array([features.get(name, 0) for name in self._feature_names])
        
        except Exception as e:
            error_msg = f"Error extracting features: {e}"
            logging.error(error_msg)
            self.processing_errors.append({
                'type': 'feature_extraction',
                'message': error_msg
            })
            
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
    
    def measure_ambient_noise(self, file_path, duration=3.0):
        """
        Measure the ambient noise level from the beginning of a recording.
        
        Args:
            file_path (str): Path to audio file
            duration (float): Duration in seconds to measure from start of file
            
        Returns:
            float: RMS level of ambient noise
        """
        try:
            y, sr = self.load_audio(file_path)
            
            # Use just the first part of the recording for ambient noise
            samples_to_use = min(int(duration * sr), len(y))
            ambient_segment = y[:samples_to_use]
            
            # Calculate RMS energy
            from librosa.feature import rms
            ambient_rms = rms(y=ambient_segment, hop_length=self.hop_length)[0]
            
            # Use the median RMS as ambient noise level (more robust than mean)
            ambient_level = np.median(ambient_rms)
            
            logging.info(f"Measured ambient noise level: {ambient_level:.6f}")
            
            # Update the class threshold based on this measurement
            self.ambient_noise_level = ambient_level
            self.sound_threshold = ambient_level * self.sound_multiplier
            self.silence_threshold = ambient_level * 1.5  # A bit higher than ambient
            
            return ambient_level
            
        except Exception as e:
            logging.error(f"Error measuring ambient noise: {e}")
            return self.silence_threshold  # Fall back to default threshold

    def chop_recording(self, file_path, output_dir, min_samples=4000, use_ambient_noise=False):
        """
        Split a WAV file into chunks based on silence detection.
        
        Args:
            file_path (str): Path to WAV file
            output_dir (str): Directory to save chunks
            min_samples (int): Minimum number of samples per chunk
            use_ambient_noise (bool): Whether to measure and use ambient noise level
            
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
        
        # Measure ambient noise if requested
        if use_ambient_noise and self.ambient_noise_level is None:
            self.measure_ambient_noise(file_path)
            logging.info(f"Using ambient noise level for silence detection: {self.ambient_noise_level:.6f}")
        
        # Calculate thresholds in samples
        min_silence_samples = int(self.min_silence_duration * sr)
        max_silence_samples = int(self.max_silence_duration * sr)
        min_chunk_samples = int(self.min_chunk_duration * sr)
        padding_samples = int(self.padding_duration * sr)
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms = np.repeat(rms, self.hop_length)
        rms = rms[:len(y)]  # Ensure same length as audio
        
        # Get the current silence threshold (may have been updated by ambient noise)
        current_silence_threshold = self.silence_threshold
        
        # Find silent regions (where RMS is below threshold)
        is_silent = rms < current_silence_threshold
        
        # Find long silences that might indicate end of recording
        silence_runs = []
        current_run = 0
        for i, silent in enumerate(is_silent):
            if silent:
                current_run += 1
            else:
                if current_run >= self.auto_stop_after_silence * sr:
                    # Found a silence that's long enough to stop recording
                    silence_runs.append((i - current_run, i))
                current_run = 0
                
        # Check if the recording ends with a long silence
        if current_run >= self.auto_stop_after_silence * sr:
            silence_runs.append((len(is_silent) - current_run, len(is_silent)))
        
        # If we found long silences, truncate the audio
        if silence_runs:
            # Use the first long silence to truncate
            y = y[:silence_runs[0][0]]
            is_silent = is_silent[:silence_runs[0][0]]
            logging.info(f"Truncated recording at {silence_runs[0][0]/sr:.2f}s due to long silence")
        
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
            
            # Extract chunk with padding
            pad_start = max(0, start - padding_samples)
            pad_end = min(len(y), end + padding_samples)
            chunk = y[pad_start:pad_end]
            
            # Apply stretching if enabled and needed
            if self.enable_stretching and len(chunk) < self.target_chunk_duration * sr:
                target_length = int(self.target_chunk_duration * sr)
                # Use librosa's time stretching
                chunk = librosa.effects.time_stretch(chunk, rate=len(chunk)/target_length)
                logging.info(f"Stretched chunk from {len(chunk)/sr:.2f}s to {target_length/sr:.2f}s")
            
            # Normalize chunk amplitude
            if np.max(np.abs(chunk)) > 0:
                chunk = chunk / np.max(np.abs(chunk)) * 0.9
                
            # Save chunk
            sf.write(chunk_path, chunk, sr)
            saved_paths.append(chunk_path)
        
        return saved_paths
    
    def load_and_preprocess_audio(self, file_path, training_mode=False):
        """
        Load and preprocess audio from a file.
        Combines loading, centering, and normalization in one step.
        
        Args:
            file_path (str): Path to audio file
            training_mode (bool): If True, uses less aggressive centering to preserve more audio content
            
        Returns:
            numpy.ndarray: Preprocessed audio
        """
        try:
            y, sr = self.load_audio(file_path)
            logging.info(f"Processing {file_path}, original length: {len(y)} samples ({len(y)/self.sr:.3f} seconds)")
            
            # For very short audios, skip centering entirely and just normalize
            min_required_samples = max(self.n_fft, self.hop_length * 4)
            if len(y) < min_required_samples * 2:  # If audio is already short
                logging.info(f"Audio already short ({len(y)} samples), skipping centering and just normalizing")
                # Simple peak normalization (avoid pyloudnorm for short audio)
                max_amp = np.max(np.abs(y))
                if max_amp > 0:
                    y_normalized = y / max_amp * 0.9
                else:
                    y_normalized = y
                return y_normalized
            
            # During training, we want to preserve more of the audio
            if training_mode:
                # Use a much lower threshold for sound detection to preserve more audio
                original_threshold = self.sound_threshold
                self.sound_threshold = 0.001  # Even lower threshold for training (previously 0.005)
                
                # Special check - if this would result in very short audio, use original audio
                try:
                    logging.info("Testing if centering would make audio too short...")
                    test_processed = self.center_audio(y)
                    if len(test_processed) < min_required_samples:
                        logging.info(f"Centering would make audio too short ({len(test_processed)} samples), using full audio")
                        # Simple peak normalization (avoid pyloudnorm for short audio)
                        max_amp = np.max(np.abs(y))
                        if max_amp > 0:
                            y_processed = y / max_amp * 0.9
                        else:
                            y_processed = y
                        
                        # Restore threshold
                        self.sound_threshold = original_threshold
                        return y_processed
                    else:
                        # Proceed with centering using lower threshold
                        y_processed = test_processed
                        # Restore threshold
                        self.sound_threshold = original_threshold
                        return y_processed
                except Exception as e:
                    # Check specifically for the pyloudnorm error
                    error_msg = str(e)
                    if "Audio must have length greater than the block size" in error_msg:
                        logging.info(f"Caught pyloudnorm error: {error_msg}, using simple peak normalization instead")
                        # Add to processing errors
                        self.processing_errors.append({
                            'type': 'loudness_normalization',
                            'file': file_path,
                            'message': error_msg,
                            'stage': 'training_preprocessing',
                            'resolution': 'Used peak normalization instead'
                        })
                        
                        # Simple peak normalization (avoid pyloudnorm for short audio)
                        max_amp = np.max(np.abs(y))
                        if max_amp > 0:
                            y_processed = y / max_amp * 0.9
                        else:
                            y_processed = y
                    else:
                        error_msg = f"Error during preprocessing: {e}, using original audio"
                        logging.error(error_msg)
                        # Add to processing errors
                        self.processing_errors.append({
                            'type': 'preprocessing',
                            'file': file_path,
                            'message': error_msg,
                            'stage': 'training_preprocessing',
                            'resolution': 'Used original audio with peak normalization'
                        })
                    
                    self.sound_threshold = original_threshold
                    return y_processed
            else:
                # For inference, use the regular centering approach
                try:
                    y_processed = self.center_audio(y)
                    return y_processed
                except Exception as e:
                    # If centering fails, fall back to basic normalization
                    error_msg = f"Error centering audio for inference: {e}, using simple normalization"
                    logging.error(error_msg)
                    # Add to processing errors
                    self.processing_errors.append({
                        'type': 'centering',
                        'file': file_path,
                        'message': str(e),
                        'stage': 'inference_preprocessing',
                        'resolution': 'Used peak normalization instead'
                    })
                    
                    max_amp = np.max(np.abs(y))
                    if max_amp > 0:
                        y_processed = y / max_amp * 0.9
                    else:
                        y_processed = y
                    return y_processed
        except Exception as e:
            error_msg = f"Error in load_and_preprocess_audio for {file_path}: {e}"
            logging.error(error_msg)
            # Add to processing errors
            self.processing_errors.append({
                'type': 'audio_loading',
                'file': file_path,
                'message': str(e),
                'stage': 'load_and_preprocess_audio',
                'resolution': 'Failed to process audio'
            })
            # Re-raise the exception to be handled by the caller
            raise
    
    def process_audio_for_cnn(self, file_path, training_mode=True):
        """
        Process audio for CNN model input.
        Combines loading, preprocessing, and mel spectrogram extraction.
        
        Args:
            file_path (str): Path to audio file
            training_mode (bool): If True, uses training-specific preprocessing
            
        Returns:
            numpy.ndarray: Mel spectrogram features
        """
        logging.info(f"===== Processing {file_path} for CNN input =====")
        try:
            # Load and preprocess audio
            y_processed = self.load_and_preprocess_audio(file_path, training_mode=training_mode)
            logging.info(f"After preprocessing: length = {len(y_processed)} samples")
            
            # Extract mel spectrogram
            mel_spec = self.extract_mel_spectrogram(y_processed)
            logging.info(f"Final mel spectrogram shape: {mel_spec.shape}")
            
            return mel_spec
        except Exception as e:
            error_msg = f"Error in process_audio_for_cnn for {file_path}: {e}"
            logging.error(error_msg)
            
            # Add detailed error information
            self.processing_errors.append({
                'type': 'cnn_processing',
                'file': file_path,
                'message': str(e),
                'stage': 'process_audio_for_cnn'
            })
            
            # Re-raise the exception to be handled by the caller
            raise
    
    def process_audio_for_rf(self, file_path, return_dict=False, training_mode=True):
        """
        Process audio for RandomForest model input.
        Combines loading, preprocessing, and classical feature extraction.
        
        Args:
            file_path (str): Path to audio file
            return_dict (bool): Whether to return features as a dictionary
            training_mode (bool): If True, uses training-specific preprocessing
            
        Returns:
            dict or numpy.ndarray: Audio features
        """
        y_processed = self.load_and_preprocess_audio(file_path, training_mode=training_mode)
        features = self.extract_classical_features(y_processed, return_dict=return_dict)
        return features
    
    def analyze_audio_directory(self, audio_dir, print_stats=True):
        """
        Analyze all audio files in a directory and return statistics about their length.
        This helps diagnose issues with audio files being too short.
        
        Args:
            audio_dir (str): Directory containing class-specific audio folders
            print_stats (bool): Whether to print statistics to console
            
        Returns:
            dict: Statistics about audio files
        """
        stats = {
            'total_files': 0,
            'too_short_files': 0,
            'valid_files': 0,
            'min_length': float('inf'),
            'max_length': 0,
            'avg_length': 0,
            'by_class': {}
        }
        
        # Check if the directory exists
        if not os.path.exists(audio_dir):
            logging.error(f"Audio directory {audio_dir} does not exist")
            return stats
        
        # Get class directories
        class_dirs = [d for d in os.listdir(audio_dir) 
                    if os.path.isdir(os.path.join(audio_dir, d))]
        
        # Calculate minimum required length
        min_required_samples = max(self.n_fft, self.hop_length * 4)
        
        # Process each class
        total_length = 0
        for class_dir in class_dirs:
            class_path = os.path.join(audio_dir, class_dir)
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            class_stats = {
                'total_files': len(wav_files),
                'too_short_files': 0,
                'valid_files': 0,
                'min_length': float('inf'),
                'max_length': 0,
                'avg_length': 0
            }
            
            class_total_length = 0
            for wav_file in wav_files:
                file_path = os.path.join(class_path, wav_file)
                
                try:
                    # Load audio
                    y, sr = self.load_audio(file_path)
                    
                    # Update stats
                    length = len(y)
                    stats['total_files'] += 1
                    class_stats['total_files'] += 1
                    class_total_length += length
                    total_length += length
                    
                    # Check if too short
                    if length < min_required_samples:
                        stats['too_short_files'] += 1
                        class_stats['too_short_files'] += 1
                    else:
                        stats['valid_files'] += 1
                        class_stats['valid_files'] += 1
                    
                    # Update min/max length
                    stats['min_length'] = min(stats['min_length'], length)
                    stats['max_length'] = max(stats['max_length'], length)
                    
                    class_stats['min_length'] = min(class_stats['min_length'], length)
                    class_stats['max_length'] = max(class_stats['max_length'], length)
                    
                except Exception as e:
                    logging.error(f"Error analyzing {file_path}: {e}")
            
            # Calculate average length for this class
            if class_stats['total_files'] > 0:
                class_stats['avg_length'] = class_total_length / class_stats['total_files']
            
            # Convert sample lengths to seconds
            class_stats['min_length_sec'] = class_stats['min_length'] / self.sr
            class_stats['max_length_sec'] = class_stats['max_length'] / self.sr
            class_stats['avg_length_sec'] = class_stats['avg_length'] / self.sr
            
            # Add to overall stats
            stats['by_class'][class_dir] = class_stats
        
        # Calculate overall average length
        if stats['total_files'] > 0:
            stats['avg_length'] = total_length / stats['total_files']
        
        # Convert sample lengths to seconds
        stats['min_length_sec'] = stats['min_length'] / self.sr
        stats['max_length_sec'] = stats['max_length'] / self.sr
        stats['avg_length_sec'] = stats['avg_length'] / self.sr
        
        # Required length in seconds
        stats['required_length_sec'] = min_required_samples / self.sr
        
        # Print stats if requested
        if print_stats:
            print("\n===== AUDIO ANALYSIS RESULTS =====")
            print(f"Total files: {stats['total_files']}")
            print(f"Too short files: {stats['too_short_files']} ({stats['too_short_files']/stats['total_files']*100:.1f}%)")
            print(f"Valid files: {stats['valid_files']} ({stats['valid_files']/stats['total_files']*100:.1f}%)")
            print(f"Min length: {stats['min_length']} samples ({stats['min_length_sec']:.3f} sec)")
            print(f"Max length: {stats['max_length']} samples ({stats['max_length_sec']:.3f} sec)")
            print(f"Avg length: {stats['avg_length']:.1f} samples ({stats['avg_length_sec']:.3f} sec)")
            print(f"Required length: {min_required_samples} samples ({stats['required_length_sec']:.3f} sec)")
            print("\nBy class:")
            
            for class_dir, class_stats in stats['by_class'].items():
                print(f"\n  {class_dir}:")
                print(f"    Total files: {class_stats['total_files']}")
                print(f"    Too short files: {class_stats['too_short_files']} ({class_stats['too_short_files']/class_stats['total_files']*100:.1f}% if applicable)")
                print(f"    Min length: {class_stats['min_length']} samples ({class_stats['min_length_sec']:.3f} sec)")
                print(f"    Max length: {class_stats['max_length']} samples ({class_stats['max_length_sec']:.3f} sec)")
                print(f"    Avg length: {class_stats['avg_length']:.1f} samples ({class_stats['avg_length_sec']:.3f} sec)")
        
        return stats 