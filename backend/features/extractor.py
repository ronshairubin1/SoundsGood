#!/usr/bin/env python3
"""
Unified Feature Extractor

This module provides a single, unified implementation for extracting features from audio data.
It extracts ALL possible features at once, which can then be selectively used by different models.

Features extracted include:
1. Mel-spectrograms (for CNN models)
2. MFCCs and statistical features (for RF models)
3. Advanced features (rhythm, spectral, tonal)

The goal is to extract features once and store them, rather than re-extracting
for each model type.
"""

import numpy as np
import librosa
import os
import json
import logging
import time
import hashlib
import h5py  # Using HDF5 for better feature caching
from pathlib import Path

# Explicitly import librosa exceptions
from librosa.util.exceptions import ParameterError as LibrosaParameterError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)

class FeatureExtractor:
    """
    Unified implementation for extracting all possible features from audio data.
    Extracts features once, which can then be used by different models.
    """
    
    def __init__(self, sample_rate=8000, n_mfcc=13, n_mels=64, n_fft=1024, hop_length=256, 
                cache_dir="backend/data/features/cache", use_cache=True):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Sample rate for audio processing
            n_mfcc: Number of MFCC coefficients to extract
            n_mels: Number of mel bands for spectrograms
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            cache_dir: Directory for caching features
            use_cache: Whether to use caching
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Set up caching
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Create cache directory if needed
        if self.use_cache and cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def _hash_file_path(self, file_path):
        """
        Create a hash of the file path for caching.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String hash of the file path
        """
        return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _check_cache(self, file_hash, model_type=None):
        """
        Check if features exist in cache and validate based on model type.
        
        Args:
            file_hash: Hash of the file path
            model_type: Type of model ('cnn', 'rf', etc.) to validate specific features
            
        Returns:
            Cached features if found and valid, None otherwise
        """
        if not self.use_cache:
            return None
        
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.h5")
        logging.info(f"[CACHE] Checking HDF5 cache at: {cache_path}, model_type={model_type}")
        print(f"[EXTRACTOR] Checking cache for: {file_hash}.h5")
        
        if os.path.exists(cache_path):
            try:
                # Load from HDF5 file
                logging.info(f"[CACHE] HDF5 cache file exists, attempting to load features")
                print(f"[EXTRACTOR] Found HDF5 cache file, loading")
                features = {}
                
                with h5py.File(cache_path, 'r') as f:
                    # Handle the metadata group
                    if 'metadata' in f:
                        features['metadata'] = {}
                        for key in f['metadata']:
                            features['metadata'][key] = f['metadata'][key][()]
                    
                    # Handle other feature groups
                    for group_name in f.keys():
                        if group_name == 'metadata':
                            continue
                            
                        if isinstance(f[group_name], h5py.Group):
                            # Handle nested dictionaries (like 'statistical', 'rhythm', etc.)
                            features[group_name] = {}
                            for key in f[group_name]:
                                features[group_name][key] = f[group_name][key][()]
                        else:
                            # Handle direct datasets (like 'mel_spectrogram')
                            features[group_name] = f[group_name][()]
                
                # Log the structure of what was loaded
                feature_keys = list(features.keys())
                logging.info(f"[CACHE] Successfully loaded features from HDF5 cache: {cache_path}")
                logging.info(f"[CACHE] Loaded feature keys: {feature_keys}")
                print(f"[EXTRACTOR] Successfully loaded features from cache: {feature_keys}")
                
                # Perform model-specific validation
                if model_type == 'cnn' and 'mel_spectrogram' not in features:
                    logging.warning(f"[CACHE] Cache missing mel_spectrogram required for CNN, will regenerate")
                    print(f"[EXTRACTOR] Cache missing CNN features, regenerating")
                    return None
                
                # General validation for all models
                if 'mel_spectrogram' not in features:
                    logging.warning(f"[CACHE] Cache is missing mel_spectrogram feature")
                    
                return features
            except Exception as e:
                logging.error(f"[CACHE] Error loading from HDF5 cache {cache_path}: {e}")
                logging.error(f"[CACHE] Error type: {type(e).__name__}, Details: {str(e)}")
                print(f"[EXTRACTOR] Failed to load from HDF5 cache: {type(e).__name__}: {str(e)}")
                # If cache file is corrupted, remove it
                try:
                    os.remove(cache_path)
                    logging.info(f"[CACHE] Removed corrupted HDF5 cache file: {cache_path}")
                    print(f"[EXTRACTOR] Removed corrupted cache file")
                except:
                    pass
                return None
        
        # Check for legacy cache formats (for backward compatibility)
        legacy_cache_path = os.path.join(self.cache_dir, f"{file_hash}.npz")
        if os.path.exists(legacy_cache_path):
            logging.info(f"[CACHE] Found legacy NPZ cache file, will convert to HDF5: {legacy_cache_path}")
            print(f"[EXTRACTOR] Found legacy cache file, converting to HDF5 format")
            try:
                # Try to load with numpy
                data = np.load(legacy_cache_path, allow_pickle=True)
                features = data['features'].item()
                # Save to new HDF5 format for future use
                self._save_to_cache(file_hash, features)
                return features
            except Exception as e:
                logging.warning(f"[CACHE] Could not convert legacy NPZ cache: {e}")
                logging.warning(f"[CACHE] Error type: {type(e).__name__}, Details: {str(e)}")
                print(f"[EXTRACTOR] Failed to convert legacy cache: {type(e).__name__}: {str(e)}")
        
        logging.info(f"[CACHE] No HDF5 or legacy cache files found for: {file_hash}")
        print(f"[EXTRACTOR] No cache files found, will extract features")
        return None
    
    def _save_to_cache(self, file_hash, features):
        """
        Save features to cache using HDF5.
        
        Args:
            file_hash: Hash of the file path
            features: Feature dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_cache:
            return False
        
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.h5")
        logging.info(f"[CACHE] Attempting to save features to HDF5 cache: {cache_path}")
        feature_keys = list(features.keys())
        logging.info(f"[CACHE] Saving feature keys: {feature_keys}")
        print(f"[EXTRACTOR] Saving features to cache: {feature_keys}")
        
        try:
            with h5py.File(cache_path, 'w') as f:
                # Store metadata
                if 'metadata' in features:
                    metadata_group = f.create_group('metadata')
                    for key, value in features['metadata'].items():
                        metadata_group.create_dataset(key, data=value)
                
                # Store all other features
                for key, value in features.items():
                    if key == 'metadata':
                        continue  # Already handled
                        
                    if isinstance(value, dict):
                        # Create a group for nested dictionaries
                        group = f.create_group(key)
                        for subkey, subvalue in value.items():
                            # Handle various data types
                            if isinstance(subvalue, (int, float, bool, np.number)):
                                group.create_dataset(subkey, data=subvalue)
                            elif isinstance(subvalue, np.ndarray):
                                group.create_dataset(subkey, data=subvalue, compression="gzip")
                            else:
                                # Convert other types to string representation
                                group.create_dataset(subkey, data=str(subvalue))
                    elif isinstance(value, np.ndarray):
                        # Store array directly with compression
                        f.create_dataset(key, data=value, compression="gzip")
                    else:
                        # Store other types directly
                        f.create_dataset(key, data=value)
                        
            logging.info(f"[CACHE] Successfully saved features to HDF5 cache: {cache_path}")
            print(f"[EXTRACTOR] Successfully saved features to cache")
            return True
        except Exception as e:
            logging.error(f"[CACHE] Error saving to HDF5 cache {cache_path}: {e}")
            import traceback
            logging.error(f"[CACHE] Traceback: {traceback.format_exc()}")
            print(f"[EXTRACTOR] Failed to save to cache: {type(e).__name__}: {str(e)}")
            return False
    
    def extract_features(self, audio_source, is_file=True, model_type=None, force_update=False):
        """
        Extract all possible features from an audio source.
        
        Args:
            audio_source: Path to audio file or audio data as numpy array
            is_file: Whether audio_source is a file path
            model_type: Optional model type to validate model-specific features
            force_update: If True, bypass cache and regenerate features
            
        Returns:
            Dictionary of extracted features
        """
        # Check cache if audio_source is a file
        file_hash = None
        if is_file:
            file_hash = self._hash_file_path(audio_source)
            # Only check cache if force_update is False
            if not force_update:
                cached_features = self._check_cache(file_hash, model_type)
                if cached_features:
                    return cached_features
            else:
                logging.info(f"Force update requested, bypassing cache for {os.path.basename(audio_source)}")
        
        try:
            # Load audio if needed with enhanced logging
            if is_file:
                file_name = os.path.basename(audio_source)
                file_size = os.path.getsize(audio_source) if os.path.exists(audio_source) else 'file not found'
                logging.info(f"Loading audio from file: {file_name}, size: {file_size} bytes")
                print(f"[EXTRACTOR] Loading audio from file: {file_name}")
                
                try:
                    audio, sr = librosa.load(audio_source, sr=self.sample_rate)
                    audio_duration = len(audio) / sr
                    audio_info = f"Audio loaded successfully: {len(audio)} samples, {sr}Hz, duration: {audio_duration:.2f}s"
                    audio_info += f", min: {np.min(audio):.3f}, max: {np.max(audio):.3f}"
                    logging.info(audio_info)
                    print(f"[EXTRACTOR] {audio_info}")
                    
                    # Validate audio data
                    if len(audio) == 0:
                        logging.error(f"Empty audio file: {file_name}")
                        return None
                        
                    if np.isnan(audio).any():
                        logging.error(f"Audio contains NaN values: {file_name}")
                        return None
                        
                    if audio_duration < 0.1:
                        logging.warning(f"Audio is very short ({audio_duration:.3f}s): {file_name}")
                except Exception as e:
                    logging.error(f"Failed to load audio: {file_name}, error: {str(e)}")
                    print(f"[EXTRACTOR] Failed to load audio: {file_name}")
                    raise
            else:
                audio = audio_source
                sr = self.sample_rate
                
            # Initialize features dictionary
            features = {
                'metadata': {
                    'sample_rate': sr,
                    'length_samples': len(audio),
                    'extraction_time': time.time()
                }
            }
            
            # --------------------
            # Mel spectrogram (for CNN)
            # --------------------
            logging.info(f"Extracting mel spectrogram with n_mels={self.n_mels}, n_fft={self.n_fft}, hop_length={self.hop_length}")
            
            try:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=sr, 
                    n_mels=self.n_mels, 
                    n_fft=self.n_fft, 
                    hop_length=self.hop_length
                )
                logging.info(f"Mel spectrogram extracted: shape={mel_spec.shape}")
                
                # Convert to dB scale
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                features['mel_spectrogram'] = mel_spec_db
                
                # Store file path in metadata if this is a file
                if is_file:
                    if 'metadata' not in features:
                        features['metadata'] = {}
                    features['metadata']['file_path'] = audio_source
                
                # Verify the mel_spectrogram was stored correctly
                if 'mel_spectrogram' not in features or features['mel_spectrogram'] is None:
                    logging.error(f"Failed to store mel_spectrogram in features dictionary")
                    raise ValueError("Failed to store mel_spectrogram")
                else:
                    logging.info(f"Successfully stored mel_spectrogram with shape {features['mel_spectrogram'].shape}")
            except Exception as e:
                logging.error(f"Failed to extract mel spectrogram: {str(e)}")
                print(f"[EXTRACTOR] Error extracting mel spectrogram: {str(e)}")
                raise
            
            # --------------------
            # MFCC features (for RF)
            # --------------------
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Calculate delta and delta2 features
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Store raw MFCC data
            features['mfccs'] = mfccs
            features['mfcc_delta'] = mfcc_delta
            features['mfcc_delta2'] = mfcc_delta2
            
            # Store statistical features
            features['statistical'] = {}
            
            # MFCC statistics
            for i in range(self.n_mfcc):
                features['statistical'][f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features['statistical'][f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                features['statistical'][f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
                features['statistical'][f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
                features['statistical'][f'mfcc_delta2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
                features['statistical'][f'mfcc_delta2_{i}_std'] = float(np.std(mfcc_delta2[i]))
            
            # --------------------
            # Formant approximation
            # --------------------
            formants = librosa.effects.preemphasis(audio)
            features['statistical']['formant_mean'] = float(np.mean(formants))
            features['statistical']['formant_std'] = float(np.std(formants))
            
            # --------------------
            # Pitch
            # --------------------
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_vals = pitches[magnitudes > np.median(magnitudes)]
            if len(pitch_vals) > 0:
                features['statistical']['pitch_mean'] = float(np.mean(pitch_vals))
                features['statistical']['pitch_std'] = float(np.std(pitch_vals))
            else:
                features['statistical']['pitch_mean'] = 0.0
                features['statistical']['pitch_std'] = 0.0
            
            # --------------------
            # Spectral Centroid
            # --------------------
            cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['statistical']['spectral_centroid_mean'] = float(np.mean(cent))
            features['statistical']['spectral_centroid_std'] = float(np.std(cent))
            
            # --------------------
            # Zero Crossing Rate
            # --------------------
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['statistical']['zcr_mean'] = float(np.mean(zcr))
            features['statistical']['zcr_std'] = float(np.std(zcr))
            
            # --------------------
            # RMS Energy
            # --------------------
            rms = librosa.feature.rms(y=audio)
            features['statistical']['rms_mean'] = float(np.mean(rms))
            features['statistical']['rms_std'] = float(np.std(rms))
            
            # --------------------
            # Spectral Rolloff
            # --------------------
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['statistical']['rolloff_mean'] = float(np.mean(rolloff))
            features['statistical']['rolloff_std'] = float(np.std(rolloff))
            
            # --------------------
            # Advanced Features - Rhythm
            # --------------------
            features['rhythm'] = {}
            
            # Tempo and beat extraction
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            features['rhythm']['tempo'] = float(tempo)
            
            # Beat strength and clarity
            features['rhythm']['onset_strength_mean'] = float(np.mean(onset_env))
            features['rhythm']['onset_strength_std'] = float(np.std(onset_env))
            
            # Pulse clarity - how clearly the rhythm is perceived
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            features['rhythm']['pulse_clarity'] = float(np.max(pulse))
            
            # Number of detected beats
            features['rhythm']['beat_count'] = len(beat_frames)
            
            # --------------------
            # Advanced Features - Spectral
            # --------------------
            features['spectral'] = {}
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=audio)
            features['spectral']['flatness_mean'] = float(np.mean(flatness))
            features['spectral']['flatness_std'] = float(np.std(flatness))
            
            # Spectral contrast - adjust parameters for low sample rate audio
            try:
                # Use fewer bands and lower fmin for low sample rate audio
                contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=4, fmin=60)
                features['spectral']['contrast_mean'] = float(np.mean(contrast))
                features['spectral']['contrast_std'] = float(np.std(contrast))
            except LibrosaParameterError as e:
                logging.warning(f"Skipping spectral contrast due to parameter error: {str(e)}")
                features['spectral']['contrast_mean'] = 0.0
                features['spectral']['contrast_std'] = 0.0
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features['spectral']['bandwidth_mean'] = float(np.mean(bandwidth))
            features['spectral']['bandwidth_std'] = float(np.std(bandwidth))
            
            # --------------------
            # Advanced Features - Tonal
            # --------------------
            features['tonal'] = {}
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['tonal']['chroma_mean'] = float(np.mean(chroma))
            
            # Individual chroma features (12 pitch classes)
            for i in range(12):
                features['tonal'][f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
                features['tonal'][f'chroma_{i}_std'] = float(np.std(chroma[i]))
            
            # Tonal centroid features (tonnetz)
            try:
                tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
                for i in range(min(6, tonnetz.shape[0])):
                    features['tonal'][f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
                    features['tonal'][f'tonnetz_{i}_std'] = float(np.std(tonnetz[i]))
            except Exception as e:
                logging.warning(f"Error extracting tonnetz features: {e}")
                for i in range(6):
                    features['tonal'][f'tonnetz_{i}_mean'] = 0.0
                    features['tonal'][f'tonnetz_{i}_std'] = 0.0
            
            # Harmonic/percussive separation
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(audio)
                features['tonal']['harmonic_energy'] = float(np.sum(y_harmonic**2))
                features['tonal']['percussive_energy'] = float(np.sum(y_percussive**2))
                features['tonal']['harmonic_percussive_ratio'] = float(features['tonal']['harmonic_energy'] / 
                                                                 (features['tonal']['percussive_energy'] + 1e-8))
            except Exception as e:
                logging.warning(f"Error extracting harmonic/percussive features: {e}")
                features['tonal']['harmonic_energy'] = 0.0
                features['tonal']['percussive_energy'] = 0.0
                features['tonal']['harmonic_percussive_ratio'] = 0.0
            
            # Cache features if audio_source is a file
            if is_file and file_hash:
                feature_groups = list(features.keys())
                feature_summary = f"Feature extraction completed successfully, extracted {len(feature_groups)} feature groups: {feature_groups}"
                logging.info(feature_summary)
                print(f"[EXTRACTOR] {feature_summary}")
                
                # Validate extracted features before caching
                feature_validation_passed = True
                for key, value in features.items():
                    if key != 'metadata' and isinstance(value, (np.ndarray, list)):
                        if isinstance(value, np.ndarray) and (np.isnan(value).any() or np.isinf(value).any()):
                            logging.warning(f"Feature '{key}' contains NaN or Inf values")
                            feature_validation_passed = False
                
                if feature_validation_passed:
                    logging.info(f"All features validated successfully, caching results")
                    print(f"[EXTRACTOR] Caching features for {os.path.basename(audio_source) if is_file else 'audio data'}")
                    self._save_to_cache(file_hash, features)
                else:
                    logging.warning(f"Features contain NaN or Inf values, skipping cache")
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            print(f"[EXTRACTOR] Feature extraction failed: {type(e).__name__}: {str(e)}")
            
            # Log specialized error diagnostics for common error types
            if isinstance(e, LibrosaParameterError):
                logging.error(f"Librosa parameter error - likely invalid audio format or empty audio")
            elif isinstance(e, ValueError) and "empty" in str(e).lower():
                logging.error(f"Empty audio data detected")
            elif isinstance(e, np.linalg.LinAlgError):
                logging.error(f"Linear algebra error - likely due to invalid audio data")
            elif isinstance(e, (OSError, IOError)):
                logging.error(f"File system error - possible file access, permission, or corruption issue")
            
            # Add audio metadata to help debug
            if 'audio' in locals() and isinstance(audio, np.ndarray):
                logging.error(f"Audio metadata at error: shape={audio.shape}, min={np.min(audio) if audio.size > 0 else 'empty'}, "  
                              f"max={np.max(audio) if audio.size > 0 else 'empty'}, "  
                              f"has_nan={np.isnan(audio).any() if audio.size > 0 else 'N/A'}")
            
            return None
    
    def extract_features_for_model(self, features, model_type):
        """
        Extract model-specific features from the full feature set.
        
        Args:
            features: Full features dictionary
            model_type: Model type ('cnn', 'rf', or 'ensemble')
            
        Returns:
            Model-specific features
        """
        if features is None:
            logging.error(f"Cannot extract model features, feature dictionary is None")
            return None
        
        try:    
            if not isinstance(features, dict):
                logging.error(f"Features must be a dictionary, got {type(features)}")
                return None
                
            feature_keys = list(features.keys())
            logging.info(f"Preparing features for model: {model_type}, available feature keys: {feature_keys}")
            print(f"[EXTRACTOR] Preparing features for model: {model_type}")
            
            if model_type == 'cnn':
                # Check if mel spectrogram exists
                if 'mel_spectrogram' not in features:
                    logging.error(f"Missing 'mel_spectrogram' key in features. Available keys: {feature_keys}")
                    print(f"[EXTRACTOR] Error: Missing mel_spectrogram for CNN")
                    raise KeyError("Missing 'mel_spectrogram' key in features")
                    
                # Get the mel spectrogram
                mel_spec = features['mel_spectrogram']
                
                # Ensure the mel spectrogram has the channel dimension for CNN
                if isinstance(mel_spec, np.ndarray) and len(mel_spec.shape) == 2:
                    logging.info(f"Adding channel dimension to mel spectrogram: {mel_spec.shape} -> {mel_spec.shape + (1,)}")
                    mel_spec = np.expand_dims(mel_spec, axis=-1)
                
                # Add file path to features if it exists in metadata for error tracking
                result = mel_spec
                
                # For training_service.py, we need to return a dictionary with both mel_spectrogram and file_path
                if isinstance(result, np.ndarray) and 'metadata' in features and 'file_path' in features['metadata']:
                    return {
                        'mel_spectrogram': result,
                        'file_path': features['metadata']['file_path']
                    }
                return result
            
            elif model_type == 'rf':
                # Return statistical features for RF
                return features['statistical']
            
            elif model_type == 'ensemble':
                # Return both mel spectrogram and statistical features
                return {
                    'cnn': features['mel_spectrogram'],
                    'rf': features['statistical']
                }
            
            else:
                logging.warning(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logging.error(f"Error extracting features for model {model_type}: {str(e)}")
            return None
    
    def get_feature_names(self):
        """
        Return a list of feature names for RF model features.
        This is maintained for backward compatibility with AudioFeatureExtractor.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # MFCC features (mean and std)
        for i in range(self.n_mfcc):
            feature_names.extend([
                f'mfcc_{i}_mean', f'mfcc_{i}_std',
                f'mfcc_delta_{i}_mean', f'mfcc_delta_{i}_std',
                f'mfcc_delta2_{i}_mean', f'mfcc_delta2_{i}_std'
            ])
        
        # Spectral features
        feature_names.extend([
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_contrast_mean', 'spectral_contrast_std',
            'spectral_flatness_mean', 'spectral_flatness_std'
        ])
        
        # Temporal features
        feature_names.extend([
            'zero_crossing_rate_mean', 'zero_crossing_rate_std',
            'rms_energy_mean', 'rms_energy_std',
            'tempo', 'beat_strength',
            'chroma_mean', 'chroma_std',
            'harmonic_mean', 'harmonic_std',
            'percussive_mean', 'percussive_std'
        ])
        
        return feature_names
    
    def extract_features_from_directory(self, input_dir, output_dir=None):
        """
        Extract features from all audio files in a directory.
        
        Args:
            input_dir (str): Directory containing audio files
            output_dir (str, optional): Directory to save extracted features
            
        Returns:
            dict: Statistics about the extraction process
        """
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        all_features = {}
        stats = {
            'total_processed': 0,
            'total_skipped': 0,
            'error_files': [],
            'error_types': {}
        }
        
        # Process all wav files
        for file_path in input_dir.glob("*.wav"):
            try:
                features = self.extract_features(file_path)
                
                if features:
                    all_features[str(file_path)] = features
                    stats['total_processed'] += 1
                    
                    # Save features if output directory is provided
                    if output_dir:
                        output_path = output_dir / f"{file_path.stem}_features.npz"
                        np.savez_compressed(output_path, features=features)
                        logging.info(f"Saved features to {output_path}")
                else:
                    stats['total_skipped'] += 1
                    stats['error_files'].append(str(file_path))
                    stats['error_types'].setdefault('feature_extraction_failed', 0)
                    stats['error_types']['feature_extraction_failed'] += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                stats['total_skipped'] += 1
                stats['error_files'].append(str(file_path))
                stats['error_types'].setdefault('processing_error', 0)
                stats['error_types']['processing_error'] += 1
        
        logging.info(f"Extracted features from {stats['total_processed']} files, skipped {stats['total_skipped']} files")
        return all_features
    
    def batch_extract_features(self, audio_dir, class_dirs, model_type='cnn', progress_callback=None):
        """
        Extract features from a batch of audio files for training.
        
        Args:
            audio_dir (str): Base directory containing class folders
            class_dirs (list): List of class directory names
            model_type (str): Type of model to extract features for ('cnn', 'rf', 'ensemble')
            progress_callback (callable): Optional callback for progress reporting
            
        Returns:
            tuple: (X, y, class_names, stats) where:
                X is a numpy array of features
                y is a numpy array of class labels
                class_names is a list of class names
                stats is a dictionary of extraction statistics
        """
        X = []
        y = []
        stats = {
            'original_counts': {},
            'processed_counts': {},
            'skipped_counts': {},
            'total_processed': 0,
            'total_skipped': 0,
            'error_files': [],
            'error_types': {}
        }
        
        logging.info(f"Batch extracting features for {model_type} model from {len(class_dirs)} classes")
        
        # Calculate total files for progress tracking
        total_files = 0
        for class_dir in class_dirs:
            class_path = os.path.join(audio_dir, class_dir)
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            total_files += len(wav_files)
            stats['original_counts'][class_dir] = len(wav_files)
            stats['processed_counts'][class_dir] = 0
            stats['skipped_counts'][class_dir] = 0
        
        files_processed = 0
        
        # Process each class
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(audio_dir, class_dir)
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            if not wav_files:
                logging.warning(f"No .wav files found in {class_path}")
                continue
            
            logging.info(f"Processing class {class_dir}: Found {len(wav_files)} audio files")
            
            # Process each file in the class directory
            for file_idx, wav_file in enumerate(wav_files):
                file_path = os.path.join(class_path, wav_file)
                
                try:
                    # Extract all features first
                    logging.debug(f"Extracting features from {file_path}")
                    all_features = self.extract_features(file_path, is_file=True)
                    
                    # Skip if no features were extracted
                    if all_features is None:
                        logging.warning(f"Skipping {file_path}: No features were extracted")
                        stats['skipped_counts'][class_dir] += 1
                        stats['total_skipped'] += 1
                        stats['error_files'].append(file_path)
                        stats['error_types'].setdefault('feature_extraction_failed', 0)
                        stats['error_types']['feature_extraction_failed'] += 1
                        continue
                        
                    # Ensure that mel_spectrogram is present for CNN models
                    if model_type == 'cnn' and 'mel_spectrogram' not in all_features:
                        logging.error(f"Missing mel_spectrogram in features for {file_path}")
                        stats['skipped_counts'][class_dir] += 1
                        stats['total_skipped'] += 1
                        stats['error_files'].append(file_path)
                        stats['error_types'].setdefault('missing_mel_spectrogram', 0)
                        stats['error_types']['missing_mel_spectrogram'] += 1
                        continue
                    
                    # For CNN models, store a complete feature dictionary with mel_spectrogram and file_path
                    if model_type == 'cnn':
                        features = {
                            'mel_spectrogram': all_features['mel_spectrogram'],
                            'file_path': file_path
                        }
                    else:
                        # For other models, use the extract_features_for_model method
                        features = self.extract_features_for_model(all_features, model_type)
                    
                    # Skip files with invalid features
                    if features is None:
                        logging.warning(f"Skipping {file_path}: Failed to extract model-specific features")
                        stats['skipped_counts'][class_dir] += 1
                        stats['total_skipped'] += 1
                        stats['error_files'].append(file_path)
                        stats['error_types'].setdefault('feature_extraction_failed', 0)
                        stats['error_types']['feature_extraction_failed'] += 1
                        continue
                    
                    # Add to dataset
                    X.append(features)
                    y.append(class_dir)
                    
                    # Update statistics
                    stats['processed_counts'][class_dir] += 1
                    stats['total_processed'] += 1
                    files_processed += 1
                    
                    # Report progress if callback provided
                    if progress_callback and (file_idx % 5 == 0 or file_idx == len(wav_files) - 1):
                        progress_percentage = int((files_processed / total_files) * 100)
                        progress_callback(
                            progress_percentage,
                            f"Processed {files_processed}/{total_files} files ({class_dir})"
                        )
                    
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
                    stats['skipped_counts'][class_dir] += 1
                    stats['total_skipped'] += 1
                    stats['error_files'].append(file_path)
                    stats['error_types'].setdefault('processing_error', 0)
                    stats['error_types']['processing_error'] += 1
            
        # Convert to numpy arrays
        X = np.array(X) if X else np.array([])
        y = np.array(y) if y else np.array([])
        
        logging.info(f"Extracted features for {len(X)} samples, skipped {stats['total_skipped']} files")
        
        return X, y, class_dirs, stats 