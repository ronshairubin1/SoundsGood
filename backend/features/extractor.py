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
from pathlib import Path

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
    
    def _check_cache(self, file_hash):
        """
        Check if features exist in cache.
        
        Args:
            file_hash: Hash of the file path
            
        Returns:
            Cached features if found, None otherwise
        """
        if not self.use_cache:
            return None
        
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.npz")
        
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                features = data['features'].item()
                logging.info(f"Loaded features from cache: {cache_path}")
                return features
            except Exception as e:
                logging.warning(f"Error loading from cache {cache_path}: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, file_hash, features):
        """
        Save features to cache.
        
        Args:
            file_hash: Hash of the file path
            features: Feature dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_cache:
            return False
        
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.npz")
        
        try:
            # Save as compressed npz file
            np.savez_compressed(cache_path, features=features)
            logging.info(f"Saved features to cache: {cache_path}")
            return True
        except Exception as e:
            logging.warning(f"Error saving to cache {cache_path}: {e}")
            return False
    
    def extract_features(self, audio_source, is_file=True):
        """
        Extract all possible features from an audio source.
        
        Args:
            audio_source: Path to audio file or audio data as numpy array
            is_file: Whether audio_source is a file path
            
        Returns:
            Dictionary of extracted features
        """
        # Check cache if audio_source is a file
        file_hash = None
        if is_file:
            file_hash = self._hash_file_path(audio_source)
            cached_features = self._check_cache(file_hash)
            if cached_features:
                return cached_features
        
        try:
            # Load audio if needed
            if is_file:
                audio, sr = librosa.load(audio_source, sr=self.sample_rate)
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
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=self.n_mels, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spectrogram'] = mel_spec_db
            
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
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral']['contrast_mean'] = float(np.mean(contrast))
            features['spectral']['contrast_std'] = float(np.std(contrast))
            
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
                self._save_to_cache(file_hash, features)
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
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
            return None
            
        if model_type == 'cnn':
            # Return mel spectrogram for CNN
            return features['mel_spectrogram']
            
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
                    
                    # Then extract model-specific features
                    features = self.extract_features_for_model(all_features, model_type)
                    
                    # Skip files with invalid features
                    if features is None:
                        logging.warning(f"Skipping {file_path}: Failed to extract features")
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