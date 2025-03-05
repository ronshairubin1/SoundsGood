import os
import logging
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from collections import Counter

from src.core.audio.processor import AudioProcessor
from backend.features.extractor import FeatureExtractor
from src.ml.augmentation_manager import augment_audio_with_repetitions
from config import Config

class DatasetPreparation:
    """
    Centralized dataset preparation for all model types.
    
    This class handles data collection, feature extraction, and dataset preparation
    for CNN and RF models, ensuring consistent preprocessing between model types.
    """
    
    def __init__(self, sample_rate=8000):
        """
        Initialize the dataset preparation service.
        
        Args:
            sample_rate (int): Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        # Initialize AudioProcessor for CNN spectrograms
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            enable_loudness_normalization=False  # Disable loudness normalization by default
        )
        # Initialize FeatureExtractor for features
        self.feature_extractor = FeatureExtractor(sr=sample_rate)
        # Stats tracking
        self.file_errors = []
        self.error_logs = []
        
    def prepare_dataset_for_cnn(self, audio_dir, classes=None):
        """
        Prepare a dataset of mel spectrograms for CNN training.
        
        Args:
            audio_dir (str): Directory containing class-specific audio folders
            classes (list): List of class names to include (optional)
            
        Returns:
            tuple: (X, y, class_names, stats)
        """
        X = []
        y = []
        class_names = []
        stats = {
            'original_counts': {},
            'augmented_counts': {},
            'processed_counts': {},
            'skipped_counts': {},
            'stretched_counts': {},  # Track files that were time-stretched
            'total_processed': 0,    # Total files successfully processed
            'total_skipped': 0,      # Total files skipped
            'total_stretched': 0,    # Total files stretched
            'total_augmented': 0,    # Track total augmented files
            'error_files': [],       # Track files with errors
            'error_types': {}        # Track types of errors
        }
        
        # Clear previous errors
        self.file_errors = []
        
        # Check if the directory exists
        if not os.path.exists(audio_dir):
            error_msg = f"Audio directory {audio_dir} does not exist"
            logging.error(error_msg)
            self.error_logs.append({'level': 'ERROR', 'message': error_msg})
            return None, None, [], stats
        
        # Get class directories - either use provided classes or scan directory
        if classes:
            # Filter out non-existent class directories
            class_dirs = []
            for class_dir in classes:
                class_path = os.path.join(audio_dir, class_dir)
                if os.path.isdir(class_path):
                    class_dirs.append(class_dir)
                else:
                    error_msg = f"Class directory {class_path} does not exist"
                    logging.warning(error_msg)
                    self.error_logs.append({'level': 'WARNING', 'message': error_msg})
                    
            if not class_dirs:
                error_msg = f"None of the specified class directories exist in {audio_dir}"
                logging.error(error_msg)
                self.error_logs.append({'level': 'ERROR', 'message': error_msg})
                return None, None, [], stats
        else:
            # Get subdirectories (each corresponds to a class)
            class_dirs = [d for d in os.listdir(audio_dir) 
                        if os.path.isdir(os.path.join(audio_dir, d))]
            
            if not class_dirs:
                error_msg = f"No class subdirectories found in {audio_dir}"
                logging.error(error_msg)
                self.error_logs.append({'level': 'ERROR', 'message': error_msg})
                return None, None, [], stats
        
        # Sort for consistent class indices
        class_dirs.sort()
        class_names = class_dirs
        
        # Process each class
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(audio_dir, class_dir)
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            if not wav_files:
                error_msg = f"No .wav files found in {class_path}"
                logging.warning(error_msg)
                self.error_logs.append({'level': 'WARNING', 'message': error_msg})
                continue
            
            # Track original file count
            stats['original_counts'][class_dir] = len(wav_files)
            stats['processed_counts'][class_dir] = 0
            stats['skipped_counts'][class_dir] = 0
            stats['stretched_counts'][class_dir] = 0
            stats['augmented_counts'][class_dir] = 0
            
            # Add a debug message about files in this class
            logging.info(f"Processing class {class_dir}: Found {len(wav_files)} audio files")
            self.error_logs.append({'level': 'INFO', 'message': f"Processing class {class_dir}: Found {len(wav_files)} audio files"})
            
            # Store original audio data for augmentation
            original_features = []  # List of (raw_audio, sr, file_path) tuples
            
            for wav_file in wav_files:
                file_path = os.path.join(class_path, wav_file)
                
                try:
                    # Reset processor errors before processing this file
                    self.audio_processor.processing_errors = []
                    
                    # Add tracking for stretching
                    original_stretched_count = self.audio_processor.stretched_count if hasattr(self.audio_processor, 'stretched_count') else 0
                    
                    # Save the raw audio for augmentation
                    if os.path.exists(file_path):
                        try:
                            raw_audio, sr = librosa.load(file_path, sr=self.audio_processor.sr)
                            original_features.append((raw_audio, sr, file_path))
                        except Exception as e:
                            logging.warning(f"Could not load {file_path} for augmentation: {e}")
                    
                    # Process audio for CNN (mel spectrogram)
                    features = self.audio_processor.process_audio_for_cnn(file_path, training_mode=True)
                    
                    # Check if the audio was stretched
                    if hasattr(self.audio_processor, 'stretched_count') and self.audio_processor.stretched_count > original_stretched_count:
                        stats['stretched_counts'][class_dir] += 1
                        stats['total_stretched'] += 1
                        # Log the stretch for this specific file
                        stretch_msg = f"Successfully stretched file: {file_path}"
                        logging.info(stretch_msg)
                        self.error_logs.append({'level': 'INFO', 'message': stretch_msg})
                    
                    # Skip features with unusual shapes or NaN values
                    if features is None or np.isnan(features).any():
                        error_msg = f"Skipping {file_path}: Invalid features (contains NaN)"
                        logging.warning(error_msg)
                        self.error_logs.append({'level': 'WARNING', 'message': error_msg})
                        
                        # Add to file errors
                        self.file_errors.append({
                            'file': file_path,
                            'class': class_dir,
                            'error': 'Invalid features (contains NaN)',
                            'stage': 'feature_extraction'
                        })
                        
                        # Add to stats
                        stats['skipped_counts'][class_dir] += 1
                        stats['total_skipped'] += 1
                        stats['error_files'].append(file_path)
                        stats['error_types'].setdefault('nan_values', 0)
                        stats['error_types']['nan_values'] += 1
                        continue
                    
                    # Add the features and label
                    X.append(features)
                    y.append(class_idx)
                    
                    # Update stats
                    stats['processed_counts'][class_dir] += 1
                    stats['total_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    logging.error(error_msg)
                    self.error_logs.append({'level': 'ERROR', 'message': error_msg})
                    
                    # Add to file errors
                    self.file_errors.append({
                        'file': file_path,
                        'class': class_dir,
                        'error': str(e),
                        'stage': 'processing'
                    })
                    
                    # Update stats
                    stats['skipped_counts'][class_dir] += 1
                    stats['total_skipped'] += 1
                    stats['error_files'].append(file_path)
                    stats['error_types'].setdefault('processing_error', 0)
                    stats['error_types']['processing_error'] += 1
            
            # Apply augmentation to successfully processed audio files
            logging.info(f"Applying augmentation to {len(original_features)} files for class {class_dir}...")
            self.error_logs.append({'level': 'INFO', 'message': f"Applying augmentation to {len(original_features)} files for class {class_dir}..."})
            
            for raw_audio, sr, file_path in original_features:
                try:
                    # Apply augmentation to get multiple variations of the audio
                    augmented_audios = augment_audio_with_repetitions(raw_audio, sr)
                    
                    if augmented_audios:
                        aug_msg = f"Created {len(augmented_audios)} augmentations for {file_path}"
                        logging.info(aug_msg)
                        self.error_logs.append({'level': 'INFO', 'message': aug_msg})
                        
                        # Process each augmented audio
                        for i, aug_audio in enumerate(augmented_audios):
                            try:
                                # Process through AudioProcessor to extract spectrogram
                                # We need to manually extract spectrogram since we have the augmented audio array
                                mel_spec = self.audio_processor.extract_mel_spectrogram(aug_audio)
                                
                                # Validate the spectrogram
                                if mel_spec is None or np.isnan(mel_spec).any():
                                    aug_error = f"Augmented audio {i} from {file_path} produced invalid features (NaN)"
                                    logging.warning(aug_error)
                                    self.error_logs.append({'level': 'WARNING', 'message': aug_error})
                                    continue
                                
                                # Make sure it has the right dimensions
                                if len(mel_spec.shape) != 3 or mel_spec.shape[2] != 1:
                                    aug_error = f"Augmented spectrogram has wrong shape: {mel_spec.shape}"
                                    logging.warning(aug_error)
                                    self.error_logs.append({'level': 'WARNING', 'message': aug_error})
                                    continue
                                
                                # Store the augmented feature
                                X.append(mel_spec)
                                y.append(class_idx)
                                stats['augmented_counts'][class_dir] += 1
                                stats['total_augmented'] += 1
                                
                            except Exception as e:
                                aug_error = f"Error processing augmented audio {i} from {file_path}: {e}"
                                logging.warning(aug_error)
                                self.error_logs.append({'level': 'WARNING', 'message': aug_error})
                    
                except Exception as e:
                    aug_error = f"Error augmenting {file_path}: {e}"
                    logging.warning(aug_error)
                    self.error_logs.append({'level': 'WARNING', 'message': aug_error})
        
        # Check if we have any data
        if not X:
            error_msg = "No valid data was processed"
            logging.error(error_msg)
            self.error_logs.append({'level': 'ERROR', 'message': error_msg})
            return None, None, [], stats
        
        # For syllable recognition, we want to:
        # 1. Preserve the entire beginning and end of sounds
        # 2. Use a reasonable maximum length that captures all sounds
        
        # Standardize spectrogram shapes
        X = self._standardize_spectrogram_shapes(X)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Record dataset shape
        stats['dataset_shape'] = {
            'X': X.shape,
            'y': y.shape
        }
        
        # Final log
        logging.info(f"Dataset prepared: X shape {X.shape}, y shape {y.shape}")
        logging.info(f"Original samples: {stats['total_processed']}, Augmented samples: {stats['total_augmented']}")
        
        return X, y, class_names, stats
    
    def _standardize_spectrogram_shapes(self, spectrograms):
        """
        Standardize the shapes of spectrograms for CNN training.
        This function ensures all spectrograms have the same dimensions,
        preserving the beginning of sounds which is critical for phoneme/syllable recognition.
        
        Args:
            spectrograms (list): List of spectrogram arrays with potentially different shapes
            
        Returns:
            list: List of spectrogram arrays with consistent shapes
        """
        if not spectrograms:
            return []
            
        # For syllable recognition, we want to:
        # 1. Preserve the entire beginning of sounds
        # 2. Use a reasonable maximum length that captures most sounds without excessive padding
        
        # Get all time steps
        all_time_steps = [spec.shape[0] for spec in spectrograms]
        
        # We could use various strategies for determining target length:
        # Option 1: Use a percentile (e.g., 90th) to avoid extreme outliers
        # target_time_steps = int(np.percentile(all_time_steps, 90))
        
        # Option 2: Use the median plus some standard deviation
        # std_dev = np.std(all_time_steps)
        # median = np.median(all_time_steps)
        # target_time_steps = int(median + std_dev)
        
        # Option 3: Simply use the maximum (most conservative, ensures no information loss)
        target_time_steps = max(all_time_steps)
        
        # Get frequency dimension and channels from the first spectrogram
        freq_bins = spectrograms[0].shape[1]
        channels = spectrograms[0].shape[2] if len(spectrograms[0].shape) > 2 else 1
        
        logging.info(f"Standardizing spectrograms to shape: ({target_time_steps}, {freq_bins}, {channels})")
        logging.info(f"Time steps distribution - Min: {min(all_time_steps)}, Max: {max(all_time_steps)}, Mean: {np.mean(all_time_steps):.1f}")
        
        # Standardize all spectrograms to the target shape, preserving the beginning
        standardized = []
        for spec in spectrograms:
            if spec.shape[0] < target_time_steps:
                # Pad at the end (preserves the beginning of the sound)
                padding = ((0, target_time_steps - spec.shape[0]), (0, 0), (0, 0))
                padded_spec = np.pad(spec, padding, mode='constant', constant_values=0)
                standardized.append(padded_spec)
            else:
                # Keep the exact length or trim from the end if needed
                standardized.append(spec[:target_time_steps, :, :])
                
        return standardized
    
    def prepare_dataset_for_rf(self, audio_dir, classes=None):
        """
        Prepare a dataset of classical audio features for RandomForest training.
        Includes data augmentation to expand the training set.
        
        Args:
            audio_dir (str): Directory containing class-specific audio folders
            classes (list): List of class names to include (optional)
            
        Returns:
            tuple: (X, y, class_names, stats, feature_names)
        """
        X = []
        y = []
        class_names = []
        stats = {
            'original_counts': {},
            'augmented_counts': {},
            'processed_counts': {},
            'skipped_counts': {},
            'total_processed': 0,
            'total_skipped': 0,
            'total_augmented': 0,
            'error_files': [],
            'error_types': {}
        }
        
        # Clear previous errors
        self.file_errors = []
        
        # Check if the directory exists
        if not os.path.exists(audio_dir):
            error_msg = f"Audio directory {audio_dir} does not exist"
            logging.error(error_msg)
            self.error_logs.append({'level': 'ERROR', 'message': error_msg})
            return None, None, [], stats, []
        
        # Get class directories - either use provided classes or scan directory
        if classes:
            # Filter out non-existent class directories
            class_dirs = []
            for class_dir in classes:
                class_path = os.path.join(audio_dir, class_dir)
                if os.path.isdir(class_path):
                    class_dirs.append(class_dir)
                else:
                    error_msg = f"Class directory {class_path} does not exist"
                    logging.warning(error_msg)
                    self.error_logs.append({'level': 'WARNING', 'message': error_msg})
                    
            if not class_dirs:
                error_msg = f"None of the specified class directories exist in {audio_dir}"
                logging.error(error_msg)
                self.error_logs.append({'level': 'ERROR', 'message': error_msg})
                return None, None, [], stats, []
        else:
            # Get subdirectories (each corresponds to a class)
            class_dirs = [d for d in os.listdir(audio_dir) 
                        if os.path.isdir(os.path.join(audio_dir, d))]
            
            if not class_dirs:
                error_msg = f"No class subdirectories found in {audio_dir}"
                logging.error(error_msg)
                self.error_logs.append({'level': 'ERROR', 'message': error_msg})
                return None, None, [], stats, []
        
        # Sort for consistent class indices
        class_dirs.sort()
        class_names = class_dirs
        
        # Counters for logging
        file_count = 0
        aug_count = 0
        
        # Process each class
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(audio_dir, class_dir)
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            if not wav_files:
                error_msg = f"No .wav files found in {class_path}"
                logging.warning(error_msg)
                self.error_logs.append({'level': 'WARNING', 'message': error_msg})
                continue
            
            # Track original file count
            stats['original_counts'][class_dir] = len(wav_files)
            stats['processed_counts'][class_dir] = 0
            stats['skipped_counts'][class_dir] = 0
            stats['augmented_counts'][class_dir] = 0
            
            # Add a debug message about files in this class
            logging.info(f"Processing class {class_dir}: Found {len(wav_files)} audio files")
            self.error_logs.append({'level': 'INFO', 'message': f"Processing class {class_dir}: Found {len(wav_files)} audio files"})
            
            for wav_file in wav_files:
                file_path = os.path.join(class_path, wav_file)
                
                try:
                    # Process audio file and extract features
                    y_audio, sr = librosa.load(file_path, sr=self.sample_rate)
                    
                    # Extract features
                    all_features = self.feature_extractor.extract_features(y_audio, is_file=False)
                    feats = self.feature_extractor.extract_features_for_model(all_features, model_type='rf')
                    
                    if feats is None:
                        error_msg = f"Skipping {file_path}: Failed to extract features"
                        logging.warning(error_msg)
                        self.error_logs.append({'level': 'WARNING', 'message': error_msg})
                        
                        # Add to stats
                        stats['skipped_counts'][class_dir] += 1
                        stats['total_skipped'] += 1
                        stats['error_files'].append(file_path)
                        stats['error_types'].setdefault('feature_extraction_failed', 0)
                        stats['error_types']['feature_extraction_failed'] += 1
                        continue
                    
                    # Assemble feature row
                    row = self._assemble_feature_row(feats, self.feature_extractor)
                    X.append(row)
                    y.append(class_dir)  # Use class name as label
                    file_count += 1
                    
                    # Update stats
                    stats['processed_counts'][class_dir] += 1
                    stats['total_processed'] += 1
                    
                    # Augment audio for RF training to improve robustness
                    augmented_versions = augment_audio_with_repetitions(
                        y_audio, sr,
                        do_noise=True,
                        noise_count=3
                    )
                    
                    logging.info(f"Created {len(augmented_versions)} augmented clips for {wav_file}")
                    
                    for aug_audio in augmented_versions:
                        aug_all_features = self.feature_extractor.extract_features(aug_audio, is_file=False)
                        aug_feats = self.feature_extractor.extract_features_for_model(aug_all_features, model_type='rf')
                        if aug_feats is not None:
                            row_aug = self._assemble_feature_row(aug_feats, self.feature_extractor)
                            X.append(row_aug)
                            y.append(class_dir)
                            aug_count += 1
                            
                            # Update stats
                            stats['augmented_counts'][class_dir] = stats['augmented_counts'].get(class_dir, 0) + 1
                            stats['total_augmented'] += 1
                        else:
                            logging.warning(f"Augmented clip failed feature extraction for {wav_file}, skipping.")
                
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    logging.error(error_msg)
                    self.error_logs.append({'level': 'ERROR', 'message': error_msg})
                    
                    # Add to file errors
                    self.file_errors.append({
                        'file': file_path,
                        'class': class_dir,
                        'error': str(e),
                        'stage': 'processing'
                    })
                    
                    # Update stats
                    stats['skipped_counts'][class_dir] += 1
                    stats['total_skipped'] += 1
                    stats['error_files'].append(file_path)
                    stats['error_types'].setdefault('processing_error', 0)
                    stats['error_types']['processing_error'] += 1
        
        # Check if we have any data
        if not X:
            error_msg = "No valid data was processed"
            logging.error(error_msg)
            self.error_logs.append({'level': 'ERROR', 'message': error_msg})
            return None, None, [], stats, []
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Record dataset shape
        stats['dataset_shape'] = {
            'X': X.shape,
            'y': y.shape
        }
        
        # Final logs for clarity
        logging.info(f"Total original samples: {file_count}")
        logging.info(f"Total augmentation samples: {aug_count}")
        logging.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
        
        # Get feature names - use the keys from the RF features
        feature_names = list(feats.keys()) if feats else []
        
        return X, y, class_names, stats, feature_names
    
    def _assemble_feature_row(self, feats, extractor):
        """
        Helper method that assembles a feature vector from extracted features.
        
        Args:
            feats (dict): Dictionary of extracted features
            extractor (FeatureExtractor): Feature extractor instance
            
        Returns:
            list: Assembled feature vector
        """
        feature_names = extractor.get_feature_names()
        row = []
        for fn in feature_names:
            row.append(feats.get(fn, 0.0))  # Get feature value or default to 0.0
        return row 