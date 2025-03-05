import os
import logging
import numpy as np
from threading import Thread
from datetime import datetime
import copy
import time
import json
import traceback
import librosa
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.core.ml_algorithms import create_model
from src.core.audio.processor import AudioProcessor
from src.core.data.dataset_preparation import DatasetPreparation
from src.services.training_analysis_service import TrainingAnalysisService
from backend.features.extractor import FeatureExtractor
from config import Config
from backend.audio import AudioPreprocessor

class TrainingService:
    """
    Service for training different types of models.
    This service handles dataset preparation, feature extraction, 
    and model training for CNN, RF, and Ensemble models.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the training service.
        
        Args:
            model_dir (str): Directory for model storage
        """
        self.model_dir = model_dir
        # Initialize dataset preparation service
        self.dataset_preparation = DatasetPreparation(sample_rate=8000)
        # Initialize training analysis service
        self.analysis_service = TrainingAnalysisService()
        # Initialize with loudness normalization disabled and MFCC normalization enabled
        self.audio_processor = AudioProcessor(
            sample_rate=8000,  # Lower sample rate for better handling of short files
            enable_loudness_normalization=False  # Disable loudness normalization by default
        )
        self.training_stats = {}
        self.training_thread = None
        self.is_training = False
        self.error_logs = []  # Store error logs
        self.file_errors = []  # Store detailed file-specific errors
        
        # Set up logging
        self.logger = logging.getLogger('TrainingService')
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG level for more detailed logs
    
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
            'stretched_counts': {},  # New: track files that were time-stretched
            'total_processed': 0,    # New: total files successfully processed
            'total_skipped': 0,      # New: total files skipped
            'total_stretched': 0,    # New: total files stretched
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
        
        # Import the augmentation module here to avoid circular imports
        from src.ml.augmentation_manager import augment_audio_with_repetitions
        
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
            
            # New: Add a debug message about files in this class
            logging.info(f"Processing class {class_dir}: Found {len(wav_files)} audio files")
            self.error_logs.append({'level': 'INFO', 'message': f"Processing class {class_dir}: Found {len(wav_files)} audio files"})
            
            original_features = []  # Store original features for augmentation
            
            for wav_file in wav_files:
                file_path = os.path.join(class_path, wav_file)
                
                try:
                    # Reset processor errors before processing this file
                    self.audio_processor.processing_errors = []
                    
                    # Add tracking for stretching
                    original_stretched_count = self.audio_processor.stretched_count if hasattr(self.audio_processor, 'stretched_count') else 0
                    
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
                    
                    # Check for expected dimensions for CNN (time_steps, mel_bands, 1)
                    if len(features.shape) != 3 or features.shape[2] != 1:
                        error_msg = f"Skipping {file_path}: Invalid shape {features.shape}, expected 3D with channel dimension"
                        logging.warning(error_msg)
                        self.error_logs.append({'level': 'WARNING', 'message': error_msg})
                        
                        # Add to file errors
                        self.file_errors.append({
                            'file': file_path,
                            'class': class_dir,
                            'error': f'Invalid shape {features.shape}, expected 3D with channel dimension',
                            'stage': 'feature_extraction'
                        })
                        
                        # Add to stats
                        stats['skipped_counts'][class_dir] += 1
                        stats['total_skipped'] += 1
                        stats['error_files'].append(file_path)
                        stats['error_types'].setdefault('invalid_shape', 0)
                        stats['error_types']['invalid_shape'] += 1
                        continue
                    
                    # Store the feature
                    X.append(features)
                    y.append(class_idx)
                    stats['processed_counts'][class_dir] += 1
                    stats['total_processed'] += 1
                    
                    # Save the raw audio for augmentation
                    raw_audio, sr = self.audio_processor.load_audio(file_path)
                    original_features.append((raw_audio, sr, file_path))
                    
                    # Check if there were any non-fatal errors during processing
                    if self.audio_processor.processing_errors:
                        for error in self.audio_processor.processing_errors:
                            # Add to file errors but mark as non-fatal
                            self.file_errors.append({
                                'file': file_path,
                                'class': class_dir,
                                'error': error.get('message', 'Unknown error'),
                                'stage': error.get('stage', 'unknown'),
                                'type': error.get('type', 'unknown'),
                                'fatal': False
                            })
                            
                            # Track error types in stats
                            error_type = error.get('type', 'unknown')
                            stats['error_types'].setdefault(error_type, 0)
                            stats['error_types'][error_type] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {e}"
                    logging.error(error_msg)
                    self.error_logs.append({'level': 'ERROR', 'message': error_msg})
                    
                    # Add to file errors
                    self.file_errors.append({
                        'file': file_path,
                        'class': class_dir,
                        'error': str(e),
                        'stage': 'processing',
                        'fatal': True
                    })
                    
                    # Add to stats
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
        
        # Log overall statistics
        stats_msg = f"Dataset preparation complete: {stats['total_processed']} files processed, " + \
                    f"{stats['total_stretched']} files stretched, {stats['total_skipped']} files skipped, " + \
                    f"{stats['total_augmented']} augmented samples created"
        logging.info(stats_msg)
        self.error_logs.append({'level': 'INFO', 'message': stats_msg})
        
        # Include detailed stats by class
        for class_dir in class_dirs:
            class_stats = f"Class {class_dir}: {stats['processed_counts'][class_dir]} processed, " + \
                        f"{stats['stretched_counts'][class_dir]} stretched, {stats['skipped_counts'][class_dir]} skipped, " + \
                        f"{stats['augmented_counts'][class_dir]} augmented " + \
                        f"out of {stats['original_counts'][class_dir]} original files"
            logging.info(class_stats)
            self.error_logs.append({'level': 'INFO', 'message': class_stats})
        
        # Log error types summary
        if stats['error_types']:
            error_summary = "Error types summary: " + ", ".join([f"{k}: {v}" for k, v in stats['error_types'].items()])
            logging.info(error_summary)
            self.error_logs.append({'level': 'INFO', 'message': error_summary})
        
        if not X:
            error_msg = "No features extracted from audio files"
            logging.error(error_msg)
            self.error_logs.append({'level': 'ERROR', 'message': error_msg})
            return None, None, [], stats
            
        # Ensure all features have the same shape
        # Find the most common number of time steps to standardize on
        time_steps_counts = Counter([x.shape[0] for x in X])
        if len(time_steps_counts) > 1:
            # Use the most common number of time steps as our standard
            target_time_steps = time_steps_counts.most_common(1)[0][0]
            logging.info(f"Standardizing all spectrograms to {target_time_steps} time steps")
            
            X_standardized = []
            for i, spec in enumerate(X):
                if spec.shape[0] != target_time_steps:
                    # Create a new array with the target shape
                    resized = np.zeros((target_time_steps, spec.shape[1], spec.shape[2]))
                    
                    # If we need more time steps, pad with zeros
                    if spec.shape[0] < target_time_steps:
                        resized[:spec.shape[0], :, :] = spec
                    else:
                        # If we need fewer time steps, use the middle section
                        start = (spec.shape[0] - target_time_steps) // 2
                        resized = spec[start:start+target_time_steps, :, :]
                    
                    X_standardized.append(resized)
                else:
                    X_standardized.append(spec)
            
            # Replace X with the standardized version
            X = X_standardized
        
        # Convert to numpy arrays
        try:
            X = np.array(X)
            y = np.array(y)
            
            logging.info(f"CNN dataset prepared: X shape {X.shape}, y shape {y.shape}")
            return X, y, class_names, stats
        except Exception as e:
            logging.error(f"Error converting dataset to numpy arrays: {e}")
            return None, None, [], stats
    
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
            'augmented_counts': {}
        }
        
        # Check if the directory exists
        if not os.path.exists(audio_dir):
            logging.error(f"Audio directory {audio_dir} does not exist")
            return None, None, [], stats, []
        
        # Get class directories - either use provided classes or scan directory
        if classes:
            class_dirs = classes
            # Confirm the class directories exist
            for class_dir in class_dirs:
                class_path = os.path.join(audio_dir, class_dir)
                if not os.path.isdir(class_path):
                    logging.warning(f"Class directory {class_path} does not exist")
        else:
            # Get subdirectories (each corresponds to a class)
            class_dirs = [d for d in os.listdir(audio_dir) 
                        if os.path.isdir(os.path.join(audio_dir, d))]
            
            if not class_dirs:
                logging.error(f"No class subdirectories found in {audio_dir}")
                return None, None, [], stats, []
        
        # Sort for consistent class indices
        class_dirs.sort()
        class_names = class_dirs
        
        # Get feature names (we'll need these for creating the X array)
        feature_names = self.audio_processor.get_feature_names()
        
        # Process each class
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(audio_dir, class_dir)
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            if not wav_files:
                logging.warning(f"No .wav files found in {class_path}")
                continue
            
            # Track original file count
            stats['original_counts'][class_dir] = len(wav_files)
            
            # Counter for augmented files
            augmented_count = 0
            
            # Process original files
            for wav_file in wav_files:
                file_path = os.path.join(class_path, wav_file)
                
                try:
                    # First, load the audio file for both original and augmentation
                    y_audio, sr = librosa.load(file_path, sr=self.audio_processor.sr)
                    
                    # Process original audio for RF (classical features)
                    features_dict = self.audio_processor.process_audio_for_rf(file_path)
                    
                    # Convert dictionary to feature vector in the right order
                    feature_vector = [features_dict[name] for name in feature_names]
                    
                    X.append(feature_vector)
                    y.append(class_idx)
                    
                    # Apply data augmentation
                    # Use the augment_audio_with_repetitions function from augmentation_manager
                    from src.ml.augmentation_manager import augment_audio_with_repetitions
                    from src.ml.constants import (
                        AUG_DO_TIME_SHIFT, AUG_TIME_SHIFT_COUNT, AUG_SHIFT_MAX,
                        AUG_DO_PITCH_SHIFT, AUG_PITCH_SHIFT_COUNT, AUG_PITCH_RANGE,
                        AUG_DO_SPEED_CHANGE, AUG_SPEED_CHANGE_COUNT, AUG_SPEED_RANGE,
                        AUG_DO_NOISE, AUG_NOISE_COUNT, AUG_NOISE_TYPE, AUG_NOISE_FACTOR
                    )
                    
                    augmented_audios = augment_audio_with_repetitions(
                        audio=y_audio,
                        sr=sr,
                        do_time_shift=AUG_DO_TIME_SHIFT,
                        time_shift_count=AUG_TIME_SHIFT_COUNT,
                        shift_max=AUG_SHIFT_MAX,
                        do_pitch_shift=AUG_DO_PITCH_SHIFT,
                        pitch_shift_count=AUG_PITCH_SHIFT_COUNT,
                        pitch_range=AUG_PITCH_RANGE,
                        do_speed_change=AUG_DO_SPEED_CHANGE,
                        speed_change_count=AUG_SPEED_CHANGE_COUNT,
                        speed_range=AUG_SPEED_RANGE,
                        do_noise=AUG_DO_NOISE,
                        noise_count=AUG_NOISE_COUNT,
                        noise_type=AUG_NOISE_TYPE,
                        noise_factor=AUG_NOISE_FACTOR
                    )
                    
                    # Process each augmented audio
                    for aug_audio in augmented_audios:
                        try:
                            # Extract features from the augmented audio
                            aug_features_dict = self.audio_processor.extract_classical_features(aug_audio)
                            
                            # Convert to feature vector
                            aug_feature_vector = [aug_features_dict[name] for name in feature_names]
                            
                            X.append(aug_feature_vector)
                            y.append(class_idx)
                            augmented_count += 1
                        except Exception as e:
                            logging.error(f"Error processing augmented audio for {file_path}: {e}")
                            
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
            
            # Update augmented counts
            stats['augmented_counts'][class_dir] = augmented_count
        
        if not X:
            logging.error("No features extracted from audio files")
            return None, None, [], stats, []
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        logging.info(f"RF dataset prepared: X shape {X.shape}, y shape {y.shape}")
        logging.info(f"Original samples: {sum(stats['original_counts'].values())}, "
                    f"Augmented samples: {sum(stats['augmented_counts'].values())}")
        
        return X, y, class_names, stats, feature_names
    
    def train_model(self, model_type, audio_dir, save=True, **kwargs):
        """
        Train a model of the specified type.
        
        Args:
            model_type (str): Type of model to train ('cnn', 'rf', or 'ensemble')
            audio_dir (str): Directory containing class-specific audio folders
            save (bool): Whether to save the model after training
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results
        """
        # Create the model
        model = create_model(model_type, model_dir=self.model_dir)
        
        # Prepare datasets based on model type
        if model_type == 'cnn':
            return self._train_cnn_model(model, audio_dir, save, **kwargs)
        elif model_type == 'rf':
            return self._train_rf_model(model, audio_dir, save, **kwargs)
        elif model_type == 'ensemble':
            return self._train_ensemble_model(model, audio_dir, save, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_cnn_model(self, data, num_classes, class_names, save=True, **kwargs):
        """Train a CNN model on the provided data"""
        start_time = time.time()
        
        # Extract data
        X_train, y_train = data['train']
        X_val, y_val = data['val']
        
        # Get some stats about our data
        train_per_class = np.bincount(np.argmax(y_train, axis=1))
        val_per_class = np.bincount(np.argmax(y_val, axis=1))
        
        self.logger.info(f"Training CNN with {X_train.shape[0]} samples, {X_val.shape[0]} validation samples")
        self.logger.info(f"Samples per class (train): {train_per_class}")
        self.logger.info(f"Samples per class (val): {val_per_class}")
        
        # Get input shape from the data
        input_shape = X_train.shape[1:]
        self.logger.info(f"Input shape: {input_shape}")
        
        # Select model architecture
        model_architecture = kwargs.get('architecture', 'default')
        
        if model_architecture == 'mobilenet':
            self.logger.info("Using MobileNetV2 architecture")
            from src.core.ml_algorithms.cnn_mobilenet import CNNMobileNet
            model = CNNMobileNet()
        else:
            self.logger.info("Using default CNN architecture")
            from src.core.ml_algorithms.cnn import CNN
            model = CNN()
        
        # Build model
        model.build(input_shape=input_shape, num_classes=num_classes)
        
        # Calculate class weights if needed
        class_weights = None
        if kwargs.get('use_class_weights', True):
            # Calculate inverse frequency for each class
            total_samples = np.sum(train_per_class)
            class_freq = train_per_class / total_samples
            class_weights = {i: 1.0 / (freq + 1e-5) for i, freq in enumerate(class_freq)}
            
            # Normalize weights
            weight_sum = sum(class_weights.values())
            class_weights = {i: (weight * len(class_weights) / weight_sum) for i, weight in class_weights.items()}
            
            self.logger.info(f"Using class weights: {class_weights}")
        
        # Train the model
        history = model.train(
            X_train, y_train, 
            X_val, y_val,
            epochs=kwargs.get('epochs', 10),
            batch_size=kwargs.get('batch_size', 32),
            class_weights=class_weights,
            verbose=1
        )
        
        # Evaluate the model
        metrics = model.evaluate(X_val, y_val)
        self.logger.info(f"Validation metrics: {metrics}")
        
        # Save the model if requested
        if save:
            dict_name = kwargs.get('dict_name', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_id = f"{dict_name.replace(' ', '_')}_cnn_{timestamp}"
            
            # Create model directory
            model_dir = os.path.join(self.config.BASE_DIR, 'data', 'models', 'cnn', model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model file
            model_file = os.path.join(model_dir, f"{model_id}.h5")
            model.save(model_file)
            
            # Prepare metadata
            metadata = {
                "class_names": class_names,
                "num_classes": num_classes,
                "input_shape": list(input_shape),
                "accuracy": float(metrics.get('accuracy', 0)),
                "created_at": datetime.now().isoformat(),
                "dictionary": dict_name,
                "architecture": model_architecture
            }
            
            # Save metadata file
            from src.ml.model_paths import save_model_metadata, update_model_registry
            metadata_file = save_model_metadata(model_dir, metadata)
            
            # Update models.json registry
            is_best = kwargs.get('is_best', False)
            update_model_registry(model_id, 'cnn', dict_name, metadata, is_best)
            
            self.logger.info(f"Model saved to {model_file}")
            self.logger.info(f"Metadata saved to {metadata_file}")
            
        # Calculate training time in seconds
        training_time = int(time.time() - start_time)
        
        # Gather training statistics
        training_stats = {
            'model_type': 'cnn',
            'input_shape': input_shape,
            'num_classes': num_classes,
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'metrics': metrics,
            'training_time': training_time
        }
        
        # Add training history if available
        if history:
            # Convert to regular Python types for JSON compatibility
            epoch_details = []
            best_val_acc = 0
            best_epoch = 0
            
            for epoch in range(len(history.history['accuracy'])):
                epoch_info = {
                    'epoch': epoch + 1,
                    'accuracy': float(history.history['accuracy'][epoch]),
                    'loss': float(history.history['loss'][epoch]),
                    'val_accuracy': float(history.history['val_accuracy'][epoch]),
                    'val_loss': float(history.history['val_loss'][epoch])
                }
                
                # Track best epoch
                if epoch_info['val_accuracy'] > best_val_acc:
                    best_val_acc = epoch_info['val_accuracy']
                    best_epoch = epoch + 1
                    
                epoch_details.append(epoch_info)
            
            training_stats['history'] = {
                'epochs': len(history.history['accuracy']),
                'accuracy': float(history.history['accuracy'][-1]),
                'loss': float(history.history['loss'][-1]),
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'val_loss': float(history.history['val_loss'][-1]),
                'best_epoch': best_epoch,
                'best_val_accuracy': best_val_acc,
                'epoch_details': epoch_details
            }
        
        # Check if early stopping was triggered
        if history and len(history.history['accuracy']) < kwargs.get('epochs', 10):
            training_stats['early_stopping'] = True
            training_stats['stopped_epoch'] = len(history.history['accuracy'])
        
        # Register the training results for later analysis
        self.analysis_service.register_training_results(
            'cnn', training_stats, metrics, class_names, X_train, X_val, y_train, y_val
        )
        
        return True, model, metrics, training_stats
    
    def _train_rf_model(self, model, audio_dir, save=True, **kwargs):
        """
        Train a Random Forest model on sound data.
        
        Args:
            data_dir (str): Directory containing sound data
            save (bool): Whether to save the model after training
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results
        """
        start_time = time.time()
        
        # Prepare data
        X, y, feature_names, class_names, stats = self.dataset_preparation.prepare_dataset_for_rf(audio_dir, classes=kwargs.get('classes', None))
        
        # Split data
        train_idx = stats['train_idx']
        val_idx = stats['val_idx']
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Number of classes
        num_classes = len(class_names)
        
        # Create model
        rf_model = RandomForestModel(model_dir=os.path.join(Config.BASE_DIR, 'data', 'models'))
        rf_model.set_class_names(class_names)
        
        # Set hyperparameters
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        
        # Build and train the model
        rf_model.build(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=kwargs.get('random_state', 42),
            feature_names=feature_names
        )
        
        rf_model.train(X_train, y_train)
        
        # Evaluate on validation set
        metrics = rf_model.evaluate(X_val, y_val)
        
        # Save the model if requested
        model_filename = None
        if save:
            dict_name = kwargs.get('dict_name', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_filename = f"{dict_name.replace(' ', '_')}_rf_{timestamp}.joblib"
            rf_model.save(model_filename)
            
            # Update the models.json registry file with the newly saved model
            self.update_model_registry(
                model_type='rf',
                model_filename=model_filename,
                metadata={
                    'num_classes': num_classes,
                    'accuracy': metrics.get('accuracy', 0),
                    'feature_names': feature_names
                }
            )
        
        # Calculate training time in seconds
        training_time = int(time.time() - start_time)
        
        # Store training stats
        self.training_stats = {
            'model_type': 'rf',
            'num_classes': num_classes,
            'class_names': class_names,
            'feature_names': feature_names,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'total_samples': len(X),
            'metrics': metrics,
            'training_time': training_time,
            'feature_importance': rf_model.feature_importance if hasattr(rf_model, 'feature_importance') else None,
            'original_counts': stats['original_counts'],
            'skipped_counts': stats['skipped_counts'],
            'total_processed': stats['total_processed'],
            'total_skipped': stats['total_skipped'],
            'rf_params': {
                'n_estimators': n_estimators,
                'max_depth': str(max_depth),
                'min_samples_split': min_samples_split
            }
        }
        
        # Register training results with the analysis service
        self.analysis_service.register_training_results(
            model_type='rf',
            stats=self.training_stats,
            metrics=metrics,
            class_names=class_names,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        return {
            'success': True,
            'model': rf_model,
            'metrics': metrics,
            'stats': self.training_stats
        }
    
    def _train_ensemble_model(self, model, audio_dir, save=True, **kwargs):
        """
        Train an ensemble model that combines CNN and RF models.
        
        Args:
            model: The Ensemble model to train
            audio_dir (str): Directory containing class-specific audio folders
            save (bool): Whether to save the model after training
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results
        """
        start_time = time.time()
        
        # First train a CNN model
        cnn_results = self._train_cnn_model(model, audio_dir, save=True, **kwargs)
        if not cnn_results.get('success', False):
            return {
                'success': False,
                'error': 'Failed to train CNN model for ensemble'
            }
        
        # Then train a RF model
        rf_results = self._train_rf_model(model, audio_dir, save=True, **kwargs)
        if not rf_results.get('success', False):
            return {
                'success': False,
                'error': 'Failed to train RF model for ensemble'
            }
        
        # Get models from results
        cnn_model = cnn_results['model']
        rf_model = rf_results['model']
        
        # Get class names from both models, ensuring they're consistent
        cnn_class_names = cnn_model.get_class_names()
        rf_class_names = rf_model.get_class_names()
        
        # Verify class names match between models
        if set(cnn_class_names) != set(rf_class_names):
            logging.warning(f"Class names mismatch between CNN and RF models: {cnn_class_names} vs {rf_class_names}")
        
        # Use CNN class names as the canonical set
        class_names = cnn_class_names
        num_classes = len(class_names)
        
        # Create ensemble model
        ensemble_model = EnsembleModel(model_dir=os.path.join(Config.BASE_DIR, 'data', 'models'))
        ensemble_model.set_class_names(class_names)
        
        # Build and combine the models
        ensemble_model.build(cnn_model=cnn_model, rf_model=rf_model, rf_weight=kwargs.get('rf_weight', 0.5))
        
        # Get the validation datasets from the individual models for evaluation
        X_val_cnn = cnn_results.get('stats', {}).get('training_data', {}).get('X_val')
        y_val_cnn = cnn_results.get('stats', {}).get('training_data', {}).get('y_val')
        X_val_rf = rf_results.get('stats', {}).get('training_data', {}).get('X_val')
        y_val_rf = rf_results.get('stats', {}).get('training_data', {}).get('y_val')
        
        # Evaluate the ensemble model
        metrics = ensemble_model.evaluate(X_val_cnn, X_val_rf, y_val_cnn)
        
        # Save the model if requested
        model_filename = None
        if save:
            dict_name = kwargs.get('dict_name', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_filename = f"{dict_name.replace(' ', '_')}_ensemble_{timestamp}.model"
            ensemble_model.save(model_filename)
            
            # Update the models.json registry file with the newly saved model
            self.update_model_registry(
                model_type='ensemble',
                model_filename=model_filename,
                metadata={
                    'num_classes': num_classes,
                    'accuracy': metrics.get('accuracy', 0),
                    'rf_weight': kwargs.get('rf_weight', 0.5),
                    'cnn_model_id': cnn_model.get_model_path().split('/')[-1].replace('.h5', ''),
                    'rf_model_id': rf_model.get_model_path().split('/')[-1].replace('.joblib', '')
                }
            )
        
        # Calculate training time in seconds
        training_time = int(time.time() - start_time)
        
        # Store training stats
        self.training_stats = {
            'model_type': 'ensemble',
            'num_classes': num_classes,
            'class_names': class_names,
            'metrics': metrics,
            'training_time': training_time,
            'cnn_accuracy': cnn_results.get('metrics', {}).get('accuracy', 0),
            'rf_accuracy': rf_results.get('metrics', {}).get('accuracy', 0),
            'ensemble_accuracy': metrics.get('accuracy', 0),
            'ensemble_params': {
                'rf_weight': kwargs.get('rf_weight', 0.5)
            }
        }
        
        # Register training results with the analysis service
        self.analysis_service.register_training_results(
            model_type='ensemble',
            stats=self.training_stats,
            metrics=metrics,
            class_names=class_names
        )
        
        return {
            'success': True,
            'model': ensemble_model,
            'metrics': metrics,
            'stats': self.training_stats
        }
    
    def train_model_async(self, model_type, audio_dir, callback=None, **kwargs):
        """
        Train a model asynchronously in a separate thread.
        
        Args:
            model_type (str): Type of model to train ('cnn', 'rf', or 'ensemble')
            audio_dir (str): Directory containing class-specific audio folders
            callback (callable): Function to call when training is complete
            **kwargs: Additional training parameters
            
        Returns:
            bool: True if training started, False otherwise
        """
        if self.is_training:
            logging.error("A training session is already in progress")
            return False
        
        # Enhanced logging - add these lines
        logging.info(f"Starting training for model_type={model_type} with audio_dir={audio_dir}")
        logging.info(f"Training parameters: {kwargs}")
        
        # Add debug logging for class directories
        dict_name = kwargs.get('dict_name', 'unknown')
        classes = kwargs.get('classes', [])
        logging.info(f"Dictionary: {dict_name}, Classes: {classes}")
        
        # Log if the specified classes exist as directories
        for class_name in classes:
            class_path = os.path.join(audio_dir, class_name)
            if os.path.isdir(class_path):
                wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
                logging.info(f"Class directory '{class_path}' exists with {len(wav_files)} WAV files")
            else:
                logging.error(f"Class directory '{class_path}' does not exist!")
        
        # Reset error logs at the start of training
        self.error_logs = []
        
        # Create a custom log handler to capture logs
        class LogHandler(logging.Handler):
            def __init__(self, log_list):
                super().__init__()
                self.log_list = log_list
            
            def emit(self, record):
                log_entry = {
                    'level': record.levelname,
                    'message': self.format(record)
                }
                self.log_list.append(log_entry)
                # Add this line to ensure logs are also shown in the console/main log
                print(f"TRAINING LOG: {record.levelname} - {self.format(record)}")
        
        # Add the custom handler to the root logger
        handler = LogHandler(self.error_logs)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(handler)
        
        def train_thread_fn():
            try:
                self.is_training = True
                logging.info(f"Training thread started for {model_type} model")
                
                # For CNN and Ensemble models, first check if we have enough data
                min_samples_per_class = 3  # Minimum samples needed per class for training
                
                if model_type in ['cnn', 'ensemble']:
                    # Get the classes from kwargs
                    dict_classes = kwargs.get('classes', [])
                    
                    if not dict_classes:
                        error_msg = "No classes specified for training. Dictionary must have classes defined."
                        logging.error(error_msg)
                        if callback:
                            callback({
                                'success': False,
                                'error': error_msg,
                                'stats': {
                                    'log_messages': self.error_logs
                                }
                            })
                        return
                    
                    logging.info(f"Checking specified class directories: {dict_classes}")
                    
                    # Check only the specified class directories
                    insufficient_classes = []
                    for class_name in dict_classes:
                        class_path = os.path.join(audio_dir, class_name)
                        if not os.path.isdir(class_path):
                            logging.error(f"Class directory doesn't exist: {class_path}")
                            insufficient_classes.append((class_name, 0))
                            continue
                            
                        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
                        logging.info(f"Class '{class_name}' has {len(wav_files)} WAV files")
                        
                        if len(wav_files) < min_samples_per_class:
                            insufficient_classes.append((class_name, len(wav_files)))
                    
                    if insufficient_classes:
                        error_msg = "Insufficient samples for training: "
                        error_msg += ", ".join([f"{cls} ({count} samples)" for cls, count in insufficient_classes])
                        error_msg += f". Each class needs at least {min_samples_per_class} samples."
                        logging.error(error_msg)
                        
                        if callback:
                            callback({
                                'success': False,
                                'error': error_msg,
                                'stats': {
                                    'log_messages': self.error_logs
                                }
                            })
                        return
                
                # Proceed with training
                logging.info(f"Starting actual model training for {model_type}")
                result = self.train_model(model_type, audio_dir, **kwargs)
                logging.info(f"Training completed with result: {result.get('success', False)}")
                
                # Add error logs to the result
                if isinstance(result, dict):
                    if 'stats' in result:
                        result['stats']['log_messages'] = self.error_logs
                
                if callback:
                    callback(result)
            except Exception as e:
                logging.error(f"Error in training thread: {e}")
                import traceback
                tb = traceback.format_exc()
                logging.error(tb)
                if callback:
                    callback({
                        'success': False,
                        'error': str(e),
                        'stats': {
                            'log_messages': self.error_logs,
                            'traceback': tb
                        }
                    })
            finally:
                # Remove the custom handler
                logging.getLogger().removeHandler(handler)
                self.is_training = False
                logging.info("Training thread completed and resources released")
        
        self.training_thread = Thread(target=train_thread_fn)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        return True
    
    def get_training_stats(self):
        """
        Get the training statistics from the last training run.
        
        Returns:
            dict: Training statistics
        """
        # Add error logs to stats
        stats = copy.deepcopy(self.training_stats)
        stats['log_messages'] = self.error_logs
        return stats
        
    def is_training_in_progress(self):
        """
        Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress, False otherwise
        """
        return self.is_training 

    def update_model_registry(self, model_type, model_filename, metadata=None):
        """
        Update the models.json registry file with the newly trained model.
        
        Args:
            model_type (str): Type of model ('cnn', 'rf', 'ensemble')
            model_filename (str): Filename of the saved model
            metadata (dict, optional): Additional metadata about the model
            
        Returns:
            bool: True if the registry was updated successfully, False otherwise
        """
        try:
            # Path to the models.json registry file
            registry_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')
            
            # Build the model information
            model_id = os.path.splitext(model_filename)[0]  # Remove file extension
            
            # Extract dictionary name from filename (format: DictName_type_timestamp.ext)
            parts = model_id.split('_')
            if len(parts) < 3:
                logging.warning(f"Model filename doesn't match expected format: {model_filename}")
                dict_name = "Unknown"
            else:
                # If there are underscores in the dictionary name, reconstruct it
                if model_type in parts:
                    type_index = parts.index(model_type)
                    dict_name = '_'.join(parts[:type_index])
                else:
                    dict_name = parts[0]
            
            # Determine file path in the registry (format: 'type/model_id/model_id.ext')
            if model_type == 'cnn':
                ext = '.h5'
            elif model_type == 'rf':
                ext = '.joblib'
            else:
                ext = '.model'
                
            file_path = f"{model_type}/{model_id}/{model_id}{ext}"
            
            # Create model entry
            model_entry = {
                'id': model_id,
                'name': f"{dict_name} {model_type.upper()} Model ({model_id.split('_')[-1]})",
                'type': model_type,
                'dictionary': dict_name,
                'file_path': file_path,
                'created_at': datetime.now().isoformat()
            }
            
            # Add any additional metadata
            if metadata:
                model_entry.update(metadata)
            
            # Load existing registry or create a new one
            registry = {'models': {}}
            if os.path.exists(registry_path):
                try:
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                except Exception as e:
                    logging.error(f"Error reading models.json, creating a new one: {e}")
                    registry = {'models': {}}
            
            # Initialize model type in registry if needed
            if 'models' not in registry:
                registry['models'] = {}
                
            if model_type not in registry['models']:
                registry['models'][model_type] = {}
            
            # Add the model to the registry
            registry['models'][model_type][model_id] = model_entry
            
            # Save the updated registry
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
                
            logging.info(f"Added model {model_id} to models.json registry")
            return True
            
        except Exception as e:
            logging.error(f"Error updating model registry: {e}", exc_info=True)
            return False 

    def train_unified(self, model_type, audio_dir, save=True, progress_callback=None, **kwargs):
        """
        Train a model using the unified feature extractor.
        
        Args:
            model_type (str): Type of model to train ('cnn', 'rf', 'ensemble')
            audio_dir (str): Directory containing class folders with audio files
            save (bool): Whether to save the trained model
            progress_callback (callable): Optional callback for progress reporting
            **kwargs: Additional model-specific parameters
            
        Returns:
            dict: Training results
        """
        logging.info(f"Starting unified {model_type} model training with FeatureExtractor")
        
        # Initialize audio processing components
        audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            sound_threshold=0.008,
            min_silence_duration=0.1,
            target_duration=1.0,
            normalize_audio=True
        )
        
        # Initialize the feature extractor based on model type
        if model_type == 'cnn':
            unified_extractor = FeatureExtractor(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=256
            )
        else:
            unified_extractor = FeatureExtractor(sample_rate=16000)
            
        logging.info("Initialized audio processing components for training")
        
        # Check if audio directory exists
        if not os.path.exists(audio_dir):
            error_msg = f"Audio directory {audio_dir} does not exist"
            logging.error(error_msg)
            return {"error": error_msg}
        
        # Get class directories
        class_dirs = [d for d in os.listdir(audio_dir) 
                     if os.path.isdir(os.path.join(audio_dir, d)) and not d.startswith('.')]
        
        if not class_dirs:
            error_msg = f"No class directories found in {audio_dir}"
            logging.error(error_msg)
            return {"error": error_msg}
        
        class_dirs.sort()
        
        if progress_callback:
            progress_callback(5, f"Found {len(class_dirs)} classes. Preprocessing and extracting features...")
        
        # First, preprocess all audio files to ensure consistency
        preprocessed_files = []
        class_labels = []
        
        try:
            # Process each class directory
            for class_idx, class_dir in enumerate(class_dirs):
                class_path = os.path.join(audio_dir, class_dir)
                wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
                
                if not wav_files:
                    logging.warning(f"No .wav files found in {class_path}")
                    continue
                
                logging.info(f"Preprocessing {len(wav_files)} files for class {class_dir}")
                
                # Process each file in the class
                for file_idx, wav_file in enumerate(wav_files):
                    file_path = os.path.join(class_path, wav_file)
                    
                    try:
                        # Preprocess the audio file
                        processed_audio = audio_preprocessor.preprocess_file(file_path)
                        
                        if processed_audio is not None:
                            preprocessed_files.append((processed_audio, file_path))
                            class_labels.append(class_dir)
                            
                            # Report progress
                            if progress_callback and (file_idx % 5 == 0 or file_idx == len(wav_files) - 1):
                                total_progress = int(5 + (class_idx * len(wav_files) + file_idx) / 
                                                  (len(class_dirs) * len(wav_files)) * 15)  # 5-20% of total
                                progress_callback(
                                    total_progress,
                                    f"Preprocessing {class_dir}: {file_idx+1}/{len(wav_files)} files"
                                )
                    except Exception as e:
                        logging.error(f"Error preprocessing {file_path}: {str(e)}")
            
            if len(preprocessed_files) == 0:
                error_msg = "No valid audio files found after preprocessing. Check audio files."
                logging.error(error_msg)
                if progress_callback:
                    progress_callback(0, f"Error: {error_msg}")
                return {'status': 'error', 'message': error_msg}
            
            if progress_callback:
                progress_callback(20, f"Preprocessing complete. Extracting features from {len(preprocessed_files)} files...")
        
            # Now extract features from the preprocessed audio
            X = []
            y = []
            
            for idx, (processed_audio, file_path) in enumerate(preprocessed_files):
                try:
                    # Extract all features
                    all_features = unified_extractor.extract_features(processed_audio, is_file=False)
                    
                    # Extract model-specific features
                    features = unified_extractor.extract_features_for_model(all_features, model_type=model_type)
                    
                    if features is not None:
                        X.append(features)
                        y.append(class_labels[idx])
                        
                        # Report progress
                        if progress_callback and (idx % 5 == 0 or idx == len(preprocessed_files) - 1):
                            total_progress = int(20 + (idx / len(preprocessed_files) * 15))  # 20-35% of total
                            progress_callback(
                                total_progress,
                                f"Extracting features: {idx+1}/{len(preprocessed_files)} files"
                            )
                except Exception as e:
                    logging.error(f"Error extracting features from {file_path}: {str(e)}")
            
            # Convert to arrays
            X = np.array(X)
            y = np.array(y)
            class_names = np.unique(y).tolist()
            
            # Create stats
            stats = {
                'total_samples': len(X),
                'class_counts': {c: np.sum(y == c) for c in class_names},
                'preprocessing_method': 'unified_audio_preprocessor',
                'feature_extractor': 'unified_feature_extractor'
            }
            
            logging.info(f"Feature extraction complete: {len(X)} samples processed with shapes: {X.shape}")
            
            # Check if we have any samples at all
            if len(X) == 0:
                error_msg = "No valid samples extracted for training. Check audio files and feature extraction process."
                logging.error(error_msg)
                if progress_callback:
                    progress_callback(0, f"Error: {error_msg}")
                return {'status': 'error', 'message': error_msg}
                
            # Check if we have enough samples for each class
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_samples_needed = 5  # Minimum needed for splitting into train/val
            
            if len(unique_classes) < 2:
                error_msg = f"Need at least 2 classes, but only found {len(unique_classes)}."
                logging.error(error_msg)
                if progress_callback:
                    progress_callback(0, f"Error: {error_msg}")
                return {'status': 'error', 'message': error_msg}
                
            for cls, count in zip(unique_classes, class_counts):
                if count < min_samples_needed:
                    error_msg = f"Class '{cls}' has only {count} samples. Need at least {min_samples_needed} samples per class."
                    logging.error(error_msg)
                    if progress_callback:
                        progress_callback(0, f"Error: {error_msg}")
                    return {'status': 'error', 'message': error_msg}
                    
            if progress_callback:
                progress_callback(35, "Feature extraction complete, starting model training")
            
            # Train the model based on type
            if model_type == 'cnn':
                # For CNN models, reshape X if needed to match expected input shape
                # (The extractor should already return with the right shape)
                if len(X.shape) == 3:  # (samples, time_steps, mel_bands)
                    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
                
                logging.info(f"Preparing data for CNN training: X shape={X.shape}, y unique values={np.unique(y)}")
                
                try:
                    # Split the data into training and validation sets (80/20 split)
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train_indices, y_val_indices = train_test_split(
                        X, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Convert class names to one-hot encoded labels
                    from tensorflow.keras.utils import to_categorical
                    
                    # Create a mapping from class names to indices
                    class_to_index = {class_name: i for i, class_name in enumerate(class_names)}
                    logging.info(f"Class to index mapping: {class_to_index}")
                    
                    # Convert string labels to indices
                    y_train = np.array([class_to_index[class_name] for class_name in y[y_train_indices]])
                    y_val = np.array([class_to_index[class_name] for class_name in y[y_val_indices]])
                    
                    # Convert to one-hot encoding
                    y_train_onehot = to_categorical(y_train, num_classes=len(class_names))
                    y_val_onehot = to_categorical(y_val, num_classes=len(class_names))
                    
                    # Log data shapes for debugging
                    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train_onehot.shape}")
                    logging.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val_onehot.shape}")
                    
                    # Create the data dictionary that _train_cnn_model expects
                    data_dict = {
                        'train': (X_train, y_train_onehot),
                        'val': (X_val, y_val_onehot)
                    }
                    
                    # Call existing CNN training method
                    if progress_callback:
                        # Create a wrapper to forward progress updates
                        def cnn_progress_callback(epoch, logs):
                            # Report model training progress (35-95%)
                            epochs = kwargs.get('epochs', 50)
                            progress = 35 + int((epoch / epochs) * 60)
                            progress_callback(progress, f"Training epoch {epoch+1}/{epochs}")
                        
                        # Add callback to kwargs
                        kwargs['custom_callback'] = cnn_progress_callback
                    
                    result = self._train_cnn_model(
                        data_dict,
                        len(class_names),
                        class_names,
                        save=save,
                        **kwargs
                    )
                except Exception as e:
                    error_msg = f"Error preparing data for CNN training: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    if progress_callback:
                        progress_callback(35, f"Error: {error_msg}")
                    return {'status': 'error', 'message': error_msg}
            elif model_type == 'rf':
                # Call existing RF training method with our dataset
                if progress_callback:
                    # This is just an estimate since RF doesn't have clear progress reporting
                    progress_callback(50, "Training Random Forest model")
                
                result = self._train_rf_model(
                    (X, y, class_names, stats),
                    save=save,
                    **kwargs
                )
                
                # Report completion for RF model
                if progress_callback and result.get('status') == 'success':
                    progress_callback(95, "Random Forest training complete, finalizing model")
            else:  # ensemble or other
                error_msg = f"Unsupported model type for unified training: {model_type}"
                logging.error(error_msg)
                if progress_callback:
                    progress_callback(35, f"Error: {error_msg}")
                return {'status': 'error', 'message': error_msg}
            
            # Report final status
            if progress_callback:
                if result.get('status') == 'success':
                    progress_callback(100, "Training complete!")
                else:
                    progress_callback(
                        min(95, current_app.training_progress if hasattr(current_app, 'training_progress') else 35), 
                        f"Training failed: {result.get('message', 'Unknown error')}"
                    )
            
            return result 
            
        except Exception as e:
            error_msg = f"Error during feature extraction: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            if progress_callback:
                progress_callback(5, f"Error: {error_msg}")
            return {'status': 'error', 'message': error_msg} 