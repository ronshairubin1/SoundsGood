import os
import logging
import numpy as np
from threading import Thread
from datetime import datetime
import copy
import time

from src.core.models import create_model
from src.core.audio.processor import AudioProcessor
from config import Config

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
        from collections import Counter
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
            'augmented_counts': {}
        }
        
        # Check if the directory exists
        if not os.path.exists(audio_dir):
            logging.error(f"Audio directory {audio_dir} does not exist")
            return None, None, [], stats
        
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
                return None, None, [], stats
        
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
            
            for wav_file in wav_files:
                file_path = os.path.join(class_path, wav_file)
                
                try:
                    # Process audio for RF (classical features)
                    features_dict = self.audio_processor.process_audio_for_rf(file_path)
                    
                    # Convert dictionary to feature vector in the right order
                    feature_vector = [features_dict[name] for name in feature_names]
                    
                    X.append(feature_vector)
                    y.append(class_idx)
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
            
            # No augmentation applied here - that will be in a separate method
            stats['augmented_counts'][class_dir] = 0
        
        if not X:
            logging.error("No features extracted from audio files")
            return None, None, [], stats
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        logging.info(f"RF dataset prepared: X shape {X.shape}, y shape {y.shape}")
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
    
    def _train_cnn_model(self, model, audio_dir, save=True, **kwargs):
        """
        Train a CNN model on audio data.
        """
        # Track start time for training duration measurement
        start_time = time.time()
        
        # Prepare dataset
        classes = kwargs.get('classes', None)
        X, y, class_names, stats = self.prepare_dataset_for_cnn(audio_dir, classes=classes)
        
        if X is None or y is None:
            return {
                'success': False,
                'error': 'Failed to prepare dataset'
            }
        
        # Set class names
        model.set_class_names(class_names)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        input_shape = X_train.shape[1:]
        num_classes = len(class_names)
        model.build(input_shape=input_shape, num_classes=num_classes)
        
        # Calculate class weights for imbalanced data
        class_weights = {}
        if kwargs.get('use_class_weights', True):
            classes, counts = np.unique(y_train, return_counts=True)
            total_samples = len(y_train)
            for i, c in enumerate(classes):
                class_weights[int(c)] = total_samples / (len(classes) * counts[i])
        
        # Train model
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 32)
        
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            class_weights=class_weights,
            save_best=True
        )
        
        # Evaluate on validation set
        metrics = model.evaluate(X_val, y_val)
        
        # Save the model if requested
        if save:
            dict_name = kwargs.get('dict_name', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{dict_name.replace(' ', '_')}_cnn_{timestamp}.h5"
            model.save(filename)
        
        # Calculate training time in seconds
        training_time = int(time.time() - start_time)
        
        # Store training stats
        self.training_stats = {
            'model_type': 'cnn',
            'input_shape': str(input_shape),
            'num_classes': num_classes,
            'class_names': class_names,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'total_samples': len(X),
            'metrics': metrics,
            'training_time': training_time,  # Add training time
            'original_counts': stats['original_counts'],
            'processed_counts': stats['processed_counts'],
            'skipped_counts': stats['skipped_counts'],
            'stretched_counts': stats['stretched_counts'],
            'total_processed': stats['total_processed'],
            'total_skipped': stats['total_skipped'],
            'total_stretched': stats['total_stretched'],
            'augmented_counts': stats['augmented_counts'],
            'model_summary': model.get_model_summary(),
            'cnn_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': kwargs.get('learning_rate', 0.001)
            }
        }
        
        # Add history to stats if available
        if hasattr(history, 'history'):
            # More detailed epoch history
            epochs_completed = len(history.history['loss'])
            epoch_details = []
            
            for i in range(epochs_completed):
                epoch_info = {
                    'epoch': i + 1,
                    'accuracy': float(history.history['accuracy'][i]),
                    'loss': float(history.history['loss'][i]),
                }
                
                if 'val_accuracy' in history.history:
                    epoch_info['val_accuracy'] = float(history.history['val_accuracy'][i])
                    
                if 'val_loss' in history.history:
                    epoch_info['val_loss'] = float(history.history['val_loss'][i])
                    
                # Calculate improvement
                if i > 0 and 'val_accuracy' in history.history:
                    epoch_info['improved'] = history.history['val_accuracy'][i] > history.history['val_accuracy'][i-1]
                else:
                    epoch_info['improved'] = True
                    
                epoch_details.append(epoch_info)
            
            self.training_stats['history'] = {
                'epochs': epochs_completed,
                'accuracy': history.history['accuracy'],
                'loss': history.history['loss'],
                'val_accuracy': history.history['val_accuracy'] if 'val_accuracy' in history.history else None,
                'val_loss': history.history['val_loss'] if 'val_loss' in history.history else None,
                'epoch_details': epoch_details
            }
            
            # Check if early stopping occurred
            self.training_stats['early_stopped'] = epochs_completed < epochs
        
        return {
            'success': True,
            'model': model,
            'metrics': metrics,
            'stats': self.training_stats
        }
    
    def _train_rf_model(self, model, audio_dir, save=True, **kwargs):
        """
        Train a RandomForest model.
        
        Args:
            model: The RF model to train
            audio_dir (str): Directory containing class-specific audio folders
            save (bool): Whether to save the model after training
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results
        """
        # Prepare dataset
        classes = kwargs.get('classes', None)
        X, y, class_names, stats, feature_names = self.prepare_dataset_for_rf(audio_dir, classes=classes)
        
        if X is None or y is None:
            return {
                'success': False,
                'error': 'Failed to prepare dataset'
            }
        
        # Set class names
        model.set_class_names(class_names)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        model.build(n_estimators=n_estimators, max_depth=max_depth)
        
        # Train model
        metrics = model.train(
            X_train, y_train,
            X_val, y_val,
            feature_names=feature_names
        )
        
        # Save the model if requested
        if save:
            dict_name = kwargs.get('dict_name', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{dict_name.replace(' ', '_')}_rf_{timestamp}.joblib"
            model.save(filename)
        
        # Store training stats
        self.training_stats = {
            'model_type': 'rf',
            'num_classes': len(class_names),
            'class_names': class_names,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'total_samples': len(X),
            'metrics': metrics,
            'original_counts': stats['original_counts'],
            'augmented_counts': stats['augmented_counts'],
            'feature_importance': metrics.get('feature_importance', {})
        }
        
        return {
            'success': True,
            'model': model,
            'metrics': metrics,
            'stats': self.training_stats
        }
    
    def _train_ensemble_model(self, model, audio_dir, save=True, **kwargs):
        """
        Train an Ensemble model.
        
        Args:
            model: The Ensemble model to train
            audio_dir (str): Directory containing class-specific audio folders
            save (bool): Whether to save the model after training
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results
        """
        # Prepare datasets for both CNN and RF
        classes = kwargs.get('classes', None)
        cnn_data = self.prepare_dataset_for_cnn(audio_dir, classes=classes)
        rf_data = self.prepare_dataset_for_rf(audio_dir, classes=classes)
        
        X_cnn, y_cnn, class_names_cnn, stats_cnn = cnn_data
        X_rf, y_rf, class_names_rf, stats_rf, feature_names = rf_data
        
        if X_cnn is None or y_cnn is None or X_rf is None or y_rf is None:
            return {
                'success': False,
                'error': 'Failed to prepare datasets'
            }
        
        # Ensure consistent class names
        if class_names_cnn != class_names_rf:
            logging.warning("Class names differ between CNN and RF datasets")
        
        # Use CNN class names as the canonical set
        class_names = class_names_cnn
        
        # Set class names
        model.set_class_names(class_names)
        
        # Split data consistently
        from sklearn.model_selection import train_test_split
        X_cnn_train, X_cnn_val, y_train, y_val = train_test_split(
            X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn
        )
        
        # Use the same indices for RF split
        X_rf_train, X_rf_val, _, _ = train_test_split(
            X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf
        )
        
        # Build CNN part of ensemble
        input_shape = X_cnn_train.shape[1:]
        num_classes = len(class_names)
        cnn_params = {
            'input_shape': input_shape,
            'num_classes': num_classes
        }
        
        # Build RF part of ensemble
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        
        # Build ensemble model
        rf_weight = kwargs.get('rf_weight', 0.5)
        model.build(cnn_params=cnn_params, rf_params=rf_params, rf_weight=rf_weight)
        
        # Prepare data for training
        X_train = {
            'cnn': X_cnn_train,
            'rf': X_rf_train
        }
        
        X_val = {
            'cnn': X_cnn_val,
            'rf': X_rf_val
        }
        
        # Calculate class weights for CNN
        class_weights = {}
        if kwargs.get('use_class_weights', True):
            classes, counts = np.unique(y_train, return_counts=True)
            total_samples = len(y_train)
            for i, c in enumerate(classes):
                class_weights[int(c)] = total_samples / (len(classes) * counts[i])
        
        # Train ensemble model
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 32)
        
        cnn_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'class_weights': class_weights,
            'save_best': True
        }
        
        rf_params = {
            'feature_names': feature_names
        }
        
        metrics = model.train(
            X_train, y_train,
            X_val, y_val,
            cnn_params=cnn_params,
            rf_params=rf_params
        )
        
        # Evaluate on validation set
        eval_metrics = model.evaluate(X_val, y_val)
        
        # Save the model if requested
        if save:
            dict_name = kwargs.get('dict_name', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{dict_name.replace(' ', '_')}_ensemble_{timestamp}.pkl"
            model.save(filename)
        
        # Store training stats
        self.training_stats = {
            'model_type': 'ensemble',
            'rf_weight': rf_weight,
            'num_classes': num_classes,
            'class_names': class_names,
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'total_samples': len(y_cnn),
            'metrics': eval_metrics,
            'cnn_metrics': metrics.get('cnn_metrics', {}),
            'rf_metrics': metrics.get('rf_metrics', {}),
            'original_counts': stats_cnn['original_counts'],
            'augmented_counts': stats_cnn['augmented_counts']
        }
        
        return {
            'success': True,
            'model': model,
            'metrics': eval_metrics,
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