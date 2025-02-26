import os
import logging
import numpy as np
from threading import Thread

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
        self.audio_processor = AudioProcessor(sample_rate=16000)  # Consistent sample rate
        self.training_stats = {}
        self.training_thread = None
        self.is_training = False
    
    def prepare_dataset_for_cnn(self, audio_dir):
        """
        Prepare a dataset of mel spectrograms for CNN training.
        
        Args:
            audio_dir (str): Directory containing class-specific audio folders
            
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
        
        # Get subdirectories (each corresponds to a class)
        class_dirs = [d for d in os.listdir(audio_dir) 
                      if os.path.isdir(os.path.join(audio_dir, d))]
        
        if not class_dirs:
            logging.error(f"No class subdirectories found in {audio_dir}")
            return None, None, [], stats
        
        # Sort for consistent class indices
        class_dirs.sort()
        class_names = class_dirs
        
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
                    # Process audio for CNN (mel spectrogram)
                    features = self.audio_processor.process_audio_for_cnn(file_path)
                    X.append(features)
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
        
        logging.info(f"CNN dataset prepared: X shape {X.shape}, y shape {y.shape}")
        return X, y, class_names, stats
    
    def prepare_dataset_for_rf(self, audio_dir):
        """
        Prepare a dataset of classical audio features for RandomForest training.
        
        Args:
            audio_dir (str): Directory containing class-specific audio folders
            
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
        Train a CNN model.
        
        Args:
            model: The CNN model to train
            audio_dir (str): Directory containing class-specific audio folders
            save (bool): Whether to save the model after training
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results
        """
        # Prepare dataset
        X, y, class_names, stats = self.prepare_dataset_for_cnn(audio_dir)
        
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
            filename = f"{dict_name.replace(' ', '_')}_cnn.h5"
            model.save(filename)
        
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
            'original_counts': stats['original_counts'],
            'augmented_counts': stats['augmented_counts'],
            'model_summary': model.get_model_summary()
        }
        
        # Add history to stats if available
        if hasattr(history, 'history'):
            self.training_stats['history'] = {
                'epochs': len(history.history['loss']),
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'] if 'val_accuracy' in history.history else None
            }
        
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
        X, y, class_names, stats, feature_names = self.prepare_dataset_for_rf(audio_dir)
        
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
            filename = f"{dict_name.replace(' ', '_')}_rf.joblib"
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
        cnn_data = self.prepare_dataset_for_cnn(audio_dir)
        rf_data = self.prepare_dataset_for_rf(audio_dir)
        
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
            filename = f"{dict_name.replace(' ', '_')}_ensemble"
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
        
        def train_thread_fn():
            try:
                self.is_training = True
                result = self.train_model(model_type, audio_dir, **kwargs)
                
                if callback:
                    callback(result)
            except Exception as e:
                logging.error(f"Error in training thread: {e}")
                if callback:
                    callback({
                        'success': False,
                        'error': str(e)
                    })
            finally:
                self.is_training = False
        
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
        return self.training_stats
        
    def is_training_in_progress(self):
        """
        Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress, False otherwise
        """
        return self.is_training 