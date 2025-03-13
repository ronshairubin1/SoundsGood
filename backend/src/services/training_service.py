import os
import logging
import sys
import numpy as np
from threading import Thread
from datetime import datetime
import copy
import time
import json
import traceback
import librosa
import tensorflow as tf
import h5py  # Using HDF5 for better feature caching
import hashlib
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from flask import current_app  # Import current_app for app context access

# Configure additional stream handler for console output
root_logger = logging.getLogger()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

from backend.src.core.ml_algorithms import create_model
# Try to import directly from backend first
try:
    from backend.audio.processor import SoundProcessor as AudioProcessor
    logging.info("Using SoundProcessor from backend.audio.processor for training")
except ImportError:
    # If not found, use the bridge which will eventually delegate to SoundProcessor
    from backend.src.core.audio.processor_bridge import AudioProcessor
    logging.info("Using AudioProcessor bridge from backend.src.core.audio.processor_bridge for training")
from backend.src.core.data.dataset_preparation import DatasetPreparation
from backend.src.services.training_analysis_service import TrainingAnalysisService
from backend.features.extractor import FeatureExtractor
from backend.config import Config
from backend.audio import AudioPreprocessor
from backend.audio.augmentor import AudioAugmentor
from backend.src.ml.model_paths import update_model_registry, synchronize_model_registry, trigger_model_registry_update
import time

class TrainingService:
    """
    Service for training different types of models.
    This service handles dataset preparation, feature extraction, 
    and model training for CNN, RF, and Ensemble models.
    
    Primary method is train_unified() which provides consistent 
    preprocessing and feature extraction across all model types 
    with direct implementations of both CNN and RF training algorithms.
    """
    
    # Class variable to store epoch-by-epoch training data
    latest_epoch_data = []
    
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
            sample_rate=8000  # Lower sample rate for better handling of short files
        )
        self.training_stats = {
            'stats': {
                'original_samples': 0,
                'augmented_samples': 0,
                'total_samples': 0
            }
        }
        self.training_thread = None
        self.is_training = False
        self.error_logs = []  # Store error logs
        self.file_errors = []  # Store detailed file-specific errors
        # Initialize features cache directory
        self.features_cache_dir = os.path.join(self.model_dir, 'features_cache')
        os.makedirs(self.features_cache_dir, exist_ok=True)
        # Initialize AudioAugmentor for additional augmentations
        self.augmentor = AudioAugmentor(sample_rate=8000)
        # Initialize feature extractor for feature extraction checks
        self.extractor = FeatureExtractor(sample_rate=8000, cache_dir=self.features_cache_dir)
    
    def generate_augmentations(self, audio_file_path, pitch_count=3, stretch_count=3, noise_count=3):
        """
        Generate augmented versions of an audio file based on specified parameters.
        
        Args:
            audio_file_path (str): Path to the audio file to augment
            pitch_count (int): Number of pitch variations to generate (0-5)
            stretch_count (int): Number of time stretch variations to generate (0-5)
            noise_count (int): Number of noise level variations to generate (0-5)
            
        Returns:
            list: Paths to generated augmented files
        """
        logging.info(f"Generating augmentations for {audio_file_path} with parameters: "
                  f"pitch={pitch_count}, stretch={stretch_count}, noise={noise_count}")
        
        # Lists to store augmentation parameters
        pitch_shifts = []
        time_stretches = []
        noise_levels = []
        
        # Generate pitch shift parameters
        if pitch_count > 0:
            pitch_steps = np.linspace(-3, 3, pitch_count)
            for step in pitch_steps:
                if step != 0:  # Skip neutral pitch shift
                    pitch_shifts.append(step)
        
        # Generate time stretch parameters
        if stretch_count > 0:
            stretch_factors = np.linspace(0.8, 1.2, stretch_count)
            for factor in stretch_factors:
                if abs(factor - 1.0) > 0.05:  # Skip neutral time stretch
                    time_stretches.append(factor)
        
        # Generate noise level parameters
        if noise_count > 0:
            noise_factors = np.linspace(0.01, 0.1, noise_count)
            for factor in noise_factors:
                if factor > 0.01:  # Skip very low noise
                    noise_levels.append(factor)
        
        # Get file info for naming the augmented files
        file_dir = os.path.dirname(audio_file_path)
        filename = os.path.basename(audio_file_path)
        filename_base, ext = os.path.splitext(filename)
        
        generated_files = []
        
        # Generate the augmentations
        total_variations = 0
        if pitch_shifts and time_stretches and noise_levels:
            total_variations = len(pitch_shifts) * len(time_stretches) * len(noise_levels)
        elif pitch_shifts and time_stretches:
            total_variations = len(pitch_shifts) * len(time_stretches)
        elif pitch_shifts and noise_levels:
            total_variations = len(pitch_shifts) * len(noise_levels)
        elif time_stretches and noise_levels:
            total_variations = len(time_stretches) * len(noise_levels)
        elif pitch_shifts:
            total_variations = len(pitch_shifts)
        elif time_stretches:
            total_variations = len(time_stretches)
        elif noise_levels:
            total_variations = len(noise_levels)
        
        logging.info(f"Will generate {total_variations} variations for {filename}")
        
        current_count = 0
        
        # Use our existing AudioAugmentor for consistent processing
        for pitch in pitch_shifts or [0]:
            for stretch in time_stretches or [1.0]:
                for noise in noise_levels or [0]:
                    # Skip if all parameters are neutral
                    if pitch == 0 and abs(stretch - 1.0) < 0.05 and noise < 0.01:
                        continue
                    
                    current_count += 1
                    aug_suffix = f"_aug_{current_count}"
                    out_filename = f"{filename_base}{aug_suffix}{ext}"
                    out_path = os.path.join(file_dir, out_filename)
                    
                    try:
                        # Load audio
                        y, sr = librosa.load(audio_file_path, sr=None)
                        
                        # Apply pitch shift if needed
                        if pitch != 0:
                            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(pitch))
                        
                        # Apply time stretch if needed
                        if abs(stretch - 1.0) >= 0.05:
                            y = librosa.effects.time_stretch(y, rate=float(stretch))
                        
                        # Apply noise if needed
                        if noise > 0.01:
                            noise_amp = noise * np.random.randn(len(y))
                            y = y + noise_amp
                        
                        # Normalize audio
                        if np.max(np.abs(y)) > 0:
                            y = y / np.max(np.abs(y))
                        
                        # Save the augmented file
                        librosa.output.write_wav(out_path, y, sr)
                        generated_files.append(out_path)
                        
                        # Extract features for the augmented file for faster training later
                        try:
                            self.extractor.extract_features(out_path)
                            logging.debug(f"Features extracted for augmented file {out_filename}")
                        except Exception as e:
                            logging.error(f"Failed to extract features for {out_filename}: {e}")
                        
                    except Exception as e:
                        logging.error(f"Error generating augmentation {current_count}: {e}")
                        continue
        
        logging.info(f"Generated {len(generated_files)} augmented files for {filename}")
        return generated_files
        
    def generate_augmentations(self, audio_dir, num_variations=3, progress_callback=None):
        """
        Generate augmented versions of all original sound files in the specified directory.
        
        Args:
            audio_dir (str): Directory containing class folders with audio files
            num_variations (int): Number of augmented versions to create per original file
            progress_callback (callable): Optional callback for progress reporting
            
        Returns:
            dict: Statistics about the augmentation process
        """
        if not os.path.isdir(audio_dir):
            logging.error(f"Audio directory not found: {audio_dir}")
            return {"error": f"Directory not found: {audio_dir}"}
            
        results = {
            "classes": {},
            "total_original_files": 0,
            "total_augmented_generated": 0,
            "files_with_errors": []
        }
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
        
        if progress_callback:
            progress_callback(0, f"Generating augmentations for {len(class_dirs)} sound classes...")
            
        # Track progress
        total_dirs = len(class_dirs)
        for idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(audio_dir, class_dir)
            logging.info(f"Processing class: {class_dir}")
            
            # Initialize class statistics
            class_stats = {
                "original_files": 0,
                "augmented_generated": 0,
                "files_with_errors": []
            }
            
            # Get all WAV files in this class directory
            all_wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            # Process only original files (not already augmented)
            original_files = [f for f in all_wav_files if "_aug_" not in f]
            class_stats["original_files"] = len(original_files)
            results["total_original_files"] += len(original_files)
            
            # Generate augmentations for each original file
            for wav_file in original_files:
                full_path = os.path.join(class_path, wav_file)
                
                try:
                    # Generate variations with different parameters
                    pitch_shifts = [1, -1, 2] if num_variations >= 3 else [1] 
                    time_stretches = [0.9, 1.1, 1.2] if num_variations >= 3 else [0.9]
                    noise_levels = [0.005, 0.01, 0.02] if num_variations >= 3 else [0.005]
                    
                    augmented_files = self.create_augmented_versions(
                        full_path, 
                        pitch_shifts=pitch_shifts[:num_variations],
                        time_stretches=time_stretches[:num_variations],
                        noise_levels=noise_levels[:num_variations]
                    )
                    
                    class_stats["augmented_generated"] += len(augmented_files)
                    results["total_augmented_generated"] += len(augmented_files)
                    
                except Exception as e:
                    error_info = {"file": wav_file, "error": str(e)}
                    class_stats["files_with_errors"].append(error_info)
                    results["files_with_errors"].append(error_info)
                    logging.error(f"Error generating augmentations for {wav_file}: {e}")
            
            # Add class statistics to results
            results["classes"][class_dir] = class_stats
            
            if progress_callback:
                progress_callback(int(100 * (idx + 1) / total_dirs), 
                                 f"Processed {idx + 1}/{total_dirs} classes")
        
        return results
        
    def analyze_sound_file_status(self, audio_dir, progress_callback=None):
        """
        Analyze sound files to check augmentation and feature extraction status.
        
        For each original recording, this method will:
        1. Count how many augmented versions exist
        2. Check if feature extraction has been performed for each file
        3. Return detailed statistics about the current state
        
        Args:
            audio_dir (str): Directory containing class folders with audio files
            progress_callback (callable): Optional callback for progress reporting
            
        Returns:
            dict: Detailed statistics about sound files and feature extraction status
        """
        if not os.path.isdir(audio_dir):
            logging.error(f"Audio directory not found: {audio_dir}")
            return {"error": f"Directory not found: {audio_dir}"}
            
        results = {
            "classes": {},
            "total_original_files": 0,
            "total_augmented_files": 0,
            "total_files_with_features": 0,
            "total_files_missing_features": 0
        }
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
        
        if progress_callback:
            progress_callback(0, f"Analyzing {len(class_dirs)} sound classes...")
            
        # Track progress
        total_dirs = len(class_dirs)
        for idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(audio_dir, class_dir)
            logging.info(f"Analyzing class: {class_dir}")
            
            # Initialize class statistics
            class_stats = {
                "original_files": [],
                "augmented_files": [],
                "files_with_features": [],
                "files_missing_features": [],
                "augmentation_ratio": 0
            }
            
            # Get all WAV files in this class directory
            all_wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            # Separate original and augmented files
            for wav_file in all_wav_files:
                full_path = os.path.join(class_path, wav_file)
                
                # Check if this is an augmented file based on filename patterns
                # The augmentation creates files with '_aug_X' pattern
                is_augmented = "_aug_" in wav_file
                
                if is_augmented:
                    class_stats["augmented_files"].append({
                        "path": full_path,
                        "filename": wav_file
                    })
                else:
                    class_stats["original_files"].append({
                        "path": full_path,
                        "filename": wav_file
                    })
                    
                # Check if feature extraction has been performed for this file
                # We'll use the feature extractor to check if features exist
                from src.ml.feature_extractor import FeatureExtractor
                feature_extractor = FeatureExtractor()
                has_features = feature_extractor.check_features_exist(full_path)
                
                if has_features:
                    class_stats["files_with_features"].append({
                        "path": full_path,
                        "filename": wav_file
                    })
                else:
                    class_stats["files_missing_features"].append({
                        "path": full_path,
                        "filename": wav_file
                    })
            
            # ALSO CHECK THE AUGMENTED DIRECTORY FOR THIS CLASS
            # Get path to augmented directory for this class
            augmented_class_path = os.path.join(Config.AUGMENTED_SOUNDS_DIR, class_dir)
            
            # Add explicit debug output
            print(f"\n\n=== DEBUG INFO ===\nAugmented directory path: {augmented_class_path}")
            print(f"Directory exists: {os.path.isdir(augmented_class_path)}")
            print(f"Parent directory exists: {os.path.isdir(Config.AUGMENTED_SOUNDS_DIR)}")
            print(f"Parent directory contents: {os.listdir(Config.AUGMENTED_SOUNDS_DIR) if os.path.isdir(Config.AUGMENTED_SOUNDS_DIR) else 'N/A'}")
            
            if os.path.isdir(augmented_class_path):
                logging.info(f"Checking augmented directory for class: {class_dir}")
                print(f"Found augmented directory for class: {class_dir}")
                
                # Get all WAV files in the augmented class directory
                aug_wav_files = [f for f in os.listdir(augmented_class_path) if f.endswith('.wav')]
                print(f"Found {len(aug_wav_files)} augmented WAV files for class {class_dir}")
                
                for wav_file in aug_wav_files:
                    full_path = os.path.join(augmented_class_path, wav_file)
                    
                    # All files in the augmented directory are considered augmented
                    class_stats["augmented_files"].append({
                        "path": full_path,
                        "filename": wav_file
                    })
                    
                    # Check if feature extraction has been performed for this file
                    has_features = feature_extractor.check_features_exist(full_path)
                    
                    if has_features:
                        class_stats["files_with_features"].append({
                            "path": full_path,
                            "filename": wav_file
                        })
                    else:
                        class_stats["files_missing_features"].append({
                            "path": full_path,
                            "filename": wav_file
                        })
            
            # Calculate augmentation ratio
            num_original = len(class_stats["original_files"])
            num_augmented = len(class_stats["augmented_files"])
            
            class_stats["augmentation_ratio"] = 0 if num_original == 0 else num_augmented / num_original
            
            # Add class statistics to results
            results["classes"][class_dir] = {
                "original_count": num_original,
                "augmented_count": num_augmented,
                "total_count": num_original + num_augmented,
                "files_with_features": len(class_stats["files_with_features"]),
                "files_missing_features": len(class_stats["files_missing_features"]),
                "augmentation_ratio": class_stats["augmentation_ratio"],
                "details": class_stats
            }
            
            # Update totals
            results["total_original_files"] += num_original
            results["total_augmented_files"] += num_augmented
            results["total_files_with_features"] += len(class_stats["files_with_features"])
            results["total_files_missing_features"] += len(class_stats["files_missing_features"])
            
            # Update progress
            if progress_callback:
                progress_callback(int((idx + 1) / total_dirs * 50), 
                                 f"Analyzed class {class_dir}: {num_original} original, {num_augmented} augmented files")
                
        return results
        
    def _check_features_exist(self, audio_file_path):
        """
        Check if features have already been extracted for a given audio file.
        
        Args:
            audio_file_path (str): Path to the audio file to check
            
        Returns:
            bool: True if features exist, False otherwise
        """
        try:
            # Get the feature cache directory
            feature_cache_dir = os.path.join(self.model_dir, 'features_cache')
            
            # Generate a cache key based on the file path and default parameters
            # This should match the key generation in FeatureExtractor
            file_hash = hashlib.md5(audio_file_path.encode()).hexdigest()
            feature_types = ['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz']
            
            # Check if feature files exist for this audio file
            for feature_type in feature_types:
                cache_file = os.path.join(feature_cache_dir, f"{file_hash}_{feature_type}.npy")
                if not os.path.exists(cache_file):
                    # If any feature type is missing, return False
                    return False
            
            # All feature files exist
            return True
        except Exception as e:
            logging.error(f"Error checking features for {audio_file_path}: {e}")
            return False
    
    def update_sound_files(self, audio_dir, target_augmentation_count=27, extract_features=True, progress_callback=None):
        """
        Update sound files by generating missing augmentations and extracting features.
        
        This method will:
        1. Generate augmentations for original recordings that don't have enough
        2. Extract features for all files that are missing feature extraction
        
        Args:
            audio_dir (str): Directory containing class folders with audio files
            target_augmentation_count (int): Target number of augmentations per original file (default is 3×3×3=27)
            extract_features (bool): Whether to extract features for files missing them
            progress_callback (callable): Optional callback for progress reporting
            
        Returns:
            dict: Results of the update operation
        """
        # First analyze the current status
        status = self.analyze_sound_file_status(audio_dir, progress_callback)
        
        if "error" in status:
            return status
            
        results = {
            "augmentations_generated": 0,
            "features_extracted": 0,
            "classes_updated": []
        }
        
        # Process each class
        class_dirs = list(status["classes"].keys())
        total_dirs = len(class_dirs)
        
        for idx, class_dir in enumerate(class_dirs):
            class_stats = status["classes"][class_dir]
            class_path = os.path.join(audio_dir, class_dir)
            class_results = {
                "name": class_dir,
                "augmentations_generated": 0,
                "features_extracted": 0
            }
            
            self.logger.info(f"Processing class: {class_dir}")
            if progress_callback:
                progress_callback(50 + int((idx + 0.5) / total_dirs * 25), 
                                 f"Processing class {class_dir}...")
            
            # First, generate missing augmentations for original files
            original_files = class_stats["details"]["original_files"]
            for orig_file in original_files:
                # Count existing augmentations for this file
                base_name = os.path.splitext(orig_file["filename"])[0]
                existing_augs = [f for f in class_stats["details"]["augmented_files"] 
                               if f["filename"].startswith(base_name + "_")]
                
                # Calculate how many augmentations to generate
                missing_augs = max(0, target_augmentation_count - len(existing_augs))
                
                if missing_augs > 0:
                    self.logger.info(f"Generating {missing_augs} augmentations for {orig_file['filename']}")
                    
                    # Create augmented versions using AudioAugmentor
                    aug_dir = os.path.join(Config.AUGMENTED_SOUNDS_DIR, class_dir)
                    os.makedirs(aug_dir, exist_ok=True)
                    
                    augmented_files = self.augmentor.augment_file(
                        orig_file["path"], 
                        aug_dir, 
                        count=missing_augs
                    )
                    
                    # Update counts
                    num_created = len(augmented_files)
                    class_results["augmentations_generated"] += num_created
                    results["augmentations_generated"] += num_created
            
            # Next, extract features for files missing them if requested
            if extract_features:
                missing_features_files = class_stats["details"]["files_missing_features"]
                
                for file_info in missing_features_files:
                    self.logger.info(f"Extracting features for {file_info['filename']}")
                    
                    # Extract features using the FeatureExtractor
                    features = self.extractor.extract_features(file_info["path"])
                    
                    if features is not None:
                        class_results["features_extracted"] += 1
                        results["features_extracted"] += 1
            
            # Add class results to overall results
            results["classes_updated"].append(class_results)
            
            # Update progress
            if progress_callback:
                progress_callback(50 + int((idx + 1) / total_dirs * 50), 
                                 f"Updated class {class_dir}: {class_results['augmentations_generated']} augmentations, "
                                 f"{class_results['features_extracted']} features")
        
        return results
        
        # Feature cache directory setup
        self.features_cache_dir = os.path.join(self.model_dir, 'features_cache')
        os.makedirs(self.features_cache_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('TrainingService')
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG level for more detailed logs
    
    # ===== UNIFIED TRAINING METHOD (PREFERRED) =====
    
    def clear_feature_cache(self, cache_dir=None):
        """
        Clear the feature cache directory to force regeneration of features.
        
        Args:
            cache_dir (str): Optional custom cache directory to clear.
                             If None, uses the default features_cache_dir.
        
        Returns:
            dict: Results of the operation
        """
        try:
            import os, shutil
            cache_dir = cache_dir or self.features_cache_dir
            
            # Make sure the directory exists
            if not os.path.exists(cache_dir):
                return {'success': True, 'message': 'Cache directory does not exist', 'cleared': 0}
            
            # Count number of files in the directory
            file_count = len([f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))])
            
            # Clear all files in the directory
            cleared_count = 0
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    cleared_count += 1
            
            logging.info(f"Cleared {cleared_count} files from feature cache at {cache_dir}")
            return {'success': True, 'message': f'Cleared {cleared_count} cached feature files', 'cleared': cleared_count}
            
        except Exception as e:
            logging.error(f"Error clearing feature cache: {str(e)}")
            return {'success': False, 'message': f'Error clearing cache: {str(e)}', 'cleared': 0}
    
    def train_unified(self, model_type, audio_dir, save=True, progress_callback=None, **kwargs):
        """
        Train a model using the unified feature extractor. This is the RECOMMENDED method
        for training new models as it provides consistent preprocessing and standardized
        feature extraction across all model types.
        
        Args:
            model_type (str): Type of model to train ('cnn', 'rf', 'ensemble')
            audio_dir (str): Directory containing class folders with audio files
            save (bool): Whether to save the trained model
            progress_callback (callable): Optional callback for progress reporting
            **kwargs: Additional model-specific parameters including:
                dict_name (str): Name of the dictionary being trained
                dict_classes (list): List of class names in the dictionary (if provided)
                clear_cache (bool): Whether to clear the feature cache before training
            
        Returns:
            dict: Training results
        """
        dict_name = kwargs.get('dict_name', '')
        dict_classes = kwargs.get('dict_classes', [])
        model_name = kwargs.get('model_name', f"{dict_name}_{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        # Whether to include augmented files in training
        include_augmented = kwargs.get('include_augmented', False)
        
        # Whether to generate additional augmentations before training
        generate_augmentations = kwargs.get('generate_augmentations', False)
        target_augmentation_count = kwargs.get('target_augmentation_count', 27)  # 3×3×3 augmentation matrix by default
        
        logging.info(f"Training with augmented files: {include_augmented}")
        logging.info(f"Generate additional augmentations: {generate_augmentations} (target count: {target_augmentation_count})")
        
        # If requested, generate additional augmentations before training
        if generate_augmentations:
            logging.info(f"Analyzing and updating sound files in {audio_dir}...")
            if progress_callback:
                progress_callback(0, "Analyzing current augmentation status...")
                
            # Update sound files - generate missing augmentations and extract features
            update_results = self.update_sound_files(
                audio_dir, 
                target_augmentation_count=target_augmentation_count,
                extract_features=True,
                progress_callback=progress_callback
            )
            
            logging.info(f"Sound file update results: {update_results}")
            if progress_callback:
                progress_callback(10, f"Generated {update_results['augmentations_generated']} new augmentations and "
                                 f"extracted features for {update_results['features_extracted']} files")
        
        logging.info(f"Starting unified {model_type} model training for dictionary {dict_name} with FeatureExtractor")
        
        # Initialize training stats dictionary
        training_stats = {
            'original_files': 0,
            'augmented_files': 0,
            'total_files': 0,
            'original_samples': 0,
            'augmented_samples': 0,
            'total_samples': 0
        }
        
        # Handle special case for EhOh - expand class names if needed
        if dict_name == 'EhOh':
            # Clear any previous class specifications and use the standard EhOh classes
            logging.info("======= SPECIAL HANDLING FOR EHOH DICTIONARY =======")
            # Force using exactly these classes for EhOh regardless of what was passed
            dict_classes = ['eh', 'oh']
            logging.info(f"EHOH CONFIG: Using only exact classes: {dict_classes}")
            # Print to console for better visibility
            print(f"[TRAINING] EHOH CONFIG: Using only exact classes: {dict_classes}")
            # Log available classes in the audio directory for debugging
            try:
                audio_dir_path = Config.get_training_sounds_dir()
                available_dirs = [d for d in os.listdir(audio_dir_path) if os.path.isdir(os.path.join(audio_dir_path, d))]
                logging.info(f"EHOH DEBUG: Available classes in audio dir: {available_dirs}")
                print(f"[TRAINING] EHOH DEBUG: Available classes in audio dir: {available_dirs}")
                # Check if 'eh' and 'oh' directories exist
                eh_exists = 'eh' in available_dirs
                oh_exists = 'oh' in available_dirs
                logging.info(f"EHOH DEBUG: 'eh' directory exists: {eh_exists}, 'oh' directory exists: {oh_exists}")
                print(f"[TRAINING] EHOH DEBUG: 'eh' exists: {eh_exists}, 'oh' exists: {oh_exists}")
            except Exception as e:
                logging.error(f"EHOH ERROR checking directories: {str(e)}")
                print(f"[TRAINING] EHOH ERROR checking directories: {str(e)}")
        
        # Check if an identical model already exists
        if dict_name and save:
            identical_model = self._check_identical_model(model_type, dict_name, kwargs)
            if identical_model:
                msg = f"A model with identical parameters already exists: {identical_model}"
                logging.info(msg)
                if progress_callback:
                    progress_callback(100, f"Training complete! Used existing model: {identical_model}")
                return {'status': 'success', 'message': msg, 'model_path': identical_model, 'used_existing': True}
        
        # Initialize audio processing components
        audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            sound_threshold=0.008,
            min_silence_duration=0.1,
            target_duration=1.0,
            normalize_audio=True
        )
        
        # Extract clear_cache parameter from kwargs
        clear_cache = kwargs.get('clear_cache', False)
        logging.info(f"Clear cache option: {clear_cache}")
        if clear_cache:
            # Clear feature cache if requested
            cache_result = self.clear_feature_cache()
            logging.info(f"Cache clearing result: {cache_result}")
            print(f"[TRAINING] Cleared feature cache: {cache_result['cleared']} files removed")
        
        # Initialize the feature extractor based on model type
        if model_type == 'cnn':
            unified_extractor = FeatureExtractor(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=256,
                use_cache=not clear_cache  # Don't use cache if clear_cache is True
            )
        else:
            unified_extractor = FeatureExtractor(
                sample_rate=16000,
                use_cache=not clear_cache  # Don't use cache if clear_cache is True
            )
            
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
            # Log all available information for debugging
            logging.info(f"Audio directory: {audio_dir}")
            logging.info(f"Available class directories: {class_dirs}")
            logging.info(f"Dictionary classes from params: {dict_classes}")
            
            # Fix: Case-insensitive class matching and more robust handling
            dict_classes_lower = [c.lower() if isinstance(c, str) else c for c in dict_classes]
            processed_classes = 0
            
            # Special handling for EhOh dictionary
            if dict_name == 'EhOh':
                # Always use the standard classes for EhOh
                dict_classes = ['eh', 'oh']
                dict_classes_lower = ['eh', 'oh']
                logging.info(f"Using standard EhOh classes: {dict_classes}")
                print(f"[TRAINING] Using standard EhOh classes: {dict_classes}")
                # Log class directories to help with debugging
                logging.info(f"Available class directories: {class_dirs}")
                print(f"[TRAINING] Available class directories: {class_dirs}")
                # Force filter class_dirs for EhOh to use only the exact 'eh' and 'oh' classes
                original_class_dirs = class_dirs.copy()
                class_dirs = [d for d in class_dirs if d == 'eh' or d == 'oh']
                logging.info(f"EHOH: Filtered class directories from {len(original_class_dirs)} to {len(class_dirs)}: {class_dirs}")
                print(f"[TRAINING] EHOH: Using only these exact classes: {class_dirs}")
            
            # For empty dict_classes, include all available classes
            if not dict_classes:
                logging.info(f"No dictionary classes provided, using all available classes: {class_dirs}")
                # Process all class directories if no dictionary classes are specified
                dict_classes_lower = [d.lower() for d in class_dirs]
            
            logging.info(f"Using dictionary classes (normalized): {dict_classes_lower}")
            
            # Process each class directory, but only if it's in the dictionary classes or if no specific classes are provided
            for class_idx, class_dir in enumerate(class_dirs):
                class_dir_lower = class_dir.lower()
                # Full path to the class directory
                class_path = os.path.join(audio_dir, class_dir)
                
                # Check if the directory exists and has files
                if not os.path.isdir(class_path):
                    logging.warning(f"Class directory {class_path} is not a valid directory. Skipping.")
                    print(f"[TRAINING] WARNING: Class directory {class_path} is not a valid directory. Skipping.")
                    continue
                    
                wav_files_count = len([f for f in os.listdir(class_path) if f.endswith('.wav')])
                logging.info(f"Class {class_dir} has {wav_files_count} WAV files")
                print(f"[TRAINING] Class {class_dir} has {wav_files_count} WAV files")
                
                # Check if the class directory matches any of the dictionary classes
                matches_dict_class = False
                if not dict_classes:
                    # If no dictionary classes specified, include all
                    matches_dict_class = True
                    logging.info(f"Including class {class_dir} as no specific dictionary classes were specified")
                    print(f"[TRAINING] Including class {class_dir} as no specific dictionary classes were specified")
                    
                    # Also check if we should check the augmented directory for this class
                    if include_augmented:
                        aug_class_path = os.path.join(Config.AUGMENTED_SOUNDS_DIR, class_dir)
                        if os.path.isdir(aug_class_path):
                            aug_wav_files_count = len([f for f in os.listdir(aug_class_path) if f.endswith('.wav')])
                            logging.info(f"Augmented directory for class {class_dir} has {aug_wav_files_count} WAV files")
                            print(f"[TRAINING] Augmented directory for class {class_dir} has {aug_wav_files_count} WAV files")
                else:
                    # Special handling for EhOh dictionary - ONLY match exact classes
                    if dict_name == 'EhOh':
                        # For EhOh, use EXACT matching only - no fuzzy/partial matching
                        # Add additional check to only include specific classes
                        if class_dir == 'eh' or class_dir == 'oh':  # Use exact case matching
                            matches_dict_class = True
                            # Enhanced logging for EhOh training
                            logging.info(f"EHOH MATCH: Class '{class_dir}' is exactly one of the required classes")
                            print(f"[TRAINING] EHOH MATCH: Using class '{class_dir}'")
                        else:
                            # Log what we're skipping for better debugging
                            logging.info(f"EHOH SKIP: Skipping class '{class_dir}' as it's not exactly 'eh' or 'oh'")
                            print(f"[TRAINING] EHOH SKIP: Ignoring class '{class_dir}' as it's not exactly 'eh' or 'oh'")
                    else:
                        # For other dictionaries, try exact match first
                        if class_dir_lower in dict_classes_lower:
                            matches_dict_class = True
                            logging.info(f"Exact match: Class '{class_dir}' matches dictionary class entry")
                            print(f"[TRAINING] Exact match: Class '{class_dir}' matches dictionary class entry")
                        else:
                            # Try partial match for vowel sounds (only for non-EhOh dictionaries)
                            for dict_class in dict_classes_lower:
                                # If the class directory contains the dictionary class name or vice versa
                                if dict_class in class_dir_lower or class_dir_lower in dict_class:
                                    matches_dict_class = True
                                    logging.info(f"Partial match: '{class_dir}' matches dictionary class '{dict_class}'")
                                    print(f"[TRAINING] Partial match: '{class_dir}' matches dictionary class '{dict_class}'")
                                    break
                
                if not matches_dict_class:
                    logging.info(f"Skipping class {class_dir} as it's not in the specified dictionary classes ({dict_classes})")
                    print(f"[TRAINING] Skipping class {class_dir} as it's not in the specified dictionary classes")
                    continue
                
                processed_classes += 1
                all_wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
                
                # Identify original vs augmented files
                original_files = []
                augmented_files = []
                
                # Augmented files have patterns like _p-2_t0.85_n0.005.wav or _aug1.wav
                for f in all_wav_files:
                    # Check for augmentation patterns in the filename
                    if "_p" in f and "_t" in f and "_n" in f:  # New systematic augmentation format
                        augmented_files.append(f)
                    elif "_aug" in f:  # Legacy augmentation format
                        augmented_files.append(f)
                    else:
                        original_files.append(f)
                
                # Log the breakdown of original vs augmented files
                logging.info(f"Class {class_dir} breakdown - Original: {len(original_files)}, Augmented: {len(augmented_files)}, Total: {len(all_wav_files)}")
                print(f"[TRAINING] Class {class_dir} breakdown - Original: {len(original_files)}, Augmented: {len(augmented_files)}, Total: {len(all_wav_files)}")
                
                # Decide which files to use based on include_augmented flag
                if include_augmented:
                    wav_files = all_wav_files
                    logging.info(f"Processing class {class_dir} with all {len(wav_files)} WAV files (including augmented)")
                    print(f"[TRAINING] Processing class {class_dir} with all {len(wav_files)} WAV files (including augmented)")
                else:
                    wav_files = original_files
                    logging.info(f"Processing class {class_dir} with {len(wav_files)} original WAV files (excluding augmented)")
                    print(f"[TRAINING] Processing class {class_dir} with {len(wav_files)} original WAV files (excluding augmented)")
                    
                # Update statistics
                training_stats['original_files'] += len(original_files)
                training_stats['augmented_files'] += len(augmented_files)
                training_stats['total_files'] += len(all_wav_files)
                
                # Update UI stats as well
                self.training_stats['stats'] = {
                    'original_samples': training_stats['original_files'],
                    'augmented_samples': training_stats['augmented_files'],
                    'total_samples': training_stats['total_files']
                }
                
                if not wav_files:
                    logging.warning(f"No .wav files found in {class_path}")
                    continue
                
                logging.info(f"Preprocessing {len(wav_files)} files for class {class_dir}")
                
                # Process each file in the class
                processed_files_count = 0
                
                # Process files from the main training directory
                for file_idx, wav_file in enumerate(wav_files):
                    file_path = os.path.join(class_path, wav_file)
                    
                    try:
                        # Preprocess the audio file
                        processed_audio = audio_preprocessor.preprocess_file(file_path)
                        
                        if processed_audio is not None:
                            preprocessed_files.append((processed_audio, file_path))
                            class_labels.append(class_dir)
                            processed_files_count += 1
                            
                            # Update statistics
                            if any(aug_pattern in wav_file for aug_pattern in ['_p', '_t', '_n', '_aug']):
                                training_stats['augmented_samples'] += 1
                            else:
                                training_stats['original_samples'] += 1
                            training_stats['total_samples'] += 1
                            
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
                
                # If augmented files are requested, process files from augmented directory
                if include_augmented:
                    aug_class_path = os.path.join(Config.AUGMENTED_SOUNDS_DIR, class_dir)
                    if os.path.isdir(aug_class_path):
                        aug_wav_files = [f for f in os.listdir(aug_class_path) if f.endswith('.wav')]
                        logging.info(f"Including {len(aug_wav_files)} augmented files from {aug_class_path}")
                        print(f"[TRAINING] Including {len(aug_wav_files)} augmented files for class {class_dir}")
                        
                        for file_idx, wav_file in enumerate(aug_wav_files):
                            file_path = os.path.join(aug_class_path, wav_file)
                            
                            try:
                                # Preprocess the augmented audio file
                                processed_audio = audio_preprocessor.preprocess_file(file_path)
                                
                                if processed_audio is not None:
                                    preprocessed_files.append((processed_audio, file_path))
                                    class_labels.append(class_dir)
                                    processed_files_count += 1
                                    
                                    # Update augmented statistics
                                    training_stats['augmented_samples'] += 1
                                    training_stats['total_samples'] += 1
                                    
                                    # Report progress
                                    if progress_callback and (file_idx % 5 == 0 or file_idx == len(aug_wav_files) - 1):
                                        progress_callback(
                                            total_progress,
                                            f"Preprocessing augmented {class_dir}: {file_idx+1}/{len(aug_wav_files)} files"
                                        )
                            except Exception as e:
                                logging.error(f"Error preprocessing augmented file {file_path}: {str(e)}")
                
                logging.info(f"Processed a total of {processed_files_count} files for class {class_dir}")
            
            # Check if we actually processed any classes
            if processed_classes == 0:
                error_msg = f"No matching classes found between dictionary classes {dict_classes} and available folders {class_dirs}"
                logging.error(error_msg)
                print(f"[TRAINING] ERROR: {error_msg}")
                
                # Provide debugging information
                logging.error(f"Dictionary name: {dict_name}")
                logging.error(f"Dict classes (original): {dict_classes}")
                logging.error(f"Dict classes (normalized): {dict_classes_lower}")
                logging.error(f"Available class directories: {class_dirs}")
                
                print(f"[TRAINING] Dictionary name: {dict_name}")
                print(f"[TRAINING] Dict classes (original): {dict_classes}")
                print(f"[TRAINING] Dict classes (normalized): {dict_classes_lower}")
                print(f"[TRAINING] Available class directories: {class_dirs}")
                
                # Check directory contents to ensure classes exist
                logging.error(f"Directory structure in {audio_dir}:")
                print(f"[TRAINING] Directory structure in {audio_dir}:")
                try:
                    dirs = os.listdir(audio_dir)
                    for d in dirs:
                        full_path = os.path.join(audio_dir, d)
                        if os.path.isdir(full_path):
                            file_count = len([f for f in os.listdir(full_path) if f.endswith('.wav')])
                            msg = f"  - {d}: {file_count} WAV files"
                            logging.error(msg)
                            print(f"[TRAINING] {msg}")
                except Exception as e:
                    logging.error(f"Error listing directory: {str(e)}")
                    print(f"[TRAINING] Error listing directory: {str(e)}")
                
                if progress_callback:
                    progress_callback(0, f"Error: {error_msg}")
                return {'status': 'error', 'message': error_msg}
            
            if len(preprocessed_files) == 0:
                error_msg = "No valid audio files found after preprocessing. Check audio files."
                logging.error(error_msg)
                if progress_callback:
                    progress_callback(0, f"Error: {error_msg}")
                return {'status': 'error', 'message': error_msg}
            
            if progress_callback:
                # Report on the regular files vs augmented files
                regular_files = [f for f in preprocessed_files if Config.TRAINING_SOUNDS_DIR in f[0]]
                augmented_files = [f for f in preprocessed_files if Config.AUGMENTED_SOUNDS_DIR in f[0]]
                logging.info(f"Processing {len(regular_files)} regular files and {len(augmented_files)} augmented files")
                progress_callback(20, f"Preprocessing complete. Extracting features from {len(preprocessed_files)} files ({len(regular_files)} regular, {len(augmented_files)} augmented)...")
        
            # Now extract features from the preprocessed audio, with caching
            X = []
            y = []
            file_paths = []  # Track file paths for dataset info
            
            # Create cache subdirectory for this set of extraction parameters
            cache_key = self._get_feature_cache_key(model_type, unified_extractor)
            feature_cache_dir = os.path.join(self.features_cache_dir, cache_key)
            os.makedirs(feature_cache_dir, exist_ok=True)
            
            logging.info(f"Feature extraction starting for {len(preprocessed_files)} preprocessed files")
            print(f"[TRAINING] Starting feature extraction for {len(preprocessed_files)} files")
            
            successful_extractions = 0
            failed_extractions = 0
            feature_extraction_errors = []
            
            for idx, (processed_audio, file_path) in enumerate(preprocessed_files):
                try:
                    # Initialize features to None at the beginning to avoid reference errors
                    features = None
                    
                    # Generate a unique identifier for this file
                    file_id = os.path.basename(file_path)
                    class_name = class_labels[idx]
                    cache_file = os.path.join(feature_cache_dir, f"{class_name}_{file_id}.h5")
                    
                    # Debug audio data
                    audio_info = f"Audio shape: {processed_audio.shape if hasattr(processed_audio, 'shape') else 'N/A'}"
                    audio_info += f", dtype: {processed_audio.dtype if hasattr(processed_audio, 'dtype') else 'N/A'}"
                    audio_info += f", min: {np.min(processed_audio) if isinstance(processed_audio, np.ndarray) else 'N/A'}"
                    audio_info += f", max: {np.max(processed_audio) if isinstance(processed_audio, np.ndarray) else 'N/A'}"
                    audio_info += f", length: {len(processed_audio) if hasattr(processed_audio, '__len__') else 'N/A'}"
                    logging.info(f"Processing audio file {idx+1}/{len(preprocessed_files)}: {file_path} - {audio_info}")
                    print(f"[TRAINING] Processing {file_id} - {audio_info}")
                    
                    # Check if features are already cached
                    if os.path.exists(cache_file):
                        try:
                            with h5py.File(cache_file, 'r') as f:
                                # For CNN model, check if we have a direct mel_spectrogram dataset
                                # CNN features are typically stored as a direct mel_spectrogram dataset
                                # rather than a nested structure like RF models
                                if model_type == 'cnn' and 'mel_spectrogram' in f:
                                    logging.info(f"Found mel_spectrogram in cache for {file_id}, using for CNN training")
                                    # For CNN, we just want the mel spectrogram directly
                                    # This is what the CNN training process expects
                                    features = {'mel_spectrogram': f['mel_spectrogram'][()]}
                                    
                                    # Add file_path attribute if available
                                    if 'file_path' in f.attrs:
                                        features['file_path'] = f.attrs['file_path']
                                else:
                                    # For RF or other models, or if mel_spectrogram is not found directly
                                    # Read all features from HDF5 file into a dictionary structure
                                    features = {}
                                    
                                    # Handle feature groups
                                    for group_name in f.keys():
                                        if isinstance(f[group_name], h5py.Group):
                                            # Handle nested dictionary (e.g., 'statistical')
                                            features[group_name] = {}
                                            for key in f[group_name]:
                                                features[group_name][key] = f[group_name][key][()]
                                        else:
                                            # Handle direct dataset (e.g., 'mel_spectrogram')
                                            features[group_name] = f[group_name][()]
                                            
                                    # For CNN, if the 'features' dataset exists instead of 'mel_spectrogram',
                                    # this is likely an old cache format - rename it to what we expect
                                    if model_type == 'cnn' and 'features' in features and 'mel_spectrogram' not in features:
                                        logging.info(f"Found 'features' key instead of 'mel_spectrogram' - converting to expected format")
                                        features['mel_spectrogram'] = features['features']
                                        del features['features']
                                        
                            logging.info(f"Loaded cached features from HDF5 for {file_path}")
                            print(f"[TRAINING] Using cached features for {file_id}")
                        except Exception as e:
                            logging.error(f"Error loading HDF5 cache for {file_id}: {e}")
                            print(f"[TRAINING] Error loading cache: {e}")
                            # If cache loading fails, we'll extract features again
                            features = None
                            
                            # Check for legacy cache format for backward compatibility
                            legacy_cache = os.path.join(feature_cache_dir, f"{class_name}_{file_id}.pkl")
                            if os.path.exists(legacy_cache):
                                try:
                                    with open(legacy_cache, 'rb') as f:
                                        features = pickle.load(f)
                                    logging.info(f"Loaded features from legacy cache for {file_path}")
                                    print(f"[TRAINING] Using legacy cached features for {file_id}")
                                except Exception as e2:
                                    logging.error(f"Error loading legacy cache: {e2}")
                                    logging.error(f"Exception type: {type(e2).__name__}")
                                    logging.error(f"Legacy cache load traceback: {traceback.format_exc()}")
                                    print(f"[TRAINING] Error loading legacy cache: {type(e2).__name__}: {e2}")
                                    features = None
                    else:
                        # Extract all features with detailed error handling
                        logging.info(f"Extracting raw features for {file_path}")
                        print(f"[TRAINING] Extracting raw features for {file_id}")
                        try:
                            logging.info(f"Starting to extract features using unified_extractor: {type(unified_extractor).__name__}")
                            print(f"[TRAINING] Using extractor type: {type(unified_extractor).__name__}")
                            # Debug the processed audio data
                            logging.info(f"Processed audio: type={type(processed_audio)}, is_file=False")
                            if isinstance(processed_audio, np.ndarray):
                                logging.info(f"Audio array: shape={processed_audio.shape}, dtype={processed_audio.dtype}, range=[{np.min(processed_audio)}, {np.max(processed_audio)}]")
                            
                            # Call the feature extraction with extra diagnostic and pass model_type
                            all_features = unified_extractor.extract_features(processed_audio, is_file=False, model_type=model_type)
                            
                            # Log the success and type of returned features
                            logging.info(f"Raw feature extraction completed successfully, returned type: {type(all_features)}")
                        except Exception as e:
                            error_msg = f"Error in raw feature extraction: {str(e)}"
                            logging.error(error_msg)
                            logging.error(f"Exception type: {type(e).__name__}")
                            logging.error(traceback.format_exc())
                            print(f"[TRAINING] CRITICAL ERROR: {error_msg}")
                            all_features = None
                        
                        # Log extracted features info
                        if all_features:
                            feature_info = f"Features extracted: {type(all_features)}"
                            if isinstance(all_features, dict):
                                for k, v in all_features.items():
                                    feature_info += f", {k}: {type(v)}"
                                    if hasattr(v, 'shape'):
                                        feature_info += f" shape={v.shape}"
                            logging.info(feature_info)
                            print(f"[TRAINING] {feature_info}")
                        else:
                            logging.warning(f"No raw features extracted for {file_path}")
                            print(f"[TRAINING] WARNING: No raw features extracted for {file_id}")
                        
                        # Extract model-specific features with enhanced error handling
                        logging.info(f"Extracting model-specific features for {model_type} from {file_path}")
                        print(f"[TRAINING] Extracting {model_type}-specific features for {file_id}")
                        
                        if all_features is None:
                            error_msg = f"Cannot extract model-specific features - raw features are None for {file_id}"
                            logging.error(error_msg)
                            print(f"[TRAINING] ERROR: {error_msg}")
                            feature_extraction_errors.append(error_msg)
                            failed_extractions += 1
                            continue
                            
                        try:
                            # Debug the input to extract_features_for_model
                            if isinstance(all_features, dict):
                                logging.info(f"all_features keys: {list(all_features.keys())}")
                                for k, v in all_features.items():
                                    if hasattr(v, 'shape'):
                                        logging.info(f"Feature '{k}': shape={v.shape}, type={type(v)}, dtype={v.dtype if hasattr(v, 'dtype') else 'N/A'}")
                                    else:
                                        logging.info(f"Feature '{k}': type={type(v)}")
                            
                            # Call the model-specific feature extraction
                            logging.info(f"Calling extract_features_for_model with model_type={model_type}")
                            features = unified_extractor.extract_features_for_model(all_features, model_type=model_type)
                            
                            # Debug the returned features
                            logging.info(f"Model-specific extraction returned: type={type(features)}")
                            
                            if features is None:
                                error_msg = f"Feature extraction returned None for {file_id}"
                                logging.error(error_msg)
                                print(f"[TRAINING] ERROR: {error_msg}")
                                feature_extraction_errors.append(error_msg)
                                failed_extractions += 1
                                continue
                        except Exception as e:
                            error_msg = f"Error extracting {model_type} features: {str(e)}"
                            logging.error(error_msg)
                            logging.error(f"Exception type: {type(e).__name__}")
                            logging.error(traceback.format_exc())
                            print(f"[TRAINING] ERROR: {error_msg}")
                            feature_extraction_errors.append(error_msg)
                            failed_extractions += 1
                            continue
                        
                        # Log model-specific feature info
                        if features is not None:
                            model_feature_info = f"Model features extracted: {type(features)}"
                            if hasattr(features, 'shape'):
                                model_feature_info += f", shape={features.shape}"
                                # Check if the features have valid dimensions
                                if any(dim == 0 for dim in features.shape):
                                    logging.error(f"Invalid feature shape: {features.shape} - has zero dimension")
                                    print(f"[TRAINING] ERROR: Invalid feature shape {features.shape} has zero dimension")
                                    failed_extractions += 1
                                    continue
                            logging.info(model_feature_info)
                            print(f"[TRAINING] {model_feature_info}")
                        else:
                            logging.warning(f"No model-specific features extracted for {file_path}")
                            print(f"[TRAINING] WARNING: No model-specific features extracted for {file_id}")
                        
                        # Cache the features
                        if features is not None:
                            try:
                                with h5py.File(cache_file, 'w') as f:
                                    # Store features in HDF5 format
                                    logging.info(f"Caching features to {cache_file}")
                                    print(f"[TRAINING] Saving features to cache: {file_id}")
                                    if isinstance(features, np.ndarray):
                                        # For CNN, this is typically just the mel spectrogram array
                                        logging.info(f"Saving mel spectrogram with shape: {features.shape}, dtype: {features.dtype}")
                                        # Store as 'mel_spectrogram' instead of 'features' to match what the training process expects
                                        f.create_dataset('mel_spectrogram', data=features, compression="gzip")
                                        # Also store the file path so we can regenerate if needed
                                        f.attrs['file_path'] = file_path
                                    elif isinstance(features, dict):
                                        # For RF, this is typically a dictionary of statistical features
                                        for key, value in features.items():
                                            if isinstance(value, (int, float, bool, np.number)):
                                                f.create_dataset(key, data=value)
                                            elif isinstance(value, np.ndarray):
                                                f.create_dataset(key, data=value, compression="gzip")
                                            else:
                                                # Convert other types to string representation
                                                f.create_dataset(key, data=str(value))
                                    else:
                                        # Fallback for any other type
                                        f.create_dataset('data', data=np.array(features))
                                    
                                logging.info(f"Cached features to HDF5 for {file_path}")
                                print(f"[TRAINING] Cached features for {file_id}")
                                successful_extractions += 1
                                logging.info(f"Successfully extracted and cached features for {file_id}")
                            except Exception as e:
                                logging.error(f"Error caching features to HDF5: {e}")
                                logging.error(f"Cache exception type: {type(e).__name__}")
                                logging.error(f"Cache error traceback: {traceback.format_exc()}")
                                print(f"[TRAINING] Error caching features: {type(e).__name__}: {e}")
                                # Count as successful extraction even if caching failed
                                successful_extractions += 1
                                logging.info(f"Features were extracted but caching failed for {file_id}")
                    
                    if features is not None:
                        X.append(features)
                        y.append(class_name)
                        file_paths.append(file_path)  # Track the file path
                        # Note: successful_extractions is already incremented when features are cached
                        logging.info(f"Added features for {file_path} to training dataset")
                        print(f"[TRAINING] Added features for {file_id} to training dataset")
                    else:
                        failed_extractions += 1
                        error_msg = f"No features extracted for {file_path}"
                        feature_extraction_errors.append(error_msg)
                        logging.error(error_msg)
                        print(f"[TRAINING] ERROR: {error_msg}")
                    
                    # Report progress
                    if progress_callback and (idx % 5 == 0 or idx == len(preprocessed_files) - 1):
                        total_progress = int(20 + (idx / len(preprocessed_files) * 15))  # 20-35% of total
                        extraction_status = f"Extracting features: {idx+1}/{len(preprocessed_files)} files (success: {successful_extractions}, failed: {failed_extractions})"
                        logging.info(f"Progress update: {extraction_status}")
                        
                        # Log recent errors if any
                        if failed_extractions > 0 and feature_extraction_errors:
                            recent_errors = feature_extraction_errors[-min(3, len(feature_extraction_errors)):]
                            for i, error in enumerate(recent_errors):
                                logging.warning(f"Recent error {i+1}: {error}")
                        
                        progress_callback(
                            total_progress,
                            extraction_status
                        )
                except Exception as e:
                    failed_extractions += 1
                    error_msg = f"Error extracting features from {file_path}: {str(e)}"
                    feature_extraction_errors.append(error_msg)
                    logging.error(error_msg)
                    logging.error(f"Exception type: {type(e).__name__}")
                    logging.error(f"Extraction error traceback: {traceback.format_exc()}")
                    print(f"[TRAINING] ERROR: Feature extraction failed for {file_id}: {type(e).__name__}: {str(e)}")
            
            # Log detailed summary of feature extraction
            extraction_summary = f"Feature extraction completed: {successful_extractions} successful, {failed_extractions} failed"
            logging.info(extraction_summary)
            print(f"[TRAINING] {extraction_summary}")
            
            # Log detailed error information if any failures occurred
            if failed_extractions > 0:
                logging.warning(f"Feature extraction encountered {len(feature_extraction_errors)} errors")
                for i, error in enumerate(feature_extraction_errors[:10]):
                    logging.warning(f"Error {i+1}: {error}")
                if len(feature_extraction_errors) > 10:
                    logging.warning(f"... and {len(feature_extraction_errors) - 10} more errors")
                print(f"[TRAINING] WARNING: {failed_extractions} files failed during extraction - check logs for details")
            print(f"[TRAINING] {extraction_summary}")
            if feature_extraction_errors:
                logging.error(f"Feature extraction errors: {feature_extraction_errors}")
                print(f"[TRAINING] Feature extraction errors: {feature_extraction_errors}")
            
            # Log feature extraction summary before array conversion
            logging.info(f"Feature extraction complete. Extracted {len(X)} samples across {len(set(y))} classes")
            print(f"[TRAINING] Feature extraction complete. Extracted {len(X)} samples across {len(set(y))} classes")
            
            if len(X) > 0:
                # Log class distribution
                class_distribution = {}
                for cls in set(y):
                    count = y.count(cls)
                    class_distribution[cls] = count
                logging.info(f"Class distribution: {class_distribution}")
                print(f"[TRAINING] Class distribution: {class_distribution}")
                
                # Log information about features
                sample_shapes = []
                for i, features in enumerate(X[:5] if len(X) > 5 else X):
                    if hasattr(features, 'shape'):
                        sample_shapes.append(f"Sample {i}: {features.shape}")
                    else:
                        sample_shapes.append(f"Sample {i}: {type(features)}")
                logging.info(f"Sample shapes: {sample_shapes}")
                print(f"[TRAINING] Sample shapes: {sample_shapes}")
            
            # Convert to arrays
            logging.info(f"Converting {len(X)} features to numpy arrays")
            print(f"[TRAINING] Converting {len(X)} features to numpy arrays")
            
            # Check if we have any samples at all before conversion
            if len(X) == 0:
                error_msg = "No valid samples extracted for training. Check audio files and feature extraction process."
                logging.error(error_msg)
                print(f"[TRAINING] ERROR: {error_msg}")
                
                # Add more diagnostic information
                logging.error(f"Dictionary name: {dict_name}, Model type: {model_type}")
                logging.error(f"Dictionary classes: {dict_classes}")
                logging.error(f"Classes processed: {processed_classes} out of {len(class_dirs)} available")
                logging.error(f"Preprocessed files count: {len(preprocessed_files)}")
                logging.error(f"Feature extraction: {successful_extractions} successful, {failed_extractions} failed")
                if len(feature_extraction_errors) > 0:
                    logging.error(f"First few errors: {feature_extraction_errors[:3]}")
                
                print(f"[TRAINING] Dictionary: {dict_name}, Model: {model_type}")
                print(f"[TRAINING] Dictionary classes: {dict_classes}")
                print(f"[TRAINING] Classes processed: {processed_classes} out of {len(class_dirs)} available")
                print(f"[TRAINING] Preprocessed files: {len(preprocessed_files)}")
                print(f"[TRAINING] Feature extraction: {successful_extractions} successful, {failed_extractions} failed")
                
                # Enhanced diagnostics for EhOh dictionary
                if dict_name == 'EhOh':
                    logging.error("======= EHOH TRAINING FAILURE DIAGNOSTICS =======")
                    logging.error(f"EHOH ERROR: Training failed with {len(X)} features and {len(set(y)) if y else 0} unique classes")
                    logging.error(f"EHOH ERROR: Dictionary classes specified: {dict_classes}")
                    logging.error(f"EHOH ERROR: Classes that were found: {set(class_labels) if class_labels else 'None'}")
                    logging.error(f"EHOH ERROR: Preprocessed files: {len(preprocessed_files)}")
                    logging.error(f"EHOH ERROR: Files after extraction: {successful_extractions} successful, {failed_extractions} failed")
                    
                    # Check audio directory contents
                    try:
                        audio_dir_path = audio_dir
                        logging.error(f"EHOH ERROR: Audio directory path: {audio_dir_path}")
                        available_dirs = [d for d in os.listdir(audio_dir_path) if os.path.isdir(os.path.join(audio_dir_path, d))]
                        logging.error(f"EHOH ERROR: Available directories: {available_dirs}")
                        
                        # For each required class, check audio files
                        for required_class in ['eh', 'oh']:
                            class_path = os.path.join(audio_dir_path, required_class)
                            if os.path.exists(class_path) and os.path.isdir(class_path):
                                wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
                                logging.error(f"EHOH ERROR: Class '{required_class}' has {len(wav_files)} WAV files")
                                if not wav_files:
                                    logging.error(f"EHOH ERROR: No WAV files found in {required_class} directory")
                            else:
                                logging.error(f"EHOH ERROR: Required class directory '{required_class}' does not exist")
                    except Exception as e:
                        logging.error(f"EHOH ERROR checking audio files: {str(e)}")
                        logging.error(traceback.format_exc())
                
                if progress_callback:
                    progress_callback(0, f"Error: {error_msg}")
                return {'status': 'error', 'message': error_msg}
            
            try:
                # Convert to arrays with detailed error handling
                X = np.array(X)
                y = np.array(y)
                class_names = np.unique(y).tolist()
                
                # Log array conversion results
                logging.info(f"Arrays created: X shape {X.shape}, y shape {y.shape}, dtype {X.dtype}")
                print(f"[TRAINING] Arrays created: X shape {X.shape}, y shape {y.shape}, dtype {X.dtype}")
                logging.info(f"Unique classes: {class_names}")
                print(f"[TRAINING] Unique classes: {class_names}")
                
                # Create stats
                stats = {
                    'total_samples': len(X),
                    'class_counts': {c: np.sum(y == c) for c in class_names},
                    'preprocessing_method': 'unified_audio_preprocessor',
                    'feature_extractor': 'unified_feature_extractor'
                }
                
                logging.info(f"Statistics: {stats}")
                print(f"[TRAINING] Statistics: {stats}")
            except Exception as e:
                error_msg = f"Error converting data to arrays: {str(e)}"
                logging.error(error_msg)
                print(f"[TRAINING] ERROR: {error_msg}")
                traceback.print_exc()
                
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
                # For CNN, we need to extract the mel spectrograms from the feature dictionaries
                # and then reshape them properly for the Conv2D layers
                logging.info(f"Processing features for CNN training")
                
                # Check if we have dictionaries with 'mel_spectrogram' keys
                try:
                    # Extract mel spectrograms from the feature dictionaries
                    mel_specs = []
                    missing_mel_files = []
                    logging.info("Extracting mel spectrograms for CNN training")
                    
                    # Check if features need to be regenerated
                    regenerate_features = False
                    
                    # Debug: print the first few elements of X to understand its structure
                    for i in range(min(5, len(X))):
                        logging.info(f"Sample {i} type: {type(X[i])}, content sample: {str(X[i])[:100]}...")
                    
                    for i, feature_dict in enumerate(X):
                        if not isinstance(feature_dict, dict):
                            logging.error(f"Sample {i} is not a dictionary: {type(feature_dict)}")
                            regenerate_features = True
                            break
                            
                        # Log the keys present in the feature dictionary
                        keys = list(feature_dict.keys())
                        if i < 5:  # Only log first few to avoid verbose output
                            logging.info(f"Sample {i} has keys: {keys}")
                        
                        if 'mel_spectrogram' in feature_dict:
                            # Check for NaN or infinite values
                            mel_spec = feature_dict['mel_spectrogram']
                            if np.isnan(mel_spec).any() or np.isinf(mel_spec).any():
                                logging.warning(f"Sample {i} contains NaN or inf values - cleaning")
                                # Replace NaNs with zeros and inf with large values
                                mel_spec = np.nan_to_num(mel_spec, nan=0.0, posinf=1.0, neginf=-1.0)
                            
                            # Ensure mel_spec isn't all zeros or constant
                            if np.std(mel_spec) < 1e-6:
                                logging.warning(f"Sample {i} has near-zero variance")
                            
                            mel_specs.append(mel_spec)
                        else:
                            # If mel_spectrogram is missing but we have file_path, track for regeneration
                            if 'file_path' in feature_dict:
                                missing_mel_files.append(feature_dict['file_path'])
                                logging.warning(f"Missing 'mel_spectrogram' in sample {i}, file: {feature_dict['file_path']}")
                            else:
                                # No way to regenerate this feature
                                logging.error(f"Sample {i} has keys {keys} but missing both 'mel_spectrogram' and 'file_path'")
                                regenerate_features = True
                                break
                    
                    # If we need to regenerate features, suggest clearing cache
                    if regenerate_features:
                        error_msg = "Feature format is incorrect. Please clear cache and try again with 'clear_cache: true'."
                        logging.error(error_msg)
                        return {'status': 'error', 'message': error_msg}
                    
                    # If we have any missing mel spectrograms, raise error
                    if missing_mel_files:
                        # Log all missing files
                        logging.error(f"Found {len(missing_mel_files)} files with missing mel spectrograms")
                        for file in missing_mel_files[:5]:  # Log first 5 for brevity
                            logging.error(f"Missing mel spectrogram: {file}")
                            
                        # Clear the cache and suggest retraining
                        logging.error("Consider clearing the feature cache and retraining to regenerate all features")
                        raise ValueError(f"Missing 'mel_spectrogram' in {len(missing_mel_files)} samples. Please clear cache and retry.")
                    
                    # Convert to numpy array and reshape
                    X = np.array(mel_specs)
                    logging.info(f"Extracted mel spectrograms: shape={X.shape}, dtype={X.dtype}")
                    
                    # Reshape to 4D: (samples, height, width, channels) if needed
                    if len(X.shape) == 3:  # (samples, time_steps, mel_bands)
                        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
                    
                    # Normalize spectrograms to range [0, 1] to improve training
                    # This is critical for neural network performance
                    X_min = X.min()
                    X_max = X.max()
                    
                    # Log normalization parameters for debugging
                    logging.info(f"Normalizing data with min={X_min:.6f}, max={X_max:.6f}")
                    
                    # Check for very small range which could cause numerical issues
                    if abs(X_max - X_min) < 1e-6:
                        logging.warning(f"WARNING: Very small data range detected: {X_max - X_min}")
                        logging.warning("This could cause numerical instability. Adding artificial variance.")
                        # Force some variance in the data
                        X = X + np.random.normal(0, 0.01, X.shape)
                        X_min = X.min()
                        X_max = X.max()
                    
                    X = (X - X_min) / (X_max - X_min + 1e-7)  # Add epsilon to avoid division by zero
                    
                    # Verify normalization worked correctly
                    logging.info(f"After normalization - min: {X.min():.6f}, max: {X.max():.6f}, mean: {X.mean():.6f}")
                    
                    logging.info(f"Reshaped and normalized for CNN: X shape={X.shape}, X range: [{X.min()}, {X.max()}]")
                except Exception as e:
                    error_msg = f"Error preparing features for CNN: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    if progress_callback:
                        progress_callback(35, f"Error: {error_msg}")
                    return {'status': 'error', 'message': error_msg}
                
                logging.info(f"Preparing data for CNN training: X shape={X.shape}, y unique values={np.unique(y)}")
                
                try:
                    # Log the classes and their counts before splitting
                    class_counts = {}
                    for cls_name in np.unique(y):
                        class_counts[cls_name] = np.sum(y == cls_name)
                    logging.info(f"Class distribution before splitting: {class_counts}")
                    
                    # Generate fixed train/val indices with stratified split to ensure class balance
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train_indices, y_val_indices = train_test_split(
                        X, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Log split sizes
                    logging.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
                    
                    # Store dataset files info in app context for later retrieval
                    dataset_files = []
                    try:
                        if file_paths and len(file_paths) == len(X):  # Check if file_paths exists and matches X length
                            for i, (feature, label, file_path) in enumerate(zip(X, y, file_paths)):
                                is_training = i in y_train_indices
                                dataset_files.append({
                                    'path': file_path,
                                    'class': label,
                                    'filename': os.path.basename(file_path),
                                    'split': 'training' if is_training else 'validation'
                                })
                            logging.info(f"Stored information for {len(dataset_files)} files in app context")
                        else:
                            # Fallback if file paths are not available or length mismatch
                            logging.warning(f"File paths unavailable or count mismatch. X: {len(X)}, paths: {len(file_paths) if file_paths else 0}")
                    except Exception as e:
                        logging.error(f"Error storing dataset files: {str(e)}")
                    
                    # Store the dataset files in the app context
                    try:
                        current_app.dataset_files = dataset_files
                        logging.info(f"Stored {len(dataset_files)} dataset files in app context")
                    except Exception as e:
                        logging.error(f"Error storing dataset files in app context: {str(e)}")
                    
                    # Also store class distribution info
                    try:
                        class_distribution = {}
                        for class_name in class_names:
                            class_distribution[class_name] = y.tolist().count(class_name)
                            
                        # Store in app context
                        current_app.class_distribution = class_distribution
                        logging.info(f"Stored class distribution in app context: {class_distribution}")
                    except Exception as e:
                        logging.error(f"Error calculating or storing class distribution: {str(e)}")
                    
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
                    
                    # Implement CNN training directly instead of using legacy method
                    if progress_callback:
                        # Reset the epoch data list at the start of training
                        TrainingService.latest_epoch_data = []
                        
                        # Create a wrapper to forward progress updates and store epoch data
                        def cnn_progress_callback(epoch, logs):
                            # Make sure we actually have the metrics
                            if not logs:
                                logging.warning(f"No logs received for epoch {epoch+1}, can't store metrics")
                                return
                                
                            # Print all available metrics in logs for debugging
                            logging.info(f"Epoch {epoch+1} logs keys: {list(logs.keys())}")
                            logging.info(f"Epoch {epoch+1} raw logs: {logs}")
                                
                            # Extract and store epoch metrics
                            current_epoch_data = {
                                'epoch': epoch + 1,  # Epoch numbers are 1-based for display
                                'loss': float(logs.get('loss', 0)),
                                'accuracy': float(logs.get('accuracy', 0)),
                                'val_loss': float(logs.get('val_loss', 0)),
                                'val_accuracy': float(logs.get('val_accuracy', 0)),
                                'lr': float(logs.get('lr', 0))
                            }
                            
                            # Add to the class variable that stores all epoch data
                            TrainingService.latest_epoch_data.append(current_epoch_data)
                            
                            # Format a detailed metrics message for the progress update
                            metrics_msg = f"loss={current_epoch_data['loss']:.4f}, acc={current_epoch_data['accuracy']:.4f}, "
                            metrics_msg += f"val_loss={current_epoch_data['val_loss']:.4f}, val_acc={current_epoch_data['val_accuracy']:.4f}"
                            
                            # Log detailed epoch results
                            logging.info(f"Epoch {epoch+1} results: {metrics_msg}")
                            
                            # Report model training progress (35-95%)
                            epochs = kwargs.get('epochs', 50)
                            progress = 35 + int((epoch / epochs) * 60)
                            progress_callback(progress, f"Training epoch {epoch+1}/{epochs}: {metrics_msg}")
                    
                    # Extract CNN-specific parameters
                    epochs = kwargs.get('epochs', 50)
                    batch_size = kwargs.get('batch_size', 32)
                    use_class_weights = kwargs.get('use_class_weights', False)
                    use_data_augmentation = kwargs.get('use_data_augmentation', False)
                    model_name = kwargs.get('model_name', f"cnn_model_{datetime.now().strftime('%Y%m%d%H%M')}")
                    
                    # Unpack the data dictionary
                    X_train, y_train_onehot = data_dict['train']
                    X_val, y_val_onehot = data_dict['val']
                    
                    # Create CNN model with improved architecture
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
                    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                    from tensorflow.keras.optimizers import Adam
                    
                    # Double-check for NaN values before creating the model
                    if np.isnan(X_train).any():
                        logging.warning("NaN values found in training data, replacing with zeros")
                        X_train = np.nan_to_num(X_train)
                    if np.isnan(X_val).any():
                        logging.warning("NaN values found in validation data, replacing with zeros")
                        X_val = np.nan_to_num(X_val)
                    
                    input_shape = X_train.shape[1:]
                    num_classes = len(class_names)
                    
                    # Log data distribution and class balance
                    for i, cls in enumerate(class_names):
                        count = (y == cls).sum()
                        logging.info(f"Class {i} ({cls}): {count} samples ({count/len(y)*100:.1f}%)")
                    
                    # Enhanced data statistics for debugging
                    logging.info(f"Data sample means - X_train: {X_train.mean():.6f}, X_val: {X_val.mean():.6f}")
                    logging.info(f"Data sample std - X_train: {X_train.std():.6f}, X_val: {X_val.std():.6f}")
                    logging.info(f"Data min/max - X_train min: {X_train.min():.6f}, max: {X_train.max():.6f}")
                    logging.info(f"Data shapes - X_train: {X_train.shape}, X_val: {X_val.shape}")
                    logging.info(f"Labels - y_train shape: {y_train_onehot.shape}, y_val shape: {y_val_onehot.shape}")
                    logging.info(f"Label sums - y_train: {np.sum(y_train_onehot, axis=0)}, y_val: {np.sum(y_val_onehot, axis=0)}")
                    
                    # Create an improved CNN model with batch normalization
                    model = Sequential([
                        # First convolution block with batch normalization
                        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_initializer='he_uniform', input_shape=input_shape),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),
                        
                        # Second block with batch normalization
                        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_initializer='he_uniform'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),
                        
                        # Third block for more capacity
                        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_initializer='he_uniform'),
                        BatchNormalization(),
                        MaxPooling2D(pool_size=(2, 2)),
                        
                        # Flatten and multilayer dense network
                        Flatten(),
                        Dense(128, activation='relu', kernel_initializer='he_uniform'),
                        BatchNormalization(),
                        Dropout(0.4),
                        Dense(64, activation='relu', kernel_initializer='he_uniform'),
                        Dropout(0.3),
                        Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')
                    ])
                    
                    # Print model summary for debugging
                    model.summary(print_fn=logging.info)
                    
                    # Always use categorical_crossentropy for one-hot encoded targets
                    # The 'sigmoid' vs 'softmax' activation is already handled in the model architecture
                    logging.info(f"Using categorical_crossentropy loss for {num_classes} classes")
                    model.compile(
                        loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for stability
                        metrics=['accuracy', 'AUC']  # Add AUC for better evaluation
                    )
                    
                    # Double-check training and validation label distributions
                    logging.info(f"Label distribution in training: {np.sum(y_train_onehot, axis=0)}")
                    logging.info(f"Label distribution in validation: {np.sum(y_val_onehot, axis=0)}")
                    
                    # Set up enhanced callbacks for better training
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
                    ]
                    
                    # Create and add our custom progress callback for epoch statistics tracking
                    if progress_callback:
                        from tensorflow.keras.callbacks import LambdaCallback
                        progress_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: cnn_progress_callback(epoch, logs))
                        callbacks.append(progress_cb)
                    
                    # Add custom callback if provided
                    if 'custom_callback' in kwargs:
                        from tensorflow.keras.callbacks import LambdaCallback
                        custom_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: kwargs['custom_callback'](epoch, logs))
                        callbacks.append(custom_cb)
                    
                    # Calculate class weights if required
                    class_weights = None
                    if use_class_weights:
                        from sklearn.utils.class_weight import compute_class_weight
                        y_integers = np.argmax(y_train_onehot, axis=1)
                        unique_classes = np.unique(y_integers)
                        
                        # Log class distribution to help diagnose issues
                        class_counts = np.bincount(y_integers)
                        logging.info(f"Training class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
                        
                        # Compute class weights to handle imbalanced datasets
                        class_weights_array = compute_class_weight(
                            class_weight='balanced',
                            classes=unique_classes,
                            y=y_integers
                        )
                        class_weights = {i: class_weights_array[i if i < len(class_weights_array) else 0] for i in range(len(class_names))}
                        logging.info(f"Class weights: {class_weights}")
                    
                    # NOTE: Data augmentation happens at recording time, not training time
                    # The use_data_augmentation flag is kept for backward compatibility
                    if use_data_augmentation:
                        logging.info("Using augmented data from recording process")
                        logging.info("Note: Data augmentation is applied at recording time, not training time")
                    
                        # Train with the dataset (which already includes augmentations)
                        history = model.fit(
                            X_train, y_train_onehot,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_val, y_val_onehot),
                            callbacks=callbacks,
                            class_weight=class_weights
                        )
                    else:
                        # Train without data augmentation
                        history = model.fit(
                            X_train, y_train_onehot,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_val, y_val_onehot),
                            callbacks=callbacks,
                            class_weight=class_weights
                        )
                    
                    # Test prediction on a TRAINING sample first to verify the model can at least memorize
                    logging.info("Testing prediction on a training sample...")
                    train_pred = model.predict(X_train[:5])
                    train_pred_classes = np.argmax(train_pred, axis=1)
                    train_true_classes = np.argmax(y_train_onehot[:5], axis=1)
                    logging.info(f"Training sample - True classes: {train_true_classes}, Predicted: {train_pred_classes}")
                    
                    # Evaluate the model with extensive diagnostics
                    logging.info("Evaluating model on validation data...")
                    # Handle all metrics returned by evaluate (now includes AUC)
                    eval_results = model.evaluate(X_val, y_val_onehot, verbose=2)
                    # Unpack metrics properly based on their count
                    if isinstance(eval_results, list):
                        if len(eval_results) == 3:  # [loss, accuracy, auc]
                            loss, accuracy, auc = eval_results
                            logging.info(f"Model evaluation - Loss: {loss}, Accuracy: {accuracy}, AUC: {auc}")
                        else:  # Just in case the metrics change
                            loss = eval_results[0]  # Loss is always first
                            accuracy = eval_results[1] if len(eval_results) > 1 else None
                            logging.info(f"Model evaluation - Loss: {loss}, Other metrics: {eval_results[1:]}")
                    else:  # Just one value returned (unlikely)
                        loss = eval_results
                        accuracy = None
                        logging.info(f"Model evaluation - Loss: {loss}")
                    
                    # Get predictions and analyze in detail
                    logging.info("Getting detailed predictions on validation set...")
                    y_pred = model.predict(X_val, verbose=0)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_true_classes = np.argmax(y_val_onehot, axis=1)
                    
                    # Show raw prediction values for first few samples to check model output distribution
                    for i in range(min(5, len(y_pred))):
                        logging.info(f"Sample {i} - Raw prediction: {y_pred[i]}, True class: {y_true_classes[i]}")
                    
                    # Calculate accuracy manually to verify
                    manual_accuracy = np.mean(y_pred_classes == y_true_classes)
                    logging.info(f"Manually calculated accuracy: {manual_accuracy:.4f} vs Keras accuracy: {accuracy:.4f}")
                    
                    # Log prediction distribution to check if model is just predicting one class
                    pred_class_counts = np.bincount(y_pred_classes, minlength=len(class_names))
                    logging.info(f"Prediction class distribution: {dict(zip(range(len(pred_class_counts)), pred_class_counts))}")
                    logging.info(f"True class distribution: {dict(zip(range(len(class_names)), np.bincount(y_true_classes, minlength=len(class_names))))}")
                    
                    # Check for constant predictions (sign of a non-learning model)
                    if len(np.unique(y_pred_classes)) == 1:
                        logging.error(f"CRITICAL ERROR: Model is predicting only one class: {np.unique(y_pred_classes)[0]}")
                        logging.error("This indicates a fundamental problem with the model or training data.")
                        
                        # Extensive diagnostic information to identify the root cause
                        logging.error(f"Raw predictions (first 5): {y_pred[:5]}")
                        logging.error(f"Training label distribution: {np.sum(y_train_onehot, axis=0)}")
                        logging.error(f"Validation label distribution: {np.sum(y_val_onehot, axis=0)}")
                        
                        # Report data statistics that might reveal issues
                        logging.error(f"Input data statistics - mean: {X_val.mean():.4f}, std: {X_val.std():.4f}, min: {X_val.min():.4f}, max: {X_val.max():.4f}")
                        
                    # Compare actual predictions vs true values in a confusion matrix-like format
                    correct = 0
                    incorrect = 0
                    confusion = {}
                    for true_cls in range(len(class_names)):
                        confusion[true_cls] = {}
                        for pred_cls in range(len(class_names)):
                            mask = (y_true_classes == true_cls) & (y_pred_classes == pred_cls)
                            count = np.sum(mask)
                            confusion[true_cls][pred_cls] = count
                            if true_cls == pred_cls:
                                correct += count
                            else:
                                incorrect += count
                    logging.info(f"Confusion matrix: {confusion}")
                    logging.info(f"Correct: {correct}, Incorrect: {incorrect}, Ratio: {correct/(correct+incorrect) if (correct+incorrect) > 0 else 'N/A'}")
                    
                    # Save the model if requested
                    if save:
                        # Use the new folder structure
                        from backend.src.ml.model_paths import create_model_folder
                        
                        # Create a dedicated folder for the model
                        folder_path, model_id = create_model_folder(model_name, 'cnn', dict_name)
    
                        # Save model in the new folder
                        model_path = os.path.join(folder_path, f"{model_name}.h5")
                        model.save(model_path)
                        logging.info(f"Model saved to {model_path}")
                        
                        # Save metadata in the same folder
                        metadata = {
                            'class_names': class_names,
                            'model_type': 'cnn',
                            'input_shape': list(input_shape),
                            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'training_stats': {
                                'accuracy': float(accuracy),
                                'loss': float(loss),
                                'training_samples': len(X_train),
                                'validation_samples': len(X_val)
                            }
                        }
                        metadata_path = os.path.join(folder_path, f"{model_name}_metadata.json")
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f)
                        
                        # Update the registry with the new folder structure
                        update_model_registry(
                            model_id=model_id,
                            model_type='cnn',
                            dictionary_name=dict_name,
                            metadata=metadata,
                            is_best=True,  # Set this model as the best for this dictionary
                            folder_path=folder_path  # Pass the folder path to use new structure
                        )
                        logging.info(f"Updated models registry with new CNN model: {model_id}")
                        
                        # Trigger update of model registry with new model
                        trigger_model_registry_update(model_type="cnn", model_id=model_id, action="update")                   
                    # Prepare result
                    result = {
                        'status': 'success',
                        'model': model,
                        'history': history.history,
                        'accuracy': float(accuracy),
                        'class_names': class_names,
                        'model_path': model_path if save else None,
                        'message': f"CNN model trained successfully with accuracy {accuracy:.2f}"
                    }
                except Exception as e:
                    error_msg = f"Error preparing data for CNN training: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    if progress_callback:
                        progress_callback(35, f"Error: {error_msg}")
                    return {'status': 'error', 'message': error_msg}
            elif model_type == 'rf':
                # Implement RF training directly instead of using legacy method
                if progress_callback:
                    # This is just an estimate since RF doesn't have clear progress reporting
                    progress_callback(50, "Training Random Forest model")
                
                try:
                    # Extract RF-specific parameters
                    n_estimators = kwargs.get('n_estimators', 100)
                    max_depth = kwargs.get('max_depth', None)
                    model_name = kwargs.get('model_name', f"rf_model_{datetime.now().strftime('%Y%m%d%H%M')}")
                    
                    # Train test split for validation
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Train Random Forest model
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import classification_report, accuracy_score
                    
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    if progress_callback:
                        progress_callback(65, "Fitting Random Forest model...")
                    
                    model.fit(X_train, y_train)
                    
                    # Evaluate the model
                    y_pred = model.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred)
                    report = classification_report(y_val, y_pred, output_dict=True)
                    
                    logging.info(f"Random Forest model trained with accuracy: {accuracy:.2f}")
                    
                    if progress_callback:
                        progress_callback(85, f"Random Forest model trained with accuracy: {accuracy:.2f}")
                    
                    # Save the model if requested
                    if save:
                        # Use joblib for more efficient scikit-learn model serialization
                        import joblib  # standalone package, preferred over sklearn.externals.joblib
                        model_dir = os.path.join(self.model_dir, 'rf')
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, f"{model_name}.joblib")
                        
                        # Joblib is more efficient for scikit-learn models with numpy arrays
                        joblib.dump(model, model_path, compress=3)
                        
                        # Save metadata
                        metadata = {
                            'class_names': list(class_names),
                            'model_type': 'rf',
                            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'training_stats': {
                                'accuracy': float(accuracy),
                                'training_samples': len(X_train),
                                'validation_samples': len(X_val),
                                'class_report': report
                            }
                        }
                        
                        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f)
                            
                        logging.info(f"Random Forest model saved to {model_path}")
                        
                        # Update the models.json registry file
                        # Extract the model_id from the model_name
                        model_id = model_name
                        # Get relative path to model file from models directory
                        rel_path = os.path.join('rf', f"{model_name}.joblib")
                        # Update the registry
                        update_model_registry(
                            model_id=model_id,
                            model_type='rf',
                            dictionary_name=dict_name,
                            metadata=metadata,
                            is_best=True  # Set this model as the best for this dictionary
                        )
                        logging.info(f"Updated models registry with new RF model: {model_id}")
                        
                        # Trigger update of model registry to sync with filesystem
                        trigger_model_registry_update(model_type="rf", model_id=model_id, action="update")
                    
                    # Prepare result dictionary
                    result = {
                        'status': 'success',
                        'model': model,
                        'accuracy': float(accuracy),
                        'class_names': list(class_names),
                        'model_path': model_path if save else None,
                        'message': f"RF model trained successfully with accuracy {accuracy:.2f}"
                    }
                except Exception as e:
                    error_msg = f"Error training Random Forest model: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    result = {
                        'status': 'error',
                        'message': error_msg
                    }
                
                # Report completion for RF model
                if progress_callback and result.get('status') == 'success':
                    progress_callback(95, "Random Forest training complete, finalizing model")
            elif model_type == 'ensemble':
                # Ensemble model implementation
                if progress_callback:
                    progress_callback(50, "Building ensemble model...")
                    
                try:
                    # Extract ensemble-specific parameters
                    cnn_model_id = kwargs.get('cnn_model_id')
                    rf_model_id = kwargs.get('rf_model_id')
                    cnn_weight = kwargs.get('cnn_weight', 0.6)  # Default to slightly higher CNN weight
                    rf_weight = kwargs.get('rf_weight', 0.4)
                    
                    # Validate that we have the required model IDs
                    if not cnn_model_id or not rf_model_id:
                        error_msg = "Ensemble model requires both CNN and RF model IDs"
                        logging.error(error_msg)
                        if progress_callback:
                            progress_callback(55, f"Error: {error_msg}")
                        return {'status': 'error', 'message': error_msg}
                        
                    # Generate a name for the ensemble model
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    model_name = f"{dict_name}_ensemble_{timestamp}"
                    
                    # Load the component models
                    if progress_callback:
                        progress_callback(60, "Loading component models...")
                        
                    # For ensemble models, we store model references and weights
                    ensemble_data = {
                        'cnn_model_id': cnn_model_id,
                        'rf_model_id': rf_model_id,
                        'weights': [cnn_weight, rf_weight]
                    }
                    
                    # Create model directory for saving
                    model_dir = os.path.join(self.model_dir, 'ensemble')
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f"{model_name}.joblib")
                    
                    # Save the ensemble model data
                    if save:
                        if progress_callback:
                            progress_callback(80, "Saving ensemble model...")
                            
                        import joblib  # Use joblib for serialization
                        joblib.dump(ensemble_data, model_path)
                        
                        # Save metadata
                        metadata = {
                            'class_names': list(class_names),
                            'model_type': 'ensemble',
                            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'cnn_id': cnn_model_id,
                            'rf_id': rf_model_id,
                            'weights': [cnn_weight, rf_weight],
                            'training_stats': {
                                'accuracy': 0.0,  # We don't have direct accuracy for ensemble without inference
                                'training_samples': len(X),
                                'validation_samples': 0
                            }
                        }
                        
                        # Save metadata to a JSON file alongside the model
                        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f)
                            
                        logging.info(f"Ensemble model saved to {model_path}")
                        
                        # Update the models.json registry file
                        # Extract the model_id from the model_name
                        model_id = model_name
                        # Get relative path to model file from models directory
                        rel_path = os.path.join('ensemble', f"{model_name}.joblib")
                        # Update the registry
                        update_model_registry(
                            model_id=model_id,
                            model_type='ensemble',
                            dictionary_name=dict_name,
                            metadata=metadata,
                            is_best=True  # Set this model as the best for this dictionary
                        )
                        logging.info(f"Updated models registry with new Ensemble model: {model_id}")
                        
                        # Trigger update of model registry to sync with filesystem
                        trigger_model_registry_update(model_type="ensemble", model_id=model_id, action="update")
                    
                    # Prepare result dictionary
                    result = {
                        'status': 'success',
                        'model_id': model_name,
                        'class_names': list(class_names),
                        'model_path': model_path if save else None,
                        'message': "Ensemble model created successfully"
                    }
                    
                    if progress_callback:
                        progress_callback(90, "Ensemble model created successfully")
                        
                except Exception as e:
                    error_msg = f"Error creating ensemble model: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    result = {
                        'status': 'error',
                        'message': error_msg
                    }
            else:  # other unsupported model types
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

    # Helper methods for feature caching and model checking
    
    def _get_feature_cache_key(self, model_type, feature_extractor):
        """Generate a cache key based on feature extraction parameters"""
        params = {
            'model_type': model_type,
            'sample_rate': feature_extractor.sample_rate
        }
        
        if model_type == 'cnn':
            params.update({
                'n_mels': feature_extractor.n_mels,
                'n_fft': feature_extractor.n_fft,
                'hop_length': feature_extractor.hop_length
            })
            
        # Create a deterministic hash from the parameters
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _check_identical_model(self, model_type, dict_name, training_params):
        """Check if a model with identical parameters already exists"""
        try:
            model_dir = os.path.join(self.model_dir, model_type)
            if not os.path.exists(model_dir):
                return None
                
            for model_file in os.listdir(model_dir):
                # Check only models for this dictionary
                if not model_file.startswith(f"{dict_name}_"):
                    continue
                    
                # Get the metadata file
                metadata_path = os.path.join(model_dir, model_file.replace('.h5', '_metadata.json').replace('.pkl', '_metadata.json'))
                if not os.path.exists(metadata_path):
                    continue
                    
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Check if training parameters match
                if metadata.get('training_params'):
                    identical = True
                    for key, value in training_params.items():
                        # Skip non-relevant parameters
                        if key in ['save', 'progress_callback', 'audio_dir', 'model_name']:
                            continue
                            
                        if key not in metadata['training_params'] or metadata['training_params'][key] != value:
                            identical = False
                            break
                            
                    if identical:
                        return os.path.join(model_dir, model_file)
        except Exception as e:
            logging.error(f"Error checking for identical model: {str(e)}")
            
        return None