#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recording service for capturing and processing training and inference audio.

This service handles:
1. Recording training data from the microphone
2. Processing and saving training examples
3. Recording audio for inference
4. Integration with the unified audio processing pipeline
"""

import os
import numpy as np
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Import our unified audio processing components
from backend.audio import AudioRecorder, AudioPreprocessor, AudioAugmentor

class RecordingService:
    """Service for handling audio recording, processing, and data management."""
    
    def __init__(self, data_dir=None, 
                training_dir=None,
                augmented_dir=None,
                temp_dir=None,
                sample_rate=16000,
                min_silence_duration=0.3,
                sound_threshold=0.008):
        
        # Import config here to avoid circular imports
        from backend.config import Config
        
        """
        Initialize the recording service.
        
        Args:
            data_dir (str): Base directory for sound data
            training_dir (str): Directory for training sounds
            augmented_dir (str): Directory for augmented sounds
            temp_dir (str): Directory for temporary recordings
            sample_rate (int): Audio sample rate
            min_silence_duration (float): Minimum silence duration for chopping
            sound_threshold (float): Threshold for detecting sound vs silence
        """
        # Use provided paths or defaults from Config
        self.data_dir = Path(data_dir if data_dir else Config.BACKEND_DATA_DIR)
        self.training_dir = Path(training_dir if training_dir else Config.TRAINING_SOUNDS_DIR)
        self.augmented_dir = Path(augmented_dir if augmented_dir else Config.AUGMENTED_SOUNDS_DIR)
        self.temp_dir = Path(temp_dir if temp_dir else Config.TEMP_DIR)
        self.sample_rate = sample_rate
        
        # Initialize the audio processing components
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            sound_threshold=sound_threshold,
            min_silence_duration=min_silence_duration
        )
        
        self.recorder = AudioRecorder(
            sample_rate=sample_rate,
            preprocessor=self.preprocessor,
            silence_threshold=sound_threshold,
            min_silence_duration=min_silence_duration
        )
        
        self.augmentor = AudioAugmentor(
            sample_rate=sample_rate,
            preprocessor=self.preprocessor
        )
        
        # Create necessary directories
        self._create_directories()
        
        logging.info(f"Initialized RecordingService with sample_rate={sample_rate}")
    
    def _create_directories(self):
        """Create necessary directories for storing audio files."""
        for directory in [self.data_dir, self.training_dir, self.augmented_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {directory}")
    
    # Audio processing is now centralized in the AudioPreprocessor component
    # All preprocessing should use self.preprocessor.preprocess_recording
    
    def get_available_classes(self):
        """
        Get a list of available sound classes from training directory.
        
        Returns:
            list: List of class names
        """
        if not self.training_dir.exists():
            return []
        
        classes = [d.name for d in self.training_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
        return sorted(classes)
    
    def get_class_sample_count(self, class_name):
        """
        Get the number of training samples for a class.
        
        Args:
            class_name (str): Name of the class
            
        Returns:
            int: Number of training samples
        """
        class_dir = self.training_dir / class_name
        if not class_dir.exists():
            return 0
        
        samples = list(class_dir.glob("*.wav"))
        return len(samples)
    
    def get_sound_metadata(self):
        """
        Get metadata for all sounds.
        
        Returns:
            dict: Sound metadata
        """
        metadata_file = self.data_dir / "sounds.json"
        if not metadata_file.exists():
            return {}
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading sound metadata: {str(e)}")
            return {}
    
    def calibrate_microphone(self, duration=1.0):
        """
        Calibrate the microphone for ambient noise.
        
        Args:
            duration (float): Duration to sample ambient noise in seconds
            
        Returns:
            float: Detected ambient noise level
        """
        try:
            noise_level = self.recorder.calibrate_ambient_noise(duration)
            logging.info(f"Calibrated microphone. Ambient noise level: {noise_level}")
            return noise_level
        except Exception as e:
            logging.error(f"Error calibrating microphone: {str(e)}")
            return None
    
    def is_augmented_file(self, filename):
        """
        Check if a file is already an augmentation.
        
        Args:
            filename (str): The filename to check
            
        Returns:
            bool: True if the file is an augmentation, False otherwise
        """
        # Check for augmentation patterns in the filename
        if "_p" in filename and "_t" in filename and "_n" in filename:  # New format with pitch/time/noise params
            return True
        elif "_aug" in filename:  # Legacy format
            return True
        # Check for randomized augmentation with unique ID
        elif "_p" in filename and "_t" in filename and "_n" in filename and "_" in filename.split("_n")[1]:
            return True
        return False
    
    def get_augmentation_status(self, directory):
        """
        Get statistics about original vs augmented files in a directory.
        
        Args:
            directory (str or Path): Directory to analyze
            
        Returns:
            dict: Statistics about original and augmented files
        """
        directory = Path(directory)
        if not directory.exists():
            return {"error": "Directory does not exist"}
            
        all_files = [f for f in directory.glob("**/*.wav")]
        
        original_files = []
        augmented_files = []
        
        for f in all_files:
            if self.is_augmented_file(f.name):
                augmented_files.append(str(f))
            else:
                original_files.append(str(f))
        
        # Calculate augmentation ratio
        orig_count = len(original_files)
        aug_count = len(augmented_files)
        ratio = aug_count / orig_count if orig_count > 0 else 0
        
        # Check augmentation pattern
        if orig_count > 0 and aug_count > 0:
            expected_ratio = 26  # 3×3×3 - 1 = 26 augmentations per original
            if abs(ratio - expected_ratio) < 0.5:
                pattern = "complete_matrix"  # Full 3×3×3 augmentation matrix
            elif abs(ratio - 3) < 0.5:
                pattern = "legacy"  # Old method with 3 augmentations
            else:
                pattern = "mixed"  # Mixed or partial augmentation
        else:
            pattern = "none"
        
        return {
            "original_count": orig_count,
            "augmented_count": aug_count,
            "total_count": orig_count + aug_count,
            "augmentation_ratio": ratio,
            "augmentation_pattern": pattern,
            "original_files": original_files,
            "augmented_files": augmented_files
        }
    
    def record_training_sounds(self, class_name, max_duration=15.0, callback=None, create_augmentations=True):
        """
        Record sounds for training a specific class.
        
        Args:
            class_name (str): Class name for the recorded sounds
            max_duration (float): Maximum recording duration in seconds
            callback (callable): Optional callback function for progress updates
            
        Returns:
            dict: Recording results with file paths and counts
        """
        # Create class directory if needed
        class_dir = self.training_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Record sound for the class
        metadata = {
            "recorded_date": datetime.now().isoformat(),
            "class": class_name,
            "original": True,
            "augmented": False
        }
        
        try:
            # Record and save segments
            recorded_files = self.recorder.record_for_training(
                class_name, 
                str(self.training_dir),
                max_duration=max_duration,
                callback=callback,
                metadata=metadata
            )
            
            # Generate augmented versions if requested and if we successfully recorded any sounds
            augmented_files = []
            if recorded_files and create_augmentations:
                if callback:
                    callback(60, "Creating augmented versions...")
                    
                # Check if we should be creating augmentations
                files_to_augment = []
                for file_path in recorded_files:
                    # Make sure we're not augmenting an already augmented file
                    if self.is_augmented_file(os.path.basename(file_path)):
                        logger.warning(f"Skipping augmentation for already augmented file: {file_path}")
                        continue
                    files_to_augment.append(file_path)
                
                if not files_to_augment:
                    logger.warning("No original files to augment - skipping augmentation")
                    if callback:
                        callback(65, "No original files to augment - skipping")
                
                # Proceed with augmentation for original files only
                for file_path in files_to_augment:
                    # Create systematic augmentations with all parameter combinations
                    # Store augmented files directly in the training directory alongside original files
                    # This ensures they're properly counted in the dashboard
                    aug_class_dir = self.training_dir / class_name
                    aug_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Define comprehensive augmentation parameters with randomization
                    # Base augmentation parameters
                    pitch_centers = [-2, 0, 2]  # Center semitone values (-2, normal, +2)
                    time_centers = [0.85, 1.0, 1.15]  # Center speed factors (slower, normal, faster)
                    noise_centers = [0, 0.005, 0.01]  # Center noise levels (none, low, medium)
                    
                    # Randomization factors (how much to randomly vary from the center values)
                    # For example: pitch_randomization = 0.5 means pitch can vary ±0.5 semitones
                    pitch_randomization = 0.5  # Semitones
                    time_randomization = 0.1   # Proportion of time stretch factor
                    noise_randomization = 0.3   # Proportion of noise level
                    
                    # Generate randomized parameters
                    import random
                    pitch_options = []
                    time_options = []
                    noise_options = []
                    
                    # Create randomized pitch values
                    for center in pitch_centers:
                        if center == 0:  # Don't randomize the "no change" option
                            pitch_options.append(0)
                        else:
                            # Random variation within ±pitch_randomization semitones
                            variation = random.uniform(-pitch_randomization, pitch_randomization)
                            pitch_options.append(center + variation)
                    
                    # Create randomized time stretch values
                    for center in time_centers:
                        if center == 1.0:  # Don't randomize the "no change" option
                            time_options.append(1.0)
                        else:
                            # Random variation within ±time_randomization % of center value
                            variation = random.uniform(-time_randomization, time_randomization) * center
                            time_options.append(center + variation)
                    
                    # Create randomized noise levels
                    for center in noise_centers:
                        if center == 0:  # Don't randomize the "no noise" option
                            noise_options.append(0)
                        else:
                            # Random variation within ±noise_randomization % of center value
                            variation = random.uniform(-noise_randomization, noise_randomization) * center
                            # Ensure noise level doesn't go negative
                            noise_options.append(max(0, center + variation))
                    
                    # Extract filename without extension
                    file_path_obj = Path(file_path)
                    base_filename = file_path_obj.stem
                    
                    # Load the original audio
                    try:
                        import librosa
                        import soundfile as sf
                        
                        # Log the augmentation process
                        logger.info(f"Creating comprehensive augmentations for {file_path}")
                        logger.info(f"Using parameters: Pitch={pitch_options}, Time={time_options}, Noise={noise_options}")
                        logger.info(f"Total combinations: {len(pitch_options) * len(time_options) * len(noise_options)}")
                        
                        # Load audio file
                        audio_data, _ = librosa.load(file_path, sr=self.augmentor.sample_rate)
                        
                        # Generate all combinations
                        aug_count = 0
                        for pitch in pitch_options:
                            for time in time_options:
                                for noise in noise_options:
                                    # Skip the no-augmentation case (which would duplicate the original)
                                    if pitch == 0 and time == 1.0 and noise == 0:
                                        continue
                                    
                                    # Create descriptive filename with augmentation parameters
                                    # Create descriptive filename with actual randomized augmentation parameters
                                    # Round values slightly to keep filenames reasonable length
                                    pitch_str = f"{pitch:.1f}".replace('.0', '') if pitch == int(pitch) else f"{pitch:.1f}"
                                    time_str = f"{time:.2f}"
                                    noise_str = f"{noise:.3f}"
                                    
                                    # Add a small random string to ensure uniqueness even with same parameters
                                    unique_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
                                    
                                    aug_filename = f"{base_filename}_p{pitch_str}_t{time_str}_n{noise_str}_{unique_id}.wav"
                                    output_path = aug_class_dir / aug_filename
                                    
                                    # Apply augmentations in sequence
                                    current_audio = audio_data.copy()
                                    
                                    # Apply pitch shift if not 0
                                    if pitch != 0:
                                        current_audio = self.augmentor.pitch_shift(current_audio, n_steps=pitch)
                                    
                                    # Apply time stretch if not 1.0
                                    if time != 1.0:
                                        current_audio = self.augmentor.time_stretch(current_audio, rate=time)
                                    
                                    # Apply noise if not 0
                                    if noise > 0:
                                        current_audio = self.augmentor.add_noise(current_audio, noise_level=noise)
                                    
                                    # Preprocess the augmented audio to ensure it's clean
                                    # Use preprocess_recording for consistency
                                    processed_segments = self.augmentor.preprocessor.preprocess_recording(current_audio)
                                    if processed_segments:
                                        current_audio = processed_segments[0]
                                    else:
                                        continue  # Skip this augmentation if processing failed
                                    
                                    # Save the augmented audio
                                    sf.write(str(output_path), current_audio, self.augmentor.sample_rate)
                                    augmented_files.append(str(output_path))
                                    aug_count += 1
                        
                        logger.info(f"Created {aug_count} augmented versions of {file_path}")
                        
                    except Exception as e:
                        logger.error(f"Error creating augmentations for {file_path}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                if callback:
                    callback(90, f"Created {len(augmented_files)} augmented versions")
            
            # Create summary of recording results
            result = {
                "class": class_name,
                "original_count": len(recorded_files),
                "augmented_count": len(augmented_files),
                "total_count": len(recorded_files) + len(augmented_files),
                "original_files": recorded_files,
                "augmented_files": augmented_files
            }
            
            logging.info(f"Recorded {result['original_count']} sounds for class '{class_name}' "
                       f"with {result['augmented_count']} augmented versions")
            
            if callback:
                callback(100, f"Recorded {result['total_count']} total sounds for class '{class_name}'")
            
            return result
            
        except Exception as e:
            logging.error(f"Error recording training sounds: {str(e)}")
            if callback:
                callback(100, f"Error: {str(e)}")
            return {
                "class": class_name,
                "error": str(e),
                "original_count": 0,
                "augmented_count": 0,
                "total_count": 0,
                "original_files": [],
                "augmented_files": []
            }
    
    def record_for_inference(self, max_duration=5.0, callback=None):
        """
        Record a single sound for inference/prediction.
        
        Args:
            max_duration (float): Maximum recording duration in seconds
            callback (callable): Optional callback function
            
        Returns:
            np.array: Preprocessed audio ready for model input
        """
        try:
            # Record and preprocess the audio
            processed_audio = self.recorder.record_for_inference(
                max_duration=max_duration,
                callback=callback
            )
            
            return processed_audio
            
        except Exception as e:
            logging.error(f"Error recording for inference: {str(e)}")
            if callback:
                callback(100, f"Error: {str(e)}")
            return None
    
    def analyze_dataset(self, dataset_dir=None):
        """
        Analyze the dataset to determine augmentation status of each class.
        
        Args:
            dataset_dir (str or Path, optional): Dataset directory to analyze. Defaults to training_dir.
            
        Returns:
            dict: Comprehensive analysis of the dataset's augmentation status
        """
        if dataset_dir is None:
            dataset_dir = self.training_dir
        
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            return {"error": "Dataset directory does not exist"}
        
        # Get all class directories
        class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        
        result = {
            "dataset_directory": str(dataset_dir),
            "total_classes": len(class_dirs),
            "classes": {},
            "overall": {
                "original_count": 0,
                "augmented_count": 0,
                "total_files": 0
            }
        }
        
        # Analyze each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_stats = self.get_augmentation_status(class_dir)
            
            # Add to result
            result["classes"][class_name] = class_stats
            
            # Update overall counts
            result["overall"]["original_count"] += class_stats["original_count"]
            result["overall"]["augmented_count"] += class_stats["augmented_count"]
            result["overall"]["total_files"] += class_stats["total_count"]
        
        # Calculate overall augmentation ratio
        if result["overall"]["original_count"] > 0:
            result["overall"]["augmentation_ratio"] = (
                result["overall"]["augmented_count"] / result["overall"]["original_count"]
            )
        else:
            result["overall"]["augmentation_ratio"] = 0
        
        # Determine if we need to create augmentations
        result["needs_augmentation"] = False
        
        # Check if any class has original files but no augmentations
        for class_name, stats in result["classes"].items():
            if stats["original_count"] > 0 and stats["augmented_count"] == 0:
                result["needs_augmentation"] = True
                break
        
        return result
        
    def regenerate_augmentations(self, dataset_dir=None, callback=None):
        """
        Regenerate all augmentations for a dataset with randomized parameters.
        
        Args:
            dataset_dir (str or Path, optional): Dataset directory. Defaults to training_dir.
            callback (callable, optional): Callback function for progress updates.
            
        Returns:
            dict: Results of regeneration process
        """
        if dataset_dir is None:
            dataset_dir = self.training_dir
        
        dataset_dir = Path(dataset_dir)
        
        # Get dataset analysis
        analysis = self.analyze_dataset(dataset_dir)
        
        if "error" in analysis:
            return {"status": "error", "message": analysis["error"]}
        
        # Track results
        result = {
            "classes_processed": 0,
            "original_files_processed": 0,
            "new_augmentations_created": 0,
            "details": {}
        }
        
        # Import required libraries
        import random
        import librosa
        import soundfile as sf
        
        # Process each class
        total_classes = len(analysis["classes"])
        for i, (class_name, stats) in enumerate(analysis["classes"].items()):
            class_result = {
                "original_count": stats["original_count"],
                "augmentations_before": stats["augmented_count"],
                "new_augmentations": 0
            }
            
            if callback:
                callback(int(10 + (i / total_classes) * 80), f"Processing class {class_name}...")
                
            # Skip classes with no original files
            if stats["original_count"] == 0:
                result["details"][class_name] = class_result
                continue
            
            # Set up augmentation directory
            aug_class_dir = self.augmented_dir / class_name
            aug_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each original file
            for orig_file in stats["original_files"]:
                file_path = Path(orig_file)
                
                if callback:
                    callback(int(10 + (i / total_classes) * 80), 
                            f"Augmenting {file_path.name} in class {class_name}...")
                
                # Define randomized augmentation parameters
                # Base augmentation parameters
                pitch_centers = [-2, 0, 2]  # Center semitone values (-2, normal, +2)
                time_centers = [0.85, 1.0, 1.15]  # Center speed factors (slower, normal, faster)
                noise_centers = [0, 0.005, 0.01]  # Center noise levels (none, low, medium)
                
                # Randomization factors
                pitch_randomization = 0.5  # Semitones
                time_randomization = 0.1   # Proportion of time stretch factor
                noise_randomization = 0.3   # Proportion of noise level
                
                # Generate randomized parameters
                pitch_options = []
                time_options = []
                noise_options = []
                
                # Create randomized pitch values
                for center in pitch_centers:
                    if center == 0:  # Don't randomize the "no change" option
                        pitch_options.append(0)
                    else:
                        variation = random.uniform(-pitch_randomization, pitch_randomization)
                        pitch_options.append(center + variation)
                
                # Create randomized time stretch values
                for center in time_centers:
                    if center == 1.0:  # Don't randomize the "no change" option
                        time_options.append(1.0)
                    else:
                        variation = random.uniform(-time_randomization, time_randomization) * center
                        time_options.append(center + variation)
                
                # Create randomized noise levels
                for center in noise_centers:
                    if center == 0:  # Don't randomize the "no noise" option
                        noise_options.append(0)
                    else:
                        variation = random.uniform(-noise_randomization, noise_randomization) * center
                        noise_options.append(max(0, center + variation))
                
                # Extract filename without extension
                base_filename = file_path.stem
                new_augs = 0
                
                try:
                    # Load the original audio
                    audio_data, _ = librosa.load(str(file_path), sr=self.augmentor.sample_rate)
                    
                    # Generate all combinations
                    for pitch in pitch_options:
                        for time in time_options:
                            for noise in noise_options:
                                # Skip the no-augmentation case
                                if pitch == 0 and time == 1.0 and noise == 0:
                                    continue
                                
                                # Format parameter strings for filename
                                pitch_str = f"{pitch:.1f}".replace('.0', '') if pitch == int(pitch) else f"{pitch:.1f}"
                                time_str = f"{time:.2f}"
                                noise_str = f"{noise:.3f}"
                                
                                # Add a small random string to ensure uniqueness
                                unique_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
                                
                                # Create filename with augmentation parameters
                                aug_filename = f"{base_filename}_p{pitch_str}_t{time_str}_n{noise_str}_{unique_id}.wav"
                                output_path = aug_class_dir / aug_filename
                                
                                # Apply augmentations in sequence
                                current_audio = audio_data.copy()
                                
                                if pitch != 0:
                                    current_audio = self.augmentor.pitch_shift(current_audio, n_steps=pitch)
                                if time != 1.0:
                                    current_audio = self.augmentor.time_stretch(current_audio, rate=time)
                                if noise > 0:
                                    current_audio = self.augmentor.add_noise(current_audio, noise_level=noise)
                                
                                # Preprocess and save
                                # Use preprocess_recording for consistency
                                processed_segments = self.augmentor.preprocessor.preprocess_recording(current_audio)
                                if processed_segments:
                                    current_audio = processed_segments[0]
                                else:
                                    continue  # Skip this augmentation if processing failed
                                sf.write(str(output_path), current_audio, self.augmentor.sample_rate)
                                new_augs += 1
                except Exception as e:
                    logger.error(f"Error creating augmentations for {file_path}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # Update counts
                class_result["new_augmentations"] += new_augs
                result["new_augmentations_created"] += new_augs
                result["original_files_processed"] += 1
            
            result["classes_processed"] += 1
            result["details"][class_name] = class_result
        
        if callback:
            callback(90, f"Finished creating {result['new_augmentations_created']} new augmentations")
        
        return result
    
    def save_training_sound(self, audio_data, class_name, is_approved=True, metadata=None):
        """
        Save an externally provided sound for training.
        
        Args:
            audio_data (np.array): Audio data to save
            class_name (str): Class name
            is_approved (bool): Whether the sound is approved for training
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: Path to saved file or None if not saved
        """
        if not is_approved:
            logging.info(f"Sound for class '{class_name}' was not approved, skipping")
            return None
        
        try:
            # Preprocess the audio to ensure consistency
            # Use preprocess_recording for consistency across the codebase
            processed_segments = self.preprocessor.preprocess_recording(audio_data)
            
            if not processed_segments:
                logging.warning(f"Failed to preprocess audio for class '{class_name}'")
                return None
                
            # Use the first processed segment
            processed_audio = processed_segments[0]
            
            # Save the processed audio
            output_path = self.preprocessor.save_training_sound(
                processed_audio,
                class_name,
                str(self.training_dir),
                metadata=metadata
            )
            
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving training sound: {str(e)}")
            return None 