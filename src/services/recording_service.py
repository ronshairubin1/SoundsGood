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
    
    def __init__(self, data_dir="data/sounds", 
                training_dir="data/sounds/training_sounds",
                augmented_dir="data/sounds/augmented_sounds",
                temp_dir="temp/recordings",
                sample_rate=16000,
                min_silence_duration=0.3,
                sound_threshold=0.008):
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
        self.data_dir = Path(data_dir)
        self.training_dir = Path(training_dir)
        self.augmented_dir = Path(augmented_dir)
        self.temp_dir = Path(temp_dir)
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
    
    def record_training_sounds(self, class_name, max_duration=15.0, callback=None):
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
            
            # Generate augmented versions if we successfully recorded any sounds
            augmented_files = []
            if recorded_files:
                if callback:
                    callback(60, "Creating augmented versions...")
                
                for file_path in recorded_files:
                    # Create 3 augmented versions of each recorded file
                    aug_class_dir = self.augmented_dir / class_name
                    aug_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    augmented = self.augmentor.augment_file(
                        file_path, 
                        str(aug_class_dir), 
                        count=3
                    )
                    augmented_files.extend(augmented)
                
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
            processed_audio = self.preprocessor.preprocess_audio(audio_data)
            
            if processed_audio is None:
                logging.warning(f"Failed to preprocess audio for class '{class_name}'")
                return None
            
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