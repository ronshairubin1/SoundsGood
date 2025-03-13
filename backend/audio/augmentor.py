#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio augmentation module for training data enhancement.

This module provides functions to augment audio data by applying various
transformations, such as pitch shifting, time stretching, noise addition, etc.
This helps generate more training data and improve model robustness.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import logging
import random
from pathlib import Path
from .preprocessor import AudioPreprocessor

class AudioAugmentor:
    """
    Audio augmentation class that applies various transformations to audio data.
    """
    
    def __init__(self, sample_rate=16000, preprocessor=None):
        """
        Initialize the AudioAugmentor.
        
        Args:
            sample_rate (int): Sample rate for audio processing
            preprocessor (AudioPreprocessor, optional): Preprocessor for generated sounds
        """
        self.sample_rate = sample_rate
        self.preprocessor = preprocessor or AudioPreprocessor(sample_rate=sample_rate)
        
        # Available augmentation methods
        self.augmentation_methods = {
            'pitch_shift': self.pitch_shift,
            'time_stretch': self.time_stretch,
            'add_noise': self.add_noise,
            'change_volume': self.change_volume,
            'reverse': self.reverse,
            'time_shift': self.time_shift
        }
        
        logging.info(f"Initialized AudioAugmentor with {len(self.augmentation_methods)} augmentation methods")
    
    def pitch_shift(self, audio_data, n_steps=None):
        """
        Shift the pitch of audio.
        
        Args:
            audio_data (np.array): Audio data
            n_steps (float, optional): Number of semitones to shift. If None, a random value is chosen.
            
        Returns:
            np.array: Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = random.uniform(-3, 3)  # Random shift between -3 and 3 semitones
            
        return librosa.effects.pitch_shift(
            audio_data, 
            sr=self.sample_rate, 
            n_steps=n_steps
        )
    
    def time_stretch(self, audio_data, rate=None):
        """
        Time-stretch audio (change speed).
        
        Args:
            audio_data (np.array): Audio data
            rate (float, optional): Stretch factor. If None, a random value is chosen.
            
        Returns:
            np.array: Time-stretched audio
        """
        if rate is None:
            rate = random.uniform(0.8, 1.2)  # Random stretch between 0.8x and 1.2x
            
        stretched_audio = librosa.effects.time_stretch(audio_data, rate=rate)
        
        # Ensure the audio is still 1 second long after stretching
        return self.preprocessor.normalize_duration(stretched_audio)
    
    def add_noise(self, audio_data, noise_level=None):
        """
        Add random noise to audio.
        
        Args:
            audio_data (np.array): Audio data
            noise_level (float, optional): Noise level relative to signal. If None, a random value is chosen.
            
        Returns:
            np.array: Noisy audio
        """
        if noise_level is None:
            noise_level = random.uniform(0.001, 0.02)  # Random noise level
            
        noise = np.random.randn(len(audio_data))
        noise_amplitude = noise_level * np.max(np.abs(audio_data))
        noisy_audio = audio_data + noise_amplitude * noise
        
        return noisy_audio
    
    def change_volume(self, audio_data, factor=None):
        """
        Change the volume of audio.
        
        Args:
            audio_data (np.array): Audio data
            factor (float, optional): Volume change factor. If None, a random value is chosen.
            
        Returns:
            np.array: Volume-adjusted audio
        """
        if factor is None:
            factor = random.uniform(0.5, 1.5)  # Random volume between 0.5x and 1.5x
            
        return audio_data * factor
    
    def reverse(self, audio_data, _=None):
        """
        Reverse the audio (play backwards).
        
        Args:
            audio_data (np.array): Audio data
            _ (None): Placeholder parameter for API consistency
            
        Returns:
            np.array: Reversed audio
        """
        return audio_data[::-1]
    
    def time_shift(self, audio_data, shift_ms=None):
        """
        Shift audio in time (circular shift).
        
        Args:
            audio_data (np.array): Audio data
            shift_ms (float, optional): Shift in milliseconds. If None, a random value is chosen.
            
        Returns:
            np.array: Time-shifted audio
        """
        if shift_ms is None:
            max_shift_ms = 100  # Maximum shift of 100ms
            shift_ms = random.uniform(-max_shift_ms, max_shift_ms)
            
        shift_samples = int(shift_ms * self.sample_rate / 1000)
        return np.roll(audio_data, shift_samples)
    
    def augment_audio(self, audio_data, method=None, param=None):
        """
        Apply a single augmentation method to audio data.
        
        Args:
            audio_data (np.array): Audio data
            method (str, optional): Augmentation method name. If None, a random method is chosen.
            param (any, optional): Parameter for the augmentation method.
            
        Returns:
            np.array: Augmented audio
        """
        if audio_data is None or len(audio_data) == 0:
            logging.warning("Cannot augment empty audio data")
            return None
            
        if method is None:
            method = random.choice(list(self.augmentation_methods.keys()))
        
        if method not in self.augmentation_methods:
            logging.warning(f"Unknown augmentation method: {method}. Using pitch_shift instead.")
            method = 'pitch_shift'
        
        try:
            # Apply the augmentation
            logging.debug(f"Applying augmentation method: {method}")
            augment_func = self.augmentation_methods[method]
            augmented_audio = augment_func(audio_data, param)
            
            if augmented_audio is None or len(augmented_audio) == 0:
                logging.warning(f"Augmentation method {method} returned empty audio")
                return None
            
            # Try simpler preprocessing first - just normalize it directly
            # This is a fallback in case the more complex preprocessing fails
            normalized_audio = self.preprocessor.normalize_amplitude(augmented_audio)
            
            # Try to preprocess through the standard route
            try:
                # Make a direct call to preprocess_audio to bypass the segment chopping 
                # that might be rejecting the audio in preprocess_recording
                processed_audio = self.preprocessor.preprocess_audio(augmented_audio)
                if processed_audio is not None and len(processed_audio) > 0:
                    return processed_audio
                else:
                    # Fall back to the normalized audio
                    logging.info(f"Direct preprocessing failed, using normalized audio instead")
                    return normalized_audio
            except Exception as e:
                logging.warning(f"Error in standard preprocessing route: {e}, falling back to simple normalization")
                return normalized_audio
                
        except Exception as e:
            logging.error(f"Error during audio augmentation with method {method}: {e}")
            return None
    
    def augment_audio_multiple(self, audio_data, methods=None, count=1):
        """
        Apply multiple augmentation methods to audio data.
        
        Args:
            audio_data (np.array): Audio data
            methods (list, optional): List of augmentation method names. If None, random methods are chosen.
            count (int): Number of augmentations to create
            
        Returns:
            list: List of augmented audio data
        """
        logging.info(f"Augment multiple called, requesting {count} variations")
        
        # Validate input data
        if audio_data is None:
            logging.error("Cannot augment None audio data")
            return []
            
        if len(audio_data) == 0:
            logging.error("Cannot augment empty audio data (length 0)")
            return []
            
        result = []
        
        for _ in range(count):
            # Start with the original audio
            current_audio = audio_data.copy()
            
            # Choose methods to apply
            if methods is None:
                # Apply 1-3 random methods
                num_methods = random.randint(1, 3)
                methods_to_apply = random.sample(list(self.augmentation_methods.keys()), num_methods)
            else:
                methods_to_apply = methods
            
            # Apply each method in sequence
            logging.debug(f"Applying methods: {methods_to_apply}")
            for method in methods_to_apply:
                previous_audio = current_audio
                current_audio = self.augment_audio(current_audio, method)
                
                if current_audio is None:
                    logging.warning(f"Method '{method}' returned None result, skipping this augmentation")
                    break
                    
                # Log transformation details
                prev_max = np.max(np.abs(previous_audio)) if previous_audio is not None else 0
                curr_max = np.max(np.abs(current_audio)) if current_audio is not None else 0
                logging.debug(f"Method '{method}' transformation - Before max: {prev_max:.6f}, After max: {curr_max:.6f}")
            
            # Add the augmented audio to the result
            if current_audio is not None:
                result.append(current_audio)
            else:
                logging.warning("Augmentation sequence resulted in None, not adding to results")
        
        return result
    
    def augment_file(self, input_file, output_dir, count=5, methods=None):
        """
        Augment an audio file and save the augmented versions.
        
        Args:
            input_file (str): Path to the input audio file
            output_dir (str): Directory to save augmented files
            count (int): Number of augmentations to create
            methods (list, optional): List of augmentation method names
            
        Returns:
            list: Paths to augmented audio files
        """
        try:
            # Check if input file exists
            if not os.path.exists(input_file):
                logging.error(f"Input file does not exist: {input_file}")
                return []
                
            # Log more details about the file
            file_size = os.path.getsize(input_file)
            logging.info(f"Loading audio file for augmentation: {input_file} (size: {file_size} bytes)")
            
            try:
                # Try loading with soundfile first, which might provide more detailed errors
                sf_data, sf_rate = sf.read(input_file)
                logging.info(f"Soundfile load successful - samples: {len(sf_data)}, rate: {sf_rate}")
            except Exception as sf_err:
                logging.warning(f"Soundfile load failed: {sf_err}, falling back to librosa")
            
            # Load with librosa
            audio_data, _ = librosa.load(input_file, sr=self.sample_rate)
            
            if audio_data is None or len(audio_data) == 0:
                logging.warning(f"Loaded empty audio from file: {input_file}")
                return []
                
            # Log detailed audio file stats for debugging
            audio_duration = len(audio_data) / self.sample_rate
            audio_max = np.max(np.abs(audio_data))
            audio_min = np.min(audio_data)
            audio_mean = np.mean(audio_data)
            audio_std = np.std(audio_data)
            audio_shape = audio_data.shape
            has_nan = np.isnan(audio_data).any()
            has_inf = np.isinf(audio_data).any()
            
            logging.info(f"Audio stats - Duration: {audio_duration:.2f}s, Shape: {audio_shape}")
            logging.info(f"Audio values - Max: {audio_max:.6f}, Min: {audio_min:.6f}, Mean: {audio_mean:.6f}, Std: {audio_std:.6f}")
            logging.info(f"Audio contains NaN: {has_nan}, contains Inf: {has_inf}")
            
            # Extract filename without extension
            input_path = Path(input_file)
            filename_base = input_path.stem
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create augmented versions
            logging.info(f"Starting augmentation for {input_file}, requesting {count} variations")
            augmented_audios = self.augment_audio_multiple(audio_data, methods, count)
            logging.info(f"Augmentation returned {len(augmented_audios)} results out of {count} requested")
            output_files = []
            
            # Save augmented versions
            for i, augmented_audio in enumerate(augmented_audios):
                if augmented_audio is None:
                    logging.warning(f"Augmentation {i+1} produced None result for {input_file}")
                    continue
                
                output_file = output_path / f"{filename_base}_aug{i+1}.wav"
                sf.write(output_file, augmented_audio, self.sample_rate)
                output_files.append(str(output_file))
                logging.info(f"Saved augmented audio to {output_file}")
            
            if not output_files:
                logging.warning(f"No successful augmentations for {input_file}")
                
            return output_files
            
        except Exception as e:
            logging.error(f"Error augmenting file {input_file}: {str(e)}")
            return []
    
    def augment_directory(self, input_dir, output_dir, count_per_file=5, class_specific=True):
        """
        Augment all audio files in a directory.
        
        Args:
            input_dir (str): Directory containing audio files
            output_dir (str): Directory to save augmented files
            count_per_file (int): Number of augmentations to create per file
            class_specific (bool): Whether to maintain class folder structure
            
        Returns:
            dict: Statistics about the augmentation process
        """
        # Check if the input directory exists
        if not os.path.exists(input_dir):
            logging.error(f"Input directory does not exist: {input_dir}")
            return {
                'input_files': 0,
                'augmented_files': 0,
                'failed_files': 0,
                'classes': {},
                'error': 'Input directory does not exist'
            }
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'input_files': 0,
            'augmented_files': 0,
            'failed_files': 0,
            'classes': {}
        }
        
        # Configure more detailed logging temporarily
        log_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        
        logging.info(f"Starting augmentation of directory: {input_dir} -> {output_dir}")
        logging.info(f"Augmentations per file: {count_per_file}, Class-specific: {class_specific}")
        
        try:
            if class_specific and input_path.is_dir():
                # Process each class directory
                class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
                logging.info(f"Found {len(class_dirs)} class directories")
                
                for class_dir in class_dirs:
                    class_name = class_dir.name
                    class_output_dir = output_path / class_name
                    class_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    stats['classes'][class_name] = {
                        'input_files': 0,
                        'augmented_files': 0,
                        'failed_files': 0
                    }
                    
                    # Process each audio file in class directory
                    wav_files = list(class_dir.glob("*.wav"))
                    logging.info(f"Found {len(wav_files)} .wav files in class '{class_name}'")
                    
                    for file_path in wav_files:
                        stats['input_files'] += 1
                        stats['classes'][class_name]['input_files'] += 1
                        
                        try:
                            augmented_files = self.augment_file(file_path, class_output_dir, count_per_file)
                            stats['augmented_files'] += len(augmented_files)
                            stats['classes'][class_name]['augmented_files'] += len(augmented_files)
                            
                            if not augmented_files:
                                stats['failed_files'] += 1
                                stats['classes'][class_name]['failed_files'] += 1
                                
                        except Exception as e:
                            logging.error(f"Error augmenting {file_path}: {str(e)}")
                            stats['failed_files'] += 1
                            stats['classes'][class_name]['failed_files'] += 1
            else:
                # Process all audio files in the directory
                wav_files = list(input_path.glob("*.wav"))
                logging.info(f"Found {len(wav_files)} .wav files in directory")
                
                for file_path in wav_files:
                    stats['input_files'] += 1
                    
                    try:
                        augmented_files = self.augment_file(file_path, output_path, count_per_file)
                        stats['augmented_files'] += len(augmented_files)
                        
                        if not augmented_files:
                            stats['failed_files'] += 1
                            
                    except Exception as e:
                        logging.error(f"Error augmenting {file_path}: {str(e)}")
                        stats['failed_files'] += 1
            
            # Log detailed stats
            for class_name, class_stats in stats['classes'].items():
                success_rate = 0
                if class_stats['input_files'] > 0:
                    success_rate = (class_stats['input_files'] - class_stats['failed_files']) / class_stats['input_files'] * 100
                logging.info(f"Class '{class_name}': {class_stats['augmented_files']} files created from {class_stats['input_files']} input files (Success rate: {success_rate:.1f}%)")
                
                # Add a warning if no files were created for this class
                if class_stats['augmented_files'] == 0 and class_stats['input_files'] > 0:
                    logging.warning(f"⚠️ NO augmented files were created for class '{class_name}' despite having {class_stats['input_files']} input files")
            
            overall_success = 0
            if stats['input_files'] > 0:
                overall_success = (stats['input_files'] - stats['failed_files']) / stats['input_files'] * 100
                
            logging.info(f"Augmentation complete: {stats['augmented_files']} files created from {stats['input_files']} input files")
            logging.info(f"Success rate: {overall_success:.1f}% ({stats['input_files'] - stats['failed_files']}/{stats['input_files']} files processed successfully)")
            
            # Add clear warning if no files were created at all
            if stats['augmented_files'] == 0 and stats['input_files'] > 0:
                logging.warning(f"⚠️ CRITICAL: NO augmented files were created despite having {stats['input_files']} input files!")
                logging.warning("Please check the logs above for specific errors in file loading or processing.")
            
            # Restore logging level
            logging.getLogger().setLevel(log_level)
            
            return stats
            
        except Exception as e:
            logging.error(f"Unexpected error during directory augmentation: {str(e)}")
            # Restore logging level
            logging.getLogger().setLevel(log_level)
            return stats 