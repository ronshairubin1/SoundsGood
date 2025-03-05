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
        if method is None:
            method = random.choice(list(self.augmentation_methods.keys()))
        
        if method not in self.augmentation_methods:
            logging.warning(f"Unknown augmentation method: {method}. Using pitch_shift instead.")
            method = 'pitch_shift'
        
        augment_func = self.augmentation_methods[method]
        augmented_audio = augment_func(audio_data, param)
        
        # Make sure the augmented audio is properly preprocessed
        return self.preprocessor.preprocess_audio(augmented_audio)
    
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
            for method in methods_to_apply:
                current_audio = self.augment_audio(current_audio, method)
            
            # Add the augmented audio to the result
            if current_audio is not None:
                result.append(current_audio)
        
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
            # Load audio file
            audio_data, _ = librosa.load(input_file, sr=self.sample_rate)
            
            # Extract filename without extension
            input_path = Path(input_file)
            filename_base = input_path.stem
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create augmented versions
            augmented_audios = self.augment_audio_multiple(audio_data, methods, count)
            output_files = []
            
            # Save augmented versions
            for i, augmented_audio in enumerate(augmented_audios):
                if augmented_audio is None:
                    continue
                
                output_file = output_path / f"{filename_base}_aug{i+1}.wav"
                sf.write(output_file, augmented_audio, self.sample_rate)
                output_files.append(str(output_file))
                logging.info(f"Saved augmented audio to {output_file}")
            
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
        
        if class_specific and input_path.is_dir():
            # Process each class directory
            for class_dir in [d for d in input_path.iterdir() if d.is_dir()]:
                class_name = class_dir.name
                class_output_dir = output_path / class_name
                class_output_dir.mkdir(parents=True, exist_ok=True)
                
                stats['classes'][class_name] = {
                    'input_files': 0,
                    'augmented_files': 0
                }
                
                # Process each audio file in class directory
                for file_path in class_dir.glob("*.wav"):
                    stats['input_files'] += 1
                    stats['classes'][class_name]['input_files'] += 1
                    
                    try:
                        augmented_files = self.augment_file(file_path, class_output_dir, count_per_file)
                        stats['augmented_files'] += len(augmented_files)
                        stats['classes'][class_name]['augmented_files'] += len(augmented_files)
                    except Exception as e:
                        logging.error(f"Error augmenting {file_path}: {str(e)}")
                        stats['failed_files'] += 1
        else:
            # Process all audio files in the directory
            for file_path in input_path.glob("*.wav"):
                stats['input_files'] += 1
                
                try:
                    augmented_files = self.augment_file(file_path, output_path, count_per_file)
                    stats['augmented_files'] += len(augmented_files)
                except Exception as e:
                    logging.error(f"Error augmenting {file_path}: {str(e)}")
                    stats['failed_files'] += 1
        
        logging.info(f"Augmentation complete: {stats['augmented_files']} files created from {stats['input_files']} input files")
        return stats 