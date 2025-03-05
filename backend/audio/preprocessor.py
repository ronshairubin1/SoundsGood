#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified audio preprocessing module for both training and inference.

This module implements a consistent preprocessing pipeline that's used during:
1. Training data preparation (chopping, finding boundaries, scaling duration)
2. Inference (applying the same preprocessing to live audio)

This ensures that audio is processed identically in both cases.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import logging
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
import tempfile

class AudioPreprocessor:
    """
    Unified audio preprocessor that handles all audio preprocessing steps
    consistently for both training and inference.
    """

    def __init__(self, sample_rate=16000, 
                 sound_threshold=0.008, 
                 min_silence_duration=0.1,
                 min_sound_duration=0.1,
                 boundary_margin_ms=20,
                 target_duration=1.0,
                 normalize_audio=True):
        """
        Initialize the AudioPreprocessor.
        
        Args:
            sample_rate (int): Sampling rate for audio processing
            sound_threshold (float): Threshold for detecting sound vs silence
            min_silence_duration (float): Minimum duration of silence to consider (seconds)
            min_sound_duration (float): Minimum duration of sound to consider valid (seconds)
            boundary_margin_ms (float): Margin to add before/after sound boundaries (milliseconds)
            target_duration (float): Target duration to scale audio to (seconds)
            normalize_audio (bool): Whether to normalize audio amplitude
        """
        self.sample_rate = sample_rate
        self.sound_threshold = sound_threshold
        self.min_silence_duration = min_silence_duration
        self.min_sound_duration = min_sound_duration
        self.boundary_margin_ms = boundary_margin_ms
        self.target_duration = target_duration
        self.normalize_audio = normalize_audio
        
        # Derived parameters
        self.boundary_margin_samples = int(self.boundary_margin_ms * self.sample_rate / 1000)
        self.min_silence_samples = int(self.min_silence_duration * self.sample_rate)
        self.min_sound_samples = int(self.min_sound_duration * self.sample_rate)
        
        logging.info(f"Initialized AudioPreprocessor: sample_rate={sample_rate}, "
                    f"sound_threshold={sound_threshold}, target_duration={target_duration}")

    def chop_audio(self, audio_data):
        """
        Chop continuous audio into individual sound segments separated by silence.
        
        Args:
            audio_data (np.array): Audio data as numpy array
            
        Returns:
            list: List of audio segments as numpy arrays
        """
        if len(audio_data) < self.min_sound_samples:
            logging.warning(f"Audio too short: {len(audio_data)/self.sample_rate:.2f}s < "
                          f"{self.min_sound_samples/self.sample_rate:.2f}s")
            return []
        
        # Calculate energy (RMS) in rolling windows
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i+frame_length]
            energy.append(np.sqrt(np.mean(np.square(frame))))
        
        energy = np.array(energy)
        sound_mask = energy > self.sound_threshold
        
        # Find continuous sound segments
        segments = []
        in_sound = False
        start_frame = 0
        
        for i, is_sound in enumerate(sound_mask):
            if is_sound and not in_sound:
                # Transition from silence to sound
                start_frame = i
                in_sound = True
            elif not is_sound and in_sound:
                # Transition from sound to silence
                # Only add if sound segment is long enough
                if i - start_frame >= self.min_sound_samples / hop_length:
                    start_sample = max(0, start_frame * hop_length)
                    end_sample = min(len(audio_data), i * hop_length)
                    segments.append((start_sample, end_sample))
                in_sound = False
        
        # Handle last segment if still in sound
        if in_sound and len(sound_mask) - start_frame >= self.min_sound_samples / hop_length:
            start_sample = max(0, start_frame * hop_length)
            end_sample = len(audio_data)
            segments.append((start_sample, end_sample))
        
        # Extract segments with additional boundary margin
        audio_segments = []
        for start, end in segments:
            # Apply margin before and after
            adj_start = max(0, start - self.boundary_margin_samples)
            adj_end = min(len(audio_data), end + self.boundary_margin_samples)
            
            # Extract segment
            segment = audio_data[adj_start:adj_end]
            audio_segments.append(segment)
            
        logging.info(f"Chopped audio into {len(audio_segments)} segments")
        return audio_segments
    
    def find_sound_boundaries(self, audio_data):
        """
        Find the precise start and end of sound in an audio segment.
        
        Args:
            audio_data (np.array): Audio data as numpy array
            
        Returns:
            tuple: (start_sample, end_sample) in samples
        """
        if len(audio_data) < self.min_sound_samples:
            logging.warning("Audio too short for boundary detection")
            return 0, len(audio_data)
        
        # Calculate energy in rolling windows
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i+frame_length]
            energy.append(np.sqrt(np.mean(np.square(frame))))
        
        energy = np.array(energy)
        sound_mask = energy > self.sound_threshold
        
        # Find the first and last frames with sound
        if not np.any(sound_mask):
            # No sound detected
            return 0, len(audio_data)
        
        first_sound_frame = np.argmax(sound_mask)
        last_sound_frame = len(sound_mask) - np.argmax(sound_mask[::-1]) - 1
        
        # Convert to sample indices
        start_sample = max(0, first_sound_frame * hop_length - self.boundary_margin_samples)
        end_sample = min(len(audio_data), (last_sound_frame + 1) * hop_length + self.boundary_margin_samples)
        
        return start_sample, end_sample
    
    def normalize_duration(self, audio_data):
        """
        Normalize audio to target duration (1 second by default).
        
        Args:
            audio_data (np.array): Audio data as numpy array
            
        Returns:
            np.array: Duration-normalized audio
        """
        if audio_data is None or len(audio_data) == 0:
            logging.warning("Empty audio for duration normalization")
            return np.zeros(int(self.target_duration * self.sample_rate))
        
        current_duration = len(audio_data) / self.sample_rate
        
        # If the audio is already close to the target duration, no need to change
        if abs(current_duration - self.target_duration) / self.target_duration < 0.05:
            return audio_data
        
        # If audio is shorter than target, pad with zeros
        if current_duration < self.target_duration:
            padding = int((self.target_duration - current_duration) * self.sample_rate)
            # Add padding equally before and after
            pad_before = padding // 2
            pad_after = padding - pad_before
            return np.pad(audio_data, (pad_before, pad_after), 'constant')
        
        # If audio is longer than target, use time stretching
        time_stretch_ratio = self.target_duration / current_duration
        return librosa.effects.time_stretch(audio_data, rate=time_stretch_ratio)
    
    def normalize_amplitude(self, audio_data):
        """
        Normalize audio amplitude to a standard level.
        
        Args:
            audio_data (np.array): Audio data as numpy array
            
        Returns:
            np.array: Amplitude-normalized audio
        """
        if not self.normalize_audio or audio_data is None or len(audio_data) == 0:
            return audio_data
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # Normalize only if there's enough sound energy
        if rms > 1e-6:
            return audio_data / rms
        
        return audio_data
    
    def preprocess_audio(self, audio_data):
        """
        Apply full preprocessing pipeline to a single audio segment.
        
        Args:
            audio_data (np.array): Audio data as numpy array
            
        Returns:
            np.array: Preprocessed audio
        """
        if audio_data is None or len(audio_data) == 0:
            logging.warning("Empty audio for preprocessing")
            return None
        
        # Step 1: Find precise sound boundaries
        start, end = self.find_sound_boundaries(audio_data)
        audio_data = audio_data[start:end]
        
        # Step 2: Check if remaining audio is long enough
        if len(audio_data) < self.min_sound_samples:
            logging.warning(f"Audio too short after boundary detection: {len(audio_data)/self.sample_rate:.2f}s")
            return None
        
        # Step 3: Normalize amplitude
        audio_data = self.normalize_amplitude(audio_data)
        
        # Step 4: Normalize duration to exactly 1 second
        audio_data = self.normalize_duration(audio_data)
        
        return audio_data
    
    def preprocess_file(self, input_file, output_file=None):
        """
        Preprocess an audio file and optionally save the result.
        
        Args:
            input_file (str): Path to input audio file
            output_file (str, optional): Path to save preprocessed audio
            
        Returns:
            np.array: Preprocessed audio data
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(input_file, sr=self.sample_rate)
            
            # Apply preprocessing
            processed_audio = self.preprocess_audio(audio_data)
            
            # Save if output file is provided
            if output_file and processed_audio is not None:
                sf.write(output_file, processed_audio, self.sample_rate)
                logging.info(f"Saved preprocessed audio to {output_file}")
            
            return processed_audio
            
        except Exception as e:
            logging.error(f"Error preprocessing file {input_file}: {str(e)}")
            return None
    
    def preprocess_recording(self, audio_data):
        """
        Preprocess a recording (possibly containing multiple sounds).
        
        Args:
            audio_data (np.array): Recorded audio data
            
        Returns:
            list: List of preprocessed audio segments
        """
        try:
            # Step 1: Chop audio into separate sound segments
            segments = self.chop_audio(audio_data)
            
            # Step 2: Preprocess each segment
            processed_segments = []
            for i, segment in enumerate(segments):
                processed = self.preprocess_audio(segment)
                if processed is not None:
                    processed_segments.append(processed)
            
            logging.info(f"Preprocessed {len(processed_segments)} segments from recording")
            return processed_segments
            
        except Exception as e:
            logging.error(f"Error preprocessing recording: {str(e)}")
            return []
    
    def save_training_sound(self, audio_data, class_name, output_dir, metadata=None):
        """
        Save a preprocessed sound for training with proper naming convention.
        
        Args:
            audio_data (np.array): Preprocessed audio data
            class_name (str): Class/category name
            output_dir (str): Base output directory
            metadata (dict, optional): Additional metadata to save
            
        Returns:
            str: Path to saved file
        """
        if audio_data is None or len(audio_data) == 0:
            logging.warning("Empty audio data, nothing saved")
            return None
        
        try:
            # Create class directory if needed
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{class_name}_{timestamp}.wav"
            output_path = os.path.join(class_dir, filename)
            
            # Save audio file
            sf.write(output_path, audio_data, self.sample_rate)
            
            # Save metadata if provided
            if metadata:
                metadata_file = os.path.join(output_dir, "sounds.json")
                
                # Load existing metadata if it exists
                all_metadata = {}
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            all_metadata = json.load(f)
                    except json.JSONDecodeError:
                        logging.warning(f"Error decoding {metadata_file}, creating new metadata")
                
                # Add current sound metadata
                all_metadata[filename] = {
                    "class": class_name,
                    "timestamp": timestamp,
                    "duration": len(audio_data) / self.sample_rate,
                    "sample_rate": self.sample_rate,
                    **metadata
                }
                
                # Save updated metadata
                with open(metadata_file, 'w') as f:
                    json.dump(all_metadata, f, indent=2)
            
            logging.info(f"Saved training sound to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving training sound: {str(e)}")
            return None 