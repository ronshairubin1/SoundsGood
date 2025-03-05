#!/usr/bin/env python3
"""
Unified Audio Chopper

This module provides a single, unified implementation for chopping audio files
into individual sound segments. It is used by both training and inference pipelines
to ensure consistent preprocessing.

The approach follows these steps:
1. Detect sound boundaries using amplitude thresholds
2. Apply minimum silence duration to differentiate sounds
3. Add 20ms padding before and after detected sounds
4. Output individual sound segments
"""

import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_processing.log"),
        logging.StreamHandler()
    ]
)

class AudioChopper:
    """
    Unified implementation for chopping audio into individual sound segments.
    Used by both training and inference to ensure consistency.
    """
    
    def __init__(self, sample_rate=8000, sound_threshold=0.03, min_silence_duration=0.5, padding_ms=20):
        """
        Initialize the audio chopper.
        
        Args:
            sample_rate: Sample rate for audio processing (Hz)
            sound_threshold: Amplitude threshold for sound detection
            min_silence_duration: Minimum silence duration to separate sounds (seconds)
            padding_ms: Padding to add before and after detected sounds (milliseconds)
        """
        self.sample_rate = sample_rate
        self.sound_threshold = sound_threshold
        self.min_silence_duration = min_silence_duration
        self.padding_samples = int(padding_ms * sample_rate / 1000)  # Convert ms to samples
        
    def detect_sound_boundaries(self, audio):
        """
        Detect the start and end boundaries of sounds in an audio segment.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            List of (start, end) tuples in samples
        """
        # Calculate RMS energy in small windows
        hop_length = int(0.01 * self.sample_rate)  # 10ms windows
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Find frames where sound is above threshold
        sound_frames = np.where(rms > self.sound_threshold)[0]
        
        if len(sound_frames) == 0:
            return []
            
        # Convert frame indices to sample indices
        sound_samples = sound_frames * hop_length
        
        # Find gaps between sounds
        min_silence_samples = int(self.min_silence_duration * self.sample_rate)
        sound_boundaries = []
        
        current_start = sound_samples[0]
        prev_sample = sound_samples[0]
        
        for sample in sound_samples[1:]:
            # If there's a gap larger than min_silence_duration
            if sample - prev_sample > min_silence_samples:
                sound_boundaries.append((current_start, prev_sample))
                current_start = sample
            prev_sample = sample
        
        # Add the last sound segment
        sound_boundaries.append((current_start, sound_samples[-1]))
        
        # Add padding to each boundary
        padded_boundaries = []
        for start, end in sound_boundaries:
            padded_start = max(0, start - self.padding_samples)
            padded_end = min(len(audio), end + self.padding_samples)
            padded_boundaries.append((padded_start, padded_end))
            
        return padded_boundaries
        
    def chop_audio_file(self, file_path, output_dir=None):
        """
        Chop an audio file into individual sound segments.
        
        Args:
            file_path: Path to the audio file
            output_dir: Directory to save chopped segments (if None, don't save)
            
        Returns:
            List of audio segments as numpy arrays
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Detect sound boundaries
            boundaries = self.detect_sound_boundaries(audio)
            
            segments = []
            
            # Extract and save each segment
            for i, (start, end) in enumerate(boundaries):
                segment = audio[start:end]
                segments.append(segment)
                
                # Save segment if output directory is provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = Path(file_path).stem
                    segment_path = os.path.join(output_dir, f"{base_name}_{i+1}.wav")
                    sf.write(segment_path, segment, self.sample_rate)
                    logging.info(f"Saved segment to {segment_path}")
            
            return segments
        
        except Exception as e:
            logging.error(f"Error chopping audio file {file_path}: {str(e)}")
            return []
            
    def chop_audio_array(self, audio):
        """
        Chop an audio array into individual sound segments.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            List of audio segments as numpy arrays
        """
        try:
            # Detect sound boundaries
            boundaries = self.detect_sound_boundaries(audio)
            
            # Extract each segment
            segments = [audio[start:end] for start, end in boundaries]
            
            return segments
        
        except Exception as e:
            logging.error(f"Error chopping audio array: {str(e)}")
            return []
            
    def chop_directory(self, input_dir, output_dir):
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save chopped segments
            
        Returns:
            Dictionary mapping original files to lists of output files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Process all wav files
        for file_path in input_dir.glob("*.wav"):
            segments = self.chop_audio_file(file_path, output_dir)
            results[str(file_path)] = len(segments)
            
        return results 