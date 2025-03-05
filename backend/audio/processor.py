#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SoundProcessor for audio preprocessing in both training and inference.

This module provides a unified approach to audio preprocessing to ensure 
consistency between training and inference pipelines.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
import logging
import os
import json
import hashlib
import tempfile
import soundfile as sf

# Default sample rate
SAMPLE_RATE = 16000

class SoundProcessor:
    """
    Unified SoundProcessor for audio preprocessing in both training and inference.
    
    This ensures consistency: the same trimming, centering, RMS normalization,
    time-stretch, and mel-spectrogram creation are used throughout the application.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, sound_threshold=0.1, min_silence_duration=0.5, 
                trim_silence=True, normalize_audio=True, n_mels=64, n_fft=1024, hop_length=256):
        """
        Initialize the SoundProcessor.
        
        Args:
            sample_rate (int): Audio sample rate
            sound_threshold (float): Threshold for sound detection
            min_silence_duration (float): Minimum duration of silence in seconds
            trim_silence (bool): Whether to trim silence from audio
            normalize_audio (bool): Whether to normalize audio
            n_mels (int): Number of mel bands
            n_fft (int): FFT window size
            hop_length (int): Hop length for feature extraction
        """
        self.sample_rate = sample_rate
        self.sound_threshold = sound_threshold  # Threshold for sound detection
        self.min_silence_duration = min_silence_duration
        self.trim_silence = trim_silence
        self.normalize_audio = normalize_audio
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def is_sound(self, audio):
        """
        Check if the audio contains a sound above the threshold.
        
        Args:
            audio (np.array): Audio data
            
        Returns:
            bool: True if sound is detected, False otherwise
        """
        if audio is None or len(audio) == 0:
            return False
            
        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(audio)))
        return rms > self.sound_threshold
    
    def detect_sound(self, audio):
        """
        Detect segments with sound above the threshold.
        
        Args:
            audio (np.array): Audio data
            
        Returns:
            np.array: Audio segments with sound
        """
        if audio is None or len(audio) == 0:
            return None
            
        # Calculate rolling RMS
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Simple rolling RMS calculation
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energy.append(np.sqrt(np.mean(np.square(frame))))
        
        energy = np.array(energy)
        mask = energy > self.sound_threshold
        
        return mask
    
    def detect_sound_boundaries(self, audio):
        """
        Detect the start and end of sound in the audio.
        
        Args:
            audio (np.array): Audio data
            
        Returns:
            tuple: (start_sample, end_sample) or (0, len(audio)) if no clear boundaries
        """
        if audio is None or len(audio) == 0:
            return (0, 0)
            
        # Create a mask of sound vs. silence
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        # Simple rolling RMS calculation
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energy.append(np.sqrt(np.mean(np.square(frame))))
        
        energy = np.array(energy)
        sound_mask = energy > self.sound_threshold
        
        # Find continuous segments of sound
        segments = []
        start = None
        min_silence_frames = int(self.min_silence_duration / (hop_length / self.sample_rate))
        
        for i, is_sound in enumerate(sound_mask):
            if is_sound and start is None:
                start = i
            elif not is_sound and start is not None:
                # Check if silence is long enough
                if i - start >= min_silence_frames:
                    segments.append((start, i))
                    start = None
        
        # Handle the case where the sound continues to the end
        if start is not None:
            segments.append((start, len(sound_mask)))
        
        # If no segments found, return full audio
        if not segments:
            return (0, len(audio))
        
        # Find the longest segment
        longest_segment = max(segments, key=lambda x: x[1] - x[0])
        
        # Convert frame indices to sample indices
        start_sample = longest_segment[0] * hop_length
        end_sample = min(longest_segment[1] * hop_length, len(audio))
        
        return (start_sample, end_sample)
    
    def normalize_duration(self, audio, target_duration=1.0):
        """
        Normalize the audio to a target duration.
        
        Args:
            audio (np.array): Audio data
            target_duration (float): Target duration in seconds
            
        Returns:
            np.array: Normalized audio
        """
        if audio is None or len(audio) == 0:
            return np.zeros(int(target_duration * self.sample_rate))
            
        current_duration = len(audio) / self.sample_rate
        
        # If audio is already the target duration (within 10%), return as is
        if abs(current_duration - target_duration) / target_duration < 0.1:
            return audio
            
        # If audio is too short, pad with zeros
        if current_duration < target_duration:
            padding = int((target_duration - current_duration) * self.sample_rate)
            padded_audio = np.pad(audio, (0, padding), 'constant')
            return padded_audio
            
        # If audio is too long, time-stretch
        time_stretch_ratio = target_duration / current_duration
        return librosa.effects.time_stretch(audio, rate=time_stretch_ratio)
    
    def center_audio(self, audio):
        """
        Center the audio by trimming or padding.
        
        Args:
            audio (np.array): Audio data
            
        Returns:
            np.array: Centered audio
        """
        if audio is None or len(audio) == 0:
            return None
            
        # Trim front and back silence
        if self.trim_silence:
            # Find sound boundaries
            start, end = self.detect_sound_boundaries(audio)
            if start < end:
                audio = audio[start:end]
        
        # Normalize RMS amplitude
        if self.normalize_audio:
            rms = np.sqrt(np.mean(np.square(audio)))
            if rms > 0:
                audio = audio / rms
        
        return audio
    
    def process_audio(self, audio):
        """
        Process audio for model input.
        
        Args:
            audio (np.array): Audio data
            
        Returns:
            np.array: Processed feature matrix
        """
        if audio is None or len(audio) == 0:
            return None
            
        try:
            # Pre-process the audio
            processed_audio = self.center_audio(audio)
            if processed_audio is None or len(processed_audio) == 0:
                return None
            
            # Convert to mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=processed_audio, 
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Convert to log scale
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Normalize to range [0, 1]
            normalized = (log_mel_spectrogram - log_mel_spectrogram.min()) / (log_mel_spectrogram.max() - log_mel_spectrogram.min() + 1e-8)
            
            return normalized
            
        except Exception as e:
            logging.error(f"Error in processing audio: {str(e)}")
            return None 