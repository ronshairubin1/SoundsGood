#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio recording and chopping module for training and inference.

This module provides a unified interface for:
1. Recording audio from the microphone
2. Chopping continuous audio into distinct sounds
3. Applying preprocessing to prepare sounds for training or prediction
"""

import os
import numpy as np
import pyaudio
import wave
import librosa
import soundfile as sf
import logging
import time
import tempfile
from datetime import datetime
from pathlib import Path
import threading
import queue
from .preprocessor import AudioPreprocessor

class AudioRecorder:
    """
    Audio recorder for capturing training samples and inference input.
    """
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1, 
                format=pyaudio.paInt16, preprocessor=None, 
                silence_threshold=0.008, min_silence_duration=0.5):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate (int): Sample rate for recording
            chunk_size (int): Size of audio chunks
            channels (int): Number of audio channels (1=mono, 2=stereo)
            format: PyAudio format constant
            preprocessor (AudioPreprocessor, optional): Preprocessor for recorded audio
            silence_threshold (float): Threshold for silence detection
            min_silence_duration (float): Minimum silence duration to stop recording (seconds)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.silence_samples = int(min_silence_duration * sample_rate)
        
        # Create audio interface
        self.pyaudio = pyaudio.PyAudio()
        
        # Create preprocessor if not provided
        self.preprocessor = preprocessor or AudioPreprocessor(
            sample_rate=sample_rate,
            sound_threshold=silence_threshold,
            min_silence_duration=min_silence_duration
        )
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.frames = []
        self.ambient_noise_level = None
        
        logging.info(f"Initialized AudioRecorder with sample_rate={sample_rate}, "
                    f"silence_threshold={silence_threshold}")
    
    def __del__(self):
        """Clean up PyAudio resources."""
        self.pyaudio.terminate()
    
    def calibrate_ambient_noise(self, duration=1.0):
        """
        Calibrate for ambient noise level.
        
        Args:
            duration (float): Duration to sample ambient noise in seconds
            
        Returns:
            float: Detected ambient noise level
        """
        logging.info(f"Calibrating ambient noise for {duration} seconds")
        
        # Open stream for calibration
        stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Record ambient noise for the specified duration
        frames = []
        chunks_to_record = int(duration * self.sample_rate / self.chunk_size)
        
        for _ in range(chunks_to_record):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        # Close the stream
        stream.stop_stream()
        stream.close()
        
        # Convert frames to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32767.0  # Convert to float
        
        # Calculate ambient noise level (RMS)
        noise_level = np.sqrt(np.mean(np.square(audio_data)))
        
        # Set ambient noise level and adjust silence threshold
        self.ambient_noise_level = noise_level
        adjusted_threshold = max(self.silence_threshold, noise_level * 1.5)
        self.preprocessor.sound_threshold = adjusted_threshold
        
        logging.info(f"Calibrated ambient noise level: {noise_level}, "
                   f"adjusted threshold: {adjusted_threshold}")
        
        return noise_level
    
    def start_recording(self, max_duration=30.0, stop_on_silence=True, 
                        callback=None, save_path=None):
        """
        Start recording audio from the microphone.
        
        Args:
            max_duration (float): Maximum recording duration in seconds
            stop_on_silence (bool): Whether to stop recording after detecting silence
            callback (callable): Optional callback function for recording status
            save_path (str, optional): Path to save the recorded audio
            
        Returns:
            bool: True if recording started successfully
        """
        if self.is_recording:
            logging.warning("Recording already in progress")
            return False
        
        self.is_recording = True
        self.frames = []
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_thread,
            args=(max_duration, stop_on_silence, callback, save_path)
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        return True
    
    def _record_thread(self, max_duration, stop_on_silence, callback, save_path):
        """
        Thread function for recording audio.
        
        Args:
            max_duration (float): Maximum recording duration in seconds
            stop_on_silence (bool): Whether to stop recording after detecting silence
            callback (callable): Optional callback function for recording status
            save_path (str, optional): Path to save the recorded audio
        """
        try:
            # Open the audio stream
            stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Calculate maximum chunks to record
            max_chunks = int(max_duration * self.sample_rate / self.chunk_size)
            
            # Variables for silence detection
            silence_counter = 0
            has_detected_sound = False
            
            # Record audio
            for i in range(max_chunks):
                if not self.is_recording:
                    break
                
                # Read audio chunk
                data = stream.read(self.chunk_size)
                self.frames.append(data)
                
                # Process chunk for silence detection if needed
                if stop_on_silence:
                    # Convert audio chunk to numpy array
                    chunk_data = np.frombuffer(data, dtype=np.int16)
                    chunk_data = chunk_data.astype(np.float32) / 32767.0  # Convert to float
                    
                    # Calculate RMS of the chunk
                    chunk_rms = np.sqrt(np.mean(np.square(chunk_data)))
                    
                    # Detect sound or silence
                    if chunk_rms > self.preprocessor.sound_threshold:
                        has_detected_sound = True
                        silence_counter = 0
                    elif has_detected_sound:
                        silence_counter += 1
                    
                    # Check if we have enough silence to stop
                    silence_chunks_needed = self.silence_samples // self.chunk_size
                    if has_detected_sound and silence_counter >= silence_chunks_needed:
                        logging.info("Detected sufficient silence after sound, stopping recording")
                        break
                
                # Call the callback with progress if provided
                if callback:
                    progress = min(100, int((i / max_chunks) * 100))
                    callback(progress, "Recording in progress")
            
            # Close the stream
            stream.stop_stream()
            stream.close()
            
            # Convert recorded frames to a single audio buffer
            audio_data = self._frames_to_array(self.frames)
            
            # Save the audio if requested
            if save_path and audio_data is not None:
                self._save_wav(save_path, audio_data)
                logging.info(f"Saved recording to {save_path}")
                
                if callback:
                    callback(100, f"Recording saved to {save_path}")
            
            logging.info(f"Recording completed: {len(self.frames)} chunks, "
                      f"{len(audio_data)/self.sample_rate:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error in recording thread: {str(e)}")
            
        finally:
            self.is_recording = False
    
    def _frames_to_array(self, frames):
        """
        Convert list of audio frames to numpy array.
        
        Args:
            frames (list): List of audio frame bytes
            
        Returns:
            np.array: Audio data as numpy array
        """
        if not frames:
            return None
        
        # Join frames and convert to numpy array
        audio_bytes = b''.join(frames)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float in range [-1, 1]
        return audio_data.astype(np.float32) / 32767.0
    
    def _save_wav(self, file_path, audio_data):
        """
        Save audio data to a WAV file.
        
        Args:
            file_path (str): Path to save the WAV file
            audio_data (np.array): Audio data
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Convert to int16
            int_data = (audio_data * 32767).astype(np.int16)
            
            # Save using soundfile
            sf.write(file_path, int_data, self.sample_rate)
            
            return True
        except Exception as e:
            logging.error(f"Error saving WAV file: {str(e)}")
            return False
    
    def stop_recording(self):
        """
        Stop the current recording.
        
        Returns:
            np.array: Recorded audio data
        """
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join()
            self.recording_thread = None
        
        return self._frames_to_array(self.frames)
    
    def record_for_training(self, class_name, output_dir, max_duration=30.0, 
                           callback=None, metadata=None):
        """
        Record audio for training data collection.
        
        Args:
            class_name (str): Class label for the recording
            output_dir (str): Directory to save training samples
            max_duration (float): Maximum recording duration in seconds
            callback (callable): Optional callback function for status updates
            metadata (dict, optional): Additional metadata to save
            
        Returns:
            list: Paths to saved audio files
        """
        # Create a temporary file for the raw recording
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        if callback:
            callback(0, f"Recording sounds for class '{class_name}'")
        
        # Record audio
        self.start_recording(
            max_duration=max_duration, 
            stop_on_silence=True,
            callback=callback,
            save_path=temp_path
        )
        
        # Wait for recording to complete
        while self.is_recording:
            time.sleep(0.1)
        
        if callback:
            callback(50, "Processing recording...")
        
        # Load the recorded audio
        audio_data, _ = librosa.load(temp_path, sr=self.sample_rate)
        
        # Chop into segments and preprocess
        segments = self.preprocessor.preprocess_recording(audio_data)
        
        if not segments:
            logging.warning("No valid sound segments found in recording")
            if callback:
                callback(100, "No valid sound segments found")
            return []
        
        # Save processed segments
        output_files = []
        for i, segment in enumerate(segments):
            # Save each segment
            output_path = self.preprocessor.save_training_sound(
                segment, 
                class_name, 
                output_dir,
                metadata=metadata
            )
            
            if output_path:
                output_files.append(output_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if callback:
            callback(100, f"Saved {len(output_files)} sound samples for class '{class_name}'")
        
        logging.info(f"Recorded {len(output_files)} training samples for class '{class_name}'")
        return output_files
    
    def record_for_inference(self, max_duration=5.0, callback=None):
        """
        Record a single sound for inference/prediction.
        
        Args:
            max_duration (float): Maximum recording duration in seconds
            callback (callable): Optional callback function
            
        Returns:
            np.array: Preprocessed audio data ready for model input
        """
        if callback:
            callback(0, "Recording sound for prediction...")
        
        # Record audio
        self.start_recording(
            max_duration=max_duration, 
            stop_on_silence=True,
            callback=callback
        )
        
        # Wait for recording to complete
        while self.is_recording:
            time.sleep(0.1)
        
        # Get the recorded audio
        audio_data = self._frames_to_array(self.frames)
        
        if audio_data is None or len(audio_data) == 0:
            logging.warning("No audio recorded")
            if callback:
                callback(100, "No audio recorded")
            return None
        
        if callback:
            callback(50, "Processing sound...")
        
        # Preprocess the audio for inference
        try:
            # Take only the first sound from the recording
            segments = self.preprocessor.chop_audio(audio_data)
            
            if not segments:
                logging.warning("No sound detected in recording")
                if callback:
                    callback(100, "No sound detected")
                return None
            
            # Preprocess the first segment
            processed_audio = self.preprocessor.preprocess_audio(segments[0])
            
            if callback:
                callback(100, "Sound processed and ready for prediction")
            
            return processed_audio
            
        except Exception as e:
            logging.error(f"Error processing audio for inference: {str(e)}")
            if callback:
                callback(100, f"Error: {str(e)}")
            return None 