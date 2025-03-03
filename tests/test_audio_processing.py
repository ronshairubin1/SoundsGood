#!/usr/bin/env python
"""
Test script to analyze specific audio files that are failing.
This helps diagnose issues with audio processing.
"""

import os
import sys
import logging
import numpy as np
import pyloudnorm as pyln
from src.core.audio.processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_file(file_path):
    """
    Test a specific audio file with detailed logging.
    """
    print(f"\n===== TESTING FILE: {file_path} =====")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        return
    
    # Create audio processor with our updated settings
    processor = AudioProcessor(sample_rate=8000)
    
    # Test loading the file
    try:
        y, sr = processor.load_audio(file_path)
        print(f"Successfully loaded audio file.")
        print(f"Audio length: {len(y)} samples ({len(y)/sr:.3f} seconds)")
        print(f"Sample rate: {sr} Hz")
        print(f"Max amplitude: {np.max(np.abs(y)):.3f}")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    # Test preprocessing (without centering)
    try:
        # Normalize but don't center
        print("\nTesting volume normalization only:")
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if np.isfinite(loudness):
            y_norm = pyln.normalize.loudness(y, loudness, -23.0)
        else:
            y_norm = y / (np.max(np.abs(y)) + 1e-8) * 0.9
        print(f"After normalization: {len(y_norm)} samples")
    except Exception as e:
        print(f"Error normalizing audio: {e}")
    
    # Test sound detection
    try:
        print("\nTesting sound detection:")
        has_sound = processor.detect_sound(y)
        print(f"Sound detected: {has_sound}")
        
        start_idx, end_idx = processor.detect_sound_boundaries(y)
        print(f"Sound boundaries: start={start_idx}, end={end_idx}")
        print(f"Sound duration: {(end_idx - start_idx)/sr:.3f} seconds")
        
        # Compare with different thresholds
        original_threshold = processor.sound_threshold
        test_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
        for threshold in test_thresholds:
            processor.sound_threshold = threshold
            has_sound = processor.detect_sound(y)
            start_idx, end_idx = processor.detect_sound_boundaries(y)
            print(f"Threshold {threshold}: Detected={has_sound}, Duration={(end_idx - start_idx)/sr:.3f}s")
        processor.sound_threshold = original_threshold
    except Exception as e:
        print(f"Error in sound detection: {e}")
    
    # Test centering
    try:
        print("\nTesting audio centering:")
        y_centered = processor.center_audio(y)
        print(f"After centering: {len(y_centered)} samples ({len(y_centered)/sr:.3f} seconds)")
    except Exception as e:
        print(f"Error centering audio: {e}")
    
    # Test full preprocessing
    try:
        print("\nTesting full preprocessing:")
        y_processed = processor.load_and_preprocess_audio(file_path, training_mode=True)
        print(f"After preprocessing: {len(y_processed)} samples ({len(y_processed)/sr:.3f} seconds)")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
    
    # Test mel spectrogram extraction
    try:
        print("\nTesting mel spectrogram extraction:")
        mel_spec = processor.extract_mel_spectrogram(y_processed)
        print(f"Mel spectrogram shape: {mel_spec.shape}")
    except Exception as e:
        print(f"Error extracting mel spectrogram: {e}")
    
    # Test full CNN processing
    try:
        print("\nTesting full CNN processing:")
        features = processor.process_audio_for_cnn(file_path, training_mode=True)
        print(f"Final features shape: {features.shape}")
        print("SUCCESS: Full processing completed without errors")
    except Exception as e:
        print(f"Error in full processing: {e}")
    
    print("\n===== TEST COMPLETE =====")

def main():
    """
    Main function to test audio processing.
    """
    # Default to a specific file that we know is failing
    default_file = "data/sounds/training_sounds/eh/eh_admin_20.wav"
    
    # Check if a file was provided as an argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    test_file(file_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())