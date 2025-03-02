#!/usr/bin/env python
"""
Script to analyze audio files and report statistics.
This helps diagnose issues with audio files being too short or problematic.
"""

import os
import sys
import logging
from src.core.audio.processor import AudioProcessor

def main():
    """Main function to analyze audio files."""
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Default audio directory
    audio_dir = 'data/sounds/training_sounds'
    
    # Check if a directory was provided as an argument
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
    
    # Check if the directory exists
    if not os.path.exists(audio_dir):
        print(f"Error: Directory '{audio_dir}' does not exist")
        return 1
    
    print(f"Analyzing audio files in '{audio_dir}'...")
    
    # Create audio processor with two different configurations for comparison
    # Default configuration - 8kHz sample rate, 512 FFT size
    processor_default = AudioProcessor(sample_rate=8000)
    
    # Alternative configuration - 16kHz sample rate, 2048 FFT size (original)
    processor_original = AudioProcessor(sample_rate=16000)
    processor_original.n_fft = 2048
    processor_original.n_mels = 128
    processor_original.hop_length = 512
    
    # Run analysis with both configurations
    print("\n\n=== ANALYSIS WITH NEW SETTINGS (8kHz, smaller block size) ===")
    processor_default.analyze_audio_directory(audio_dir)
    
    print("\n\n=== ANALYSIS WITH ORIGINAL SETTINGS (16kHz, larger block size) ===")
    processor_original.analyze_audio_directory(audio_dir)
    
    # Print guidance
    print("\n\n=== RECOMMENDATION ===")
    print("With the new settings (8kHz sample rate, 512 FFT size):")
    print("- Each audio sample needs to be at least 0.064 seconds long (512/8000)")
    print("- More of your audio files should now be usable for training")
    print("- The spectrograms will have lower frequency resolution but better time resolution")
    print("- Time-stretching has been improved to handle very short files better")
    
    print("\nTo use these new settings:")
    print("1. The code has been updated to use these settings by default")
    print("2. Run the application normally with 'python run.py'")
    print("3. Train your model as usual - more of your files should be processed successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 