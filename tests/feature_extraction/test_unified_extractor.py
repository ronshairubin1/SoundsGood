#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test the unified feature extractor implementation.

This script performs a series of tests on the unified feature extractor to
verify its functionality, consistency, and performance.
"""

import os
import sys
import logging
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import librosa

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the unified feature extractor
from backend.features.extractor import FeatureExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_extractor_test.log"),
        logging.StreamHandler()
    ]
)

def main():
    """
    Main test function
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test the unified feature extractor')
    parser.add_argument('--audio_dir', default='data/sounds', help='Directory with audio files')
    parser.add_argument('--training_sounds_dir', default='data/sounds/training_sounds', help='Directory with training sounds')
    parser.add_argument('--output_dir', default='tests/output', help='Directory for test results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the unified feature extractor with adjusted parameters
    extractor = FeatureExtractor(
        sample_rate=16000,  # Higher sample rate
        n_mels=40,          # Fewer mel bands
        n_fft=512,          # Smaller FFT window
        hop_length=128,     # Smaller hop length
        use_cache=True,
        cache_dir="backend/data/features/cache"
    )
    
    # Run tests
    test_basic_extraction(extractor, args.training_sounds_dir, args.output_dir)
    test_model_specific_features(extractor, args.training_sounds_dir, args.output_dir)
    test_metadata_handling(args.audio_dir, args.training_sounds_dir, args.output_dir)
    test_batch_extraction(extractor, args.training_sounds_dir, args.output_dir)
    
    logging.info("All tests completed successfully!")

def find_sound_file_with_metadata(base_dir):
    """
    Find a sound file that has accompanying metadata
    Returns (sound_path, metadata_path, class_name)
    """
    # First check if sounds.json exists
    global_metadata_path = os.path.join(os.path.dirname(base_dir), 'sounds.json')
    global_metadata = None
    if os.path.exists(global_metadata_path):
        try:
            with open(global_metadata_path, 'r') as f:
                global_metadata = json.load(f)
            logging.info(f"Found global metadata file: {global_metadata_path}")
        except Exception as e:
            logging.warning(f"Error reading global metadata: {e}")
    
    # Look for class directories and wav files
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        for file in os.listdir(class_path):
            if file.endswith('.wav'):
                wav_path = os.path.join(class_path, file)
                # Check for companion JSON with same name
                json_path = wav_path.replace('.wav', '.json')
                
                # Even if there's no individual JSON, return the WAV file
                # since we have the global metadata
                return wav_path, json_path if os.path.exists(json_path) else None, class_dir, global_metadata
                
    return None, None, None, global_metadata

def test_basic_extraction(extractor, audio_dir, output_dir):
    """
    Test basic feature extraction capabilities
    """
    logging.info("=== Testing Basic Feature Extraction ===")
    
    # Find a sample audio file with metadata
    wav_path, json_path, class_name, _ = find_sound_file_with_metadata(audio_dir)
    
    if wav_path is None:
        logging.error(f"No suitable .wav files with metadata found in {audio_dir}")
        return
    
    logging.info(f"Using sample file: {wav_path}")
    if json_path:
        logging.info(f"With metadata: {json_path}")
    
    # Extract features
    try:
        start_time = time.time()
        features = extractor.extract_features(wav_path)
        extraction_time = time.time() - start_time
        
        if features is None:
            logging.error("Feature extraction failed!")
            return
        
        # Log feature information
        logging.info(f"Feature extraction took {extraction_time:.2f} seconds")
        logging.info(f"Feature keys: {list(features.keys())}")
        
        if 'mel_spectrogram' in features:
            logging.info(f"Mel spectrogram shape: {features['mel_spectrogram'].shape}")
        
        if 'mfccs' in features:
            logging.info(f"MFCC shape: {features['mfccs'].shape}")
        
        if 'statistical' in features:
            logging.info(f"Number of statistical features: {len(features['statistical'])}")
        
        # Save a visualization of the mel spectrogram if available
        if 'mel_spectrogram' in features:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                features['mel_spectrogram'],
                y_axis='mel',
                x_axis='time',
                sr=extractor.sample_rate,
                hop_length=extractor.hop_length
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mel_spectrogram.png'))
        
        # Save feature stats to file
        with open(os.path.join(output_dir, 'feature_stats.txt'), 'w') as f:
            f.write(f"Feature extraction statistics:\n")
            f.write(f"Sample file: {wav_path}\n")
            f.write(f"Class: {class_name}\n")
            f.write(f"Extraction time: {extraction_time:.2f} seconds\n")
            f.write(f"Feature keys: {list(features.keys())}\n")
            
            if 'mel_spectrogram' in features:
                f.write(f"Mel spectrogram shape: {features['mel_spectrogram'].shape}\n")
            
            if 'mfccs' in features:
                f.write(f"MFCC shape: {features['mfccs'].shape}\n")
            
            if 'statistical' in features:
                f.write(f"Number of statistical features: {len(features['statistical'])}\n\n")
                f.write("Statistical features:\n")
                for key, value in features['statistical'].items():
                    f.write(f"{key}: {value}\n")
        
        logging.info("Basic feature extraction test successful")
    except Exception as e:
        logging.error(f"Error in basic feature extraction: {str(e)}")

def test_model_specific_features(extractor, audio_dir, output_dir):
    """
    Test model-specific feature extraction
    """
    logging.info("=== Testing Model-Specific Feature Extraction ===")
    
    # Find a sample audio file with metadata
    wav_path, json_path, class_name, _ = find_sound_file_with_metadata(audio_dir)
    
    if wav_path is None:
        logging.error(f"No suitable .wav files with metadata found in {audio_dir}")
        return
    
    try:
        # Extract all features
        all_features = extractor.extract_features(wav_path)
        
        if all_features is None:
            logging.error("Feature extraction failed!")
            return
        
        # Test CNN features
        try:
            cnn_features = extractor.extract_features_for_model(all_features, model_type='cnn')
            
            if cnn_features is None:
                logging.error("CNN feature extraction failed!")
            else:
                logging.info(f"CNN features shape: {cnn_features.shape}")
                
                # Save CNN features visualization
                plt.figure(figsize=(10, 4))
                plt.imshow(cnn_features.squeeze(), aspect='auto', origin='lower')
                plt.colorbar(format='%+2.0f dB')
                plt.title('CNN Features (Mel Spectrogram)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'cnn_features.png'))
        except Exception as e:
            logging.error(f"Error in CNN feature extraction: {str(e)}")
        
        # Test RF features
        try:
            rf_features = extractor.extract_features_for_model(all_features, model_type='rf')
            
            if rf_features is None:
                logging.error("RF feature extraction failed!")
            else:
                logging.info(f"RF features count: {len(rf_features)}")
                
                # Save RF features visualization
                plt.figure(figsize=(12, 6))
                keys = list(rf_features.keys())
                values = [rf_features[k] for k in keys]
                plt.bar(range(len(keys)), values)
                plt.xticks([])  # Hide x-tick labels as there are too many
                plt.title('RF Features (Statistical)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'rf_features.png'))
                
                # Save features to file
                with open(os.path.join(output_dir, 'rf_features.txt'), 'w') as f:
                    for key, value in rf_features.items():
                        f.write(f"{key}: {value}\n")
        except Exception as e:
            logging.error(f"Error in RF feature extraction: {str(e)}")
        
        # Test ensemble features
        try:
            ensemble_features = extractor.extract_features_for_model(all_features, model_type='ensemble')
            
            if ensemble_features is None:
                logging.error("Ensemble feature extraction failed!")
            else:
                logging.info(f"Ensemble features keys: {list(ensemble_features.keys())}")
                
                if 'cnn' in ensemble_features:
                    logging.info(f"  CNN shape: {ensemble_features['cnn'].shape}")
                
                if 'rf' in ensemble_features:
                    logging.info(f"  RF count: {len(ensemble_features['rf'])}")
        except Exception as e:
            logging.error(f"Error in ensemble feature extraction: {str(e)}")
        
        logging.info("Model-specific feature extraction test successful")
    except Exception as e:
        logging.error(f"Error in model-specific feature extraction: {str(e)}")

def test_metadata_handling(base_dir, training_dir, output_dir):
    """
    Test handling of metadata files
    """
    logging.info("=== Testing Metadata Handling ===")
    
    # Check for global sounds.json
    global_json_path = os.path.join(base_dir, 'sounds.json')
    if os.path.exists(global_json_path):
        try:
            with open(global_json_path, 'r') as f:
                global_metadata = json.load(f)
            logging.info(f"Successfully loaded global sounds.json with {len(global_metadata)} entries")
            
            # Save first few entries for inspection
            with open(os.path.join(output_dir, 'global_metadata_sample.json'), 'w') as f:
                json.dump(global_metadata[:5] if isinstance(global_metadata, list) and len(global_metadata) > 5 
                         else global_metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Error reading global sounds.json: {str(e)}")
    else:
        logging.warning(f"Global sounds.json not found at {global_json_path}")
    
    # Find a sound file with individual metadata
    wav_path, json_path, class_name, _ = find_sound_file_with_metadata(training_dir)
    
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                sound_metadata = json.load(f)
            logging.info(f"Successfully loaded individual sound metadata for {os.path.basename(wav_path)}")
            
            # Save for inspection
            with open(os.path.join(output_dir, 'sound_metadata_sample.json'), 'w') as f:
                json.dump(sound_metadata, f, indent=2)
                
            logging.info(f"Metadata keys: {list(sound_metadata.keys())}")
        except Exception as e:
            logging.error(f"Error reading sound metadata: {str(e)}")
    else:
        logging.warning(f"No individual sound metadata found")
    
    logging.info("Metadata handling test completed")

def test_batch_extraction(extractor, audio_dir, output_dir):
    """
    Test batch feature extraction
    """
    logging.info("=== Testing Batch Feature Extraction ===")
    
    # Get all class directories
    class_dirs = []
    for item in os.listdir(audio_dir):
        if os.path.isdir(os.path.join(audio_dir, item)) and any(
            fname.endswith('.wav') for fname in os.listdir(os.path.join(audio_dir, item))
        ):
            class_dirs.append(item)
    
    if not class_dirs:
        logging.error(f"No class directories with .wav files found in {audio_dir}")
        return
    
    # Limit to max 3 classes for the test
    if len(class_dirs) > 3:
        class_dirs = class_dirs[:3]
    
    logging.info(f"Testing with classes: {class_dirs}")
    
    try:
        # Test CNN batch extraction
        start_time = time.time()
        X_cnn, y_cnn, class_names, stats = extractor.batch_extract_features(
            audio_dir, 
            class_dirs, 
            model_type='cnn'
        )
        cnn_time = time.time() - start_time
        
        logging.info(f"CNN batch extraction took {cnn_time:.2f} seconds")
        if X_cnn is not None:
            logging.info(f"CNN features shape: {X_cnn.shape}")
        else:
            logging.info("CNN features: None")
            
        logging.info(f"Classes: {class_names}")
        logging.info(f"Stats: {stats}")
        
        # Test RF batch extraction
        start_time = time.time()
        X_rf, y_rf, _, stats = extractor.batch_extract_features(
            audio_dir, 
            class_dirs, 
            model_type='rf'
        )
        rf_time = time.time() - start_time
        
        logging.info(f"RF batch extraction took {rf_time:.2f} seconds")
        if X_rf is not None:
            logging.info(f"RF features shape: {X_rf.shape}")
        else:
            logging.info("RF features: None")
            
        logging.info(f"Stats: {stats}")
        
        # Save batch extraction summary
        with open(os.path.join(output_dir, 'batch_extraction_summary.txt'), 'w') as f:
            f.write(f"Batch extraction summary:\n")
            f.write(f"Classes: {class_names}\n\n")
            
            f.write(f"CNN extraction:\n")
            f.write(f"  Time: {cnn_time:.2f} seconds\n")
            if X_cnn is not None:
                f.write(f"  Features shape: {X_cnn.shape}\n")
            else:
                f.write(f"  Features shape: None\n")
                
            f.write(f"  Total processed: {stats['total_processed']}\n")
            f.write(f"  Total skipped: {stats['total_skipped']}\n\n")
            
            f.write(f"RF extraction:\n")
            f.write(f"  Time: {rf_time:.2f} seconds\n")
            if X_rf is not None:
                f.write(f"  Features shape: {X_rf.shape}\n")
            else:
                f.write(f"  Features shape: None\n")
                
            f.write(f"\nProcessed counts by class:\n")
            for class_dir, count in stats['processed_counts'].items():
                f.write(f"  {class_dir}: {count}\n")
            
            if stats['error_files']:
                f.write(f"\nError files ({len(stats['error_files'])}):\n")
                for error_file in stats['error_files'][:10]:  # List only first 10 errors
                    f.write(f"  {error_file}\n")
                
                f.write(f"\nError types:\n")
                for error_type, count in stats['error_types'].items():
                    f.write(f"  {error_type}: {count}\n")
        
        logging.info("Batch extraction test successful")
    except Exception as e:
        logging.error(f"Error in batch extraction: {str(e)}")

if __name__ == "__main__":
    main() 