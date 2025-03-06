#!/usr/bin/env python3
"""
Test script for the unified FeatureExtractor to diagnose feature extraction issues.
"""

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the unified feature extractor
from backend.features.extractor import FeatureExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction_test.log"),
        logging.StreamHandler()
    ]
)

def main():
    """
    Test the unified FeatureExtractor with our new improvements:
    - Improved feature extraction for all feature types
    - Consistent preprocessing for all feature types
    - Better handling of short sounds
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test feature extraction')
    parser.add_argument('--sample', default='test_audio.wav', help='Sample audio file to test')
    parser.add_argument('--output_dir', default='test_output', help='Directory to save test output')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the unified feature extractor
    logging.info("Initializing FeatureExtractor...")
    extractor = FeatureExtractor(
        sample_rate=8000,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )
    logging.info("FeatureExtractor initialized successfully")
    
    # Test file path
    sample_path = args.sample
    if not os.path.exists(sample_path):
        logging.error(f"Sample file not found: {sample_path}")
        sys.exit(1)
    
    # Test 1: Basic feature extraction
    logging.info("TEST 1: Basic feature extraction")
    test_basic_extraction(extractor, sample_path, args.output_dir)
    
    # Test 2: Model-specific feature extraction
    logging.info("TEST 2: Model-specific feature extraction")
    test_model_specific_extraction(extractor, sample_path, args.output_dir)
    
    # Test 3: Visualization
    logging.info("TEST 3: Feature visualization")
    test_feature_visualization(extractor, sample_path, args.output_dir)
    
    logging.info("ALL TESTS PASSED: Unified FeatureExtractor is working correctly!")

def test_basic_extraction(extractor, sample_path, output_dir):
    """Test basic feature extraction"""
    start_time = time.time()
    
    # Extract all features
    features = extractor.extract_features(sample_path, is_file=True)
    
    extraction_time = time.time() - start_time
    logging.info(f"Feature extraction took {extraction_time:.2f} seconds")
    
    # Check if features were extracted successfully
    if features is None:
        logging.error("Feature extraction failed")
        sys.exit(1)
    
    # Log feature shapes and statistics
    logging.info(f"Mel spectrogram shape: {features['mel_spectrogram'].shape}")
    logging.info(f"MFCC shape: {features['mfccs'].shape}")
    logging.info(f"Number of statistical features: {len(features['statistical'])}")
    
    # Save features
    np.savez_compressed(os.path.join(output_dir, "all_features.npz"), features=features)
    logging.info(f"Saved features to {os.path.join(output_dir, 'all_features.npz')}")

def test_model_specific_extraction(extractor, sample_path, output_dir):
    """Test model-specific feature extraction"""
    # Extract all features first
    all_features = extractor.extract_features(sample_path, is_file=True)
    
    # Test CNN feature extraction
    cnn_features = extractor.extract_features_for_model(all_features, model_type='cnn')
    logging.info(f"CNN features shape: {cnn_features.shape}")
    np.save(os.path.join(output_dir, "cnn_features.npy"), cnn_features)
    
    # Test RF feature extraction
    rf_features = extractor.extract_features_for_model(all_features, model_type='rf')
    logging.info(f"RF features count: {len(rf_features)}")
    np.savez_compressed(os.path.join(output_dir, "rf_features.npz"), features=rf_features)
    
    # Test direct model-specific extraction
    cnn_features_direct = extractor.extract_features_for_model(
        extractor.extract_features(sample_path, is_file=True), 
        model_type='cnn'
    )
    
    rf_features_direct = extractor.extract_features_for_model(
        extractor.extract_features(sample_path, is_file=True), 
        model_type='rf'
    )
    
    # Consistency check
    cnn_equal = np.array_equal(cnn_features, cnn_features_direct)
    rf_equal = all(rf_features[k] == rf_features_direct[k] for k in rf_features.keys())
    
    logging.info(f"CNN features consistent: {cnn_equal}")
    logging.info(f"RF features consistent: {rf_equal}")
    
    assert cnn_equal, "CNN features are not consistent"
    assert rf_equal, "RF features are not consistent"
    
    logging.info("✓ Consistent feature extraction")

def test_feature_visualization(extractor, sample_path, output_dir):
    """Test feature visualization"""
    features = extractor.extract_features(sample_path, is_file=True)
    
    # Visualize mel spectrogram
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
    
    # Visualize MFCC features
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        features['mfccs'],
        x_axis='time',
        sr=extractor.sample_rate,
        hop_length=extractor.hop_length
    )
    plt.colorbar()
    plt.title('MFCC Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mfcc_features.png'))
    
    # Visualize statistical features
    plt.figure(figsize=(12, 6))
    feat_names = list(features['statistical'].keys())
    feat_values = [features['statistical'][k] for k in feat_names]
    plt.bar(range(len(feat_names)), feat_values)
    plt.xticks(range(len(feat_names)), feat_names, rotation=90)
    plt.title('Statistical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_features.png'))
    
    logging.info(f"Saved visualizations to {output_dir}")

if __name__ == "__main__":
    main() 