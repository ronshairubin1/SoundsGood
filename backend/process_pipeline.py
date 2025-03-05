#!/usr/bin/env python3
"""
Unified Processing Pipeline Demo

This script demonstrates the full audio processing pipeline:
1. Chopping audio files into individual sounds
2. Preprocessing each sound file
3. Augmenting the training data
4. Extracting unified features
5. Preparing model-specific features
6. Training different models

It uses the backend modules we've created for a single, unified approach
to ensure consistency across training and inference.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append('.')

# Import our backend modules
from backend.audio.chopper import AudioChopper
from backend.audio.preprocessor import AudioPreprocessor
from backend.audio.augmentor import AudioAugmentor
from backend.features.extractor import FeatureExtractor
from backend.features.cnn_features import CNNFeaturePreparation
from backend.features.rf_features import RFFeaturePreparation
from backend.data.dataset_manager import DatasetManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def process_pipeline(input_dir, output_base_dir, class_dirs=None):
    """
    Run the complete processing pipeline.
    
    Args:
        input_dir: Directory with raw audio files
        output_base_dir: Base directory for output
        class_dirs: List of class directories to process (if None, process all)
    """
    # Initialize components
    dataset_manager = DatasetManager(base_dir=output_base_dir)
    chopper = AudioChopper()
    preprocessor = AudioPreprocessor()
    augmentor = AudioAugmentor()
    feature_extractor = FeatureExtractor()
    cnn_preparer = CNNFeaturePreparation()
    rf_preparer = RFFeaturePreparation()
    
    # Get class directories
    input_dir = Path(input_dir)
    if class_dirs is None:
        class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    else:
        class_dirs = [input_dir / d for d in class_dirs]
        
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        logging.info(f"Processing class: {class_name}")
        
        # Make sure the class exists in the dataset
        if class_name not in dataset_manager.metadata['sound_classes']:
            dataset_manager.add_class(class_name)
            
        # Step 1: Chop audio files
        logging.info("Step 1: Chopping audio files")
        wav_files = list(class_dir.glob("*.wav"))
        
        for wav_file in wav_files:
            logging.info(f"Chopping {wav_file}")
            
            # Define output directory
            chop_output_dir = dataset_manager.chopped_sounds_dir / class_name
            
            # Chop the file
            chopped_segments = chopper.chop_audio_file(wav_file, chop_output_dir)
            
            # Add chopped files to dataset
            for i, segment_path in enumerate(chop_output_dir.glob(f"{wav_file.stem}_*.wav")):
                dataset_manager.add_file(segment_path, class_name)
                
        # Step 2: Preprocess chopped files
        logging.info("Step 2: Preprocessing chopped files")
        chopped_files = list(dataset_manager.chopped_sounds_dir.glob(f"{class_name}/*.wav"))
        
        for chopped_file in chopped_files:
            logging.info(f"Preprocessing {chopped_file}")
            
            # Define output directory
            preprocessed_output_dir = dataset_manager.raw_sounds_dir / class_name
            output_path = preprocessed_output_dir / chopped_file.name
            
            # Preprocess the file
            preprocessor.preprocess_file(chopped_file, output_path)
            
            # Update file status
            dataset_manager.update_file_status(output_path, 'preprocessed')
            
        # Step 3: Augment preprocessed files for training
        logging.info("Step 3: Augmenting training files")
        preprocessed_files = list(dataset_manager.raw_sounds_dir.glob(f"{class_name}/*.wav"))
        
        for preprocessed_file in preprocessed_files:
            logging.info(f"Augmenting {preprocessed_file}")
            
            # Define output directory
            augmented_output_dir = dataset_manager.augmented_sounds_dir / class_name / preprocessed_file.stem
            
            # Augment the file
            augmentor.augment_file(preprocessed_file, augmented_output_dir)
            
            # Update file status
            dataset_manager.update_file_status(preprocessed_file, 'augmented')
            
        # Step 4: Extract unified features from all files
        logging.info("Step 4: Extracting unified features")
        all_sound_files = (
            list(dataset_manager.raw_sounds_dir.glob(f"{class_name}/*.wav")) +
            list(dataset_manager.augmented_sounds_dir.glob(f"{class_name}/**/*.wav"))
        )
        
        for sound_file in all_sound_files:
            logging.info(f"Extracting features from {sound_file}")
            
            # Extract features
            features = feature_extractor.extract_features(sound_file)
            
            # Save features
            dataset_manager.save_features(sound_file, features, 'unified')
            
        # Step 5: Prepare model-specific features
        logging.info("Step 5: Preparing model-specific features")
        all_features = []
        
        for sound_file in all_sound_files:
            # Load unified features
            features = dataset_manager.load_features(sound_file, 'unified')
            
            if features is not None:
                # Prepare CNN features
                cnn_features = cnn_preparer.prepare_features(features)
                dataset_manager.save_features(sound_file, {'features': cnn_features}, 'cnn')
                
                # Prepare RF features
                rf_features = rf_preparer.prepare_features(features)
                dataset_manager.save_features(sound_file, {'features': rf_features}, 'rf')
                
                # Store for later use
                features['label'] = class_name
                all_features.append(features)
                
    # Return collected features for further use
    return all_features

def visualize_features(features, output_dir):
    """
    Visualize some of the extracted features.
    
    Args:
        features: List of feature dictionaries
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a figure for mel spectrograms
    plt.figure(figsize=(12, 8))
    
    for i, feature_dict in enumerate(features[:4]):  # Show first 4 examples
        if 'mel_spectrogram' in feature_dict:
            plt.subplot(2, 2, i+1)
            plt.title(f"Class: {feature_dict['label']}")
            plt.imshow(feature_dict['mel_spectrogram'], aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'mel_spectrograms.png')
    
    # Create a figure for MFCC features
    plt.figure(figsize=(12, 8))
    
    for i, feature_dict in enumerate(features[:4]):  # Show first 4 examples
        if 'mfccs' in feature_dict:
            plt.subplot(2, 2, i+1)
            plt.title(f"Class: {feature_dict['label']}")
            plt.imshow(feature_dict['mfccs'], aspect='auto', origin='lower')
            plt.colorbar()
            
    plt.tight_layout()
    plt.savefig(output_dir / 'mfccs.png')
    
    # Plot statistical features
    if features and 'statistical' in features[0]:
        # Get feature names
        feature_names = list(features[0]['statistical'].keys())
        
        # Group features by class
        class_features = {}
        for feature_dict in features:
            class_name = feature_dict['label']
            if class_name not in class_features:
                class_features[class_name] = []
            class_features[class_name].append(feature_dict['statistical'])
            
        # Plot some key features
        key_features = ['spectral_centroid_mean', 'zcr_mean', 'rms_mean', 'formant_mean']
        
        plt.figure(figsize=(15, 10))
        
        for i, feature_name in enumerate(key_features):
            plt.subplot(2, 2, i+1)
            plt.title(feature_name)
            
            for class_name, feat_list in class_features.items():
                values = [f[feature_name] for f in feat_list if feature_name in f]
                plt.hist(values, alpha=0.5, label=class_name)
                
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_features.png')

def main():
    parser = argparse.ArgumentParser(description='Run the unified audio processing pipeline')
    parser.add_argument('--input', '-i', default='data/sounds/raw', help='Input directory with raw audio files')
    parser.add_argument('--output', '-o', default='backend/data', help='Base output directory')
    parser.add_argument('--classes', '-c', nargs='*', help='Specific classes to process')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize extracted features')
    
    args = parser.parse_args()
    
    # Run the processing pipeline
    start_time = time.time()
    features = process_pipeline(args.input, args.output, args.classes)
    end_time = time.time()
    
    logging.info(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Visualize features if requested
    if args.visualize and features:
        logging.info("Generating feature visualizations")
        visualize_features(features, f"{args.output}/feature_plots")
        
    logging.info("Done!")

if __name__ == "__main__":
    main() 