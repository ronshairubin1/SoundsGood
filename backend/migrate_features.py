#!/usr/bin/env python3
"""
Feature Migration Script

This script migrates feature data from legacy formats to the new unified format.
It can:
1. Convert features from the old AudioFeatureExtractor format
2. Convert features from the UnifiedFeatureExtractor format
3. Convert features from the ComprehensiveFeatureExtractor format

Usage:
    python backend/migrate_features.py --input_dir legacy_features/ --output_dir backend/data/features/unified
"""

import os
import sys
import numpy as np
import logging
import argparse
import glob
from pathlib import Path
import json
import time

# Add parent directory to path for imports
sys.path.append('.')

# Import the new feature extractor
from backend.features.extractor import FeatureExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_migration.log"),
        logging.StreamHandler()
    ]
)

def migrate_audio_feature_extractor(feature_file, output_dir):
    """
    Migrate features from the AudioFeatureExtractor format.
    
    Args:
        feature_file: Path to the feature file
        output_dir: Output directory
        
    Returns:
        Path to the migrated file
    """
    try:
        # Load the old format features
        data = np.load(feature_file, allow_pickle=True)
        
        if 'features' in data:
            old_features = data['features'].item()
        else:
            # Try loading as a flat dictionary
            old_features = {}
            for key in data.keys():
                old_features[key] = data[key]
        
        # Create the new format features
        new_features = {
            'metadata': {
                'migration_time': time.time(),
                'original_format': 'AudioFeatureExtractor',
                'original_file': str(feature_file)
            },
            'statistical': {}
        }
        
        # Map statistical features
        for key, value in old_features.items():
            if isinstance(value, (int, float)):
                new_features['statistical'][key] = float(value)
        
        # Save to the new location
        output_path = Path(output_dir) / f"{Path(feature_file).stem}_migrated.npz"
        np.savez_compressed(output_path, features=new_features)
        logging.info(f"Migrated {feature_file} to {output_path}")
        
        return output_path
    
    except Exception as e:
        logging.error(f"Error migrating {feature_file}: {e}")
        return None

def migrate_unified_extractor(feature_file, output_dir):
    """
    Migrate features from the UnifiedFeatureExtractor format.
    
    Args:
        feature_file: Path to the feature file
        output_dir: Output directory
        
    Returns:
        Path to the migrated file
    """
    try:
        # Load the old format features
        data = np.load(feature_file, allow_pickle=True)
        
        if 'features' in data:
            old_features = data['features'].item()
        else:
            # Different format, try to load keys
            old_features = {}
            for key in data.keys():
                old_features[key] = data[key]
        
        # Create the new format features
        new_features = {
            'metadata': {
                'migration_time': time.time(),
                'original_format': 'UnifiedFeatureExtractor',
                'original_file': str(feature_file)
            }
        }
        
        # Map CNN features if present
        if 'cnn_features' in old_features:
            new_features['mel_spectrogram'] = old_features['cnn_features']
        
        # Map MFCCs if present in first_mfcc_features
        if 'first_mfcc_features' in old_features:
            first_mfcc = old_features['first_mfcc_features']
            if 'mfccs' in first_mfcc:
                new_features['mfccs'] = first_mfcc['mfccs']
            if 'mfcc_delta' in first_mfcc:
                new_features['mfcc_delta'] = first_mfcc['mfcc_delta']
            if 'mfcc_delta2' in first_mfcc:
                new_features['mfcc_delta2'] = first_mfcc['mfcc_delta2']
        
        # Map statistical features
        new_features['statistical'] = {}
        if 'rf_features' in old_features:
            for key, value in old_features['rf_features'].items():
                if isinstance(value, (int, float)):
                    new_features['statistical'][key] = float(value)
        
        # Map any other features
        for key in ['rhythm', 'spectral', 'tonal']:
            if key in old_features:
                new_features[key] = old_features[key]
        
        # Save to the new location
        output_path = Path(output_dir) / f"{Path(feature_file).stem}_migrated.npz"
        np.savez_compressed(output_path, features=new_features)
        logging.info(f"Migrated {feature_file} to {output_path}")
        
        return output_path
    
    except Exception as e:
        logging.error(f"Error migrating {feature_file}: {e}")
        return None

def migrate_comprehensive_extractor(feature_file, output_dir):
    """
    Migrate features from the ComprehensiveFeatureExtractor format.
    
    Args:
        feature_file: Path to the feature file
        output_dir: Output directory
        
    Returns:
        Path to the migrated file
    """
    try:
        # Load the old format features
        data = np.load(feature_file, allow_pickle=True)
        
        if 'features' in data:
            old_features = data['features'].item()
        else:
            # Different format, try to load keys
            old_features = {}
            for key in data.keys():
                old_features[key] = data[key]
        
        # Create the new format features
        new_features = {
            'metadata': {
                'migration_time': time.time(),
                'original_format': 'ComprehensiveFeatureExtractor',
                'original_file': str(feature_file)
            }
        }
        
        # Map mel spectrogram
        if 'mel_spectrogram' in old_features:
            new_features['mel_spectrogram'] = old_features['mel_spectrogram']
        
        # Map MFCCs
        if 'mfccs' in old_features:
            new_features['mfccs'] = old_features['mfccs']
        
        if 'mfcc_delta' in old_features:
            new_features['mfcc_delta'] = old_features['mfcc_delta']
        
        if 'mfcc_delta2' in old_features:
            new_features['mfcc_delta2'] = old_features['mfcc_delta2']
        
        # Map statistical features
        new_features['statistical'] = {}
        for key_source in ['rf_features', 'statistical']:
            if key_source in old_features:
                for key, value in old_features[key_source].items():
                    if isinstance(value, (int, float)):
                        new_features['statistical'][key] = float(value)
        
        # Map advanced features
        if 'advanced_features' in old_features:
            adv = old_features['advanced_features']
            
            # Map rhythm features
            if 'rhythm' in adv:
                new_features['rhythm'] = adv['rhythm']
            
            # Map spectral features
            if 'spectral' in adv:
                new_features['spectral'] = adv['spectral']
            
            # Map tonal features
            if 'tonal' in adv:
                new_features['tonal'] = adv['tonal']
        
        # Save to the new location
        output_path = Path(output_dir) / f"{Path(feature_file).stem}_migrated.npz"
        np.savez_compressed(output_path, features=new_features)
        logging.info(f"Migrated {feature_file} to {output_path}")
        
        return output_path
    
    except Exception as e:
        logging.error(f"Error migrating {feature_file}: {e}")
        return None

def detect_format(feature_file):
    """
    Detect the format of a feature file
    
    Args:
        feature_file: Path to the feature file
        
    Returns:
        Format string: 'audio_feature_extractor', 'unified_extractor', 'comprehensive_extractor', or 'unknown'
    """
    try:
        data = np.load(feature_file, allow_pickle=True)
        
        # Get the keys
        if 'features' in data:
            features = data['features'].item()
            keys = features.keys()
            
            # Check for our specific unified format (it has cnn_features, rf_features and metadata)
            if 'cnn_features' in keys and 'rf_features' in keys and 'metadata' in keys:
                return 'unified_extractor'
                
            # Check for AudioFeatureExtractor format
            if any(key.startswith('mfcc_') and key.endswith('_mean') for key in keys):
                return 'audio_feature_extractor'
            
            # Check for ComprehensiveFeatureExtractor format
            if 'advanced_features' in keys:
                return 'comprehensive_extractor'
            
        else:
            keys = data.keys()
            
            # Some specific format checks that match our legacy implementations
            if 'mel_spectrogram' in keys and 'mfccs' in keys:
                return 'comprehensive_extractor'
                
            if 'features' in keys:
                # Try to look at the features structure
                features = data['features'].item()
                if 'cnn_features' in features and 'rf_features' in features:
                    return 'unified_extractor'
        
        logging.warning(f"Format detection: file {feature_file} has keys {keys}")
        return 'unknown'
    
    except Exception as e:
        logging.error(f"Error detecting format for {feature_file}: {e}")
        return 'unknown'

def migrate_directory(input_dir, output_dir):
    """
    Migrate all feature files in a directory
    
    Args:
        input_dir: Input directory containing feature files
        output_dir: Output directory for migrated files
        
    Returns:
        Number of successfully migrated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all npz files
    input_dir = Path(input_dir)
    feature_files = list(input_dir.glob("**/*.npz"))
    
    # Statistics
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    logging.info(f"Found {len(feature_files)} feature files")
    
    for file_path in feature_files:
        # Detect format
        format_type = detect_format(file_path)
        logging.info(f"Detected format for {file_path}: {format_type}")
        
        if format_type == 'audio_feature_extractor':
            result = migrate_audio_feature_extractor(file_path, output_dir)
        elif format_type == 'unified_extractor':
            result = migrate_unified_extractor(file_path, output_dir)
        elif format_type == 'comprehensive_extractor':
            result = migrate_comprehensive_extractor(file_path, output_dir)
        else:
            logging.warning(f"Unknown format for {file_path}, skipping")
            skipped_count += 1
            continue
        
        if result:
            success_count += 1
        else:
            error_count += 1
    
    logging.info(f"Migration complete. Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}")
    return success_count

def main():
    parser = argparse.ArgumentParser(description='Migrate features from legacy formats to the new unified format.')
    parser.add_argument('--input_dir', required=True, help='Directory containing legacy feature files')
    parser.add_argument('--output_dir', required=True, help='Directory for migrated feature files')
    
    args = parser.parse_args()
    
    migrate_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 