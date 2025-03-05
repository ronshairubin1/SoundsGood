#!/usr/bin/env python3
"""
Create Unified Feature Dataset

This script creates a comprehensive dataset containing features extracted from all sound files 
in data/sounds/training_sounds/, along with their corresponding metadata.

The dataset is saved in a structured format that can be used for training various models.
"""

import os
import sys
import json
import numpy as np
import time
import logging
from pathlib import Path
from tqdm import tqdm

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the unified feature extractor
from backend.features.extractor import FeatureExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("create_dataset.log"),
        logging.StreamHandler()
    ]
)

def load_metadata():
    """
    Load metadata from centralized JSON files
    """
    metadata = {
        'sounds': {},
        'dataset': {}
    }
    
    # Load sounds.json
    sounds_path = 'data/sounds/sounds.json'
    if os.path.exists(sounds_path):
        try:
            with open(sounds_path, 'r') as f:
                sounds_data = json.load(f)
                if 'sounds' in sounds_data:
                    metadata['sounds'] = sounds_data['sounds']
                else:
                    metadata['sounds'] = sounds_data  # In case the data is not nested under 'sounds'
            logging.info(f"Loaded metadata for {len(metadata['sounds'])} sounds from {sounds_path}")
        except Exception as e:
            logging.error(f"Error loading sounds metadata: {e}")
    else:
        logging.warning(f"Sounds metadata file not found: {sounds_path}")
    
    # Load dataset_metadata.json if it exists
    dataset_path = 'data/dataset_metadata.json'
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, 'r') as f:
                metadata['dataset'] = json.load(f)
            logging.info(f"Loaded dataset metadata with {len(metadata['dataset'])} entries from {dataset_path}")
        except Exception as e:
            logging.error(f"Error loading dataset metadata: {e}")
    
    return metadata

def get_sound_id(file_path, metadata):
    """
    Get the sound ID from a file path using the metadata
    """
    # First try direct path matching
    rel_path = str(file_path)
    if rel_path.startswith('./'):
        rel_path = rel_path[2:]
    
    for sound_id, sound_data in metadata['sounds'].items():
        if 'path' in sound_data and sound_data['path'] == rel_path:
            return sound_id, sound_data
    
    # Try matching based on filename and class
    filename = os.path.basename(file_path)
    class_name = os.path.basename(os.path.dirname(file_path))
    
    for sound_id, sound_data in metadata['sounds'].items():
        if ('file_name' in sound_data and sound_data['file_name'] == filename and
            'class' in sound_data and sound_data['class'] == class_name):
            return sound_id, sound_data
    
    # Construct a default ID if no match is found
    default_id = f"{class_name}_{os.path.splitext(filename)[0]}"
    return default_id, None

def create_dataset(training_dir='data/sounds/training_sounds', 
                   output_dir='unified_dataset',
                   feature_types=None):
    """
    Create a unified dataset with features and metadata
    
    Args:
        training_dir: Directory containing class subdirectories with sound files
        output_dir: Directory to save the dataset
        feature_types: List of feature types to include, or None for all
    """
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    metadata = load_metadata()
    
    # Initialize the feature extractor
    extractor = FeatureExtractor(
        sample_rate=16000,
        n_mels=40,
        n_fft=512,
        hop_length=128,
        use_cache=True,
        cache_dir=os.path.join(output_dir, 'cache')
    )
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(training_dir) 
                 if os.path.isdir(os.path.join(training_dir, d))]
    
    logging.info(f"Found {len(class_dirs)} class directories: {class_dirs}")
    
    # Collect dataset information
    dataset = {
        'features': {},
        'metadata': {},
        'sound_ids': [],
        'class_mapping': {},  # Maps class names to indices
        'classes': [],
        'feature_types': feature_types or ['mel_spectrogram', 'mfccs', 'statistical', 'rhythm', 'spectral', 'tonal'],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create class mapping
    for i, class_name in enumerate(sorted(class_dirs)):
        dataset['class_mapping'][class_name] = i
        dataset['classes'].append(class_name)
    
    # Process each class
    total_processed = 0
    total_skipped = 0
    error_files = []
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        
        if not wav_files:
            logging.warning(f"No WAV files found in {class_path}")
            continue
        
        logging.info(f"Processing class {class_name}: {len(wav_files)} files")
        
        # Process each sound file
        for wav_file in tqdm(wav_files, desc=f"Processing {class_name}"):
            wav_path = os.path.join(class_path, wav_file)
            
            try:
                # Extract features
                features = extractor.extract_features(wav_path)
                
                if features is None:
                    logging.warning(f"No features extracted for {wav_path}")
                    total_skipped += 1
                    continue
                
                # Get sound ID and metadata
                sound_id, sound_meta = get_sound_id(wav_path, metadata)
                
                # Store features
                dataset['features'][sound_id] = {}
                for feature_type in dataset['feature_types']:
                    if feature_type in features:
                        dataset['features'][sound_id][feature_type] = features[feature_type]
                
                # Store metadata
                dataset['metadata'][sound_id] = {
                    'class': class_name,
                    'class_index': dataset['class_mapping'][class_name],
                    'file_path': wav_path
                }
                
                # Add any additional metadata from the central files
                if sound_meta:
                    for key, value in sound_meta.items():
                        if key not in ['class', 'file_path']:  # Don't overwrite these fields
                            dataset['metadata'][sound_id][key] = value
                
                # Add to sound IDs list
                dataset['sound_ids'].append(sound_id)
                
                total_processed += 1
                
            except Exception as e:
                logging.error(f"Error processing {wav_path}: {str(e)}")
                error_files.append(wav_path)
                total_skipped += 1
    
    # Save dataset metadata
    dataset_info = {
        'total_sounds': total_processed,
        'classes': dataset['classes'],
        'class_counts': {cls: dataset['sound_ids'].count(id) for cls, id in dataset['class_mapping'].items()},
        'feature_types': dataset['feature_types'],
        'created_at': dataset['created_at'],
        'processing_time': time.time() - start_time
    }
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save features by class for easier access
    os.makedirs(os.path.join(output_dir, 'features_by_class'), exist_ok=True)
    
    for class_name in class_dirs:
        class_sound_ids = [sid for sid in dataset['sound_ids'] 
                          if dataset['metadata'][sid]['class'] == class_name]
        
        if not class_sound_ids:
            continue
        
        # Save class-specific features
        class_data = {
            'sound_ids': class_sound_ids,
            'class_index': dataset['class_mapping'][class_name],
            'features': {sid: dataset['features'][sid] for sid in class_sound_ids},
            'metadata': {sid: dataset['metadata'][sid] for sid in class_sound_ids}
        }
        
        np.savez_compressed(
            os.path.join(output_dir, 'features_by_class', f"{class_name}_features.npz"),
            **class_data
        )
    
    # Save the complete dataset
    np.savez_compressed(
        os.path.join(output_dir, 'unified_dataset.npz'),
        metadata=dataset['metadata'],
        features=dataset['features'],
        sound_ids=dataset['sound_ids'],
        class_mapping=dataset['class_mapping'],
        classes=dataset['classes'],
        feature_types=dataset['feature_types'],
        created_at=dataset['created_at']
    )
    
    logging.info(f"Dataset creation completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Processed {total_processed} files, skipped {total_skipped} files")
    logging.info(f"Saved unified dataset to {os.path.join(output_dir, 'unified_dataset.npz')}")
    
    if error_files:
        logging.warning(f"Errors occurred while processing {len(error_files)} files")
        with open(os.path.join(output_dir, 'error_files.txt'), 'w') as f:
            for file_path in error_files:
                f.write(f"{file_path}\n")
    
    return dataset

def main():
    """
    Main function
    """
    import argparse
    parser = argparse.ArgumentParser(description='Create unified feature dataset')
    parser.add_argument('--training_dir', default='data/sounds/training_sounds',
                      help='Directory containing class subdirectories with sound files')
    parser.add_argument('--output_dir', default='backend/data/features/unified',
                      help='Directory to save the dataset')
    parser.add_argument('--feature_types', nargs='+',
                      help='List of feature types to include (default: all)')
    args = parser.parse_args()
    
    create_dataset(
        training_dir=args.training_dir,
        output_dir=args.output_dir,
        feature_types=args.feature_types
    )

if __name__ == "__main__":
    main() 