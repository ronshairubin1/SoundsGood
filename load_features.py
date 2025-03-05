#!/usr/bin/env python3
import os
import numpy as np
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_dataset_arrays(dataset_dir='.'):
    """
    Load dataset arrays from the unified dataset file
    
    Args:
        dataset_dir: Directory containing the unified dataset file
        
    Returns:
        Dictionary with dataset arrays
    """
    unified_dataset_path = os.path.join(dataset_dir, "unified_dataset.npz")
    if os.path.exists(unified_dataset_path):
        logging.info(f"Loading unified dataset from {unified_dataset_path}")
        return np.load(unified_dataset_path, allow_pickle=True)
    
    # If unified dataset doesn't exist, check for individual feature files
    cnn_features_path = os.path.join(dataset_dir, "cnn_features.npz")
    rf_features_path = os.path.join(dataset_dir, "rf_features.npz")
    
    if os.path.exists(cnn_features_path) and os.path.exists(rf_features_path):
        logging.info(f"Loading individual feature files from {dataset_dir}")
        cnn_data = np.load(cnn_features_path, allow_pickle=True)
        rf_data = np.load(rf_features_path, allow_pickle=True)
        
        # Combine data
        return {
            'cnn_features': cnn_data['features'],
            'rf_features': rf_data['features'],
            'labels': cnn_data['labels'],  # Both should have the same labels
            'filenames': cnn_data['filenames'],  # Both should have the same filenames
            'class_names': cnn_data['class_names']  # Both should have the same class names
        }
    
    logging.error(f"No feature files found in {dataset_dir}")
    return None

def load_feature_metadata(dataset_dir='.'):
    """
    Load feature metadata from the feature_metadata.json file
    
    Args:
        dataset_dir: Directory containing the feature_metadata.json file
        
    Returns:
        Dictionary with feature metadata
    """
    metadata_path = os.path.join(dataset_dir, "feature_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    logging.error(f"No metadata file found in {dataset_dir}")
    return {}

def load_features_for_file(filename, dataset_dir='.'):
    """
    Load features for a specific file
    
    Args:
        filename: Name of the file to load features for
        dataset_dir: Directory containing the feature files
        
    Returns:
        Dictionary with features for the file
    """
    # Load dataset arrays
    dataset = load_dataset_arrays(dataset_dir)
    if dataset is None:
        return None
    
    # Find the index of the file
    if 'filenames' not in dataset:
        logging.error("Dataset does not contain filenames")
        return None
    
    filenames = dataset['filenames']
    try:
        idx = np.where(filenames == filename)[0][0]
    except IndexError:
        logging.error(f"File {filename} not found in dataset")
        return None
    
    # Extract features for this file
    features = {}
    
    # CNN features
    if 'cnn_features' in dataset:
        features['cnn'] = dataset['cnn_features'][idx]
    
    # RF features
    if 'rf_features' in dataset:
        features['rf'] = dataset['rf_features'][idx]
    
    # Advanced features
    for feature_type in ['rhythm', 'spectral', 'tonal']:
        feature_key = f'{feature_type}_features'
        if feature_key in dataset:
            features[feature_type] = dataset[feature_key][idx]
    
    # Add label
    if 'labels' in dataset:
        features['label'] = dataset['labels'][idx]
    
    return features

def get_feature_vectors(features, feature_types):
    """
    Get feature vectors for specified feature types
    
    Args:
        features: Dictionary with features for a file
        feature_types: List of feature types to include
        
    Returns:
        Concatenated feature vector
    """
    if features is None:
        return None
    
    feature_vectors = []
    
    for feature_type in feature_types:
        if feature_type in features:
            # For CNN features, flatten the 2D array
            if feature_type == 'cnn' and len(features[feature_type].shape) > 1:
                feature_vectors.append(features[feature_type].flatten())
            else:
                feature_vectors.append(features[feature_type])
        else:
            logging.warning(f"Feature type {feature_type} not found in features")
    
    if not feature_vectors:
        return None
    
    # Concatenate feature vectors
    return np.concatenate(feature_vectors)

def get_dataset_arrays_by_type(feature_types, dataset_dir='.'):
    """
    Get dataset arrays for specified feature types
    
    Args:
        feature_types: List of feature types to include
        dataset_dir: Directory containing the feature files
        
    Returns:
        X, y arrays for model training
    """
    # Load dataset arrays
    dataset = load_dataset_arrays(dataset_dir)
    if dataset is None:
        return None, None
    
    # Get filenames and labels
    if 'filenames' not in dataset or 'labels' not in dataset:
        logging.error("Dataset missing required fields (filenames or labels)")
        return None, None
    
    filenames = dataset['filenames']
    labels = dataset['labels']
    
    # Prepare feature vectors
    X = []
    y = []
    
    for i, filename in enumerate(filenames):
        try:
            # Load features for this file
            features = load_features_for_file(filename, dataset_dir)
            if features is None:
                logging.warning(f"Could not load features for {filename}")
                continue
            
            # Get feature vector with selected types
            feature_vector = get_feature_vectors(features, feature_types)
            if feature_vector is None or len(feature_vector) == 0:
                logging.warning(f"No features extracted for {filename} with types {feature_types}")
                continue
            
            X.append(feature_vector)
            y.append(labels[i])
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
    
    if len(X) == 0:
        logging.error("No valid feature vectors extracted")
        return None, None
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and display feature information")
    parser.add_argument('--feature-dir', default='.', help='Directory containing feature files')
    args = parser.parse_args()
    
    # Load metadata
    metadata = load_feature_metadata(args.feature_dir)
    print("\nFeature Metadata:")
    for key, value in metadata.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    
    # Load dataset
    dataset = load_dataset_arrays(args.feature_dir)
    if dataset is not None:
        print("\nDataset Contents:")
        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Print class distribution
        if 'labels' in dataset and 'class_names' in dataset:
            labels = dataset['labels']
            class_names = dataset['class_names']
            
            print("\nClass Distribution:")
            for i, class_name in enumerate(class_names):
                count = np.sum(labels == i)
                print(f"  {class_name}: {count} samples")
    
    # Example of loading features for different models
    print("\nFeature Vector Sizes for Different Combinations:")
    feature_combinations = [
        ['rf'],
        ['rf', 'rhythm'],
        ['rf', 'spectral'],
        ['rf', 'tonal'],
        ['rf', 'rhythm', 'spectral', 'tonal']
    ]
    
    for combo in feature_combinations:
        X, y = get_dataset_arrays_by_type(combo, args.feature_dir)
        if X is not None:
            print(f"  {'+'.join(combo)}: {X.shape[1]} features") 