import os
import json
import numpy as np

def load_feature_metadata():
    """Load the feature metadata"""
    with open('feature_metadata.json', 'r') as f:
        return json.load(f)

def load_features_for_file(filename):
    """Load features for a specific file"""
    # Extract just the filename without the path
    basename = os.path.basename(filename)
    
    # Check if the file exists with just the basename
    feature_file = f"{basename}.npz"
    if os.path.exists(feature_file):
        data = np.load(feature_file, allow_pickle=True)
        return data['features'].item()  # Convert from 0d array to dict
    
    # If not found, try with the full path structure
    if '/' in filename:
        # For filenames like "ah/ah_admin_1.wav", try "ah_admin_1.wav.npz"
        sound_file = os.path.basename(filename)
        feature_file = f"{sound_file}.npz"
        if os.path.exists(feature_file):
            data = np.load(feature_file, allow_pickle=True)
            return data['features'].item()
    
    return None

def get_features_for_model(features, model_type):
    """
    Extract model-specific features from the unified feature set
    
    Args:
        features: The unified feature set
        model_type: 'cnn', 'rf', or 'ensemble'
        
    Returns:
        The appropriate features for the model
    """
    if model_type.lower() == 'cnn':
        return features.get('cnn_features')
    elif model_type.lower() == 'rf':
        # For RF, we might want to exclude the first MFCC coefficient
        rf_features = features.get('rf_features', {})
        # Filter out first MFCC coefficient if needed
        # rf_features = {k: v for k, v in rf_features.items() if 'mfcc_0' not in k}
        return rf_features
    elif model_type.lower() == 'ensemble':
        return {
            'cnn_features': features.get('cnn_features'),
            'rf_features': features.get('rf_features')
        }
    else:
        return None

def load_all_features():
    """
    Load all features and organize by filename
    
    Returns:
        Dictionary mapping filenames to feature sets
    """
    metadata = load_feature_metadata()
    result = {}
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.npz') and file != 'unified_dataset.npz':
                try:
                    data = np.load(file, allow_pickle=True)
                    features = data['features'].item()
                    result[os.path.splitext(file)[0]] = features
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    return result

def load_dataset_arrays():
    """
    Load and organize dataset into feature arrays by type
    
    Returns:
        Dictionary with CNN features, RF features, and labels
    """
    # Load unified dataset file if it exists
    if os.path.exists('unified_dataset.npz'):
        return np.load('unified_dataset.npz')
    
    # Otherwise, load from individual files
    features = load_all_features()
    metadata = load_feature_metadata()
    
    # Prepare arrays
    X_cnn = []
    X_rf = []
    labels = []
    filenames = []
    
    # Process files
    for filename, feature_set in features.items():
        # Get label from filename
        label = None
        for filepath in metadata.get('filenames', []):
            if os.path.basename(filepath) == filename:
                class_name = filepath.split(os.sep)[0]
                label = metadata['class_to_idx'].get(class_name)
                break
        
        if label is not None:
            if 'cnn_features' in feature_set:
                X_cnn.append(feature_set['cnn_features'])
            if 'rf_features' in feature_set:
                # Convert dict to vector using consistent feature order
                rf_names = sorted(metadata['rf_feature_names'])
                rf_vector = [feature_set['rf_features'].get(name, 0) for name in rf_names]
                X_rf.append(rf_vector)
            
            labels.append(label)
            filenames.append(filename)
    
    # Convert to numpy arrays
    dataset = {
        'X_cnn': np.array(X_cnn) if X_cnn else None,
        'X_rf': np.array(X_rf) if X_rf else None,
        'labels': np.array(labels),
        'filenames': np.array(filenames),
        'class_names': np.array(metadata['class_names'])
    }
    
    return dataset


def get_advanced_features(features, feature_type=None):
    """
    Extract advanced features from the unified feature set
    
    Args:
        features: The unified feature set
        feature_type: Optional specific type ('rhythm', 'spectral', 'tonal', or None for all)
        
    Returns:
        The requested advanced features
    """
    if 'advanced_features' not in features:
        return None
        
    if feature_type is None:
        return features['advanced_features']
    elif feature_type in features['advanced_features']:
        return features['advanced_features'][feature_type]
    else:
        return None
        
def get_feature_vectors(features, feature_types):
    """
    Extract and concatenate feature vectors based on specified feature types
    
    Args:
        features: Dictionary containing all features for a file
        feature_types: List of feature types to include (e.g., ['rf', 'rhythm'])
        
    Returns:
        Concatenated feature vector with selected feature types
    """
    metadata = load_feature_metadata()
    feature_vector = []
    
    # Add RF features if requested
    if 'rf' in feature_types and 'rf_features' in features:
        rf_features = features['rf_features']
        # Ensure consistent feature order using metadata
        if 'rf_feature_names' in metadata:
            rf_names = metadata['rf_feature_names']
            rf_values = [rf_features.get(name, 0) for name in rf_names]
            feature_vector.extend(rf_values)
    
    # Add CNN features if requested (flatten if needed)
    if 'cnn' in feature_types and 'cnn_features' in features:
        cnn_features = features['cnn_features']
        if isinstance(cnn_features, np.ndarray):
            feature_vector.extend(cnn_features.flatten())
    
    # Add rhythm features if requested
    if 'rhythm' in feature_types and 'rhythm_features' in features:
        rhythm_features = features['rhythm_features']
        if isinstance(rhythm_features, dict):
            if 'rhythm_features' in metadata:
                rhythm_names = metadata['rhythm_features']
                rhythm_values = [rhythm_features.get(name, 0) for name in rhythm_names]
                feature_vector.extend(rhythm_values)
    
    # Add spectral features if requested
    if 'spectral' in feature_types and 'spectral_features' in features:
        spectral_features = features['spectral_features']
        if isinstance(spectral_features, dict):
            if 'spectral_features' in metadata:
                spectral_names = metadata['spectral_features']
                spectral_values = [spectral_features.get(name, 0) for name in spectral_names]
                feature_vector.extend(spectral_values)
    
    # Add tonal features if requested
    if 'tonal' in feature_types and 'tonal_features' in features:
        tonal_features = features['tonal_features']
        if isinstance(tonal_features, dict):
            if 'tonal_features' in metadata:
                tonal_names = metadata['tonal_features']
                tonal_values = [tonal_features.get(name, 0) for name in tonal_names]
                feature_vector.extend(tonal_values)
    
    return np.array(feature_vector)
