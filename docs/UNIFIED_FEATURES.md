# Unified Feature Extraction System

This document explains the new unified feature extraction system for sound classification.

## Overview

The unified feature extraction system is designed to:

1. Extract ALL possible features from audio files (model-independent)
2. Store features on a per-file basis in a structured format
3. Preserve raw features including those typically excluded (like 1st MFCC coefficient)
4. Allow flexible selection and filtering when using features for model training
5. Support easy addition of new feature types

## Key Components

The system consists of these main components:

### 1. Comprehensive Feature Extractor (`unified_feature_extractor.py`)

This is the core component that:
- Extracts both CNN and RF features from audio files
- Preserves all extracted features, including those normally excluded
- Creates individual feature files for each audio sample
- Generates metadata about the feature extraction process

### 2. Advanced Feature Extractor (`advanced_feature_extractor.py`)

This component enhances the basic features with:
- Rhythm features (tempo, beat strength, pulse clarity)
- Advanced spectral features (flatness, contrast, bandwidth)
- Tonal features (chroma, harmonic/percussive separation)

### 3. Helper Utilities (`load_features.py`)

Generated automatically to help load and manipulate features:
- Functions to load individual file features
- Functions to load dataset arrays for training
- Utilities to filter and select specific feature types

## Directory Structure

```
data/
├── sounds/
│   └── training_sounds/ (original WAV files)
├── dataset_metadata.json (dataset tracking)
├── unified_features/
│   ├── feature_metadata.json (feature information)
│   ├── load_features.py (utilities)
│   ├── unified_dataset.npz (optional consolidated file)
│   └── *.npz (individual feature files, one per audio file)
└── enhanced_features/
    ├── feature_metadata.json (enhanced feature information)
    ├── load_features.py (enhanced utilities)
    └── *.npz (individual feature files with advanced features)
```

## How Features Are Stored

### 1. Individual File Storage

Each audio file has a corresponding NPZ file containing all its features:

```
filename.npz
└── features (dictionary with):
    ├── cnn_features: Mel spectrogram (shape depends on audio length)
    ├── rf_features: Dictionary of statistical features 
    ├── first_mfcc_features: The first MFCC coefficient features (normally excluded)
    ├── advanced_features: 
    │   ├── rhythm: Tempo and rhythm features
    │   ├── spectral: Advanced spectral features
    │   └── tonal: Tonal and harmonic features
    └── metadata: Information about the file and processing
```

### 2. Feature Types

The system extracts and organizes these feature categories:

**CNN Features**:
- Mel spectrograms (time-frequency representations)
- Standardized to shape (223, 64, 1)

**RF Features**:
- MFCCs (including first coefficient, separately stored)
- Spectral features (centroid, bandwidth, contrast, rolloff)
- Zero crossing rate
- RMS energy
- Pitch statistics

**Advanced Features**:
- **Rhythm**: Tempo, onset strength, pulse clarity
- **Spectral**: Flatness, contrast, bandwidth
- **Tonal**: Chroma, tonnetz, harmonic/percussive ratio

## Usage Examples

### 1. Extracting Unified Features

```bash
# Extract basic unified features
python unified_feature_extractor.py

# Customize extraction
python unified_feature_extractor.py --data-dir data --output-dir my_features
```

### 2. Adding Advanced Features

```bash
# Add advanced features to unified features
python advanced_feature_extractor.py --unified-dir unified_features --output-dir enhanced_features
```

### 3. Loading and Using Features in Python

```python
import os
import numpy as np
from enhanced_features.load_features import load_features_for_file, get_feature_vectors

# Load features for a specific file
features = load_features_for_file("dog_bark_01.wav")

# Get features for CNN model
cnn_features = features.get('cnn_features')

# Get features for RF model (excluding first MFCC)
rf_features = features.get('rf_features')

# Get advanced features
rhythm_features = features.get('advanced_features', {}).get('rhythm', {})
spectral_features = features.get('advanced_features', {}).get('spectral', {})

# Create feature vectors for model training with selected feature types
# Combine RF features with rhythm and spectral features
feature_vector = get_feature_vectors(features, ['rf', 'rhythm', 'spectral'])
```

### 4. Training Models with Different Feature Combinations

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from enhanced_features.load_features import load_dataset_arrays, get_feature_vectors

# Load all features
dataset = load_dataset_arrays()
labels = dataset['labels']
filenames = dataset['filenames']

# Generate feature vectors from different combinations
X_basic = []  # Basic RF features only
X_rhythm = []  # RF + rhythm features
X_all = []  # All features combined

for i, filename in enumerate(filenames):
    features = load_features_for_file(filename)
    
    # Create different feature combinations
    X_basic.append(get_feature_vectors(features, ['rf']))
    X_rhythm.append(get_feature_vectors(features, ['rf', 'rhythm']))
    X_all.append(get_feature_vectors(features, ['rf', 'rhythm', 'spectral', 'tonal']))

# Convert to numpy arrays
X_basic = np.array(X_basic)
X_rhythm = np.array(X_rhythm)
X_all = np.array(X_all)

# Train models with different feature sets
model_basic = RandomForestClassifier().fit(X_basic, labels)
model_rhythm = RandomForestClassifier().fit(X_rhythm, labels)
model_all = RandomForestClassifier().fit(X_all, labels)

# Compare performance
# ...
```

## Adding New Feature Types

To add new feature types to the system:

1. Create a new extractor method in `AdvancedFeatureExtractor`:
```python
def extract_new_features(self, y):
    features = {}
    # Extract features here
    features['new_feature1'] = value1
    features['new_feature2'] = value2
    return features
```

2. Add the extraction to the `enhance_feature_set` method:
```python
new_features = self.extract_new_features(y)
enhanced_features['advanced_features']['new_category'] = new_features
```

3. Update the metadata and helper utilities to include the new features.

## Benefits of This Approach

This unified system provides several key benefits:

1. **Extract once, use many ways**: All possible features are extracted just once
2. **Model flexibility**: Features can be mixed and matched for different models
3. **Future-proofing**: Adding new feature types doesn't require re-extracting base features
4. **Flexibility in feature selection**: Features like the 1st MFCC coefficient are preserved but can be excluded when needed
5. **Experiment-friendly**: Easy to try different feature combinations without re-processing audio

## Technical Notes

- All feature values are stored as Python native types (float, list) for JSON compatibility
- Numpy arrays are converted to serializable formats during storage
- File identification is maintained through consistent naming
- Feature extractors handle errors gracefully to prevent pipeline failures