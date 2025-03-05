# SoundsEasy Backend

This folder contains the unified backend implementation for the SoundsEasy application, focused on providing a consistent approach to audio processing, feature extraction, and model training.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Audio Processing](#audio-processing)
3. [Feature Extraction](#feature-extraction)
4. [Data Management](#data-management)
5. [Training](#training)
6. [Inference](#inference)
7. [Migration Tools](#migration-tools)

## Architecture Overview

The backend is organized into the following components:

```
backend/
├── data/                   # Data storage and processing
│   ├── features/           # Feature extraction results
│   │   ├── cache/          # Feature cache for faster reuse
│   │   ├── unified/        # Unified feature datasets
│   │   ├── enhanced/       # Enhanced feature sets
│   │   ├── plots/          # Feature visualization plots
│   │   ├── results/        # Analysis results
│   │   └── model_specific/ # Model-specific feature formatting
│   ├── sounds/             # Sound data storage
│   ├── models/             # Model storage
│   └── dictionaries/       # Dictionaries for preprocessing
├── features/               # Feature extraction code
│   ├── extractor.py        # Unified feature extractor
│   ├── rf_features.py      # Random Forest specific features
│   ├── cnn_features.py     # CNN specific features
│   └── README.md           # Documentation for feature extraction
├── audio/                  # Audio processing utilities
├── inference/              # Inference-related components
├── training/               # Training-related components
├── legacy/                 # Legacy code (for backward compatibility)
│   └── feature_extractors/ # Old feature extraction implementations
├── process_pipeline.py     # Processing pipeline utilities
├── migrate_features.py     # Tools for migrating legacy features
├── move_legacy_extractors.py # Scripts for code organization
└── test_feature_unification.py # Test scripts for feature unification
```

## Audio Processing

Audio processing is handled by three main components:

1. **AudioChopper**: Divides continuous audio into individual sound segments
2. **AudioPreprocessor**: Performs preprocessing on audio files (normalization, centering, etc.)
3. **AudioAugmentor**: Creates augmented versions of audio files for training

## Feature Extraction

The codebase has been refactored to use a single, unified feature extraction approach. This ensures:

1. **Consistency**: The same features are used across training and inference
2. **Performance**: Features are cached and reused when possible
3. **Maintainability**: A single source of truth for feature extraction
4. **Flexibility**: Support for different feature types for different models

### Unified Feature Extractor

The central `FeatureExtractor` class provides a unified interface for extracting features from audio data:

```python
from backend.features.extractor import FeatureExtractor

# Initialize the extractor
extractor = FeatureExtractor(
    sample_rate=16000,
    n_mfcc=13,
    n_mels=40,
    n_fft=512,
    hop_length=128,
    cache_dir="backend/data/features/cache",
    use_cache=True
)

# Extract features from a file
features = extractor.extract_features("path/to/audio.wav")

# Extract features from audio data
features = extractor.extract_features(audio_data, is_file=False)

# Get model-specific features
cnn_features = extractor.extract_features_for_model(features, model_type='cnn')
rf_features = extractor.extract_features_for_model(features, model_type='rf')
```

### Migration from Legacy Code

Legacy feature extractors have been moved to `backend/legacy/feature_extractors/`. Import forwarders have been set up in the original locations to maintain backward compatibility.

For new code, always use the unified `FeatureExtractor` from `backend.features.extractor`.

### Feature Directory Structure

All feature-related files are organized in a standardized directory structure:

- `backend/data/features/cache`: Cached features for performance optimization
- `backend/data/features/unified`: Unified datasets for training
- `backend/data/features/enhanced`: Enhanced feature sets
- `backend/data/features/plots`: Feature visualization plots
- `backend/data/features/results`: Analysis results
- `backend/data/features/model_specific`: Model-specific feature formatting

### Testing

The unified feature extractor is thoroughly tested in:
- `tests/feature_extraction/test_unified_extractor.py`
- `tests/feature_extraction/test_feature_unification.py`
- `tests/webapp/test_webapp_feature_extractor.py`

## Data Management

The `DatasetManager` class handles:

- Organizing sound files by class
- Tracking file status (raw, preprocessed, augmented)
- Storing and retrieving features
- Generating datasets for training

## Training

Training is organized into model-specific trainers that all share common infrastructure.

## Inference

The inference component uses the same unified feature extraction as training, ensuring consistency.

## Migration Tools

To facilitate the transition to this new architecture, migration tools are provided:

- `migrate_features.py`: Converts legacy feature files to the new unified format
- `move_legacy_extractors.py`: Moves legacy feature extractors to a legacy folder

## Consistency Between Training and Inference

A key advantage of this architecture is the consistency between training and inference:

1. Both use the SAME chopping logic
2. Both use the SAME preprocessing logic
3. Both use the SAME feature extraction logic
4. Both use the SAME feature preparation logic

This ensures that models perform as expected in production. 