# Unified Model Architecture

## Overview

The unified model architecture standardizes feature extraction, preprocessing, and model training/inference workflows across the application. This ensures consistency between training and inference, improves performance through caching, and simplifies the codebase by reducing duplication.

## Core Components

### FeatureExtractor

The `FeatureExtractor` class in `backend.features.extractor` is the central component for feature extraction:

```python
from backend.features.extractor import FeatureExtractor

# Create an extractor
extractor = FeatureExtractor(sample_rate=16000)

# Extract all features
all_features = extractor.extract_features(audio_file)

# Get model-specific features
cnn_features = extractor.extract_features_for_model(all_features, model_type='cnn')
rf_features = extractor.extract_features_for_model(all_features, model_type='rf')
```

Key methods:
- `extract_features(audio_source, is_file=True)`: Extracts all features from audio
- `extract_features_for_model(features, model_type)`: Gets model-specific features
- `get_feature_names()`: Returns feature names for RF models

### SoundProcessor

The `SoundProcessor` class in `backend.audio.processor` handles audio preprocessing:

```python
from backend.audio.processor import SoundProcessor

# Create a processor
processor = SoundProcessor(sample_rate=16000)

# Process audio
processed_audio = processor.preprocess_file(audio_file)
```

Key methods:
- `preprocess_file(file_path)`: Preprocesses audio from a file
- `process_audio(audio)`: Processes audio data for model input
- `detect_sound_boundaries(audio)`: Detects start/end of sound in audio

### TrainingService

The `TrainingService` class in `src.services.training_service` handles model training:

```python
from src.services.training_service import TrainingService

# Create a service
service = TrainingService()

# Train a model
result = service.train_unified(
    model_type='cnn',
    audio_dir='path/to/audio',
    save=True,
    class_names=['class1', 'class2']
)
```

Key methods:
- `train_unified(model_type, audio_dir, ...)`: Trains a model with unified preprocessing
- `_train_cnn_model(data_dict, ...)`: Trains a CNN model
- `_train_rf_model(data_tuple, ...)`: Trains a Random Forest model

### Inference Components

The inference components include:

- `SoundDetector`: For CNN-based inference
- `SoundDetectorRF`: For RF-based inference
- `SoundDetectorEnsemble`: For ensemble inference

## Workflow

### Training Workflow

1. User selects dictionary and model type
2. `train_unified_model` route processes request
3. `TrainingService.train_unified()` is called
4. Audio files are preprocessed with `AudioPreprocessor`
5. Features are extracted with `FeatureExtractor`
6. Model is trained based on type (CNN, RF, Ensemble)
7. Model and metadata are saved

### Inference Workflow

1. User uploads audio or starts listening
2. Audio is preprocessed with `SoundProcessor`
3. Features are extracted with `FeatureExtractor`
4. Model makes prediction
5. Results are returned to user

## Benefits

- **Consistency**: Same preprocessing and feature extraction for training and inference
- **Performance**: Feature caching improves speed
- **Maintainability**: Reduced code duplication
- **Flexibility**: Support for multiple model types

## Known Issues and Next Steps

### Known Issues

1. **CNN Training Shape Mismatch**: The CNN training process encounters an error with feature shapes: "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions."

2. **RF Training Parameter Issue**: The RF training process has a parameter mismatch in the `_train_rf_model` method, which is missing the `audio_dir` parameter.

3. **Feature Extraction Inconsistency**: Some routes still use different approaches for feature extraction, which can lead to inconsistencies between training and inference.

### Next Steps

1. **Fix CNN Training**: Update the feature extraction process to ensure consistent feature shapes for CNN training.

2. **Fix RF Training**: Update the `_train_rf_model` method to handle the correct parameters or update the calling code.

3. **Complete Route Updates**: Ensure all routes use the unified approach consistently.

4. **Comprehensive Testing**: Create a test suite that validates the entire workflow from feature extraction to inference.

5. **Documentation**: Update API documentation to reflect the unified architecture.

## Future Improvements

- **Hyperparameter Optimization**: Add automated hyperparameter tuning
- **Transfer Learning**: Implement transfer learning for better performance
- **Model Versioning**: Improve model versioning and tracking
- **Feature Selection**: Add feature selection to improve model performance 