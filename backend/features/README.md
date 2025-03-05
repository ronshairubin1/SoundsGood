# Feature Extraction

The feature extraction module provides a unified approach to extracting features from audio data. It's designed to extract ALL possible features in one pass, which can then be selectively used by different models.

## Table of Contents

1. [Unified Feature Extractor](#unified-feature-extractor)
2. [Feature Types](#feature-types)
3. [Usage Examples](#usage-examples)
4. [Model-Specific Feature Preparation](#model-specific-feature-preparation)
5. [Technical Details](#technical-details)

## Unified Feature Extractor

The `FeatureExtractor` class in `extractor.py` is the single source of truth for all feature extraction in the system. It extracts a comprehensive set of features from audio data, which can then be selectively used for different model types.

### Key Features

- **Extract Once, Use Many Times**: Features are extracted once and stored, rather than re-extracted for each model.
- **Caching**: Features can be cached to avoid recomputation.
- **Comprehensive Feature Set**: Extracts all possible features (spectral, temporal, tonal, etc.).
- **Model-Specific Selection**: Easy selection of features for specific model types.

### Basic Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| sample_rate | Audio sample rate | 8000 Hz |
| n_mfcc | Number of MFCC coefficients | 13 |
| n_mels | Number of mel bands | 64 |
| n_fft | FFT window size | 1024 |
| hop_length | Hop length for feature extraction | 256 |
| cache_dir | Directory for feature caching | "backend/data/features/cache" |
| use_cache | Whether to use caching | True |

## Feature Types

The `FeatureExtractor` extracts three main types of features:

### 1. Spectral Features (for CNN Models)

- **Mel Spectrogram**: Time-frequency representation with mel scaling
- Time dimension: Audio duration in frames
- Frequency dimension: 64 mel bands by default
- Shape: (frames, mel_bands)

### 2. Statistical Features (for RF Models)

- **MFCC Features**:
  - 13 MFCC coefficients (mean and std) - 26 features
  - MFCC delta features (mean and std) - 26 features
  - MFCC delta2 features (mean and std) - 26 features

- **Spectral Features**:
  - Spectral centroid (mean and std)
  - Spectral rolloff (mean and std)
  - Spectral flatness (mean and std)
  - Spectral contrast (mean and std)
  - Spectral bandwidth (mean and std)

- **Temporal Features**:
  - Zero-crossing rate (mean and std)
  - RMS energy (mean and std)

- **Pitch Features**:
  - Pitch (mean and std)
  - Formant (mean and std)

### 3. Advanced Features

- **Rhythm Features**:
  - Tempo
  - Beat strength (onset strength)
  - Pulse clarity
  - Beat count

- **Tonal Features**:
  - Chroma (12 pitch classes)
  - Harmonic/percussive energy ratio
  - Tonal centroid (tonnetz)

## Usage Examples

### Basic Usage

```python
from backend.features.extractor import FeatureExtractor

# Initialize the extractor
extractor = FeatureExtractor()

# Extract all features from a file
features = extractor.extract_features("sound.wav")

# Extract features from audio data
import librosa
audio, sr = librosa.load("sound.wav", sr=8000)
features = extractor.extract_features(audio, is_file=False)
```

### Using Model-Specific Features

```python
# Extract all features first
all_features = extractor.extract_features("sound.wav")

# Get CNN-specific features
cnn_features = extractor.extract_features_for_model(all_features, model_type='cnn')

# Get RF-specific features
rf_features = extractor.extract_features_for_model(all_features, model_type='rf')

# Get both CNN and RF features for ensemble models
ensemble_features = extractor.extract_features_for_model(all_features, model_type='ensemble')
```

### Batch Processing for Training

```python
# Batch extract features for all classes
X, y, class_names, stats = extractor.batch_extract_features(
    "training_data/",
    ["class1", "class2", "class3"],
    model_type='cnn'
)

# Print statistics about the extraction
print(f"Processed {stats['total_processed']} files, skipped {stats['total_skipped']} files")
print(f"Class counts: {stats['processed_counts']}")
```

### Processing a Directory

```python
# Extract features from all files in a directory
features_dict = extractor.extract_features_from_directory(
    "sounds/",
    output_dir="features/"
)
```

## Model-Specific Feature Preparation

After extracting features, model-specific preparation is handled by dedicated classes:

### CNN Feature Preparation

```python
from backend.features.cnn_features import CNNFeaturePreparation

# Initialize the preparer
cnn_preparer = CNNFeaturePreparation(target_shape=(223, 64, 1))

# Prepare CNN features
cnn_ready_features = cnn_preparer.prepare_features(all_features)
```

### RF Feature Preparation

```python
from backend.features.rf_features import RFFeaturePreparation

# Initialize the preparer
rf_preparer = RFFeaturePreparation(exclude_first_mfcc=True)

# Prepare RF features
rf_ready_features = rf_preparer.prepare_features(all_features)
```

## Technical Details

### Feature Extraction Process

1. **Load Audio**: Load audio from file or use provided audio data
2. **Extract Mel Spectrogram**: Create mel spectrogram with specified parameters
3. **Extract MFCCs**: Extract MFCC coefficients and their delta features
4. **Extract Statistical Features**: Calculate statistical features (mean, std) for various audio characteristics
5. **Extract Advanced Features**: Calculate rhythm, spectral, and tonal features
6. **Cache Results**: Save features to cache for future use

### Caching

Features are cached to avoid recomputation:

- **Memory Cache**: Features are stored in memory for quick access during the same session
- **Disk Cache**: Features are also stored on disk for persistence between sessions
- **Hash-Based**: Files are hashed based on their path to create unique cache keys

### Feature Format

Features are returned as a nested dictionary structure:

```
{
    'metadata': {
        'sample_rate': 8000,
        'length_samples': 8000,
        'extraction_time': 1630000000.0
    },
    'mel_spectrogram': numpy.ndarray,
    'mfccs': numpy.ndarray,
    'mfcc_delta': numpy.ndarray,
    'mfcc_delta2': numpy.ndarray,
    'statistical': {
        'mfcc_0_mean': 0.123,
        'mfcc_0_std': 0.456,
        ...
    },
    'rhythm': {
        'tempo': 120.0,
        'pulse_clarity': 0.789,
        ...
    },
    'spectral': {
        'flatness_mean': 0.234,
        ...
    },
    'tonal': {
        'chroma_mean': 0.345,
        ...
    }
}
``` 