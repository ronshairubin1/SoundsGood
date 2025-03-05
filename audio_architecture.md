# SoundsEasy Audio Processing Architecture

## Architecture Overview

```
                        ┌─────────────────┐
                        │ Audio Capture   │
                        └────────┬────────┘
                                 ▼
                        ┌─────────────────┐
                        │  Preprocessor   │ ◄── Common Component
                        └────────┬────────┘
                                 │
           ┌────────────────────┴────────────────────┐
           ▼                                         ▼
┌─────────────────────┐                   ┌─────────────────────┐
│  Training Pipeline  │                   │ Inference Pipeline  │
└─────────┬───────────┘                   └─────────┬───────────┘
          │                                         │
          ▼                                         ▼
┌─────────────────────┐                   ┌─────────────────────┐
│   Audio Chunker     │                   │  Model Prediction   │
└─────────┬───────────┘                   └─────────┬───────────┘
          │                                         │
          ▼                                         ▼
┌─────────────────────┐                   ┌─────────────────────┐
│ User Verification & │                   │   User Feedback     │
│  File Organization  │                   └─────────┬───────────┘
└─────────┬───────────┘                             │
          │                                         ▼
          ▼                                ┌─────────────────────┐
┌─────────────────────┐                   │ Data Collection for  │
│  Data Augmentation  │                   │  Model Improvement   │
└─────────┬───────────┘                   └─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Feature Extraction  │
│  (Model-specific)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Model Training    │
└─────────────────────┘
```

## Implementation Structure

```
src/
├── core/
│   ├── audio/
│   │   ├── capture.py        # Audio capture (common)
│   │   ├── preprocessor.py   # Preprocessing (common)
│   │   └── chunker.py        # Audio chunking (training only)
│   │
│   └── utils/
│       └── config.py         # Configuration management
│
├── training/
│   ├── file_organizer.py     # User verification & file organization
│   ├── augmentor.py          # Data augmentation
│   ├── features/
│   │   ├── base_extractor.py # Base feature extraction
│   │   ├── cnn_features.py   # CNN-specific features (mel spectrograms)
│   │   └── rf_features.py    # RF-specific features (classical features)
│   │
│   └── trainers/
│       ├── base_trainer.py   # Base model trainer
│       ├── cnn_trainer.py    # CNN-specific training
│       └── rf_trainer.py     # RF-specific training
│
├── inference/
│   ├── base_detector.py      # Base detector with common audio handling
│   ├── models/
│   │   ├── model_interface.py # Interface for all model types
│   │   ├── cnn_model.py      # CNN model implementation
│   │   └── rf_model.py       # RF model implementation
│   │
│   ├── feedback.py           # User feedback mechanisms
│   └── data_collector.py     # Collection for model improvement
│
└── routes/
    ├── training_routes.py    # API routes for training
    └── inference_routes.py   # API routes for inference
```

## Key Components

### 1. Common Components

#### Audio Preprocessor (preprocessor.py)
```python
class AudioPreprocessor:
    """Handles all audio preprocessing common to both training and inference."""
    
    def __init__(self, sample_rate=44100, normalize=True):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.ambient_noise_level = None
        
    def measure_ambient_noise(self, duration=3):
        """Measure ambient noise from microphone."""
        # Implementation for measuring ambient noise
        # ... existing code ...
        return ambient_noise_level
        
    def measure_noise_from_buffer(self, audio_data):
        """Measure ambient noise from provided audio buffer."""
        # Calculate noise statistics
        self.ambient_noise_level = np.mean(np.abs(audio_data))
        # Adjust thresholds based on noise
        self.sound_threshold = self.ambient_noise_level * 3.0
        
    def preprocess(self, audio_data):
        """Preprocess audio data (normalize, filter, etc.)."""
        # Common preprocessing steps
        if self.normalize:
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
        # Apply noise reduction if we have ambient noise data
        if self.ambient_noise_level is not None:
            # Simple noise gate
            noise_gate = self.ambient_noise_level * 1.5
            audio_data[np.abs(audio_data) < noise_gate] = 0
            
        return audio_data
```

### 2. Training Components

#### Audio Chunker (chunker.py)
```python
class AudioChunker:
    """Identifies and extracts individual sound instances from longer recordings."""
    
    def __init__(self, preprocessor, min_duration=0.5, max_silence=0.3):
        self.preprocessor = preprocessor
        self.min_duration = min_duration
        self.max_silence = max_silence
        
    def chunk_audio(self, audio_data, sample_rate):
        """Split audio into individual chunks based on silence."""
        # Preprocess the audio
        processed_audio = self.preprocessor.preprocess(audio_data)
        
        # Use energy levels to identify silence
        energy = np.abs(processed_audio)
        threshold = self.preprocessor.sound_threshold
        
        # Find segments above threshold
        is_sound = energy > threshold
        
        # Convert to chunks
        chunks = []
        in_sound = False
        start_idx = 0
        
        for i in range(len(is_sound)):
            if is_sound[i] and not in_sound:
                # Start of sound
                start_idx = i
                in_sound = True
            elif not is_sound[i] and in_sound:
                # End of sound
                end_idx = i
                duration = (end_idx - start_idx) / sample_rate
                
                if duration >= self.min_duration:
                    # Add padding
                    pad = int(0.1 * sample_rate)
                    chunks.append((
                        max(0, start_idx - pad),
                        min(len(processed_audio), end_idx + pad)
                    ))
                    
                in_sound = False
                
        return [audio_data[start:end] for start, end in chunks]
```

#### Feature Extraction Base (base_extractor.py)
```python
class BaseFeatureExtractor:
    """Base class for feature extraction."""
    
    def __init__(self, preprocessor, sample_rate=44100):
        self.preprocessor = preprocessor
        self.sample_rate = sample_rate
        
    def extract_features(self, audio_data):
        """Extract features from audio data."""
        # Preprocess audio
        processed_audio = self.preprocessor.preprocess(audio_data)
        
        # Feature extraction happens in subclasses
        raise NotImplementedError
```

#### CNN Feature Extractor (cnn_features.py)
```python
class CNNFeatureExtractor(BaseFeatureExtractor):
    """Extract mel spectrogram features for CNN models."""
    
    def __init__(self, preprocessor, n_mels=128, n_fft=2048):
        super().__init__(preprocessor)
        self.n_mels = n_mels
        self.n_fft = n_fft
        
    def extract_features(self, audio_data):
        """Extract mel spectrogram features."""
        # Preprocess audio
        processed_audio = self.preprocessor.preprocess(audio_data)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=processed_audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
```

### 3. Inference Components

#### Base Detector (base_detector.py)
```python
class BaseDetector:
    """Base detector that handles audio capture and processing."""
    
    def __init__(self, preprocessor, model, callback=None):
        """
        Initialize the detector.
        
        Args:
            preprocessor: AudioPreprocessor instance
            model: Model instance for prediction
            callback: Callback function for prediction results
        """
        self.preprocessor = preprocessor
        self.model = model
        self.callback = callback
        
        # Audio capture state
        self.is_listening = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.buffer = []
        
        # Get parameters from preprocessor
        self.sample_rate = preprocessor.sample_rate
        self.sound_threshold = preprocessor.sound_threshold
        
    def start_listening(self, use_ambient_noise=False):
        """Start audio capture and listening."""
        if self.is_listening:
            return False
            
        # Measure ambient noise if requested
        if use_ambient_noise:
            self._measure_ambient_noise()
            
        # Start audio capture
        self.is_listening = True
        self.buffer = []
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )
        
        return True
```

#### Model Interface (model_interface.py)
```python
class ModelInterface:
    """Interface for all model types."""
    
    def __init__(self, model_path, class_names, feature_extractor):
        """
        Initialize the model interface.
        
        Args:
            model_path: Path to model file
            class_names: List of class names
            feature_extractor: Feature extractor for this model type
        """
        self.model_path = model_path
        self.class_names = class_names
        self.feature_extractor = feature_extractor
        self.model = self._load_model()
        
    def _load_model(self):
        """Load model from file."""
        # Implemented by subclasses
        raise NotImplementedError
        
    def predict(self, audio_data):
        """Make prediction on audio data."""
        # Extract features
        features = self.feature_extractor.extract_features(audio_data)
        
        # Model-specific prediction
        return self._model_predict(features)
        
    def _model_predict(self, features):
        """Model-specific prediction."""
        # Implemented by subclasses
        raise NotImplementedError
```

### 4. Route Integration

#### Inference Routes (inference_routes.py)
```python
@app.route('/api/ml/start_listening', methods=['POST'])
def start_listening():
    """Start real-time listening for audio prediction."""
    data = request.json
    model_id = data.get('model_id')
    use_ambient_noise = data.get('use_ambient_noise', False)
    
    # Get model type and path
    model_type, model_path = get_model_info(model_id)
    class_names = get_class_names(model_id)
    
    # Create preprocessor
    preprocessor = AudioPreprocessor()
    
    # Create feature extractor based on model type
    if model_type == 'cnn':
        feature_extractor = CNNFeatureExtractor(preprocessor)
    else:  # RF or other
        feature_extractor = RFFeatureExtractor(preprocessor)
    
    # Create model
    if model_type == 'cnn':
        model = CNNModel(model_path, class_names, feature_extractor)
    else:  # RF
        model = RFModel(model_path, class_names, feature_extractor)
    
    # Create detector with callback
    global sound_detector
    sound_detector = BaseDetector(
        preprocessor, 
        model,
        callback=prediction_callback
    )
    
    # Start listening
    success = sound_detector.start_listening(
        use_ambient_noise=use_ambient_noise
    )
    
    if success:
        return jsonify({
            'status': 'success',
            'message': f'Listening started with {model_type}',
            'sound_classes': class_names
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to start listening'
        }), 500
```

## Implementation Strategy

This architecture provides a clear separation of concerns while ensuring consistency between training and inference. Here's a phased approach to implementation:

### Phase 1: Minimal Refactoring (Current Status)
- ✅ Add ambient noise to all detector classes
- ✅ Ensure consistent API usage across detectors

### Phase 2: Extract Common Components
1. Create the common `AudioPreprocessor` class
2. Update existing detectors to use this preprocessor
3. Standardize feature extraction methods

### Phase 3: Restructure Training Pipeline
1. Implement the `AudioChunker` for training
2. Create model-specific feature extractors
3. Update training code to use these components

### Phase 4: Implement Base Detector
1. Create the `BaseDetector` class
2. Update existing detectors to inherit from it
3. Test for feature parity

### Phase 5: Complete Integration
1. Update routes to use the new architecture
2. Ensure backward compatibility with existing models
3. Add comprehensive tests 