# SoundClassifier v09

## Project Overview
This is a sophisticated sound classification system designed to recognize and classify various vocal sounds. The system uses a combination of machine learning models, including CNN and Random Forest classifiers, to achieve high accuracy in sound recognition. The application provides a web interface for training models, managing sound dictionaries, and performing real-time inference.

## Project Structure

### Core Components
```
src/
├── core/
│   ├── models/         # Core model implementations
│   │   ├── cnn.py     # CNN model architecture
│   │   ├── rf.py      # Random Forest model
│   │   ├── ensemble.py # Ensemble model combining CNN and RF
│   │   └── base.py    # Base model class and shared functionality
│   └── audio/
│       └── processor.py # Audio processing utilities
├── ml/                 # Machine learning pipeline
├── api/                # API endpoints
├── routes/             # Web routes
├── services/           # Business logic
└── templates/          # HTML templates
```

### Key Directories and Files

#### Machine Learning (`src/ml/`)
- `cnn_classifier.py`: CNN model implementation for sound classification
- `rf_classifier.py`: Random Forest classifier implementation
- `ensemble_classifier.py`: Combines multiple models for improved accuracy
- `audio_processing.py`: Audio preprocessing and feature extraction
- `feature_extractor.py`: MFCC and other audio feature extraction
- `trainer.py`: Model training orchestration
- `inference.py`: Real-time inference engine
- `data_augmentation.py`: Audio data augmentation techniques
- `model_paths.py`: Model file path management
- `constants.py`: ML-related constants and configurations

#### API Layer (`src/api/`)
- `ml_api.py`: Machine learning operations API
- `user_api.py`: User management API
- `dashboard_api.py`: Dashboard data API
- `dictionary_api.py`: Sound dictionary management API

#### Services (`src/services/`)
- `training_service.py`: Orchestrates model training workflow
- `inference_service.py`: Manages real-time inference
- `dictionary_service.py`: Sound dictionary CRUD operations
- `user_service.py`: User authentication and management

#### Web Routes (`src/routes/`)
- `ml_routes.py`: Machine learning interface routes
- `train_app.py`: Training interface routes

#### Templates (`src/templates/`)
Organized HTML templates for various functionalities:
- Model Management: `model_summary.html`, `model_status.html`, `train_model.html`
- Sound Management: `sounds_management.html`, `upload_sounds.html`, `record.html`
- Dictionary Management: `dictionary_manager.html`, `manage_dictionaries.html`
- Analytics: `view_analysis.html`, `inference_statistics.html`
- User Interface: `dashboard.html`, `login.html`, `register.html`

## Data Flow and Architecture

### Authentication Flow
1. Users register/login through `user_api.py`
2. `user_service.py` handles authentication
3. Session management ensures secure access to features

### Training Flow
1. Users upload sounds through the web interface
2. `dictionary_service.py` manages sound organization
3. `training_service.py` orchestrates the training process:
   - Audio preprocessing
   - Feature extraction
   - Model training
   - Performance evaluation
4. Results are stored and displayed through the dashboard

### Inference Flow
1. Real-time audio capture
2. `audio_processing.py` handles preprocessing
3. `inference_service.py` manages model predictions
4. Results are displayed through the web interface

## Key Features

### Model Management
- Multiple model architectures (CNN, RF, Ensemble)
- Model versioning and comparison
- Performance metrics tracking
- Model export and import capabilities

### Sound Management
- Sound recording and upload
- Automatic sound validation
- Sound dictionary organization
- Data augmentation options

### Training Management
- Configurable training parameters
- Real-time training progress monitoring
- Cross-validation support
- Performance visualization

### Inference
- Real-time sound classification
- Confidence score display
- Performance statistics
- Batch processing capability

## Configuration

### Environment Setup
- Python 3.10+ required
- Virtual environment recommended
- Dependencies in requirements.txt
- Environment variables in .env

### Key Configuration Files
- `config.py`: Application configuration
- `model_paths.py`: Model storage configuration
- `constants.py`: ML pipeline parameters

For detailed information about the configuration system, see [CONFIGURATION.md](CONFIGURATION.md).

## Development Guidelines

### Code Organization
- Follow the established module structure
- Keep ML logic in `src/ml/`
- Place business logic in `src/services/`
- Use appropriate API endpoints in `src/api/`

### Best Practices
- Write unit tests for new features
- Document code changes
- Follow PEP 8 style guide
- Use type hints
- Handle errors appropriately

### Legacy Code
Legacy code is maintained in the `legacy/` directory for reference but is not actively used. New development should follow the current architecture in `src/`.

## Deployment

### Prerequisites
- Python 3.10+
- Required system libraries for audio processing
- Sufficient storage for model files
- GPU recommended for training

### Setup Steps
1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Configure environment variables
5. Initialize database
6. Run migrations
7. Start application

### Production Considerations
- Use appropriate WSGI server
- Configure proper logging
- Set up monitoring
- Implement backup strategy
- Configure SSL/TLS

## Future Development

### Planned Features
- Additional model architectures
- Enhanced data augmentation
- Improved real-time processing
- Extended API capabilities
- Advanced analytics

### Known Issues
- Document any current limitations
- List planned improvements
- Note performance considerations

## Support and Documentation
- API documentation available in `/docs`
- Training guide in `/docs/training`
- Model documentation in `/docs/models`
- Configuration guide in `/docs/config`

## Contributing
- Fork the repository
- Create feature branch
- Follow coding standards
- Submit pull request
- Include tests and documentation

This codebase represents a sophisticated sound classification system with emphasis on modularity, scalability, and maintainability. The architecture supports both research and production use cases, with clear separation of concerns and well-defined interfaces between components. 

Code Ontology - SoundClassifier v09

1. Core Models
src/core/models/base.py
    class BaseModel
        --> def preprocess_audio()
        --> def train()
        --> def predict()

src/core/models/cnn.py
    class CNNModel (inherits BaseModel)
        --> def build_model()
        --> def train()
        --> def predict()
        --> def preprocess_audio()

src/core/models/rf.py
    class RFModel (inherits BaseModel)
        --> def train()
        --> def predict()
        --> def extract_features()

src/core/models/ensemble.py
    class EnsembleModel (inherits BaseModel)
        --> def train()
        --> def predict()
        Uses: CNNModel, RFModel

2. Audio Processing
src/core/audio/processor.py
    class AudioProcessor
        --> def process_audio()
        --> def extract_features()
        --> def normalize_audio()

3. ML Components
src/ml/audio_processing.py
    --> def process_audio_for_cnn()
    --> def process_audio_for_rf()
    Uses: AudioProcessor

src/ml/feature_extractor.py
    class FeatureExtractor
        --> def extract_mfcc()
        --> def extract_spectral_features()
    Uses: audio_processing.py

src/ml/trainer.py
    class ModelTrainer
        --> def train_model()
        --> def validate_model()
        --> def save_model()
    Uses: CNNClassifier, RFClassifier, FeatureExtractor

src/ml/inference.py
    class InferenceEngine
        --> def predict()
        --> def batch_predict()
    Uses: CNNModel, RFModel, EnsembleModel

4. Services Layer
src/services/training_service.py
    class TrainingService
        --> def train_model_async()
        --> def get_training_status()
    Uses: ModelTrainer, DictionaryService

src/services/inference_service.py
    class InferenceService
        --> def predict()
        --> def get_prediction_stats()
    Uses: InferenceEngine, DictionaryService

src/services/dictionary_service.py
    class DictionaryService
        --> def get_dictionary()
        --> def update_dictionary()
        --> def validate_sounds()

src/services/user_service.py
    class UserService
        --> def authenticate()
        --> def register()
        --> def get_user_stats()

5. API Layer
src/api/ml_api.py
    --> @route('/api/ml/train')
    --> @route('/api/ml/predict')
    Uses: TrainingService, InferenceService

src/api/dictionary_api.py
    --> @route('/api/dictionary')
    Uses: DictionaryService

src/api/user_api.py
    --> @route('/api/user')
    Uses: UserService

src/api/dashboard_api.py
    --> @route('/api/dashboard')
    Uses: TrainingService, InferenceService

6. Routes Layer
src/routes/ml_routes.py
    --> @route('/train')
    --> @route('/predict')
    Uses: ml_api.py, dictionary_api.py
    Templates: train_model.html, predict.html

src/routes/train_app.py
    --> @route('/training')
    --> @route('/models')
    Uses: training_service.py, dictionary_service.py
    Templates: training.html, model_summary.html

7. Data Flow
User Request
    --> Routes Layer
        --> API Layer
            --> Services Layer
                --> ML Components
                    --> Core Components
                        --> Results
                            --> Services Layer
                                --> API Layer
                                    --> Routes Layer
                                        --> Templates
                                            --> User Response

8. Template Structure
templates/
├── base.html                    # Base template with common layout
├── model_management/
│   ├── model_summary.html      # Model details and statistics
│   ├── model_status.html       # Training status and progress
│   └── train_model.html        # Training interface
├── sound_management/
│   ├── sounds_management.html  # Sound file management
│   ├── upload_sounds.html     # Sound upload interface
│   └── record.html            # Sound recording interface
├── dictionary_management/
│   ├── dictionary_manager.html # Dictionary CRUD operations
│   └── manage_dictionaries.html # Dictionary list and management
├── analytics/
│   ├── view_analysis.html     # Analysis results
│   └── inference_statistics.html # Prediction statistics
└── user_interface/
    ├── dashboard.html         # Main user dashboard
    ├── login.html            # User login
    └── register.html         # User registration 