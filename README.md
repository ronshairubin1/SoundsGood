# Sound Classifier

A comprehensive speech recognition web application designed for speech therapy and analysis. This application provides tools for audio recording, processing, feature extraction, model training, and prediction.

## Overview

The Sound Classifier application is structured with a clean architecture that separates concerns into the following layers:

- **Core**: Contains the fundamental business logic and domain models
- **Services**: Implements application services that orchestrate core functionality
- **API**: Provides HTTP endpoints for interacting with the application
- **UI**: Web interface for users to interact with the system

## Project Structure

```
SoundClassifier/
├── main.py                # Main Flask application entry point
├── config.py              # Application configuration
├── requirements.txt       # Python dependencies
├── static/                # Static assets (CSS, JS, images)
│   ├── css/               # CSS files
│   ├── js/                # JavaScript files
│   └── img/               # Images
├── src/                   # Source code
│   ├── templates/         # HTML templates
│   ├── api/               # API routes and handlers
│   │   ├── __init__.py
│   │   ├── ml_api.py      # Machine learning API endpoints
│   │   ├── dictionary_api.py # Dictionary management endpoints
│   │   ├── user_api.py    # User management endpoints
│   │   └── dashboard_api.py # Dashboard data endpoints
│   ├── core/              # Core domain logic
│   │   ├── __init__.py
│   │   ├── audio/         # Audio processing
│   │   │   ├── __init__.py
│   │   │   └── processor.py  # Audio processing functionality
│   │   ├── features/      # Feature extraction
│   │   │   └── __init__.py
│   │   └── models/        # ML model definitions
│   │       ├── __init__.py
│   │       ├── base.py    # Base model interface
│   │       ├── cnn.py     # CNN model implementation
│   │       ├── rf.py      # Random Forest model implementation
│   │       └── ensemble.py # Ensemble model implementation
│   └── services/          # Application services
│       ├── __init__.py
│       ├── dictionary_service.py # Dictionary management service
│       ├── user_service.py # User management service
│       ├── training_service.py # Model training service
│       └── inference_service.py # Model inference service
├── dictionaries/          # Sound dictionaries storage
├── sounds/                # Sound samples organized by class
├── models/                # Directory for trained ML models
├── temp/                  # Temporary file storage
└── uploads/               # Directory for uploaded audio files
```

## Features

- **Audio Recording**: Record audio directly from the browser
- **Audio Processing**: Sound detection, centering, and normalization
- **Audio Chunking**: Split recordings into smaller chunks based on silence detection
- **Feature Extraction**: Extract features for different model types (CNN, RandomForest)
- **Model Training**: Train CNN, RandomForest, and Ensemble models
- **Prediction**: Make predictions using trained models
- **Analytics**: View and analyze prediction results

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/ronshairubin1/SoundsGood.git
   cd sound-classifier
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5001
   ```

## Usage

### Training a Model

1. Navigate to the Training page
2. Select a dictionary (class folder containing audio samples)
3. Choose model type (CNN, RandomForest, or Ensemble)
4. Configure training parameters
5. Start training
6. View training progress and results

### Making Predictions

1. Navigate to the Predict page
2. Record audio or upload an audio file
3. Select a trained model
4. Make a prediction
5. View prediction results with confidence scores

## API Documentation

The application provides RESTful API endpoints for training models, making predictions, and managing audio files:

### Training Endpoints

- `POST /api/ml/train`: Start model training
- `GET /api/ml/train/status`: Check training status
- `GET /api/ml/train/stats`: Get training statistics

### Prediction Endpoints

- `POST /api/ml/predict`: Make a prediction with an audio file
- `POST /api/ml/predict/batch`: Make predictions with multiple audio files
- `GET /api/ml/predict/last`: Get the last prediction result

### Model Management Endpoints

- `GET /api/ml/models`: List available trained models
- `DELETE /api/ml/models/<model_type>/<model_name>`: Delete a trained model

### Audio Analysis Endpoints

- `POST /api/ml/analyze`: Analyze audio characteristics without making a prediction

## Architecture

The application follows a clean architecture approach:

1. **Core Layer**: Contains domain entities and business rules without external dependencies
2. **Service Layer**: Orchestrates use cases and coordinates between core and API
3. **API Layer**: Exposes functionality through HTTP endpoints
4. **Infrastructure Layer**: Implements external concerns like storage and UI

Key components in the current architecture:

1. **Sound Dictionaries**: Collections of sound classes used for model training
2. **Sound Classes**: Categories of sounds (e.g., phonemes, words, environmental sounds)
3. **Dictionary Service**: Manages dictionary metadata and organization
4. **Audio Processing**: Handles recording, chunking, and preprocessing of audio
5. **User Management**: Handles authentication and user-specific settings
6. **Dashboard**: Provides overview of system status and statistics

This architecture ensures:
- Separation of concerns
- Testability
- Maintainability
- Flexibility to swap implementations

## License

This project is licensed under the MIT License - see the LICENSE file for details. 