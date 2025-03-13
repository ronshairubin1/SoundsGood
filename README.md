# SoundsEasy Sound Classifier System

A comprehensive sound classification system for detecting and categorizing audio samples using multiple machine learning approaches.

## System Overview

This application enables audio classification through several ML model architectures:
- Convolutional Neural Networks (CNN)
- Random Forest (RF) 
- Ensemble models

The system provides a complete pipeline including:
1. Recording and preprocessing of sound files
2. Feature extraction
3. Model training
4. Inference and real-time detection
5. Analysis of model performance

## Directory Structure

```
SoundClassifier_v09/
├── backend/               # Enhanced backend components
│   ├── audio/             # Audio processing modules
│   │   └── processor.py   # Core SoundProcessor implementation
│   ├── data/              # Data storage
│   │   ├── features/      # Feature storage
│   │   └── models/        # Model storage (folder-based structure)
│   └── features/          # Feature extraction components
├── config.py              # Configuration settings
├── run.py                 # Application entry point
├── src/                   # Core application code
│   ├── api/               # API endpoints
│   ├── ml/                # Machine learning components
│   │   ├── model_paths.py # Model path resolution
│   │   └── ...
│   ├── routes/            # Flask route definitions
│   │   ├── ml_routes.py   # ML-related routes
│   │   └── ...
│   ├── services/          # Business logic
│   │   ├── inference_service.py  # Inference handling
│   │   ├── training_service.py   # Training handling
│   │   └── ...
│   └── templates/         # Frontend HTML templates
├── static/                # Static assets (CSS, JS, favicon)
├── docs/                  # Documentation
└── legacy/                # Legacy code (being phased out)
```

## Key Components

### Model Architecture

The system has been refactored to use a folder-based model storage structure:
- Models are stored in dedicated folders: `backend/data/models/{model_name}_{model_type}_{timestamp}/`
- This replaces the older approach of storing models directly in model_type subdirectories
- The system maintains a registry in a models.json file

### Core Services

#### Training Service (`src/services/training_service.py`)
- Handles model training workflows
- Manages feature extraction and caching
- Supports various augmentation strategies
- Produces training metrics and statistics

#### Inference Service (`src/services/inference_service.py`)
- Loads trained models
- Processes audio for predictions
- Supports real-time inference
- Provides confidence scores and statistics

### Web Interface

The application provides a Flask-based web interface with routes for:
- Training models (`/ml/train`)
- Making predictions (`/ml/predict`)
- Managing dictionaries/datasets
- Viewing model performance
- Real-time listening/detection

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries (install via `pip install -r requirements.txt`)
- Audio input capability for recording features

### Running the Application

1. Start the application:
   ```
   python run.py
   ```

2. Access the web interface at `http://localhost:5000`

### Training a Model

1. Navigate to the training page (`/ml/train`)
2. Select a dictionary/dataset
3. Choose model parameters
4. Start training

### Using a Trained Model

1. Navigate to the prediction page (`/ml/predict`)
2. Select a trained model
3. Either:
   - Upload audio for analysis
   - Use real-time recording

## Development Notes

### Current Cleanup Process

We are currently in the process of cleaning up the codebase:
- Legacy code is being moved to the `legacy/` directory
- The model storage structure has been enhanced to use dedicated folders
- Duplicate code is being consolidated
- New features are being added to the backend/ directory structure

### Code Organization Guidelines

When continuing development:
1. Place new feature code in the appropriate module under `backend/`
2. Update imports in existing files to reference these new locations
3. After confirming functionality, move deprecated code to `legacy/`
4. Update documentation to reflect changes

### Key Files for New Developers

- **`config.py`**: System-wide configuration settings
- **`src/ml/model_paths.py`**: Logic for locating and managing model files
- **`src/services/training_service.py`**: Core training workflow
- **`src/services/inference_service.py`**: Core inference workflow
- **`src/routes/ml_routes.py`**: Web interface routes for ML functions

## API Reference

The system provides several API endpoints:

### Training Endpoints
- `/ml/train` - Train a new model
- `/ml/training_status` - Check training progress

### Inference Endpoints
- `/ml/predict` - Make predictions on uploaded audio
- `/ml/predict_rf` - Make predictions with Random Forest models
- `/ml/predict_cnn` - Make predictions with CNN models
- `/ml/predict_ensemble` - Make predictions with Ensemble models

### Monitoring Endpoints
- `/ml/inference_statistics` - Get performance metrics
- `/ml/training_analysis` - Get training metrics and analysis

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path is correct and the model exists in `backend/data/models/`
2. **Audio processing errors**: Check that the sample rate matches the expected configuration
3. **Training failures**: Verify that the dataset is properly structured and contains sufficient examples

### Debugging

- Check application logs in the terminal
- Use browser developer tools for frontend issues
- For model-specific issues, examine the model's metadata JSON file

## Next Steps for Development

1. Complete the migration of legacy code to modern structure
2. Enhance documentation and add code comments
3. Improve test coverage
4. Optimize feature extraction pipeline
5. Enhance model comparison capabilities
